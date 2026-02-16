"""
Real-Data RL Trading (PPO + GAE) with Regret vs Best Constant Rebalanced Portfolio (BCRP)

Install:
  pip install torch numpy pandas yfinance

Run:
  python rl_trading_ppo_realdata.py

Notes:
- Internet access is needed at runtime for yfinance. If you can't use yfinance, set USE_YFINANCE=False
  and provide a CSV with Date + one column per ticker of adjusted close prices.
"""

import math
import os
from dataclasses import dataclass
from typing import Dict, Tuple, Optional, List

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F

# ----------------------------
# Config
# ----------------------------

USE_YFINANCE = True  # set False to use CSV input
CSV_PATH = "prices.csv"  # used if USE_YFINANCE=False
TICKERS = ["SPY", "QQQ", "TLT"]  # multi-asset portfolio
START = "2012-01-01"
END = "2025-12-31"

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
SEED = 0

# Env
WINDOW = 40                      # lookback length for observations
TRAN_COST_BPS = 10               # turnover cost in basis points (10 bps = 0.001)
RISK_LAMBDA = 0.0                # try 0.05 or 0.1 later
EPS = 1e-12

# PPO training
TOTAL_UPDATES = 400              # number of PPO updates
ROLLOUT_LEN = 256                # steps collected per update
GAMMA = 0.99
LAMBDA_GAE = 0.95
CLIP_EPS = 0.20
ENTROPY_COEF = 0.01
VALUE_COEF = 0.5
LR = 3e-4
MAX_GRAD_NORM = 1.0
PPO_EPOCHS = 6
MINIBATCH_SIZE = 256

# Split (chronological)
TRAIN_FRAC = 0.70
VAL_FRAC = 0.15   # remaining is test


# ----------------------------
# Utils
# ----------------------------

def set_seed(seed: int):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def to_torch(x, device=DEVICE, dtype=torch.float32):
    return torch.tensor(x, device=device, dtype=dtype)

def softplus_min(x, min_val=1e-3):
    return F.softplus(x) + min_val

def simplex_project(v: np.ndarray) -> np.ndarray:
    """
    Euclidean projection onto the probability simplex {w>=0, sum w=1}.
    """
    v = v.astype(np.float64)
    n = v.size
    u = np.sort(v)[::-1]
    cssv = np.cumsum(u) - 1
    ind = np.arange(1, n + 1)
    cond = u - cssv / ind > 0
    if not np.any(cond):
        # fallback to uniform
        return np.ones(n) / n
    rho = ind[cond][-1]
    theta = cssv[cond][-1] / rho
    w = np.maximum(v - theta, 0)
    s = w.sum()
    if s <= 0:
        return np.ones(n) / n
    return w / s

def sharpe_ratio(x: np.ndarray, annualization: int = 252) -> float:
    x = np.asarray(x, dtype=np.float64)
    if x.size < 2:
        return 0.0
    mu = x.mean()
    sd = x.std(ddof=1)
    if sd < 1e-12:
        return 0.0
    return float((mu / sd) * math.sqrt(annualization))

def max_drawdown(equity: np.ndarray) -> float:
    equity = np.asarray(equity, dtype=np.float64)
    peak = np.maximum.accumulate(equity)
    dd = (equity / (peak + EPS)) - 1.0
    return float(dd.min())

# ----------------------------
# Data loader
# ----------------------------

def load_prices_yfinance(tickers: List[str], start: str, end: str) -> pd.DataFrame:
    import yfinance as yf
    df = yf.download(tickers, start=start, end=end, auto_adjust=True, progress=False)
    # yfinance returns multi-index columns for multiple tickers.
    if isinstance(df.columns, pd.MultiIndex):
        # prefer "Close" when auto_adjust True; sometimes it's "Close" still
        if ("Close" in df.columns.levels[0]):
            prices = df["Close"].copy()
        elif ("Adj Close" in df.columns.levels[0]):
            prices = df["Adj Close"].copy()
        else:
            # fallback: first level might be "Price"
            prices = df.xs(df.columns.levels[0][0], axis=1, level=0)
    else:
        # single ticker
        prices = df[["Close"]].copy()
        prices.columns = tickers
    prices = prices.dropna()
    prices = prices.astype(np.float64)
    return prices

def load_prices_csv(path: str, tickers: Optional[List[str]] = None) -> pd.DataFrame:
    df = pd.read_csv(path)
    if "Date" in df.columns:
        df["Date"] = pd.to_datetime(df["Date"])
        df = df.set_index("Date")
    df = df.sort_index()
    if tickers is not None:
        df = df[tickers]
    df = df.dropna()
    return df.astype(np.float64)

def prices_to_returns(prices: pd.DataFrame) -> pd.DataFrame:
    rets = prices.pct_change().dropna()
    return rets

# ----------------------------
# Environment
# ----------------------------

class PortfolioEnv:
    """
    Multi-asset portfolio rebalancing environment (no gym dependency).

    State:
      - lookback window of asset returns (WINDOW x N)
      - current portfolio weights (N,)
    Action:
      - new weights on simplex (N,) (Dirichlet policy outputs weights)
    Reward:
      - log portfolio growth minus turnover cost minus optional risk penalty

    Turnover cost:
      cost = tc * sum(|w_new - w_old|)
    """
    def __init__(self, returns: np.ndarray, window: int = 40,
                 trans_cost_bps: float = 10.0, risk_lambda: float = 0.0):
        self.returns = returns.astype(np.float32)  # shape [T, N]
        self.T, self.N = self.returns.shape
        self.window = window
        self.tc = float(trans_cost_bps) / 1e4
        self.risk_lambda = float(risk_lambda)
        self.reset()

    def reset(self, start_idx: Optional[int] = None):
        if start_idx is None:
            self.t = self.window
        else:
            self.t = max(self.window, int(start_idx))
        self.w = np.ones(self.N, dtype=np.float32) / self.N
        self.equity = 1.0
        self.equity_curve = [self.equity]
        return self._get_obs()

    def _get_obs(self):
        hist = self.returns[self.t - self.window:self.t]  # [W, N]
        # normalize returns within window (z-score per asset)
        mu = hist.mean(axis=0, keepdims=True)
        sd = hist.std(axis=0, keepdims=True) + 1e-6
        z = (hist - mu) / sd
        obs = {
            "ret_window": z.astype(np.float32),     # [W, N]
            "weights": self.w.copy().astype(np.float32)  # [N]
        }
        return obs

    def step(self, w_new: np.ndarray):
        w_new = np.asarray(w_new, dtype=np.float32)
        w_new = np.clip(w_new, 0.0, 1.0)
        s = w_new.sum()
        if s <= 0:
            w_new = np.ones(self.N, dtype=np.float32) / self.N
        else:
            w_new = w_new / s

        # turnover cost
        turnover = float(np.abs(w_new - self.w).sum())
        cost = self.tc * turnover

        rt = self.returns[self.t]  # [N]
        gross = 1.0 + float(np.dot(w_new, rt))
        gross = max(gross, 1e-6)

        # risk penalty: variance of last window portfolio returns (using previous weights or new)
        risk_pen = 0.0
        if self.risk_lambda > 0:
            hist = self.returns[self.t - self.window:self.t]
            port_hist = hist @ w_new
            risk_pen = float(np.var(port_hist))

        reward = float(np.log(gross) - cost - self.risk_lambda * risk_pen)

        self.equity *= gross * math.exp(-cost)  # apply cost multiplicatively
        self.equity_curve.append(self.equity)
        self.w = w_new

        self.t += 1
        done = (self.t >= self.T)
        info = {
            "equity": self.equity,
            "turnover": turnover,
            "cost": cost,
            "gross": gross,
        }
        return self._get_obs(), reward, done, info


# ----------------------------
# Best Constant Rebalanced Portfolio (BCRP) in hindsight
# ----------------------------

def bcrp_projected_grad_ascent(returns: np.ndarray, iters: int = 2000, lr: float = 0.05) -> Tuple[np.ndarray, float]:
    """
    Solve:
      maximize_w  sum_t log(1 + w^T r_t)
      s.t. w on simplex

    This is concave in w for gross returns > 0, so projected gradient ascent works well.

    returns: [T, N]
    returns value: (w*, best_portfolio_value)
    """
    T, N = returns.shape
    w = np.ones(N, dtype=np.float64) / N

    for _ in range(iters):
        g = 1.0 + returns @ w  # [T]
        g = np.maximum(g, 1e-8)
        # gradient of sum log(g_t) wrt w: sum_t r_t / g_t
        grad = (returns / g[:, None]).sum(axis=0)  # [N]
        # ascent step then project
        w = simplex_project(w + lr * grad)

    # compute value
    g = 1.0 + returns @ w
    g = np.maximum(g, 1e-8)
    pv = float(np.prod(g))
    return w.astype(np.float32), pv


# ----------------------------
# PPO Agent (Dirichlet policy for simplex actions)
# ----------------------------

class PolicyValueNet(nn.Module):
    def __init__(self, window: int, n_assets: int, hidden: int = 256):
        super().__init__()
        self.window = window
        self.n = n_assets
        in_dim = window * n_assets + n_assets  # flattened return window + current weights
        self.trunk = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
        )
        # policy head -> Dirichlet concentration params (alpha > 0)
        self.pi = nn.Linear(hidden, n_assets)
        # value head
        self.v = nn.Linear(hidden, 1)

    def forward(self, obs_ret_window: torch.Tensor, obs_weights: torch.Tensor):
        # obs_ret_window: [B, W, N], obs_weights: [B, N]
        x = torch.cat([obs_ret_window.reshape(obs_ret_window.size(0), -1), obs_weights], dim=-1)
        h = self.trunk(x)
        alpha = softplus_min(self.pi(h), 1e-3) + 1.0  # shift upward to avoid overly spiky early behavior
        value = self.v(h)
        return alpha, value

def dirichlet_logprob(alpha: torch.Tensor, a: torch.Tensor) -> torch.Tensor:
    # a is on simplex, shape [B, N]
    dist = torch.distributions.Dirichlet(alpha)
    return dist.log_prob(a)

def dirichlet_entropy(alpha: torch.Tensor) -> torch.Tensor:
    dist = torch.distributions.Dirichlet(alpha)
    return dist.entropy()

@dataclass
class RolloutBatch:
    obs_w: torch.Tensor          # [T, N]
    obs_x: torch.Tensor          # [T, W, N]
    act: torch.Tensor            # [T, N]
    logp: torch.Tensor           # [T]
    val: torch.Tensor            # [T]
    rew: torch.Tensor            # [T]
    done: torch.Tensor           # [T]
    adv: torch.Tensor            # [T]
    ret: torch.Tensor            # [T]

def compute_gae(rewards: torch.Tensor, values: torch.Tensor, dones: torch.Tensor,
                gamma: float, lam: float) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    rewards: [T]
    values: [T+1] (bootstrapped final)
    dones: [T] (1.0 if done else 0.0)
    """
    T = rewards.size(0)
    adv = torch.zeros(T, device=rewards.device)
    gae = 0.0
    for t in reversed(range(T)):
        nonterminal = 1.0 - dones[t]
        delta = rewards[t] + gamma * values[t + 1] * nonterminal - values[t]
        gae = delta + gamma * lam * nonterminal * gae
        adv[t] = gae
    ret = adv + values[:-1]
    return adv, ret

def collect_rollout(env: PortfolioEnv, net: PolicyValueNet, rollout_len: int) -> RolloutBatch:
    obs = env._get_obs()
    obs_x_list, obs_w_list = [], []
    act_list, logp_list, val_list = [], [], []
    rew_list, done_list = [], []

    for _ in range(rollout_len):
        x = to_torch(obs["ret_window"][None, ...])     # [1,W,N]
        w = to_torch(obs["weights"][None, ...])        # [1,N]
        with torch.no_grad():
            alpha, value = net(x, w)
            dist = torch.distributions.Dirichlet(alpha)
            a = dist.sample()                          # [1,N]
            logp = dist.log_prob(a)                    # [1]
        a_np = a.squeeze(0).cpu().numpy()

        obs2, r, done, info = env.step(a_np)

        obs_x_list.append(x.squeeze(0))
        obs_w_list.append(w.squeeze(0))
        act_list.append(a.squeeze(0))
        logp_list.append(logp.squeeze(0))
        val_list.append(value.squeeze(0).squeeze(-1))
        rew_list.append(torch.tensor(r, device=DEVICE, dtype=torch.float32))
        done_list.append(torch.tensor(1.0 if done else 0.0, device=DEVICE, dtype=torch.float32))

        obs = obs2
        if done:
            # if episode ends early, reset and continue collecting
            obs = env.reset()

    # bootstrap value for last state
    x = to_torch(obs["ret_window"][None, ...])
    w = to_torch(obs["weights"][None, ...])
    with torch.no_grad():
        _, v_last = net(x, w)
    v_last = v_last.squeeze(0).squeeze(-1)

    obs_x = torch.stack(obs_x_list, dim=0)  # [T,W,N]
    obs_w = torch.stack(obs_w_list, dim=0)  # [T,N]
    act = torch.stack(act_list, dim=0)      # [T,N]
    logp = torch.stack(logp_list, dim=0)    # [T]
    val = torch.stack(val_list, dim=0)      # [T]
    rew = torch.stack(rew_list, dim=0)      # [T]
    done = torch.stack(done_list, dim=0)    # [T]

    values_boot = torch.cat([val, v_last.view(1)], dim=0)  # [T+1]
    adv, ret = compute_gae(rew, values_boot, done, GAMMA, LAMBDA_GAE)

    # normalize advantages
    adv = (adv - adv.mean()) / (adv.std() + 1e-8)

    return RolloutBatch(obs_w=obs_w, obs_x=obs_x, act=act, logp=logp, val=val, rew=rew, done=done, adv=adv, ret=ret)

def ppo_update(net: PolicyValueNet, optim: torch.optim.Optimizer, batch: RolloutBatch):
    T = batch.act.size(0)
    idx = torch.randperm(T, device=DEVICE)

    for _ in range(PPO_EPOCHS):
        for start in range(0, T, MINIBATCH_SIZE):
            mb = idx[start:start + MINIBATCH_SIZE]
            x = batch.obs_x[mb]
            w = batch.obs_w[mb]
            a = batch.act[mb]
            old_logp = batch.logp[mb].detach()
            adv = batch.adv[mb].detach()
            ret = batch.ret[mb].detach()

            alpha, value = net(x, w)
            logp = dirichlet_logprob(alpha, a)
            entropy = dirichlet_entropy(alpha)

            ratio = torch.exp(logp - old_logp)
            surr1 = ratio * adv
            surr2 = torch.clamp(ratio, 1.0 - CLIP_EPS, 1.0 + CLIP_EPS) * adv
            policy_loss = -torch.min(surr1, surr2).mean()

            value_loss = F.mse_loss(value.squeeze(-1), ret)
            entropy_loss = -entropy.mean()

            loss = policy_loss + VALUE_COEF * value_loss + ENTROPY_COEF * entropy_loss

            optim.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(net.parameters(), MAX_GRAD_NORM)
            optim.step()

def eval_policy(env: PortfolioEnv, net: PolicyValueNet, deterministic: bool = True) -> Dict[str, float]:
    obs = env.reset()
    done = False
    rewards = []
    eq = [env.equity]
    port_rets = []

    while not done:
        x = to_torch(obs["ret_window"][None, ...])
        w = to_torch(obs["weights"][None, ...])
        with torch.no_grad():
            alpha, value = net(x, w)
            dist = torch.distributions.Dirichlet(alpha)
            if deterministic:
                # mean of Dirichlet = alpha / sum(alpha)
                a = (alpha / alpha.sum(dim=-1, keepdim=True)).squeeze(0)
            else:
                a = dist.sample().squeeze(0)
        a_np = a.cpu().numpy()
        obs, r, done, info = env.step(a_np)
        rewards.append(r)
        eq.append(info["equity"])
        port_rets.append(info["gross"] - 1.0)

    rewards = np.array(rewards, dtype=np.float64)
    eq = np.array(eq, dtype=np.float64)
    port_rets = np.array(port_rets, dtype=np.float64)

    return {
        "log_return": float(rewards.sum()),
        "final_equity": float(eq[-1]),
        "sharpe": sharpe_ratio(port_rets),
        "max_drawdown": max_drawdown(eq),
    }


# ----------------------------
# Main
# ----------------------------

def split_returns(rets: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    T = len(rets)
    i_train = int(T * TRAIN_FRAC)
    i_val = int(T * (TRAIN_FRAC + VAL_FRAC))
    r_train = rets.iloc[:i_train].to_numpy()
    r_val = rets.iloc[i_train:i_val].to_numpy()
    r_test = rets.iloc[i_val:].to_numpy()
    return r_train, r_val, r_test

def main():
    set_seed(SEED)

    # ---- load data
    if USE_YFINANCE:
        prices = load_prices_yfinance(TICKERS, START, END)
    else:
        prices = load_prices_csv(CSV_PATH, tickers=TICKERS)

    rets = prices_to_returns(prices)
    rets = rets.replace([np.inf, -np.inf], np.nan).dropna()
    assert rets.shape[0] > (WINDOW + 300), "Not enough data after cleaning."

    r_train, r_val, r_test = split_returns(rets)
    n_assets = r_train.shape[1]

    # ---- environments
    env_train = PortfolioEnv(r_train, window=WINDOW, trans_cost_bps=TRAN_COST_BPS, risk_lambda=RISK_LAMBDA)
    env_val = PortfolioEnv(r_val, window=WINDOW, trans_cost_bps=TRAN_COST_BPS, risk_lambda=RISK_LAMBDA)
    env_test = PortfolioEnv(r_test, window=WINDOW, trans_cost_bps=TRAN_COST_BPS, risk_lambda=RISK_LAMBDA)

    # ---- model
    net = PolicyValueNet(window=WINDOW, n_assets=n_assets).to(DEVICE)
    optim = torch.optim.Adam(net.parameters(), lr=LR)

    # ---- training loop
    history = []
    for upd in range(1, TOTAL_UPDATES + 1):
        batch = collect_rollout(env_train, net, ROLLOUT_LEN)
        ppo_update(net, optim, batch)

        if upd % 20 == 0 or upd == 1:
            val_stats = eval_policy(env_val, net, deterministic=True)
            history.append({"update": upd, **val_stats})
            print(
                f"Upd {upd:04d} | "
                f"VAL equity={val_stats['final_equity']:.3f} | "
                f"VAL logret={val_stats['log_return']:.4f} | "
                f"VAL Sharpe={val_stats['sharpe']:.2f} | "
                f"VAL MDD={val_stats['max_drawdown']:.2%}"
            )

    # ---- final evaluation on test
    test_stats = eval_policy(env_test, net, deterministic=True)

    # ---- regret vs BCRP on test
    w_star, bcrp_pv = bcrp_projected_grad_ascent(r_test.astype(np.float64), iters=2500, lr=0.03)
    agent_pv = test_stats["final_equity"]
    regret_log = float(np.log(bcrp_pv + EPS) - np.log(agent_pv + EPS))

    print("\n=== TEST RESULTS ===")
    print(f"Agent final equity: {agent_pv:.4f}")
    print(f"Agent log return  : {test_stats['log_return']:.6f}")
    print(f"Agent Sharpe      : {test_stats['sharpe']:.3f}")
    print(f"Agent max drawdown: {test_stats['max_drawdown']:.2%}")
    print("\n=== REGRET vs BCRP (hindsight optimum constant weights) ===")
    print(f"BCRP weights      : {np.round(w_star, 4)}  (tickers={TICKERS})")
    print(f"BCRP final equity : {bcrp_pv:.4f}")
    print(f"Log-regret        : {regret_log:.6f}")

    # ---- save a small CSV for resume metrics
    out = pd.DataFrame(history)
    out.to_csv("val_history.csv", index=False)
    print("\nSaved validation history to val_history.csv")

if __name__ == "__main__":
    main()
