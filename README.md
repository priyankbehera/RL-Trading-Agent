
## Problem Setting

This project implements:

- Multi-asset portfolio allocation
- PPO with clipped objective
- GAE for stable advantage estimation
- Dirichlet policy (naturally outputs valid portfolio weights)
- Transaction cost modeling
- Optional risk penalty
- Chronological train / validation / test split
- Regret analysis vs BCRP benchmark
- Sharpe ratio & Maximum Drawdown evaluation

At each timestep the agent:

1. Observes a rolling window of past asset returns  
2. Chooses new portfolio weights (sum to 1, no shorting)  
3. Pays transaction costs proportional to turnover  
4. Receives reward:

    r_t = log(1 + w_t^T r_t) − transaction_cost − λ * risk_penalty

The objective is long-term growth of capital.

## Benchmark: BCRP (Best Constant Rebalanced Portfolio)

We compute the hindsight optimal constant weights:

    max_w  Σ_t log(1 + w^T r_t)

using projected gradient ascent.

We report log regret:

    Log Regret = log(V_BCRP) − log(V_Agent)

Lower regret is better.
