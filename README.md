# Pocket Hedge v2  
A lightweight Python engine that evaluates multiple hedge allocations and identifies the most efficient downside protection for a given portfolio.

## Overview
Pocket Hedge v2 extends the earlier JS-based portfolio analyzer and focuses on **systematic hedge evaluation**.  
It compares hedge combinations, measures their impact on drawdowns and CAGR, and ranks them using a simple risk-efficiency score.

This version:
- Builds all valid hedge-weight combinations under user-defined constraints
- Adjusts the base portfolio to allocate hedge exposure
- Computes CAGR, max drawdown, and hedge efficiency
- Returns the best-performing hedge strategy with full weight breakdown

## Features
- **Automatic Hedge Generation** — inverse ETFs, bonds, gold, etc.  
- **Risk Metrics** — log returns, CAGR, max drawdown, downside improvement  
- **Scoring Function** — evaluates drawdown reduction vs. growth cost  
- **Visualization** — optional cumulative return comparison (matplotlib)

## Hedge Scoring (Intuition)
A hedge is considered “better” if:
- It meaningfully **reduces drawdowns**, and  
- The **CAGR penalty** is relatively small.

A simple ratio is used:

```
score = (drawdown_improvement) / (CAGR_loss + eps)
```

Higher score → more efficient protection.

## Usage
```
result = run_engine(
    portfolio=PORTFOLIO,
    hedge_universe=HEDGE_UNIVERSE,
    max_hedge=0.2,
    start="2024-01-01",
    end="2025-01-01"
)

pretty_print_result(result)
```

Example output includes:
- Best hedge strategy  
- Score breakdown  
- Weight allocation  
- Sorted comparison table  

## Input Structure
### Portfolio example
```
PORTFOLIO = [
    {"ticker": "BTC-USD", "amount": 1000},
    {"ticker": "NVDA",    "amount": 500},
    {"ticker": "TSLA",    "amount": 1200},
    {"ticker": "AAPL",    "amount": 900},
]
```

### Hedge universe example
```
HEDGE_UNIVERSE = [
    {"ticker": "SH",  "type": "inverse", "max_weight": 0.4},
    {"ticker": "TLT", "type": "bond",    "max_weight": 0.6},
    {"ticker": "GLD", "type": "gold",    "max_weight": 0.5},
]
```

## Output Example
```
===== Best Hedge Strategy =====
Name   : SH_20%
Score  : 14.2381
DD Gain: 12.4%
CAGR Δ : 2.1%
Weights:
  - BTC-USD: 0.3140
  - NVDA:    0.1570
  - TSLA:    0.3768
  - AAPL:    0.1410
  - SH:      0.2000
```

## Requirements
- Python 3.9+
- numpy, pandas, yfinance, matplotlib

Install:
```
pip install numpy pandas yfinance matplotlib
```

## Notes
- Data is sourced through Yahoo Finance (`yfinance`).
- This project focuses on structural risk analysis, not trading signals.
- Hedge scoring is intentionally simple for interpretability.

