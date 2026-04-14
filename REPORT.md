# Research Report

**Project Title:** Standardized Abnormal Trading Volume (SATV)  
**Author(s):** Maxwell, Sam  
**Date:** April 9, 2026  
**Version:** Draft 1  

---

## 1. Summary

This project studies whether **Standardized Abnormal Trading Volume (SATV)** can predict future cross-sectional equity returns and whether it can be used in a practical systematic trading framework. The research started as a CRSP-based proof of concept using portfolio sorts and then expanded into a more realistic implementation using Barra-style daily data, cross-sectional standardization, and alpha generation for mean-variance portfolio optimization.

The core idea is that when a stock’s trading activity becomes unusually high relative to its own historical norm, that abnormality may reflect information arrival, investor attention, disagreement, or temporary price pressure. If prices do not fully adjust immediately, this abnormal volume may contain predictive information about future returns.

The main takeaway is that SATV is strongest as a **short-horizon standalone signal**. Daily performance is very strong, but the signal weakens noticeably at weekly and monthly frequencies. That decay pattern suggests SATV mostly captures fresh, short-lived effects rather than a slow-moving characteristic. However, SATV still appears valuable at the monthly horizon when used as a **conditioning variable for momentum**, since the interaction term between standardized SATV and momentum performs much better than standalone monthly SATV.

Overall, the evidence suggests SATV is a serious candidate for inclusion in a systematic equity strategy, both as a standalone short-horizon alpha and as a feature that improves slower-moving signals like momentum. :contentReference[oaicite:0]{index=0} :contentReference[oaicite:1]{index=1}

### Key Metrics

| Metric | Value | Notes |
|------|------|------|
| Primary Metric | Daily Sharpe = 3.35 gross / 2.40 after costs | Best standalone SATV result; remains strong after transaction cost adjustment |
| Secondary Metric | Weekly Sharpe = 0.93 | Performance declines materially at weekly frequency |
| Other | Monthly Sharpe = 0.46 | Standalone monthly SATV is weak |
| Other | Monthly SATV × Momentum Sharpe = 1.51 | Interaction performs much better than standalone monthly SATV |
| Other | IC = 0.0068 | Used in alpha scaling framework |
| Other | Turnover = 3.15 | Daily SATV backtest turnover in the cost-adjusted result | :contentReference[oaicite:2]{index=2} :contentReference[oaicite:3]{index=3}

## 2. Data Requirements

Describe data dependencies.

**Sources**
- CRSP-style equity data for initial proof-of-concept portfolio sorts
- Barra-style daily asset data for the full production-style implementation and optimization framework :contentReference[oaicite:4]{index=4}

**Rate of Availability**
- Daily data for signal construction and main backtests
- Weekly and monthly rebalancing versions derived from the daily framework for horizon comparison :contentReference[oaicite:5]{index=5}

**Inputs Required**
- Returns
- Prices
- Daily trading volume
- Market capitalization
- Predicted beta
- Specific risk
- Shares outstanding, or enough information to infer shares outstanding from market capitalization and price :contentReference[oaicite:6]{index=6}

**Preprocessing**
- Apply a $5 price filter to exclude low-priced securities more affected by microstructure distortions
- Require non-missing signal values, predicted beta, and specific risk before names enter the backtest
- Compute turnover as volume divided by shares outstanding
- Estimate rolling historical turnover mean and standard deviation using roughly one trading year of history
- Shift the rolling window by 21 trading days to avoid recent contamination and look-ahead bias
- Clip extreme values and standardize the signal cross-sectionally by date before portfolio construction :contentReference[oaicite:7]{index=7}

---

## 3. Approach / System Design

This project builds and evaluates a signal based on **abnormal turnover**. The economic intuition is that trading volume spikes can reflect information arrival, changing investor attention, disagreement, or speculative intensity. If prices adjust gradually, abnormal volume may help forecast future returns. SATV may also improve momentum by identifying when momentum is supported by meaningful information flow rather than simple drift. :contentReference[oaicite:8]{index=8}

The signal construction process is:

- Compute **turnover** as  
  \[
  \text{Turnover}_{i,t} = \frac{\text{Volume}_{i,t}}{\text{Shares Outstanding}_{i,t}}
  \]

- Approximate shares outstanding as  
  \[
  \text{Shares Outstanding}_{i,t} \approx \frac{\text{Market Cap}_{i,t}}{\text{Price}_{i,t}}
  \]

- Define SATV as the standardized deviation of current turnover from its trailing historical distribution using a lagged rolling window

- Clip and standardize SATV cross-sectionally by date for use in portfolio construction

- Construct momentum using cumulative log returns over approximately one year, excluding the most recent month

- Form the interaction signal as the product of cross-sectionally standardized SATV and momentum:
  \[
  \text{Interaction}_{i,t} = Z^{SATV}_{i,t} \cdot Z^{MOM}_{i,t}
  \]

For portfolio implementation, the cross-sectional score is mapped into an expected return forecast using an assumed information coefficient and specific risk:
\[
\alpha_{i,t} = \text{score}_{i,t} \cdot IC \cdot \sigma^{specific}_{i,t}
\]

This makes the signal directly usable inside a mean-variance optimization framework instead of only through simple sorts. The design tradeoff is clear: daily SATV captures the strongest effect, but also creates the highest turnover and the greatest sensitivity to trading costs. Monthly standalone SATV is weak, but monthly SATV × momentum is much more promising. :contentReference[oaicite:9]{index=9} :contentReference[oaicite:10]{index=10}

---

## 4. Code Structure

If signal research:

```text
sf-signal/
├── src/
│   ├── framework/
│   │   ├── ew_dash.py            # Equal-weight dashboard (do not edit)
│   │   ├── opt_dash.py           # Optimal portfolio dashboard (do not edit)
│   │   └── run_backtest.py       # Run the backtest (edit config only)
│   └── signal/
│       └── create_signal.py      # Your signal implementation (edit this)
├── data/
│   ├── signal.parquet            # Output: Your signal
│   └── weights/                  # Output: Backtest weights
└── README.md
