# StableWatch
**Stablecoin Depeg Risk Monitor** — FRE 6871 Final Project

> Can macroeconomic stress indicators and stablecoin price behavior predict depeg events before they occur?

Applies statistical modeling and machine learning to five stablecoins (USDC, USDT, DAI, FRAX, UST) over 2020–2023. Primary validation: UST collapse, May 2022 (~$40B wiped in under a week).

---

## Quickstart

**1. Run the full analysis pipeline**
```r
source("StableWatch.R")
```
Pulls data from APIs, runs all modeling sections, writes processed data and model artifacts to disk.

**2. Render the report**
```r
rmarkdown::render("StableWatch.Rmd")
```

**3. Launch the interactive dashboard**
```r
shiny::runApp("shiny/")
```
Requires step 1 to have run first.

---

## Data Sources

| Source | What | Access |
|--------|------|--------|
| [CoinGecko API](https://www.coingecko.com/en/api) | Daily price, volume, market cap | Free (demo key) |
| [FRED API](https://fred.stlouisfed.org) | DXY, SOFR, VIX, HY spreads | Free (API key required) |

---

## Methods

- OLS & logistic regression (depeg prediction)
- Bootstrap resampling, ANOVA
- ARIMA + GARCH (volatility modeling)
- PCA, K-means clustering (regime detection)
- XGBoost + SHAP (classification + explainability)
- MICE (missing data imputation)
- PerformanceAnalytics (drawdown, Sharpe, CVaR)

---

## Project Structure

```
StableWatch/
├── StableWatch.R        # Main analysis pipeline (Sections 0–16)
├── StableWatch.Rmd      # R Markdown report source
├── shiny/
│   ├── app.R            # Entry point
│   ├── ui.R             # Dashboard layout
│   └── server.R         # Reactive logic + model loading
├── data/
│   ├── raw/             # API pulls (.rds) — gitignored
│   └── processed/       # stablewatch_master.rds — gitignored
└── output/
    ├── models/          # Fitted model objects (.rds) — gitignored
    ├── plots/           # All figures (.png)
    └── tables/          # Summary tables (.csv)
```
