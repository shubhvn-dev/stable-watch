# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

StableWatch is an academic R project (FRE 6871 final) analyzing stablecoin depeg risk using macroeconomic indicators. Primary validation event: UST collapse May 2022. Coins: USDC, USDT, DAI, FRAX, UST. Period: 2020–2023.

## Running the Analysis

**Full pipeline (required before Shiny app):**
```r
source("StableWatch.R")
```
This ingests data from APIs, runs all 17 sections, and writes artifacts to `data/processed/`, `output/models/`, `output/plots/`, `output/tables/`.

**Render the report:**
```r
rmarkdown::render("StableWatch.Rmd")
```

**Launch the Shiny dashboard:**
```r
shiny::runApp("shiny/")
```
Must run `StableWatch.R` first — Shiny loads pre-built `.rds` model files.

## Architecture

### Pipeline flow
```
StableWatch.R  →  data/processed/stablewatch_master.rds
                  output/models/{lm_macro, lm_full, glm_macro, glm_full,
                                 xgb_depeg, kmeans_clusters, kmeans_regimes}.rds
                  output/plots/*.png
                  output/tables/*.csv
```

### Key data object: `master`
Central `data.frame` built in Section 3. Columns include:
- `date`, `coin` — time + stablecoin identifier
- `price`, `peg_dev` — raw price and deviation from $1 peg
- `peg_dev_z`, `vol_7d`, `vol_30d`, `vol_spike` — engineered features
- `dxy`, `sofr`, `vix` — macro indicators from FRED
- `depeg` — binary target (1 = peg deviation > threshold)
- `coin_num` — integer encoding of `coin` factor (used by XGBoost)

### Shiny app (`shiny/`)
Split into `ui.R` + `server.R`, launched by `app.R`. The server loads `master` and model `.rds` files via relative paths (`../data/processed/`, `../output/models/`). Must be run from the `shiny/` directory or via `shiny::runApp("shiny/")` from project root.

### Section structure of `StableWatch.R`
| Section | Content |
|---------|---------|
| 0 | Setup, packages, API keys, output dirs |
| 1 | Data ingestion (CoinGecko + FRED APIs) |
| 2 | Cleaning, preprocessing |
| 3 | Feature engineering → `master` |
| 4 | Descriptive stats, basic plots |
| 5 | Statistical tests |
| 6 | OLS regression + ANOVA |
| 7 | Bootstrap / resampling |
| 8 | GLM / logistic regression |
| 9 | Intermediate plots |
| 10 | PCA |
| 11 | ARIMA + GARCH |
| 12 | Missing data (MICE imputation) |
| 13 | K-means cluster analysis |
| 14 | XGBoost + SHAP explainability |
| 15 | Performance analytics |
| 16 | Conclusion |

## API Keys

Stored at the top of `StableWatch.R` and mirrored in the Rmd setup chunk:
- `CG_API_KEY` — CoinGecko (demo tier, free)
- `FRED_API_KEY` — FRED (free at fred.stlouisfed.org)

Both are set in-script. Replace if regenerated.

## Package Conflicts

Many packages overlap on function names. Conflicts are resolved explicitly via `conflicted::conflict_prefer()` in Section 0. Key rules:
- `filter`, `lag`, `select`, `recode` → `dplyr`
- `first`, `last` → `xts`
- `skewness`, `kurtosis` → `moments` (use `PerformanceAnalytics::skewness()` explicitly when needed)
- `cbind`, `rbind` → `base`
- `reduce` → `rugarch` (use `purrr::reduce()` explicitly when needed)

## Data / Output Git Behavior

`data/raw/`, `data/processed/`, `output/models/` are gitignored — only `.gitkeep` placeholders are tracked. `output/plots/` and `output/tables/` PNG/CSV files **are** tracked.
