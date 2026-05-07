# ============================================================
# StableWatch: Stablecoin Depeg Risk Monitor
# FRE 6871 — Final Project
# ============================================================
#
# RESEARCH QUESTION:
# Can macroeconomic stress indicators and stablecoin on-chain
# behavior predict depeg events before they occur — and did
# these signals fire ahead of the UST collapse in May 2022?
#
# DATA SOURCES:
#   1. CoinGecko API (coingecko.com/en/api) — free, public
#      Provides daily price, volume, and market cap for
#      USDC, USDT, DAI, FRAX, and UST (2020–2023)
#
#   2. FRED API (fred.stlouisfed.org) — free, requires key
#      Provides macroeconomic indicators: DXY, SOFR, and
#      high-yield credit spreads as market stress proxies#
# STRUCTURE:
#   Section 0  — Setup & Configuration
#   Section 1  — Data Ingestion
#   Section 2  — Data Cleaning & Preprocessing
#   Section 3  — Feature Engineering
#   Section 4  — Basic Graphs & Descriptive Statistics
#   Section 5  — Statistical Tests
#   Section 6  — Regression & ANOVA
#   Section 7  — Resampling & Bootstrapping
#   Section 8  — GLM / Logistic Regression
#   Section 9  — Intermediate Graphs
#   Section 10 — Principal Component Analysis
#   Section 11 — Time Series Modeling
#   Section 12 — Missing Data
#   Section 13 — Cluster Analysis
#   Section 14 — Classification & ML (XGBoost + SHAP)
#   Section 15 — Performance Analytics
#   Section 16 — Conclusion & Insights
# ============================================================


# ============================================================
# SECTION 0 — SETUP & CONFIGURATION
# ============================================================
#
# Before any analysis can begin, we need to:
#   1. Install and load all required packages
#   2. Set our API credentials
#   3. Create the output folder structure
#   4. Define global parameters used throughout the analysis
#
# Running this section first ensures that every subsequent
# section has the tools and paths it needs without errors.
# ============================================================

# --- 0.1 Install & Load Packages ----------------------------
#
# We check whether each package is already installed before
# attempting to install it. This prevents unnecessary
# reinstallation on repeated runs and is considered best
# practice in reproducible R workflows.

packages <- c(
  # Data ingestion & API
  "httr2",          # Modern HTTP requests for CoinGecko API
  "jsonlite",       # Parse JSON responses from APIs
  "fredr",          # FRED API wrapper
  
  # Data manipulation
  "tidyverse",      # Core data wrangling (dplyr, tidyr, purrr)
  "lubridate",      # Date/time handling
  "zoo",            # Rolling calculations and time series tools
  "xts",            # Extensible time series objects
  
  # Statistics & modeling
  "moments",        # Skewness and kurtosis
  "car",            # Levene's test, ANOVA utilities
  "boot",           # Bootstrap resampling
  "forecast",       # ARIMA modeling
  "rugarch",        # GARCH modeling
  "mice",           # Multiple imputation for missing data
  "naniar",         # Visualize missing data patterns
  
  # Machine learning
  "xgboost",        # XGBoost classification
  "caret",          # Train/test split, confusion matrix
  "pROC",           # ROC curves and AUC
  "shapviz",        # SHAP explainability for XGBoost
  
  # Dimensionality reduction
  "factoextra",     # PCA visualization (scree plot, biplot)
  
  # Clustering
  "cluster",        # K-means and cluster diagnostics
  
  # Performance analytics
  "PerformanceAnalytics",  # Financial performance metrics
  "quantmod",              # Download financial data
  
  # Visualization
  "ggplot2",        # Core plotting (part of tidyverse)
  "ggcorrplot",     # Correlation heatmaps
  "patchwork",      # Combine multiple ggplots
  "scales",         # Axis formatting helpers
  "knitr",          # Tables for report output
  "kableExtra"      # Enhanced table formatting
)

# Install any packages not yet on the machine
new_packages <- packages[!(packages %in% installed.packages()[, "Package"])]
if (length(new_packages) > 0) {
  message("Installing missing packages: ", paste(new_packages, collapse = ", "))
  install.packages(new_packages, dependencies = TRUE)
}

# Load all packages silently
invisible(lapply(packages, library, character.only = TRUE))

# --- 0.1b Resolve Package Conflicts --------------------------
#
# When multiple packages export functions with the same name,
# R uses whichever was loaded last — which can cause silent
# bugs that are hard to trace. We resolve every conflict
# explicitly here so behavior is predictable throughout the
# entire script regardless of load order.
#
# We use the conflicted package to declare the winner for each
# conflicted function name. Any unresolved conflict will throw
# a clear error rather than silently using the wrong function.

library(conflicted)

# dplyr wins for data manipulation verbs
conflict_prefer("filter",     "dplyr")
conflict_prefer("lag",        "dplyr")
conflict_prefer("select",     "dplyr")
conflict_prefer("recode",     "dplyr")
conflict_prefer("group_rows", "dplyr")

# xts wins for time series first/last
conflict_prefer("first", "xts")
conflict_prefer("last",  "xts")

# stats wins for foundational stats functions
conflict_prefer("cov",    "stats")
conflict_prefer("var",    "stats")
conflict_prefer("smooth", "stats")

# moments wins for skewness/kurtosis
# PerformanceAnalytics versions will be called explicitly as
# PerformanceAnalytics::skewness() where needed
conflict_prefer("skewness", "moments")
conflict_prefer("kurtosis", "moments")

# boot wins for logit (over car)
conflict_prefer("logit", "boot")

# scales wins for discard (over purrr)
conflict_prefer("discard", "scales")

# base wins for cbind/rbind (over mice)
conflict_prefer("cbind", "base")
conflict_prefer("rbind", "base")

# rugarch wins for reduce
# purrr::reduce() will be called explicitly where needed
conflict_prefer("reduce", "rugarch")

# graphics wins for legend (over PerformanceAnalytics)
conflict_prefer("legend", "graphics")

# suppress the xts lag warning — we will use stats::lag()
# explicitly in the time series section
options(xts.warn_dplyr_breaks_lag = FALSE)

message("All packages loaded and conflicts resolved.")


# --- 0.2 API Credentials ------------------------------------
#
# We store API keys as variables at the top of the script so
# they are easy to find and update. Never hard-code credentials
# deep inside functions — this is a standard practice in
# production data pipelines.
#
# IMPORTANT: Replace these with your own keys if regenerated.
# CoinGecko demo key: free tier, sufficient for this project.
# FRED key: free at fred.stlouisfed.org

CG_API_KEY   <- "CG-GycdehXyHNXoy2EwaicA8d4c"    # CoinGecko
FRED_API_KEY <- "ebb6ec9000ecec7d134427d007962e6c" # FRED

# Register FRED key with the fredr package
fredr_set_key(FRED_API_KEY)

message("API credentials configured.")


# --- 0.3 Folder Structure -----------------------------------
#
# We programmatically create all output directories the script
# needs. Using showWarnings = FALSE means no error is thrown
# if the folder already exists — safe to run multiple times.

dirs <- c(
  "data/raw",
  "data/processed",
  "output/plots",
  "output/tables",
  "output/models",
  "report"
)

invisible(lapply(dirs, dir.create, recursive = TRUE, showWarnings = FALSE))

message("Output folders ready.")


# --- 0.4 Global Parameters ----------------------------------
#
# Centralizing parameters here means we only need to change
# them in one place if we want to update the analysis window,
# add a new coin, or adjust the depeg threshold.

# Date range for the full analysis
START_DATE <- as.Date("2020-01-01")
END_DATE   <- as.Date("2023-12-31")

# Stablecoins to analyze with their CoinGecko IDs
COINS <- c(
  "usd-coin",          # USDC
  "tether",            # USDT
  "dai",               # DAI
  "frax",              # FRAX
  "terrausd"           # UST — the collapsed algorithmic stablecoin
)

# Human-readable labels for plotting
COIN_LABELS <- c(
  "usd-coin"  = "USDC",
  "tether"    = "USDT",
  "dai"       = "DAI",
  "frax"      = "FRAX",
  "terrausd"  = "UST"
)

# FRED series IDs for macroeconomic variables
FRED_SERIES <- c(
  "DTWEXBGS",  # DXY — Nominal Broad US Dollar Index
  "SOFR",      # SOFR — Secured Overnight Financing Rate
  "VIXCLS"     # VIX — CBOE Volatility Index (market fear gauge)
  # Replaces BAMLH0A0HYM2 which only starts May 2023
)

# Depeg threshold: a stablecoin is considered "depegged"
# if its price deviates more than 0.5% from $1.00
# This is the standard threshold used by risk desks at
# stablecoin protocols and crypto exchanges
DEPEG_THRESHOLD <- 0.005

# Known stress events — used to annotate plots and validate models
STRESS_EVENTS <- data.frame(
  label = c("UST Collapse", "USDC/SVB Scare"),
  date  = as.Date(c("2022-05-09", "2023-03-10")),
  coin  = c("UST", "USDC")
)

message("Global parameters set.")
message("--- StableWatch Setup Complete ---")
message("Analysis window : ", START_DATE, " to ", END_DATE)
message("Coins tracked   : ", paste(COIN_LABELS, collapse = ", "))
message("Depeg threshold : ", DEPEG_THRESHOLD * 100, "%")
message("Ready to begin data ingestion.")

# ============================================================
# SECTION 1 — DATA INGESTION
# (See Chapter 1: Getting Started with R — Importing Data)
# ============================================================
#
# Our analysis requires two data sources:
#
#   1. CoinGecko API — daily price, volume, and market cap
#      for five stablecoins: USDC, USDT, DAI, FRAX, and UST.
#      This is our primary dataset. Stablecoins are designed
#      to maintain a $1.00 peg, so any deviation from that
#      value is our core signal of interest.
#
#   2. FRED API — macroeconomic context variables: the US
#      Dollar Index (DXY), the Secured Overnight Financing
#      Rate (SOFR), and a high-yield credit spread. These
#      tell us about broader financial stress conditions that
#      may coincide with or predict stablecoin stress events.
#
# We implement local caching: the first run pulls from the
# APIs and saves the results as .rds files. Every subsequent
# run loads from disk instead. This means:
#   - No rate limiting during iterative development
#   - The script runs fast after the first pull
#   - The script works offline after the first run
#
# To force a fresh API pull, delete the files in data/raw/.
# ============================================================


# --- 1.1 Stablecoin Price Data via Yahoo Finance ------------
#
# The CoinGecko demo API restricts historical data to the past
# 365 days, which excludes the UST collapse in May 2022 — the
# central event of our analysis. We therefore source price and
# volume data from Yahoo Finance via the quantmod package, which
# provides free, unlimited historical daily OHLCV data for
# crypto pairs going back to each coin's listing date.
#
# Yahoo Finance ticker format for crypto: SYMBOL-USD
# e.g. USDT-USD, USDC-USD, DAI-USD, FRAX-USD, UST-USD
#
# We extract the daily closing price and volume for each coin,
# convert to a tidy long-format dataframe, and cache locally.
#
# Note: Yahoo Finance does not provide market cap data. For our
# analysis this is not a limitation — peg deviation, rolling
# volatility, and volume-price divergence are all derivable
# from price and volume alone.
# (See Chapter 1: Getting Started with R — Importing Data)

# Yahoo Finance tickers corresponding to our five stablecoins
TICKERS <- c(
  "USDC-USD",  # USDC
  "USDT-USD",  # USDT
  "DAI-USD",   # DAI
  "FRAX-USD",  # FRAX
  "UST-USD"    # UST (Terra USD — collapsed May 2022)
)

# Map tickers back to our coin labels for consistency
TICKER_LABELS <- c(
  "USDC-USD" = "USDC",
  "USDT-USD" = "USDT",
  "DAI-USD"  = "DAI",
  "FRAX-USD" = "FRAX",
  "UST-USD"  = "UST"
)

fetch_yahoo <- function(ticker) {
  # getSymbols() with auto.assign = FALSE returns the xts object
  # directly rather than assigning it to an environment. This is
  # the cleanest approach for programmatic use inside a function.
  # Cl() and Vo() are quantmod helpers that extract the Close and
  # Volume columns regardless of how the columns are named.
  message("  Fetching Yahoo Finance data for: ", ticker)
  
  tryCatch({
    xts_obj <- getSymbols(
      ticker,
      src         = "yahoo",
      from        = START_DATE,
      to          = END_DATE,
      auto.assign = FALSE
    )
    
    df <- data.frame(
      date   = as.Date(index(xts_obj)),
      price  = as.numeric(Cl(xts_obj)),
      volume = as.numeric(Vo(xts_obj)),
      ticker = ticker,
      coin   = TICKER_LABELS[ticker]
    )
    
    message("    Got ", nrow(df), " rows for ", ticker,
            " (", min(df$date), " to ", max(df$date), ")")
    return(df)
    
  }, error = function(e) {
    warning("Failed to fetch ", ticker, ": ", e$message)
    return(NULL)
  })
}


# --- 1.2 Load or Cache Yahoo Finance Data -------------------
#
# Same caching pattern as before — pull once, save to disk,
# load from cache on every subsequent run.

yf_cache_path <- "data/raw/yahoo_raw.rds"

if (file.exists(yf_cache_path)) {
  message("Loading stablecoin data from cache...")
  cg_raw <- readRDS(yf_cache_path)
} else {
  message("Pulling stablecoin data from Yahoo Finance...")
  
  cg_raw <- purrr::map_dfr(TICKERS, fetch_yahoo)
  
  saveRDS(cg_raw, yf_cache_path)
  message("Stablecoin data saved to cache.")
}

# Ensure date is Date type and filter to analysis window
cg_raw <- cg_raw |>
  mutate(date = as.Date(date)) |>
  dplyr::filter(date >= START_DATE, date <= END_DATE)

message("Stablecoin data loaded: ", nrow(cg_raw), " rows, ",
        n_distinct(cg_raw$coin), " coins.")
message("Date range: ", min(cg_raw$date), " to ", max(cg_raw$date))
print(table(cg_raw$coin))


# --- 1.3 FRED Data Pull -------------------------------------
#
# The fredr package provides a clean R interface to the FRED
# API. Each series is pulled individually and then combined.
# FRED returns daily observations for most series, but
# weekends and holidays are omitted — we handle this gap
# in Section 2 during preprocessing.
#
# Series we pull:
#   DTWEXBGS — Nominal Broad US Dollar Index (DXY proxy)
#   SOFR     — Secured Overnight Financing Rate
#   VIXCLS   — CBOE Volatility Index (market fear gauge)
#              Selected over BAMLH0A0HYM2 (high-yield spread)
#              which only has data from May 2023 on FRED —
#              far too short to cover the UST collapse in 2022

fetch_fred <- function(series_id) {
  message("  Fetching FRED series: ", series_id)
  
  fredr(
    series_id         = series_id,
    observation_start = START_DATE,
    observation_end   = END_DATE,
    frequency         = "d"        # daily frequency
  ) |>
    dplyr::select(date, value) |>
    rename(!!series_id := value)   # rename value col to series ID
}


# --- 1.4 Load or Cache FRED Data ----------------------------

fred_cache_path <- "data/raw/fred_raw.rds"

if (file.exists(fred_cache_path)) {
  message("Loading FRED data from cache...")
  fred_raw <- readRDS(fred_cache_path)
} else {
  message("Pulling FRED data from API...")
  
  fred_list <- purrr::map(FRED_SERIES, fetch_fred)
  
  # Join all three series on date using reduce + full_join
  # so no dates are dropped even if one series has gaps
  fred_raw <- purrr::reduce(fred_list, full_join, by = "date")
  
  saveRDS(fred_raw, fred_cache_path)
  message("FRED data saved to cache.")
}

# Rename columns to friendlier names for use throughout the script
fred_raw <- fred_raw |>
  rename(
    dxy  = DTWEXBGS,
    sofr = SOFR,
    vix  = VIXCLS
  )

message("FRED data loaded: ", nrow(fred_raw), " rows, ",
        ncol(fred_raw) - 1, " macro variables.")


# --- 1.5 First Look at the Raw Data -------------------------
#
# Before any cleaning or transformation, we take a first look
# at the structure of both datasets. This is standard practice
# when loading any new dataset — it confirms the data arrived
# in the expected shape and helps identify immediate issues
# such as wrong column types or obviously missing values.
# (See Chapter 1: Getting Started with R — Exploring Data)

message("--- Stablecoin Raw Data Structure ---")
str(cg_raw)

message("--- Stablecoin First 6 Rows ---")
print(head(cg_raw))

message("--- Stablecoin Dimensions ---")
message("Rows: ", nrow(cg_raw), " | Columns: ", ncol(cg_raw))

message("--- FRED Raw Data Structure ---")
str(fred_raw)

message("--- FRED First 6 Rows ---")
print(head(fred_raw))

message("--- FRED Dimensions ---")
message("Rows: ", nrow(fred_raw), " | Columns: ", ncol(fred_raw))

message("--- Date Range Check ---")
message("Stablecoins : ", min(cg_raw$date), " to ", max(cg_raw$date))
message("FRED        : ", min(fred_raw$date, na.rm = TRUE),
        " to ", max(fred_raw$date, na.rm = TRUE))

message("--- Coin Coverage ---")
print(table(cg_raw$coin))

message("Section 1 complete. Raw data loaded and inspected.")

# ============================================================
# SECTION 2 — DATA CLEANING & PREPROCESSING
# (See Chapter 2: Basic Data Management)
# ============================================================
#
# Raw data is rarely analysis-ready. In this section we:
#
#   1. Handle structural missingness in UST (post-delisting NAs)
#   2. Forward-fill weekend/holiday gaps in FRED macro data
#   3. Align both datasets to a common daily date spine
#   4. Cast all columns to their correct types
#   5. Merge stablecoin and macro data into one master dataframe
#   6. Save the cleaned master to data/processed/ for use in
#      all subsequent sections
#
# The goal is a single, clean, analysis-ready dataframe that
# every downstream section can rely on without repeating
# cleaning steps.
# ============================================================


# --- 2.1 Handle UST Post-Delisting NAs ----------------------
#
# As we discovered in Section 1, UST has 448 consecutive NA
# values starting 2022-10-10. This is structural missingness —
# Yahoo Finance stops reporting after the coin was effectively
# delisted following the May 2022 collapse. We do not impute
# these values since they represent genuine absence of a market.
# Instead we truncate UST's series at its last valid observation.
# We will formally examine and visualize this in Section 4 (EDA).
#
# All other coins have no mid-series NAs — only FRAX and UST
# have shorter series due to later listing dates, which is
# expected and handled naturally by the master dataframe join.

ust_end_date <- cg_raw |>
  dplyr::filter(coin == "UST", !is.na(price)) |>
  summarise(last_date = max(date)) |>
  pull(last_date)

message("UST last valid observation: ", ust_end_date)

# Remove post-delisting rows from UST only
cg_clean <- cg_raw |>
  dplyr::filter(!(coin == "UST" & date > ust_end_date))

message("Rows after UST truncation: ", nrow(cg_clean),
        " (removed ", nrow(cg_raw) - nrow(cg_clean), " NA rows)")


# --- 2.2 Build a Complete Daily Date Spine ------------------
#
# FRED data only contains weekdays — weekends and holidays are
# absent. Stablecoin markets trade 24/7, so they have entries
# for every calendar day. To align the two datasets we first
# build a complete daily sequence covering our full window,
# then join everything onto it.
#
# This approach ensures no dates are silently dropped when we
# merge, making any remaining gaps explicit and visible.

date_spine <- data.frame(
  date = seq.Date(START_DATE, END_DATE, by = "day")
)

message("Date spine: ", nrow(date_spine), " days (",
        START_DATE, " to ", END_DATE, ")")


# --- 2.3 Forward-Fill FRED Weekend Gaps ---------------------
#
# Macro indicators like DXY and SOFR do not change on weekends
# — the last observed value carries forward until the next
# trading day. Forward-filling is the standard treatment for
# this type of gap in financial time series.
# (See Chapter 2: Advanced Data Management — Missing Values)
#
# We use zoo::na.locf() (Last Observation Carried Forward)
# which is purpose-built for this pattern.

fred_filled <- date_spine |>
  left_join(fred_raw, by = "date") |>
  arrange(date) |>
  mutate(
    dxy  = zoo::na.locf(dxy,  na.rm = FALSE),
    sofr = zoo::na.locf(sofr, na.rm = FALSE),
    vix  = zoo::na.locf(vix,  na.rm = FALSE)
  ) |>
  # Backward-fill the leading NAs (Jan 1 holiday has no prior
  # value to carry forward — we fill from the next available day)
  mutate(
    dxy  = zoo::na.locf(dxy,  fromLast = TRUE, na.rm = FALSE),
    sofr = zoo::na.locf(sofr, fromLast = TRUE, na.rm = FALSE),
    vix  = zoo::na.locf(vix,  fromLast = TRUE, na.rm = FALSE)
  )

message("FRED after forward-fill: ", nrow(fred_filled), " rows")
message("Remaining FRED NAs after fill:")
message("  dxy  : ", sum(is.na(fred_filled$dxy)))
message("  sofr : ", sum(is.na(fred_filled$sofr)))
message("  vix  : ", sum(is.na(fred_filled$vix)))


# --- 2.4 Clean and Type-Cast Stablecoin Data ----------------
#
# We ensure all columns are the correct type before merging.
# The coin and ticker columns should be factors for efficient
# grouping and plotting. Price and volume should be numeric.
# Date should be Date class (already confirmed in Section 1).

cg_clean <- cg_clean |>
  mutate(
    coin   = factor(coin,   levels = c("USDC", "USDT", "DAI",
                                       "FRAX", "UST")),
    ticker = factor(ticker),
    price  = as.numeric(price),
    volume = as.numeric(volume)
  )

message("Coin factor levels: ", paste(levels(cg_clean$coin),
                                      collapse = ", "))


# --- 2.5 Build the Master Dataframe -------------------------
#
# We join the stablecoin data onto the date spine, then bring
# in the FRED macro variables. The result is a long-format
# dataframe with one row per coin per day, with macro variables
# repeated across all coins for each date.
#
# We use a left join from the stablecoin data onto the macro
# data so that crypto-only dates (weekends) retain their
# stablecoin observations with forward-filled macro values.

master <- cg_clean |>
  left_join(fred_filled, by = "date") |>
  arrange(coin, date)

message("Master dataframe: ", nrow(master), " rows x ",
        ncol(master), " columns")
message("Coins in master : ", paste(levels(master$coin),
                                    collapse = ", "))
message("Columns         : ", paste(names(master), collapse = ", "))


# --- 2.6 Verify No Unexpected NAs ---------------------------
#
# After cleaning we do a final NA audit on every column.
# We expect zero NAs — forward-fill handles weekends and the
# backward-fill handles the single leading holiday row.

na_summary <- master |>
  summarise(across(everything(), ~ sum(is.na(.)))) |>
  tidyr::pivot_longer(everything(),
                      names_to  = "column",
                      values_to = "na_count") |>
  dplyr::filter(na_count > 0)

message("--- NA Audit After Cleaning ---")
if (nrow(na_summary) == 0) {
  message("No unexpected NAs found.")
} else {
  print(na_summary)
}


# --- 2.7 Save Processed Data --------------------------------
#
# We save the cleaned master dataframe to data/processed/ so
# every downstream section can load it directly without
# re-running the cleaning pipeline.

master_path <- "data/processed/stablewatch_master.rds"
saveRDS(master, master_path)
message("Master dataframe saved to: ", master_path)

message("Section 2 complete. Master dataframe ready for analysis.")

# ============================================================
# SECTION 3 — FEATURE ENGINEERING
# (See Chapter 2: Advanced Data Management)
# ============================================================
#
# Raw price and volume data alone are not enough to detect
# depeg risk. We need to derive features that capture the
# signals a risk analyst would actually look for:
#
#   1. Peg deviation     — how far is the price from $1.00?
#   2. Peg deviation Z-score — is the deviation unusual
#                          relative to the coin's own history?
#   3. Rolling volatility — is price stability increasing
#                          or breaking down?
#   4. Log returns       — for time series and GARCH modeling
#   5. Volume spike      — is trading volume abnormally high?
#                          (a key early warning signal)
#   6. Depeg label       — binary outcome variable: 1 if peg
#                          deviation exceeds our 0.5% threshold
#   7. Stress event flag — marks known crisis dates for use
#                          in model validation and plotting
#
# All features are computed per coin using grouped operations
# so each coin's rolling statistics are self-contained.
# ============================================================


# --- 3.1 Peg Deviation --------------------------------------
#
# A stablecoin's fundamental promise is to hold a $1.00 peg.
# Peg deviation measures how far the daily closing price strays
# from that target. We use the absolute deviation so that
# deviations above and below $1.00 are treated equally.
#
# peg_dev = |price - 1.00|
#
# This is the central variable in our analysis — every model
# and test downstream uses it directly or as a basis for
# derived features.

master <- master |>
  mutate(peg_dev = abs(price - 1.00))

message("Peg deviation computed.")
message("Max peg deviation overall: ",
        round(max(master$peg_dev, na.rm = TRUE), 4))


# --- 3.2 Log Returns ----------------------------------------
#
# Log returns are the standard transformation for financial
# price series before time series modeling. They are:
#   - Approximately normally distributed
#   - Additive across time periods
#   - Stationary (unlike raw prices)
#
# log_return = log(price_t / price_{t-1})
#
# We compute them within each coin group so the first
# observation per coin is NA (no prior price to compare to).
# (See Chapter 5: Time Series)

master <- master |>
  group_by(coin) |>
  arrange(date) |>
  mutate(log_return = log(price / dplyr::lag(price))) |>
  ungroup()

message("Log returns computed.")
message("NA log returns (expected — one per coin): ",
        sum(is.na(master$log_return)))


# --- 3.3 Rolling Volatility ---------------------------------
#
# A stablecoin under stress becomes more volatile before it
# breaks its peg entirely. Rolling volatility captures this
# buildup and is one of our key early-warning features.
#
# We compute two windows:
#   vol_7d  — 7-day rolling standard deviation of log returns
#             (short-term noise, responsive to sudden moves)
#   vol_30d — 30-day rolling standard deviation of log returns
#             (medium-term trend, smoother signal)
#
# We use zoo::rollapply() with fill = NA so early observations
# that lack a full window are marked NA rather than estimated
# from insufficient data.

master <- master |>
  group_by(coin) |>
  arrange(date) |>
  mutate(
    vol_7d  = zoo::rollapply(log_return, width = 7,
                             FUN = sd, fill = NA,
                             align = "right", na.rm = TRUE),
    vol_30d = zoo::rollapply(log_return, width = 30,
                             FUN = sd, fill = NA,
                             align = "right", na.rm = TRUE)
  ) |>
  ungroup()

message("Rolling volatility computed (7d and 30d).")


# --- 3.4 Peg Deviation Z-Score ------------------------------
#
# Raw peg deviation tells us the size of the deviation, but
# not whether it is unusual for that specific coin. A Z-score
# normalizes peg deviation relative to each coin's own rolling
# mean and standard deviation over the past 30 days.
#
# peg_dev_z = (peg_dev - rolling_mean_30d) / rolling_sd_30d
#
# A Z-score above 2 means the current deviation is more than
# 2 standard deviations above normal for that coin — a strong
# signal of emerging stress regardless of the coin's baseline
# volatility level.

master <- master |>
  group_by(coin) |>
  arrange(date) |>
  mutate(
    roll_mean_30 = zoo::rollapply(peg_dev, width = 30,
                                  FUN = mean, fill = NA,
                                  align = "right", na.rm = TRUE),
    roll_sd_30   = zoo::rollapply(peg_dev, width = 30,
                                  FUN = sd, fill = NA,
                                  align = "right", na.rm = TRUE),
    peg_dev_z    = (peg_dev - roll_mean_30) / roll_sd_30
  ) |>
  dplyr::select(-roll_mean_30, -roll_sd_30) |>
  ungroup()

message("Peg deviation Z-score computed.")


# --- 3.5 Volume Spike Indicator -----------------------------
#
# Unusual trading volume often precedes or accompanies a depeg
# event. When market participants sense risk they trade more —
# either rushing to exit or arbitraging the peg deviation.
#
# We define a volume spike as a day where volume exceeds the
# coin's own 30-day rolling average by more than 2 standard
# deviations. This is a binary flag (1 = spike, 0 = normal).
#
# vol_spike = 1 if volume > (roll_mean_vol + 2 * roll_sd_vol)

master <- master |>
  group_by(coin) |>
  arrange(date) |>
  mutate(
    roll_mean_vol = zoo::rollapply(volume, width = 30,
                                   FUN = mean, fill = NA,
                                   align = "right", na.rm = TRUE),
    roll_sd_vol   = zoo::rollapply(volume, width = 30,
                                   FUN = sd, fill = NA,
                                   align = "right", na.rm = TRUE),
    vol_spike     = as.integer(
      volume > (roll_mean_vol + 2 * roll_sd_vol)
    )
  ) |>
  dplyr::select(-roll_mean_vol, -roll_sd_vol) |>
  ungroup()

message("Volume spike indicator computed.")
message("Total volume spike days: ", sum(master$vol_spike, na.rm = TRUE))


# --- 3.6 Binary Depeg Label ---------------------------------
#
# Our classification models need a binary outcome variable.
# We define a depeg event as any day where peg deviation
# exceeds our DEPEG_THRESHOLD of 0.5% (set in Section 0).
#
# This threshold is consistent with industry practice — most
# stablecoin risk desks define a depeg as deviation > 0.5%.
# UST exceeded this threshold on May 9, 2022 and never
# recovered — making it our primary validation event.
#
# depeg = 1 if peg_dev > DEPEG_THRESHOLD, else 0

master <- master |>
  mutate(depeg = as.integer(peg_dev > DEPEG_THRESHOLD))

message("Depeg label computed.")
message("Depeg event days by coin:")
master |>
  group_by(coin) |>
  summarise(
    total_days  = n(),
    depeg_days  = sum(depeg, na.rm = TRUE),
    depeg_rate  = round(mean(depeg, na.rm = TRUE) * 100, 2)
  ) |>
  print()


# --- 3.7 Stress Event Flag ----------------------------------
#
# We create a binary flag marking the two known stress events
# defined in Section 0: the UST collapse (May 9, 2022) and
# the USDC/SVB scare (March 10, 2023). These flags are used
# to annotate plots and validate that our models fire around
# the correct dates.

master <- master |>
  mutate(
    stress_event = as.integer(date %in% STRESS_EVENTS$date)
  )

message("Stress event flags set on: ",
        paste(STRESS_EVENTS$date, collapse = ", "))


# --- 3.8 Review Engineered Features -------------------------
#
# We take a final look at the master dataframe to confirm all
# features were computed correctly before saving.

message("--- Final Master Dataframe Structure ---")
str(master)

message("--- Feature Summary for USDT (stable reference coin) ---")
master |>
  dplyr::filter(coin == "USDT") |>
  dplyr::select(peg_dev, log_return, vol_7d, vol_30d,
                peg_dev_z, vol_spike, depeg) |>
  summary() |>
  print()

message("--- Feature Summary for UST (collapsed coin) ---")
master |>
  dplyr::filter(coin == "UST") |>
  dplyr::select(peg_dev, log_return, vol_7d, vol_30d,
                peg_dev_z, vol_spike, depeg) |>
  summary() |>
  print()

# Save updated master with engineered features
saveRDS(master, master_path)
message("Master dataframe with features saved to: ", master_path)
message("Section 3 complete. Features ready for analysis.")