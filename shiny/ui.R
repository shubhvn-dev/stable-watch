# ============================================================
# StableWatch Shiny App — ui.R
# Stablecoin Depeg Risk Monitor
# Shubhan Kadam | sk12159
# FRE 6871 — Final Project
# ============================================================

library(shiny)
library(shinydashboard)
library(plotly)
library(DT)

# Custom CSS — dark fintech terminal aesthetic
custom_css <- "
  /* Import fonts */
  @import url('https://fonts.googleapis.com/css2?family=Space+Mono:wght@400;700&family=IBM+Plex+Sans:wght@300;400;500;600&display=swap');

  /* Base */
  body, .content-wrapper, .main-sidebar, .left-side {
    background-color: #0a0e1a !important;
    color: #e2e8f0 !important;
    font-family: 'IBM Plex Sans', sans-serif !important;
  }

  /* Sidebar */
  .main-sidebar { background-color: #0d1224 !important; border-right: 1px solid #1e2a45; }
  .sidebar-menu > li > a { color: #94a3b8 !important; font-size: 13px; letter-spacing: 0.03em; }
  .sidebar-menu > li.active > a,
  .sidebar-menu > li > a:hover { color: #38bdf8 !important; background: rgba(56,189,248,0.08) !important; }
  .sidebar-menu > li > a .fa { color: #38bdf8 !important; }

  /* Header */
  .main-header .logo {
    background-color: #0d1224 !important;
    border-bottom: 1px solid #1e2a45 !important;
    font-family: 'Space Mono', monospace !important;
    font-size: 15px !important;
    color: #38bdf8 !important;
    letter-spacing: 0.1em;
  }
  .main-header .navbar { background-color: #0d1224 !important; border-bottom: 1px solid #1e2a45; }
  .main-header .navbar .nav > li > a { color: #94a3b8 !important; }

  /* Value boxes */
  .small-box { border-radius: 4px !important; border: 1px solid #1e2a45; }
  .small-box h3 { font-family: 'Space Mono', monospace !important; font-size: 22px !important; }
  .small-box p  { font-size: 11px !important; text-transform: uppercase; letter-spacing: 0.08em; }

  /* Boxes */
  .box {
    background: #0d1224 !important;
    border: 1px solid #1e2a45 !important;
    border-radius: 4px !important;
    box-shadow: none !important;
  }
  .box-header {
    background: transparent !important;
    border-bottom: 1px solid #1e2a45 !important;
    padding: 10px 15px !important;
  }
  .box-title {
    font-family: 'Space Mono', monospace !important;
    font-size: 11px !important;
    text-transform: uppercase;
    letter-spacing: 0.12em;
    color: #38bdf8 !important;
  }

  /* Inputs */
  .selectize-input, .selectize-dropdown,
  .form-control {
    background: #131929 !important;
    border: 1px solid #1e2a45 !important;
    color: #e2e8f0 !important;
    border-radius: 3px !important;
    font-family: 'IBM Plex Sans', sans-serif !important;
  }
  .selectize-dropdown-content .option { color: #e2e8f0 !important; }
  .selectize-dropdown-content .option.active { background: #1e3a5f !important; }
  label { color: #94a3b8 !important; font-size: 11px !important; text-transform: uppercase; letter-spacing: 0.06em; }

  /* Slider */
  .irs-bar, .irs-bar-edge { background: #38bdf8 !important; border-color: #38bdf8 !important; }
  .irs-slider { background: #38bdf8 !important; border-color: #38bdf8 !important; }
  .irs-from, .irs-to, .irs-single { background: #1e3a5f !important; color: #38bdf8 !important; font-family: 'Space Mono', monospace !important; font-size: 10px !important; }

  /* DataTable */
  .dataTables_wrapper { color: #94a3b8 !important; }
  table.dataTable { background: #0d1224 !important; color: #e2e8f0 !important; border: none !important; font-size: 12px !important; }
  table.dataTable thead th { background: #131929 !important; color: #38bdf8 !important; border-bottom: 1px solid #1e2a45 !important; font-family: 'Space Mono', monospace !important; font-size: 10px !important; text-transform: uppercase; letter-spacing: 0.06em; }
  table.dataTable tbody tr { background: #0d1224 !important; }
  table.dataTable tbody tr:hover { background: #131929 !important; }
  table.dataTable tbody td { border-top: 1px solid #1a2035 !important; }
  .dataTables_filter input, .dataTables_length select { background: #131929 !important; color: #e2e8f0 !important; border: 1px solid #1e2a45 !important; }

  /* Tab panels */
  .nav-tabs { border-bottom: 1px solid #1e2a45 !important; }
  .nav-tabs > li > a { color: #64748b !important; background: transparent !important; border: none !important; font-size: 12px; letter-spacing: 0.04em; }
  .nav-tabs > li.active > a { color: #38bdf8 !important; border-bottom: 2px solid #38bdf8 !important; background: transparent !important; }

  /* Risk badges */
  .badge-low    { background: #166534; color: #4ade80; padding: 3px 8px; border-radius: 3px; font-size: 11px; font-family: 'Space Mono', monospace; }
  .badge-medium { background: #7c2d12; color: #fb923c; padding: 3px 8px; border-radius: 3px; font-size: 11px; font-family: 'Space Mono', monospace; }
  .badge-high   { background: #7f1d1d; color: #f87171; padding: 3px 8px; border-radius: 3px; font-size: 11px; font-family: 'Space Mono', monospace; }

  /* Monospace numbers */
  .mono { font-family: 'Space Mono', monospace !important; }

  /* Scrollbar */
  ::-webkit-scrollbar { width: 6px; }
  ::-webkit-scrollbar-track { background: #0a0e1a; }
  ::-webkit-scrollbar-thumb { background: #1e2a45; border-radius: 3px; }

  /* Section divider */
  hr { border-color: #1e2a45 !important; }

  /* Plotly bg fix */
  .js-plotly-plot .plotly .main-svg { background: transparent !important; }
"

# Coin colors consistent with main analysis
COIN_COLORS <- c(
  "USDC" = "#2563eb",
  "USDT" = "#16a34a",
  "DAI"  = "#d97706",
  "FRAX" = "#7c3aed",
  "UST"  = "#dc2626"
)

# ============================================================
# UI DEFINITION
# ============================================================

ui <- dashboardPage(
  skin = "black",

  # ----- HEADER -----
  dashboardHeader(
    title = "STABLEWATCH",
    tags$li(class = "dropdown",
            tags$li(class = "dropdown",
                    style = "padding: 10px 15px; color: #64748b; font-size: 11px; font-family: 'Space Mono', monospace;",
                    "DEPEG RISK MONITOR"))
  ),

  # ----- SIDEBAR -----
  dashboardSidebar(
    tags$style(HTML(custom_css)),
    sidebarMenu(
      id = "tabs",
      menuItem("Risk Overview",    tabName = "overview",    icon = icon("gauge-high")),
      menuItem("Historical Analysis", tabName = "historical", icon = icon("chart-line")),
      menuItem("Model Backtest",   tabName = "backtest",    icon = icon("flask")),
      menuItem("SHAP Explainer",   tabName = "shap",        icon = icon("microscope")),
      menuItem("Performance",      tabName = "performance", icon = icon("chart-bar"))
    ),

    hr(),

    # Coin selector
    selectInput("selected_coin", "COIN",
                choices  = c("USDC","USDT","DAI","FRAX","UST"),
                selected = "UST",
                multiple = FALSE),

    # Date range
    dateRangeInput("date_range", "DATE RANGE",
                   start = "2020-11-25",
                   end   = "2023-12-31",
                   min   = "2020-01-01",
                   max   = "2023-12-31"),

    hr(),

    # Threshold slider
    sliderInput("threshold", "ALERT THRESHOLD",
                min = 0.1, max = 0.9, value = 0.66, step = 0.01),

    tags$div(
      style = "padding: 10px 15px; color: #475569; font-size: 10px; font-family: 'Space Mono', monospace; line-height: 1.6;",
      "Shubhan Kadam · sk12159",
      tags$br(),
      "FRE 6871 · Final Project",
      tags$br(),
      "Data: Yahoo Finance + FRED",
      tags$br(),
      "2020 – 2023"
    )
  ),

  # ----- BODY -----
  dashboardBody(

    tabItems(

      # ==================================================
      # TAB 1: RISK OVERVIEW
      # ==================================================
      tabItem(tabName = "overview",

        # Value boxes row
        fluidRow(
          valueBoxOutput("vbox_auc",         width = 3),
          valueBoxOutput("vbox_sensitivity",  width = 3),
          valueBoxOutput("vbox_depeg_rate",   width = 3),
          valueBoxOutput("vbox_coin_risk",    width = 3)
        ),

        # Main chart + regime proportions
        fluidRow(
          box(title = "PEG DEVIATION OVER TIME",
              width = 8, solidHeader = FALSE,
              plotlyOutput("peg_dev_plot", height = "340px")),

          box(title = "RISK REGIME DISTRIBUTION",
              width = 4, solidHeader = FALSE,
              plotlyOutput("regime_pie", height = "340px"))
        ),

        # Feature distributions
        fluidRow(
          box(title = "VOLATILITY SIGNALS",
              width = 6, solidHeader = FALSE,
              plotlyOutput("vol_plot", height = "280px")),

          box(title = "PEG DEVIATION Z-SCORE",
              width = 6, solidHeader = FALSE,
              plotlyOutput("zscore_plot", height = "280px"))
        )
      ),

      # ==================================================
      # TAB 2: HISTORICAL ANALYSIS
      # ==================================================
      tabItem(tabName = "historical",

        fluidRow(
          box(title = "DESCRIPTIVE STATISTICS — ALL COINS",
              width = 12, solidHeader = FALSE,
              DTOutput("desc_table"))
        ),

        fluidRow(
          box(title = "BOOTSTRAP 95% CONFIDENCE INTERVALS: MEAN PEG DEVIATION",
              width = 6, solidHeader = FALSE,
              plotlyOutput("bootstrap_plot", height = "320px")),

          box(title = "DEPEG RATE BY COIN",
              width = 6, solidHeader = FALSE,
              plotlyOutput("depeg_rate_plot", height = "320px"))
        ),

        fluidRow(
          box(title = "PEG DEVIATION DISTRIBUTION (LOG SCALE)",
              width = 12, solidHeader = FALSE,
              plotlyOutput("box_plot", height = "320px"))
        )
      ),

      # ==================================================
      # TAB 3: MODEL BACKTEST
      # ==================================================
      tabItem(tabName = "backtest",

        fluidRow(
          box(title = "XGBOOST PREDICTED DEPEG PROBABILITY — UST PRE-COLLAPSE",
              width = 12, solidHeader = FALSE,
              tags$p(style = "color:#64748b; font-size:12px; margin-bottom:8px;",
                     "Model trained on USDC/USDT/DAI/FRAX — never saw UST data.
                      Adjust threshold slider in sidebar to change alert level."),
              plotlyOutput("backtest_plot", height = "380px"))
        ),

        fluidRow(
          box(title = "MODEL PERFORMANCE METRICS",
              width = 4, solidHeader = FALSE,
              DTOutput("perf_table")),

          box(title = "ROC CURVE",
              width = 4, solidHeader = FALSE,
              plotlyOutput("roc_plot", height = "280px")),

          box(title = "CLUSTER REGIME PROPORTIONS",
              width = 4, solidHeader = FALSE,
              plotlyOutput("regime_bar", height = "280px"))
        )
      ),

      # ==================================================
      # TAB 4: SHAP EXPLAINER
      # ==================================================
      tabItem(tabName = "shap",

        fluidRow(
          box(title = "SHAP FEATURE IMPORTANCE",
              width = 6, solidHeader = FALSE,
              tags$p(style = "color:#64748b; font-size:12px; margin-bottom:8px;",
                     "Mean absolute SHAP value — higher = more influential.
                      vol_30d dominates; vol_spike contributes near zero."),
              plotlyOutput("shap_bar", height = "340px")),

          box(title = "FEATURE IMPORTANCE INTERPRETATION",
              width = 6, solidHeader = FALSE,
              tags$div(
                style = "padding: 10px; font-size: 13px; line-height: 1.8;",
                tags$p(style = "color:#38bdf8; font-family:'Space Mono',monospace; font-size:11px;",
                       "TOP PREDICTORS"),
                tags$hr(style="border-color:#1e2a45;"),
                tags$p(tags$span(style="color:#fb923c; font-family:'Space Mono',monospace;", "vol_30d"),
                       tags$span(style="color:#94a3b8;", " — 30-day rolling volatility. The dominant signal (SHAP=3.3).
                                  Rising medium-term vol precedes depeg events by days to weeks.")),
                tags$p(tags$span(style="color:#fb923c; font-family:'Space Mono',monospace;", "peg_dev_z"),
                       tags$span(style="color:#94a3b8;", " — Normalized Z-score vs 30-day history. Each +1 SD
                                  multiplies depeg odds by 2.56× (GLM OR=2.56).")),
                tags$p(tags$span(style="color:#fb923c; font-family:'Space Mono',monospace;", "vol_7d"),
                       tags$span(style="color:#94a3b8;", " — Short-term volatility. More responsive but noisier
                                  than the 30-day window.")),
                tags$hr(style="border-color:#1e2a45;"),
                tags$p(style="color:#64748b; font-size:11px;",
                       "PCA confirms PC1 (coin stress) and PC2 (macro conditions)
                        are orthogonal — both dimensions matter independently.")
              ))
        ),

        fluidRow(
          box(title = "FEATURE DISTRIBUTIONS: DEPEG vs STABLE DAYS",
              width = 12, solidHeader = FALSE,
              plotlyOutput("feat_dist_plot", height = "300px"))
        )
      ),

      # ==================================================
      # TAB 5: PERFORMANCE ANALYTICS
      # ==================================================
      tabItem(tabName = "performance",

        fluidRow(
          box(title = "VALUE AT RISK & EXPECTED SHORTFALL (95%)",
              width = 12, solidHeader = FALSE,
              DTOutput("var_table"))
        ),

        fluidRow(
          box(title = "LOG RETURN DISTRIBUTION",
              width = 6, solidHeader = FALSE,
              plotlyOutput("return_dist", height = "320px")),

          box(title = "ROLLING VOLATILITY COMPARISON",
              width = 6, solidHeader = FALSE,
              plotlyOutput("rolling_vol_plot", height = "320px"))
        ),

        fluidRow(
          box(title = "RESEARCH CONCLUSION",
              width = 12, solidHeader = FALSE,
              tags$div(
                style = "padding: 15px; line-height: 1.9;",
                tags$p(style = "color:#38bdf8; font-family:'Space Mono',monospace; font-size:11px;",
                       "RESEARCH QUESTION: Can macro + on-chain signals predict stablecoin depegs?"),
                tags$hr(style="border-color:#1e2a45;"),
                tags$p(tags$span(style="color:#4ade80; font-family:'Space Mono',monospace;", "YES — "),
                       tags$span(style="color:#94a3b8;",
                                 "For volatility-driven stress episodes. XGBoost (AUC=0.9983, sensitivity=94.9%)
                                  correctly fires on UST's 2021 depegs. vol_30d, peg_dev_z, and vol_7d are the
                                  dominant leading indicators.")),
                tags$p(tags$span(style="color:#f87171; font-family:'Space Mono',monospace;", "NO — "),
                       tags$span(style="color:#94a3b8;",
                                 "For reflexive algorithmic collapse. UST's May 2022 collapse showed ZERO
                                  High Risk cluster days in the 60-day runup (vs 11.1% baseline). The collapse
                                  was a sudden bank-run — deceptively calm until instant failure.")),
                tags$hr(style="border-color:#1e2a45;"),
                tags$p(style="color:#64748b; font-size:12px;",
                       "Signals that would detect algorithmic collapse: LUNA supply inflation rate,
                        Anchor Protocol deposit outflows, UST/LUNA market cap ratio,
                        Curve 4pool imbalance. These require on-chain data beyond price-based models.")
              ))
        )
      )
    )
  )
)