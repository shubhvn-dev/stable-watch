# ============================================================
# StableWatch Shiny App — app.R
# Stablecoin Depeg Risk Monitor
# Shubhan Kadam | sk12159
# FRE 6871 — Final Project
#
# USAGE:
#   1. Run StableWatch.R first to generate:
#      - data/processed/stablewatch_master.rds
#      - output/models/xgb_depeg.rds
#      - output/models/glm_full.rds
#      - output/models/kmeans_regimes.rds
#
#   2. Launch the app:
#      shiny::runApp("shiny/")
#      OR open this file in RStudio and click "Run App"
#
# PACKAGES REQUIRED:
#   shiny, shinydashboard, plotly, DT, tidyverse, zoo, xts,
#   xgboost, shapviz, PerformanceAnalytics, boot, pROC, moments
# ============================================================

library(shiny)
library(shinydashboard)

# Source ui and server
source("ui.R")
source("server.R")

# Launch
shinyApp(ui = ui, server = server)