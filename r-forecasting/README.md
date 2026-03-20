# R Time Series Forecasting - Mid-Columbia Hub Price Projections

An R-based extension to the Energy Risk Analytics project, adding time series forecasting models to support Seattle City Light's wholesale power procurement planning and rate strategy.

## Models

| Model | Type | Strengths |
|-------|------|-----------|
| ARIMA (auto-selected) | Classical parametric | Captures autocorrelation and trend; fast daily updates |
| Facebook Prophet | Additive decomposition | Handles seasonality well; robust to missing data |

## Analysis Pipeline

1. Exploratory analysis - daily/monthly price trends, LOESS smoothing, volatility bands
2. STL decomposition - isolate trend, seasonal, and residual components
3. Stationarity testing - Augmented Dickey-Fuller test
4. Train/test split - 90-day holdout for backtesting
5. ARIMA fitting - auto.arima() with AIC-based parameter selection
6. Prophet fitting - yearly + weekly seasonality with changepoint detection
7. Accuracy comparison - RMSE, MAE, MAPE on holdout set
8. Forward forecast - 90-day projection with 80% and 95% confidence intervals
9. Residual diagnostics - temporal patterns and distribution analysis

## Getting Started
```r
install.packages(c("tidyverse", "forecast", "prophet", "tseries", "lubridate", "scales", "patchwork", "Metrics"))
```

Then open forecast_analysis.R in RStudio and run interactively.

## Connection to Python Dashboard

- Python (Plotly Dash): Real-time monitoring, interactive risk dashboards, credit assessment
- R (forecast + prophet): Statistical forecasting, model comparison, forward projections
- Tableau: Executive-level reporting with native calculated fields

Built by Shubh Dhar - MSIM, University of Washington Information School
