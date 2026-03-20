# =============================================================================
# Mid-Columbia Hub Electricity Price Forecasting
# Seattle City Light - Energy Risk Analytics Extension
# Author: Shubh Dhar | UW iSchool MSIM
# =============================================================================
#
# PURPOSE:
# Time series forecasting of wholesale electricity prices at the Mid-Columbia
# (Mid-C) hub using ARIMA and Prophet models. This analysis supports SCL's
# rate planning and wholesale power procurement strategy by projecting
# short-term price trends and quantifying forecast uncertainty.
#
# MODELS USED:
# 1. ARIMA (Auto-Regressive Integrated Moving Average) - classical parametric
# 2. Facebook Prophet - additive decomposition with seasonality
#
# =============================================================================

# --- 1. INSTALL & LOAD PACKAGES ----------------------------------------------
# Run this block once to install required packages, then comment it out

# install.packages(c("tidyverse", "forecast", "prophet", "tseries",
#                     "lubridate", "scales", "patchwork", "Metrics"))

library(tidyverse)    # Data wrangling + ggplot2 visualization
library(forecast)     # ARIMA modeling and auto.arima()
library(prophet)      # Facebook Prophet forecasting
library(tseries)      # ADF test for stationarity
library(lubridate)    # Date manipulation
library(scales)       # Axis formatting for plots
library(patchwork)    # Combine multiple ggplot2 plots
library(Metrics)      # RMSE, MAE, MAPE calculations

# --- 2. LOAD DATA ------------------------------------------------------------
# Load the price data from the parent project's CSV
# Adjust path if running from a different directory

prices <- read_csv("../prices.csv") %>%
  mutate(date = as.Date(date)) %>%
  arrange(date)

cat("Dataset loaded:", nrow(prices), "rows\n")
cat("Date range:", as.character(min(prices$date)), "to",
    as.character(max(prices$date)), "\n")
cat("Price range: $", round(min(prices$price_mwh), 2), "to $",
    round(max(prices$price_mwh), 2), "/MWh\n")

# --- 3. EXPLORATORY DATA ANALYSIS -------------------------------------------

# 3a. Daily price time series
p_daily <- ggplot(prices, aes(x = date, y = price_mwh)) +
  geom_line(color = "#2196F3", alpha = 0.6, linewidth = 0.3) +
  geom_smooth(method = "loess", span = 0.1, color = "#FF5722",
              se = FALSE, linewidth = 0.8) +
  labs(
    title = "Mid-Columbia Hub Wholesale Electricity Prices (2022-2024)",
    subtitle = "Daily prices with LOESS trend line",
    x = NULL, y = "Price ($/MWh)"
  ) +
  scale_x_date(date_breaks = "3 months", date_labels = "%b %Y") +
  theme_minimal(base_size = 11) +
  theme(
    plot.title = element_text(face = "bold", size = 13),
    axis.text.x = element_text(angle = 45, hjust = 1)
  )

ggsave("outputs/01_daily_prices.png", p_daily, width = 10, height = 5, dpi = 150)

# 3b. Monthly aggregation for cleaner trend view
monthly <- prices %>%
  mutate(month = floor_date(date, "month")) %>%
  group_by(month) %>%
  summarise(
    avg_price = mean(price_mwh, na.rm = TRUE),
    sd_price  = sd(price_mwh, na.rm = TRUE),
    min_price = min(price_mwh, na.rm = TRUE),
    max_price = max(price_mwh, na.rm = TRUE),
    .groups = "drop"
  )

p_monthly <- ggplot(monthly, aes(x = month)) +
  geom_ribbon(aes(ymin = avg_price - sd_price, ymax = avg_price + sd_price),
              fill = "#2196F3", alpha = 0.15) +
  geom_line(aes(y = avg_price), color = "#1565C0", linewidth = 1) +
  geom_point(aes(y = avg_price), color = "#1565C0", size = 2) +
  labs(
    title = "Monthly Average Electricity Prices with Volatility Band",
    subtitle = "Shaded region = +/- 1 standard deviation",
    x = NULL, y = "Avg Price ($/MWh)"
  ) +
  scale_x_date(date_breaks = "3 months", date_labels = "%b %Y") +
  theme_minimal(base_size = 11) +
  theme(
    plot.title = element_text(face = "bold", size = 13),
    axis.text.x = element_text(angle = 45, hjust = 1)
  )

ggsave("outputs/02_monthly_prices.png", p_monthly, width = 10, height = 5, dpi = 150)

# 3c. Seasonal decomposition
# Convert to time series object (weekly frequency approximation = 365.25/7)
price_ts <- ts(prices$price_mwh, frequency = 365)

# STL decomposition (Seasonal-Trend using LOESS)
decomp <- stl(price_ts, s.window = "periodic")

# Plot decomposition
png("outputs/03_decomposition.png", width = 1000, height = 700, res = 150)
plot(decomp, main = "STL Decomposition of Mid-C Electricity Prices")
dev.off()

# --- 4. STATIONARITY TESTING ------------------------------------------------
# ARIMA requires (or benefits from) stationary data
# ADF test: null hypothesis = series has a unit root (non-stationary)

adf_result <- adf.test(prices$price_mwh)
cat("\n--- Augmented Dickey-Fuller Test ---\n")
cat("Test statistic:", adf_result$statistic, "\n")
cat("P-value:", adf_result$p.value, "\n")
cat("Conclusion:",
    ifelse(adf_result$p.value < 0.05,
           "Series is stationary (reject null)",
           "Series is non-stationary (fail to reject null)"), "\n")

# --- 5. TRAIN/TEST SPLIT ----------------------------------------------------
# Hold out last 90 days for forecast evaluation

split_date <- max(prices$date) - 90

train <- prices %>% filter(date <= split_date)
test  <- prices %>% filter(date > split_date)

cat("\nTrain set:", nrow(train), "days (",
    as.character(min(train$date)), "to", as.character(max(train$date)), ")\n")
cat("Test set:", nrow(test), "days (",
    as.character(min(test$date)), "to", as.character(max(test$date)), ")\n")

# --- 6. ARIMA MODEL ---------------------------------------------------------
# auto.arima() automatically selects optimal (p, d, q) parameters
# using AIC/BIC criteria

train_ts <- ts(train$price_mwh, frequency = 365)

cat("\nFitting ARIMA model (this may take a minute)...\n")
arima_model <- auto.arima(
  train_ts,
  seasonal      = TRUE,
  stepwise      = TRUE,   # Faster search
  approximation = FALSE,  # Exact MLE for better accuracy
  trace         = TRUE    # Show model selection process
)

cat("\n--- ARIMA Model Summary ---\n")
print(summary(arima_model))

# Generate 90-day forecast
arima_forecast <- forecast(arima_model, h = nrow(test), level = c(80, 95))

# Extract forecast values into a data frame
arima_results <- tibble(
  date       = test$date,
  actual     = test$price_mwh,
  forecast   = as.numeric(arima_forecast$mean),
  lower_80   = as.numeric(arima_forecast$lower[, 1]),
  upper_80   = as.numeric(arima_forecast$upper[, 1]),
  lower_95   = as.numeric(arima_forecast$lower[, 2]),
  upper_95   = as.numeric(arima_forecast$upper[, 2])
)

# --- 7. PROPHET MODEL -------------------------------------------------------
# Prophet requires columns named 'ds' (date) and 'y' (value)

prophet_train <- train %>%
  select(ds = date, y = price_mwh)

# Fit Prophet with yearly + weekly seasonality
prophet_model <- prophet(
  prophet_train,
  yearly.seasonality  = TRUE,
  weekly.seasonality  = TRUE,
  daily.seasonality   = FALSE,
  changepoint.prior.scale = 0.05  # Controls trend flexibility
)

# Create future dates dataframe for prediction
future_dates <- make_future_dataframe(prophet_model, periods = nrow(test))

# Generate forecast
prophet_forecast <- predict(prophet_model, future_dates)

# Extract test period predictions
prophet_results <- prophet_forecast %>%
  filter(ds > split_date) %>%
  select(ds, yhat, yhat_lower, yhat_upper) %>%
  mutate(ds = as.Date(ds)) %>%
  left_join(test %>% select(date, price_mwh), by = c("ds" = "date")) %>%
  rename(
    date     = ds,
    forecast = yhat,
    lower_95 = yhat_lower,
    upper_95 = yhat_upper,
    actual   = price_mwh
  )

# Save Prophet component plots
png("outputs/04_prophet_components.png", width = 1000, height = 800, res = 150)
prophet_plot_components(prophet_model, prophet_forecast)
dev.off()

# --- 8. MODEL COMPARISON & ACCURACY -----------------------------------------

# Calculate error metrics for both models
arima_metrics <- tibble(
  Model = "ARIMA",
  RMSE  = rmse(arima_results$actual, arima_results$forecast),
  MAE   = mae(arima_results$actual, arima_results$forecast),
  MAPE  = mape(arima_results$actual, arima_results$forecast) * 100
)

prophet_metrics <- tibble(
  Model = "Prophet",
  RMSE  = rmse(prophet_results$actual, prophet_results$forecast),
  MAE   = mae(prophet_results$actual, prophet_results$forecast),
  MAPE  = mape(prophet_results$actual, prophet_results$forecast) * 100
)

comparison <- bind_rows(arima_metrics, prophet_metrics)

cat("\n--- Model Comparison (90-Day Holdout) ---\n")
print(comparison)

# --- 9. FORECAST VISUALIZATION -----------------------------------------------

# 9a. ARIMA forecast plot
p_arima <- ggplot() +
  # Training data
  geom_line(data = train %>% tail(180), aes(x = date, y = price_mwh),
            color = "gray50", alpha = 0.5, linewidth = 0.4) +
  # 95% confidence interval
  geom_ribbon(data = arima_results,
              aes(x = date, ymin = lower_95, ymax = upper_95),
              fill = "#2196F3", alpha = 0.15) +
  # 80% confidence interval
  geom_ribbon(data = arima_results,
              aes(x = date, ymin = lower_80, ymax = upper_80),
              fill = "#2196F3", alpha = 0.25) +
  # Actual vs forecast lines
  geom_line(data = arima_results, aes(x = date, y = actual, color = "Actual"),
            linewidth = 0.7) +
  geom_line(data = arima_results, aes(x = date, y = forecast, color = "ARIMA Forecast"),
            linewidth = 0.8, linetype = "dashed") +
  # Vertical line at split
  geom_vline(xintercept = split_date, linetype = "dotted", color = "gray40") +
  annotate("text", x = split_date - 5, y = max(arima_results$upper_95, na.rm = TRUE),
           label = "Forecast Start", hjust = 1, size = 3, color = "gray40") +
  scale_color_manual(values = c("Actual" = "#333333", "ARIMA Forecast" = "#2196F3")) +
  labs(
    title = paste0("ARIMA ", arima_model$arma[1], ",",
                   arima_model$arma[6], ",", arima_model$arma[2],
                   " - 90-Day Forecast"),
    subtitle = paste0("RMSE: $", round(arima_metrics$RMSE, 2),
                      "/MWh | MAE: $", round(arima_metrics$MAE, 2),
                      "/MWh | MAPE: ", round(arima_metrics$MAPE, 1), "%"),
    x = NULL, y = "Price ($/MWh)", color = NULL
  ) +
  theme_minimal(base_size = 11) +
  theme(
    plot.title = element_text(face = "bold", size = 13),
    legend.position = "bottom"
  )

ggsave("outputs/05_arima_forecast.png", p_arima, width = 10, height = 5.5, dpi = 150)

# 9b. Prophet forecast plot
p_prophet <- ggplot() +
  geom_line(data = train %>% tail(180), aes(x = date, y = price_mwh),
            color = "gray50", alpha = 0.5, linewidth = 0.4) +
  geom_ribbon(data = prophet_results,
              aes(x = date, ymin = lower_95, ymax = upper_95),
              fill = "#FF9800", alpha = 0.2) +
  geom_line(data = prophet_results, aes(x = date, y = actual, color = "Actual"),
            linewidth = 0.7) +
  geom_line(data = prophet_results, aes(x = date, y = forecast, color = "Prophet Forecast"),
            linewidth = 0.8, linetype = "dashed") +
  geom_vline(xintercept = split_date, linetype = "dotted", color = "gray40") +
  annotate("text", x = split_date - 5, y = max(prophet_results$upper_95, na.rm = TRUE),
           label = "Forecast Start", hjust = 1, size = 3, color = "gray40") +
  scale_color_manual(values = c("Actual" = "#333333", "Prophet Forecast" = "#FF9800")) +
  labs(
    title = "Facebook Prophet - 90-Day Forecast",
    subtitle = paste0("RMSE: $", round(prophet_metrics$RMSE, 2),
                      "/MWh | MAE: $", round(prophet_metrics$MAE, 2),
                      "/MWh | MAPE: ", round(prophet_metrics$MAPE, 1), "%"),
    x = NULL, y = "Price ($/MWh)", color = NULL
  ) +
  theme_minimal(base_size = 11) +
  theme(
    plot.title = element_text(face = "bold", size = 13),
    legend.position = "bottom"
  )

ggsave("outputs/06_prophet_forecast.png", p_prophet, width = 10, height = 5.5, dpi = 150)

# 9c. Side-by-side comparison
p_combined <- p_arima / p_prophet +
  plot_annotation(
    title = "Model Comparison: ARIMA vs. Prophet",
    subtitle = "90-day holdout forecast on Mid-Columbia Hub prices",
    theme = theme(
      plot.title = element_text(face = "bold", size = 15),
      plot.subtitle = element_text(size = 11, color = "gray40")
    )
  )

ggsave("outputs/07_model_comparison.png", p_combined, width = 10, height = 10, dpi = 150)

# 9d. Residual diagnostics
arima_residuals <- tibble(
  date = arima_results$date,
  residual = arima_results$actual - arima_results$forecast
)

p_resid_ts <- ggplot(arima_residuals, aes(x = date, y = residual)) +
  geom_hline(yintercept = 0, linetype = "dashed", color = "gray50") +
  geom_line(color = "#2196F3", alpha = 0.7) +
  geom_point(color = "#2196F3", size = 1, alpha = 0.5) +
  labs(title = "ARIMA Forecast Residuals Over Time", x = NULL, y = "Residual ($/MWh)") +
  theme_minimal(base_size = 11) +
  theme(plot.title = element_text(face = "bold"))

p_resid_hist <- ggplot(arima_residuals, aes(x = residual)) +
  geom_histogram(bins = 25, fill = "#2196F3", alpha = 0.7, color = "white") +
  geom_vline(xintercept = 0, linetype = "dashed", color = "red") +
  labs(title = "Residual Distribution", x = "Residual ($/MWh)", y = "Count") +
  theme_minimal(base_size = 11) +
  theme(plot.title = element_text(face = "bold"))

p_diagnostics <- p_resid_ts | p_resid_hist

ggsave("outputs/08_residual_diagnostics.png", p_diagnostics, width = 12, height = 4.5, dpi = 150)

# --- 10. FORWARD FORECAST (FULL DATASET) ------------------------------------
# Retrain on ALL data and produce forward-looking 90-day forecast

cat("\nRetraining on full dataset for forward forecast...\n")

full_ts <- ts(prices$price_mwh, frequency = 365)
final_arima <- auto.arima(full_ts, seasonal = TRUE, stepwise = TRUE)
final_forecast <- forecast(final_arima, h = 90, level = c(80, 95))

# Build forecast dataframe
future_dates_seq <- seq(max(prices$date) + 1, by = "day", length.out = 90)

forward_forecast <- tibble(
  date     = future_dates_seq,
  forecast = as.numeric(final_forecast$mean),
  lower_80 = as.numeric(final_forecast$lower[, 1]),
  upper_80 = as.numeric(final_forecast$upper[, 1]),
  lower_95 = as.numeric(final_forecast$lower[, 2]),
  upper_95 = as.numeric(final_forecast$upper[, 2])
)

p_forward <- ggplot() +
  geom_line(data = prices %>% tail(180), aes(x = date, y = price_mwh),
            color = "gray50", linewidth = 0.4) +
  geom_ribbon(data = forward_forecast,
              aes(x = date, ymin = lower_95, ymax = upper_95),
              fill = "#4CAF50", alpha = 0.15) +
  geom_ribbon(data = forward_forecast,
              aes(x = date, ymin = lower_80, ymax = upper_80),
              fill = "#4CAF50", alpha = 0.25) +
  geom_line(data = forward_forecast, aes(x = date, y = forecast),
            color = "#2E7D32", linewidth = 1, linetype = "dashed") +
  geom_vline(xintercept = max(prices$date), linetype = "dotted", color = "gray40") +
  annotate("text", x = max(prices$date) + 3, y = max(forward_forecast$upper_95),
           label = "Forecast", hjust = 0, size = 3.5, fontface = "bold", color = "#2E7D32") +
  labs(
    title = "90-Day Forward Price Forecast - Mid-Columbia Hub",
    subtitle = paste0("Projected average: $",
                      round(mean(forward_forecast$forecast), 2),
                      "/MWh | 95% CI: $",
                      round(min(forward_forecast$lower_95), 2), " - $",
                      round(max(forward_forecast$upper_95), 2), "/MWh"),
    x = NULL, y = "Price ($/MWh)"
  ) +
  theme_minimal(base_size = 11) +
  theme(plot.title = element_text(face = "bold", size = 13))

ggsave("outputs/09_forward_forecast.png", p_forward, width = 10, height = 5.5, dpi = 150)

# --- 11. SUMMARY TABLE -------------------------------------------------------

cat("\n============================================================\n")
cat("FORECAST SUMMARY\n")
cat("============================================================\n")
cat("Forward 90-day projected average: $",
    round(mean(forward_forecast$forecast), 2), "/MWh\n")
cat("95% confidence interval: $",
    round(min(forward_forecast$lower_95), 2), "- $",
    round(max(forward_forecast$upper_95), 2), "/MWh\n")
cat("Best backtest model:", comparison$Model[which.min(comparison$RMSE)],
    "(lowest RMSE)\n")
cat("All plots saved to outputs/ directory\n")
cat("============================================================\n")

# --- 12. EXPORT FORECAST TO CSV ----------------------------------------------

write_csv(forward_forecast, "outputs/forward_forecast_90day.csv")
write_csv(comparison, "outputs/model_comparison_metrics.csv")

cat("\nDone! Forecast data exported to outputs/\n")
