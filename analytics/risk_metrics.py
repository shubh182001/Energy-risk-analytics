"""
risk_metrics.py
---------------
Core risk analytics for wholesale electricity price data.
Computes standard energy trading risk metrics used in mid-office functions.
"""

import numpy as np
import pandas as pd
from scipy import stats


def compute_returns(prices: pd.Series) -> pd.Series:
    """Daily log returns of electricity prices."""
    return np.log(prices / prices.shift(1)).dropna()


def rolling_volatility(prices: pd.Series, window: int = 30) -> pd.Series:
    """
    Annualized rolling price volatility.
    Standard measure of market risk in energy trading.
    """
    returns = compute_returns(prices)
    return returns.rolling(window).std() * np.sqrt(252)


def historical_var(prices: pd.Series, confidence: float = 0.95, window: int = 252) -> pd.Series:
    """
    Historical Value at Risk (VaR) — rolling window.
    VaR answers: "What is the maximum loss we expect with X% confidence?"
    Used by risk teams to set position limits and capital reserves.
    """
    returns = compute_returns(prices)
    var_series = returns.rolling(window).quantile(1 - confidence)
    return var_series


def parametric_var(prices: pd.Series, confidence: float = 0.95, window: int = 30) -> pd.Series:
    """
    Parametric (variance-covariance) VaR assuming normal distribution.
    Faster to compute than historical VaR, commonly used for daily reporting.
    """
    returns = compute_returns(prices)
    rolling_mean = returns.rolling(window).mean()
    rolling_std = returns.rolling(window).std()
    z = stats.norm.ppf(1 - confidence)
    return rolling_mean + z * rolling_std


def expected_shortfall(prices: pd.Series, confidence: float = 0.95, window: int = 252) -> pd.Series:
    """
    Expected Shortfall (CVaR) — average loss beyond VaR threshold.
    More conservative than VaR; preferred by regulators post-2008.
    """
    returns = compute_returns(prices)
    def es_calc(x):
        threshold = np.quantile(x, 1 - confidence)
        tail = x[x <= threshold]
        return tail.mean() if len(tail) > 0 else np.nan
    return returns.rolling(window).apply(es_calc, raw=True)


def rolling_stats(prices: pd.Series, window: int = 30) -> pd.DataFrame:
    """
    Compute rolling mean, std, min, max for price trend analysis.
    Used for trading performance monitoring dashboards.
    """
    return pd.DataFrame({
        "rolling_mean": prices.rolling(window).mean(),
        "rolling_std":  prices.rolling(window).std(),
        "rolling_min":  prices.rolling(window).min(),
        "rolling_max":  prices.rolling(window).max(),
        "rolling_upper_band": prices.rolling(window).mean() + 2 * prices.rolling(window).std(),
        "rolling_lower_band": prices.rolling(window).mean() - 2 * prices.rolling(window).std(),
    })


def price_spike_detector(prices: pd.Series, z_threshold: float = 2.5) -> pd.Series:
    """
    Flag price spikes using z-score method.
    Energy markets experience sudden spikes due to demand surges and transmission constraints.
    """
    returns = compute_returns(prices)
    z_scores = np.abs(stats.zscore(returns.dropna()))
    spike_flags = pd.Series(False, index=returns.index)
    spike_flags[returns.index[z_scores > z_threshold]] = True
    return spike_flags.reindex(prices.index, fill_value=False)


def summary_stats(prices: pd.Series) -> dict:
    """High-level summary statistics for executive reporting."""
    returns = compute_returns(prices)
    var_95 = float(np.percentile(returns.dropna(), 5))
    return {
        "mean_price":       round(prices.mean(), 2),
        "median_price":     round(prices.median(), 2),
        "std_price":        round(prices.std(), 2),
        "min_price":        round(prices.min(), 2),
        "max_price":        round(prices.max(), 2),
        "annualized_vol":   round(returns.std() * np.sqrt(252) * 100, 2),
        "var_95_pct":       round(var_95 * 100, 2),
        "skewness":         round(float(stats.skew(returns.dropna())), 3),
        "kurtosis":         round(float(stats.kurtosis(returns.dropna())), 3),
        "spike_count":      int(price_spike_detector(prices).sum()),
    }


if __name__ == "__main__":
    import sys, os
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
    df = pd.read_csv("../data/prices.csv", parse_dates=["date"])
    prices = df.set_index("date")["price_mwh"]

    stats_out = summary_stats(prices)
    print("=== Summary Statistics ===")
    for k, v in stats_out.items():
        print(f"  {k}: {v}")
