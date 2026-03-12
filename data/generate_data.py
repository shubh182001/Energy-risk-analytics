"""
generate_data.py
----------------
Generates synthetic Pacific Northwest wholesale electricity price data
mimicking real EIA Mid-Columbia (Mid-C) hub price patterns.

To use real EIA data instead, replace this with EIA API v2 calls:
  GET https://api.eia.gov/v2/electricity/wholesale-power/...
  (requires free API key from https://www.eia.gov/opendata/)
"""

import numpy as np
import pandas as pd
from datetime import datetime, timedelta

np.random.seed(42)

def generate_electricity_prices(start="2022-01-01", end="2024-12-31"):
    """
    Simulate hourly wholesale electricity prices ($/MWh) for Mid-Columbia hub.
    Includes realistic seasonality, daily peaks, volatility spikes, and mean reversion.
    """
    dates = pd.date_range(start=start, end=end, freq="D")
    n = len(dates)

    # Base price with mean reversion (Ornstein-Uhlenbeck process)
    mu = 45.0       # long-run mean $/MWh
    theta = 0.12    # mean reversion speed
    sigma = 8.0     # base volatility

    prices = np.zeros(n)
    prices[0] = mu
    for t in range(1, n):
        dW = np.random.normal(0, 1)
        prices[t] = prices[t-1] + theta * (mu - prices[t-1]) + sigma * dW

    # Seasonal component: higher in winter (heating) and summer (cooling)
    day_of_year = np.array([d.timetuple().tm_yday for d in dates])
    seasonal = (
        6 * np.sin(2 * np.pi * day_of_year / 365 + np.pi)   # winter peak
        + 4 * np.sin(4 * np.pi * day_of_year / 365)          # summer secondary peak
    )

    # Occasional price spikes (demand surges, transmission constraints)
    spikes = np.zeros(n)
    spike_idx = np.random.choice(n, size=int(n * 0.03), replace=False)
    spikes[spike_idx] = np.random.exponential(30, size=len(spike_idx))

    prices = np.maximum(prices + seasonal + spikes, 5.0)  # floor at $5

    df = pd.DataFrame({
        "date": dates,
        "price_mwh": prices.round(2),
        "volume_mwh": np.random.normal(1500, 300, n).clip(500).round(0),
        "region": "Mid-Columbia (Mid-C)",
        "season": [
            "Winter" if d.month in [12,1,2] else
            "Spring" if d.month in [3,4,5] else
            "Summer" if d.month in [6,7,8] else "Fall"
            for d in dates
        ]
    })
    return df


def generate_counterparty_data():
    """
    Simulate financial data for 15 energy trading counterparties.
    Metrics mirror what a credit analyst would review in annual counterparty review.
    """
    np.random.seed(7)
    companies = [
        "Puget Sound Energy", "Pacific Power", "Portland General Electric",
        "Avista Corporation", "NV Energy", "Idaho Power",
        "PacifiCorp", "Bonneville Power", "Powerex Corp",
        "Shell Energy", "BP Energy", "EDF Trading",
        "Macquarie Energy", "Goldman Sachs Commodities", "Morgan Stanley Energy"
    ]

    data = []
    for company in companies:
        current_ratio      = round(np.random.uniform(0.8, 2.8), 2)
        debt_to_equity     = round(np.random.uniform(0.3, 3.5), 2)
        interest_coverage  = round(np.random.uniform(1.5, 10.0), 2)
        net_profit_margin  = round(np.random.uniform(-0.05, 0.25), 3)
        revenue_b          = round(np.random.uniform(0.5, 25.0), 2)  # billions
        credit_utilization = round(np.random.uniform(0.05, 0.90), 2)
        years_relationship = int(np.random.randint(1, 20))

        # Score: higher is riskier (0-100 scale)
        risk_score = (
            max(0, (2.0 - current_ratio) * 10)
            + min(30, debt_to_equity * 8)
            + max(0, (5 - interest_coverage) * 4)
            + max(0, (-net_profit_margin) * 100)
            + credit_utilization * 20
            - min(10, years_relationship * 0.5)
        )
        risk_score = round(np.clip(risk_score + np.random.normal(0, 3), 5, 95), 1)

        if risk_score < 30:
            rating = "A"
        elif risk_score < 50:
            rating = "BBB"
        elif risk_score < 70:
            rating = "BB"
        else:
            rating = "B"

        credit_limit_mm = round(
            max(5, (100 - risk_score) * np.random.uniform(1.5, 2.5)), 1
        )

        data.append({
            "counterparty": company,
            "current_ratio": current_ratio,
            "debt_to_equity": debt_to_equity,
            "interest_coverage": interest_coverage,
            "net_profit_margin": net_profit_margin,
            "revenue_b": revenue_b,
            "credit_utilization": credit_utilization,
            "years_relationship": years_relationship,
            "risk_score": risk_score,
            "internal_rating": rating,
            "credit_limit_mm": credit_limit_mm
        })

    return pd.DataFrame(data)


if __name__ == "__main__":
    prices_df = generate_electricity_prices()
    prices_df.to_csv("prices.csv", index=False)
    print(f"Generated {len(prices_df)} days of price data")

    cp_df = generate_counterparty_data()
    cp_df.to_csv("counterparties.csv", index=False)
    print(f"Generated {len(cp_df)} counterparty records")
    print(cp_df[["counterparty", "risk_score", "internal_rating", "credit_limit_mm"]].to_string())
