# ⚡ Energy Risk Analytics & Reporting Dashboard

An interactive risk analytics platform for wholesale electricity trading, built to demonstrate core competencies aligned with energy trading mid-office functions — including market risk monitoring, price analytics, and counterparty credit assessment.

> **Context:** Built as a portfolio project for the Seattle City Light Energy Risk Analytics & Reporting Graduate Intern role. The data simulates Pacific Northwest Mid-Columbia (Mid-C) hub wholesale electricity prices — the exact market Seattle City Light transacts in.

---

## 📊 Dashboard Preview

Three-tab interactive dashboard built with Python + Plotly Dash:

| Tab | What It Shows |
|-----|---------------|
| **Market Overview** | Price trends, Bollinger Bands, spike detection, seasonal analysis, trading volume |
| **Risk Metrics** | Rolling volatility, Historical VaR, Parametric VaR, Expected Shortfall (CVaR), return distribution |
| **Credit Assessment** | Counterparty risk scoring, internal ratings, credit limit recommendations, watchlist flags |

---

## 🗂️ Project Structure

```
energy-risk-analytics/
├── data/
│   ├── generate_data.py        # Synthetic data generator (swap for EIA API)
│   ├── prices.csv              # Daily electricity prices, 2022–2024
│   └── counterparties.csv      # Counterparty financial statements
├── analytics/
│   └── risk_metrics.py         # VaR, CVaR, volatility, spike detection
├── models/
│   └── credit_model.py         # Counterparty credit scoring model
├── dashboard/
│   └── app.py                  # Plotly Dash application
├── requirements.txt
└── README.md
```

---

## 🚀 Getting Started

### 1. Clone the repo
```bash
git clone https://github.com/YOUR_USERNAME/energy-risk-analytics.git
cd energy-risk-analytics
```

### 2. Install dependencies
```bash
pip install -r requirements.txt
```

### 3. Generate data
```bash
python data/generate_data.py
```

### 4. Run the dashboard
```bash
python dashboard/app.py
```

Open **http://127.0.0.1:8050** in your browser.

---

## 📐 Methodology

### Risk Metrics

**Historical VaR (95%):**
Answers "what is the worst daily loss we'd expect 95% of the time?"
Computed as the 5th percentile of a rolling 252-day return window.

```
VaR_hist = Percentile(Returns_window, 5%)
```

**Parametric VaR (95%):**
Assumes normally distributed returns — faster for daily reporting.
```
VaR_param = μ + z_{0.05} × σ
```

**Expected Shortfall / CVaR:**
Average of all losses beyond the VaR threshold — more conservative and preferred by modern risk frameworks (Basel III).
```
CVaR = E[Loss | Loss > VaR]
```

**Annualized Volatility:**
```
σ_annual = σ_daily × √252
```

### Credit Scoring Model

Weighted scoring across five financial dimensions:

| Factor | Weight | Rationale |
|--------|--------|-----------|
| Leverage (D/E ratio) | 25% | Primary solvency indicator |
| Liquidity (Current ratio) | 20% | Short-term payment ability |
| Coverage (Interest coverage) | 20% | Debt service capacity |
| Profitability (Net margin) | 15% | Business sustainability |
| Credit utilization | 15% | Current exposure relative to limits |
| Relationship tenure | 5% | Historical reliability adjustment |

Composite score (0–100) maps to internal ratings: **A / BBB / BB / B**

---

## 🔌 Using Real EIA Data

To replace synthetic data with real EIA Mid-Columbia hub prices:

1. Get a free API key at [eia.gov/opendata](https://www.eia.gov/opendata/)
2. Replace the `generate_electricity_prices()` function in `data/generate_data.py` with:

```python
import requests

def fetch_eia_prices(api_key, start="2022-01-01", end="2024-12-31"):
    url = "https://api.eia.gov/v2/electricity/wholesale-power/data/"
    params = {
        "api_key": api_key,
        "frequency": "daily",
        "data[0]": "price",
        "facets[location][]": "MIDC",   # Mid-Columbia hub
        "start": start,
        "end": end,
        "sort[0][column]": "period",
        "sort[0][direction]": "asc",
    }
    r = requests.get(url, params=params)
    return pd.DataFrame(r.json()["response"]["data"])
```

---

## 🛠️ Tech Stack

| Tool | Purpose |
|------|---------|
| Python | Core language |
| Pandas / NumPy | Data manipulation |
| SciPy | Statistical computations (VaR, distributions) |
| scikit-learn | Data preprocessing for credit model |
| Plotly Dash | Interactive dashboard |
| EIA API (optional) | Real wholesale electricity price data |

---

## 💡 Key Takeaways for Interview

1. **Market Risk:** Price volatility in PNW electricity markets is seasonal and spike-prone — risk teams need rolling VaR + CVaR to set position limits and capital reserves.

2. **Credit Risk:** Annual counterparty review involves scoring financial ratios against internal benchmarks — this model automates that process with a transparent, auditable scoring framework.

3. **ETRM Alignment:** A platform like this could integrate with an Energy Trading & Risk Management (ETRM) system (e.g., OPIS, Triple Point) to pull live trade data and generate automated daily risk reports — reducing manual work for risk analysts.

---

*Built by Shubh Dhar — MSIM, University of Washington Information School*
