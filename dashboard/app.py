"""
app.py — Energy Risk Analytics Dashboard
-----------------------------------------
Interactive dashboard for Seattle City Light's wholesale energy trading
risk monitoring. Built with Plotly Dash + Python.

Run: python dashboard/app.py
Then open: http://127.0.0.1:8050
"""

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import dash
from dash import dcc, html, Input, Output, dash_table
import warnings
warnings.filterwarnings("ignore")

from analytics.risk_metrics import (
    rolling_volatility, historical_var, parametric_var,
    expected_shortfall, rolling_stats, price_spike_detector, summary_stats
)
from models.credit_model import run_credit_assessment

# ── Load Data ─────────────────────────────────────────────────────────────────
BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

prices_df = pd.read_csv(os.path.join(BASE, "data", "prices.csv"), parse_dates=["date"])
cp_df     = pd.read_csv(os.path.join(BASE, "data", "counterparties.csv"))
credit_df = run_credit_assessment(cp_df)

prices = prices_df.set_index("date")["price_mwh"]

# Pre-compute risk metrics
vol_30   = rolling_volatility(prices, 30)
var_hist = historical_var(prices, 0.95, 252)
var_par  = parametric_var(prices, 0.95, 30)
cvar     = expected_shortfall(prices, 0.95, 252)
rstats   = rolling_stats(prices, 30)
spikes   = price_spike_detector(prices)
sumstats = summary_stats(prices)

# ── Color Palette ─────────────────────────────────────────────────────────────
C = {
    "bg":       "#0f1117",
    "surface":  "#1a1d27",
    "border":   "#2d3147",
    "accent":   "#4f8ef7",
    "green":    "#22c55e",
    "red":      "#ef4444",
    "yellow":   "#f59e0b",
    "purple":   "#a855f7",
    "text":     "#e2e8f0",
    "muted":    "#64748b",
}

FONT = "Inter, system-ui, sans-serif"

def card(children, style=None):
    base = {
        "background": C["surface"],
        "border": f"1px solid {C['border']}",
        "borderRadius": "12px",
        "padding": "20px",
    }
    if style:
        base.update(style)
    return html.Div(children, style=base)

def kpi(label, value, unit="", color=None):
    return html.Div([
        html.Div(label, style={"fontSize": "11px", "color": C["muted"],
                               "textTransform": "uppercase", "letterSpacing": "0.08em",
                               "marginBottom": "6px"}),
        html.Div([
            html.Span(str(value), style={"fontSize": "28px", "fontWeight": "700",
                                          "color": color or C["text"]}),
            html.Span(f" {unit}", style={"fontSize": "13px", "color": C["muted"]}),
        ])
    ])

# ── App Layout ────────────────────────────────────────────────────────────────
app = dash.Dash(__name__, title="Energy Risk Analytics | SCL")

app.layout = html.Div(style={
    "background": C["bg"], "minHeight": "100vh",
    "fontFamily": FONT, "color": C["text"], "padding": "24px"
}, children=[

    # Header
    html.Div([
        html.Div([
            html.H1("⚡ Energy Risk Analytics Dashboard",
                    style={"margin": 0, "fontSize": "22px", "fontWeight": "700"}),
            html.Div("Mid-Columbia Hub · Pacific Northwest · 2022–2024",
                     style={"color": C["muted"], "fontSize": "13px", "marginTop": "4px"}),
        ]),
        html.Div([
            html.Div("LIVE DEMO", style={
                "background": C["accent"], "color": "#fff",
                "borderRadius": "6px", "padding": "4px 12px",
                "fontSize": "11px", "fontWeight": "700", "letterSpacing": "0.1em"
            })
        ])
    ], style={"display": "flex", "justifyContent": "space-between",
              "alignItems": "center", "marginBottom": "24px"}),

    # Tabs
    dcc.Tabs(id="tabs", value="market", style={"marginBottom": "20px"},
             colors={"border": C["border"], "primary": C["accent"], "background": C["surface"]},
             children=[
        dcc.Tab(label="📈 Market Overview",    value="market",
                style={"color": C["muted"], "background": C["surface"]},
                selected_style={"color": C["text"], "background": C["bg"], "borderTop": f"2px solid {C['accent']}"}),
        dcc.Tab(label="⚠️ Risk Metrics",       value="risk",
                style={"color": C["muted"], "background": C["surface"]},
                selected_style={"color": C["text"], "background": C["bg"], "borderTop": f"2px solid {C['accent']}"}),
        dcc.Tab(label="🏦 Credit Assessment",  value="credit",
                style={"color": C["muted"], "background": C["surface"]},
                selected_style={"color": C["text"], "background": C["bg"], "borderTop": f"2px solid {C['accent']}"}),
    ]),

    html.Div(id="tab-content"),

    # Date Range Selector (shared)
    card([
        html.Div("Date Range Filter", style={"fontSize": "12px", "color": C["muted"], "marginBottom": "10px"}),
        dcc.DatePickerRange(
            id="date-range",
            min_date_allowed=prices_df["date"].min(),
            max_date_allowed=prices_df["date"].max(),
            start_date=prices_df["date"].min(),
            end_date=prices_df["date"].max(),
            style={"color": C["text"]},
        )
    ], style={"marginTop": "20px"}),
])


# ── Tab Routing ───────────────────────────────────────────────────────────────
@app.callback(Output("tab-content", "children"), Input("tabs", "value"))
def render_tab(tab):
    if tab == "market":
        return market_layout()
    elif tab == "risk":
        return risk_layout()
    elif tab == "credit":
        return credit_layout()


def market_layout():
    return html.Div([
        # KPI Row
        html.Div([
            card(kpi("Avg Price", f"${sumstats['mean_price']}", "/ MWh")),
            card(kpi("Price Range", f"${sumstats['min_price']} – ${sumstats['max_price']}", "MWh")),
            card(kpi("Annualized Volatility", f"{sumstats['annualized_vol']}%", "", C["yellow"])),
            card(kpi("Price Spikes Detected", sumstats["spike_count"], "events", C["red"])),
            card(kpi("95% VaR (daily)", f"{sumstats['var_95_pct']}%", "", C["purple"])),
        ], style={"display": "grid", "gridTemplateColumns": "repeat(5,1fr)", "gap": "12px", "marginBottom": "20px"}),

        # Price chart + Bollinger bands
        card([
            html.H3("Wholesale Electricity Price — Mid-Columbia Hub",
                    style={"margin": "0 0 16px", "fontSize": "14px", "fontWeight": "600"}),
            dcc.Graph(id="price-chart", figure=make_price_chart(), style={"height": "380px"})
        ], {"marginBottom": "16px"}),

        # Volume + Seasonality
        html.Div([
            card([
                html.H3("Daily Trading Volume", style={"margin": "0 0 12px", "fontSize": "14px", "fontWeight": "600"}),
                dcc.Graph(id="volume-chart", figure=make_volume_chart(), style={"height": "260px"})
            ]),
            card([
                html.H3("Price by Season", style={"margin": "0 0 12px", "fontSize": "14px", "fontWeight": "600"}),
                dcc.Graph(id="season-chart", figure=make_season_chart(), style={"height": "260px"})
            ]),
        ], style={"display": "grid", "gridTemplateColumns": "1fr 1fr", "gap": "16px"}),
    ])


def risk_layout():
    return html.Div([
        html.Div([
            card(kpi("Historical VaR (95%)", f"{abs(round(var_hist.dropna().iloc[-1]*100, 2))}%", "daily loss", C["red"])),
            card(kpi("Parametric VaR (95%)", f"{abs(round(var_par.dropna().iloc[-1]*100, 2))}%", "daily loss", C["yellow"])),
            card(kpi("Expected Shortfall", f"{abs(round(cvar.dropna().iloc[-1]*100, 2))}%", "CVaR", C["purple"])),
            card(kpi("30-day Volatility", f"{round(vol_30.dropna().iloc[-1]*100, 1)}%", "annualized", C["accent"])),
        ], style={"display": "grid", "gridTemplateColumns": "repeat(4,1fr)", "gap": "12px", "marginBottom": "20px"}),

        card([
            html.H3("Rolling Volatility & Value at Risk (VaR)",
                    style={"margin": "0 0 16px", "fontSize": "14px", "fontWeight": "600"}),
            dcc.Graph(figure=make_var_chart(), style={"height": "380px"})
        ], {"marginBottom": "16px"}),

        card([
            html.H3("Return Distribution & Risk Profile",
                    style={"margin": "0 0 16px", "fontSize": "14px", "fontWeight": "600"}),
            dcc.Graph(figure=make_return_dist(), style={"height": "300px"})
        ]),
    ])


def credit_layout():
    rating_colors = {"A": C["green"], "BBB": C["accent"], "BB": C["yellow"], "B": C["red"]}

    return html.Div([
        html.Div([
            card(kpi("Counterparties Assessed", len(credit_df))),
            card(kpi("A-Rated", len(credit_df[credit_df["model_rating"]=="A"]), "", C["green"])),
            card(kpi("BBB-Rated", len(credit_df[credit_df["model_rating"]=="BBB"]), "", C["accent"])),
            card(kpi("BB/B-Rated", len(credit_df[credit_df["model_rating"].isin(["BB","B"])]), "", C["red"])),
            card(kpi("Total Credit Exposure", f"${round(credit_df['recommended_limit_mm'].sum(), 0):.0f}", "MM")),
        ], style={"display": "grid", "gridTemplateColumns": "repeat(5,1fr)", "gap": "12px", "marginBottom": "20px"}),

        html.Div([
            card([
                html.H3("Risk Score by Counterparty", style={"margin": "0 0 12px", "fontSize": "14px", "fontWeight": "600"}),
                dcc.Graph(figure=make_credit_bar(), style={"height": "380px"})
            ]),
            card([
                html.H3("Credit Limit vs Risk Score", style={"margin": "0 0 12px", "fontSize": "14px", "fontWeight": "600"}),
                dcc.Graph(figure=make_credit_scatter(), style={"height": "380px"})
            ]),
        ], style={"display": "grid", "gridTemplateColumns": "1fr 1fr", "gap": "16px", "marginBottom": "16px"}),

        card([
            html.H3("Counterparty Credit Summary Table",
                    style={"margin": "0 0 16px", "fontSize": "14px", "fontWeight": "600"}),
            dash_table.DataTable(
                data=credit_df[[
                    "counterparty", "current_ratio", "debt_to_equity",
                    "interest_coverage", "composite_risk_score",
                    "model_rating", "recommended_limit_mm", "review_flag"
                ]].round(2).to_dict("records"),
                columns=[
                    {"name": "Counterparty",       "id": "counterparty"},
                    {"name": "Current Ratio",      "id": "current_ratio"},
                    {"name": "D/E Ratio",          "id": "debt_to_equity"},
                    {"name": "Interest Coverage",  "id": "interest_coverage"},
                    {"name": "Risk Score",         "id": "composite_risk_score"},
                    {"name": "Rating",             "id": "model_rating"},
                    {"name": "Credit Limit ($MM)", "id": "recommended_limit_mm"},
                    {"name": "Review Flag",        "id": "review_flag"},
                ],
                style_table={"overflowX": "auto"},
                style_cell={"background": C["surface"], "color": C["text"],
                            "border": f"1px solid {C['border']}", "textAlign": "left",
                            "padding": "10px 14px", "fontSize": "13px"},
                style_header={"background": C["bg"], "fontWeight": "700",
                              "color": C["muted"], "border": f"1px solid {C['border']}"},
                style_data_conditional=[
                    {"if": {"filter_query": "{model_rating} = 'A'", "column_id": "model_rating"},
                     "color": C["green"], "fontWeight": "700"},
                    {"if": {"filter_query": "{model_rating} = 'BBB'", "column_id": "model_rating"},
                     "color": C["accent"], "fontWeight": "700"},
                    {"if": {"filter_query": "{model_rating} = 'BB'", "column_id": "model_rating"},
                     "color": C["yellow"], "fontWeight": "700"},
                    {"if": {"filter_query": "{model_rating} = 'B'", "column_id": "model_rating"},
                     "color": C["red"], "fontWeight": "700"},
                    {"if": {"filter_query": "{review_flag} = True"},
                     "background": "rgba(239,68,68,0.08)"},
                ],
                sort_action="native",
                page_size=15,
            )
        ]),
    ])


# ── Chart Builders ────────────────────────────────────────────────────────────
def dark_layout(fig):
    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(family=FONT, color=C["text"], size=12),
        margin=dict(l=10, r=10, t=10, b=10),
        legend=dict(bgcolor="rgba(0,0,0,0)", borderwidth=0),
        xaxis=dict(gridcolor=C["border"], zerolinecolor=C["border"]),
        yaxis=dict(gridcolor=C["border"], zerolinecolor=C["border"]),
    )
    return fig


def make_price_chart():
    fig = go.Figure()
    spike_idx = prices.index[spikes]

    # Bollinger bands
    fig.add_trace(go.Scatter(x=rstats.index, y=rstats["rolling_upper_band"],
                             fill=None, line=dict(width=0), showlegend=False, name="Upper Band"))
    fig.add_trace(go.Scatter(x=rstats.index, y=rstats["rolling_lower_band"],
                             fill="tonexty", fillcolor="rgba(79,142,247,0.08)",
                             line=dict(width=0), name="Bollinger Band (±2σ)"))

    # Price line
    fig.add_trace(go.Scatter(x=prices.index, y=prices.values,
                             line=dict(color=C["accent"], width=1.5),
                             name="Electricity Price ($/MWh)"))

    # Rolling mean
    fig.add_trace(go.Scatter(x=rstats.index, y=rstats["rolling_mean"],
                             line=dict(color=C["yellow"], width=2, dash="dash"),
                             name="30-day Rolling Mean"))

    # Spike markers
    fig.add_trace(go.Scatter(x=spike_idx, y=prices[spike_idx],
                             mode="markers", marker=dict(color=C["red"], size=7, symbol="triangle-up"),
                             name="Price Spike"))

    fig.update_yaxes(title_text="$/MWh")
    return dark_layout(fig)


def make_volume_chart():
    fig = go.Figure(go.Bar(
        x=prices_df["date"], y=prices_df["volume_mwh"],
        marker_color=C["accent"], opacity=0.7, name="Volume (MWh)"
    ))
    fig.update_yaxes(title_text="MWh")
    return dark_layout(fig)


def make_season_chart():
    season_stats = prices_df.groupby("season")["price_mwh"].agg(["mean", "std"]).reset_index()
    season_order = ["Winter", "Spring", "Summer", "Fall"]
    season_stats["season"] = pd.Categorical(season_stats["season"], categories=season_order, ordered=True)
    season_stats = season_stats.sort_values("season")

    colors_map = {"Winter": C["accent"], "Spring": C["green"], "Summer": C["red"], "Fall": C["yellow"]}
    fig = go.Figure(go.Bar(
        x=season_stats["season"],
        y=season_stats["mean"].round(2),
        error_y=dict(type="data", array=season_stats["std"].round(2), visible=True),
        marker_color=[colors_map[s] for s in season_stats["season"]],
        name="Avg Price"
    ))
    fig.update_yaxes(title_text="Avg $/MWh")
    return dark_layout(fig)


def make_var_chart():
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True,
                        row_heights=[0.6, 0.4], vertical_spacing=0.08)

    fig.add_trace(go.Scatter(x=prices.index, y=prices.values,
                             line=dict(color=C["accent"], width=1.2),
                             name="Price"), row=1, col=1)

    fig.add_trace(go.Scatter(x=vol_30.index, y=(vol_30 * 100).values,
                             line=dict(color=C["yellow"], width=1.5),
                             name="30-day Vol (%)"), row=2, col=1)

    fig.add_trace(go.Scatter(x=var_hist.index, y=(var_hist.abs() * 100).values,
                             line=dict(color=C["red"], width=1.5, dash="dash"),
                             name="Hist VaR 95%"), row=2, col=1)

    fig.add_trace(go.Scatter(x=cvar.index, y=(cvar.abs() * 100).values,
                             line=dict(color=C["purple"], width=1.5, dash="dot"),
                             name="CVaR 95%"), row=2, col=1)

    fig.update_yaxes(title_text="$/MWh", row=1, col=1,
                     gridcolor=C["border"], zerolinecolor=C["border"])
    fig.update_yaxes(title_text="% Daily Loss", row=2, col=1,
                     gridcolor=C["border"], zerolinecolor=C["border"])
    fig.update_xaxes(gridcolor=C["border"])

    fig.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                      font=dict(family=FONT, color=C["text"]),
                      legend=dict(bgcolor="rgba(0,0,0,0)"),
                      margin=dict(l=10, r=10, t=10, b=10))
    return fig


def make_return_dist():
    returns = np.log(prices / prices.shift(1)).dropna() * 100
    var_line = float(np.percentile(returns, 5))

    fig = go.Figure()
    fig.add_trace(go.Histogram(x=returns, nbinsx=60, name="Daily Returns",
                               marker_color=C["accent"], opacity=0.7))
    fig.add_vline(x=var_line, line=dict(color=C["red"], dash="dash", width=2),
                  annotation_text=f"VaR 95%: {var_line:.2f}%",
                  annotation_font_color=C["red"])
    fig.update_xaxes(title_text="Daily Return (%)", gridcolor=C["border"])
    fig.update_yaxes(title_text="Frequency", gridcolor=C["border"])
    return dark_layout(fig)


def make_credit_bar():
    df_sorted = credit_df.sort_values("composite_risk_score")
    color_map = {"A": C["green"], "BBB": C["accent"], "BB": C["yellow"], "B": C["red"]}
    fig = go.Figure(go.Bar(
        x=df_sorted["composite_risk_score"],
        y=df_sorted["counterparty"],
        orientation="h",
        marker_color=[color_map.get(r, C["muted"]) for r in df_sorted["model_rating"]],
        text=df_sorted["model_rating"],
        textposition="outside",
    ))
    fig.update_xaxes(title_text="Composite Risk Score (0=safest, 100=riskiest)", gridcolor=C["border"])
    fig.update_yaxes(tickfont=dict(size=11), gridcolor=C["border"])
    return dark_layout(fig)


def make_credit_scatter():
    color_map = {"A": C["green"], "BBB": C["accent"], "BB": C["yellow"], "B": C["red"]}
    fig = go.Figure()
    for rating in ["A", "BBB", "BB", "B"]:
        sub = credit_df[credit_df["model_rating"] == rating]
        if len(sub):
            fig.add_trace(go.Scatter(
                x=sub["composite_risk_score"],
                y=sub["recommended_limit_mm"],
                mode="markers+text",
                text=sub["counterparty"].str.split().str[-1],
                textposition="top center",
                textfont=dict(size=9),
                marker=dict(color=color_map[rating], size=12),
                name=f"Rating: {rating}"
            ))
    fig.update_xaxes(title_text="Risk Score", gridcolor=C["border"])
    fig.update_yaxes(title_text="Credit Limit ($MM)", gridcolor=C["border"])
    return dark_layout(fig)


if __name__ == "__main__":
    print("Starting Energy Risk Analytics Dashboard...")
    print("Open http://127.0.0.1:8050 in your browser")
    app.run(debug=True, host="0.0.0.0", port=8050)
