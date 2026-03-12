"""
credit_model.py
---------------
Counterparty credit risk scoring model.
Mirrors the kind of internal credit limit assessment model used by
energy trading risk teams (like Seattle City Light's Risk Oversight Division).

Approach: Weighted scoring of key financial ratios, normalized to a 0-100
risk score. Lower score = lower risk = higher credit limit approved.
"""

import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler


# ── Weight configuration ──────────────────────────────────────────────────────
# Based on standard energy credit risk frameworks (e.g., Moody's, S&P methodology)
WEIGHTS = {
    "liquidity_score":   0.20,   # current ratio
    "leverage_score":    0.25,   # debt-to-equity
    "coverage_score":    0.20,   # interest coverage ratio
    "profitability_score": 0.15, # net profit margin
    "utilization_score": 0.15,   # credit line utilization
    "relationship_score": 0.05,  # years of trading relationship
}

# Maximum credit limits by rating tier (in $MM)
CREDIT_LIMITS = {
    "A":   200.0,
    "BBB": 120.0,
    "BB":   60.0,
    "B":    20.0,
}


def score_counterparty(row: pd.Series) -> dict:
    """
    Score a single counterparty on each financial dimension.
    Each component scored 0-100 (0 = best, 100 = worst / most risky).
    """
    # Liquidity: current ratio (>2 = excellent, <1 = danger)
    liquidity = np.clip((2.0 - row["current_ratio"]) / 2.0 * 100, 0, 100)

    # Leverage: debt/equity (lower is better, >3 is high risk)
    leverage = np.clip(row["debt_to_equity"] / 3.5 * 100, 0, 100)

    # Coverage: interest coverage ratio (>5 = comfortable, <2 = stressed)
    coverage = np.clip((5.0 - row["interest_coverage"]) / 5.0 * 100, 0, 100)

    # Profitability: net profit margin (negative = risky)
    profitability = np.clip((-row["net_profit_margin"] + 0.05) / 0.30 * 100, 0, 100)

    # Credit utilization: higher = riskier
    utilization = row["credit_utilization"] * 100

    # Relationship tenure: longer = more trust (inverse risk)
    relationship = np.clip((20 - row["years_relationship"]) / 20 * 100, 0, 100)

    composite = (
        WEIGHTS["liquidity_score"]    * liquidity
        + WEIGHTS["leverage_score"]   * leverage
        + WEIGHTS["coverage_score"]   * coverage
        + WEIGHTS["profitability_score"] * profitability
        + WEIGHTS["utilization_score"] * utilization
        + WEIGHTS["relationship_score"] * relationship
    )

    # Map composite score to internal rating
    if composite < 30:
        rating = "A"
    elif composite < 50:
        rating = "BBB"
    elif composite < 70:
        rating = "BB"
    else:
        rating = "B"

    # Credit limit: inversely scaled by risk, capped by rating tier
    base_limit = (100 - composite) * 2.5
    credit_limit = round(min(base_limit, CREDIT_LIMITS[rating]), 1)

    return {
        "liquidity_score":     round(liquidity, 1),
        "leverage_score":      round(leverage, 1),
        "coverage_score":      round(coverage, 1),
        "profitability_score": round(profitability, 1),
        "utilization_score":   round(utilization, 1),
        "relationship_score":  round(relationship, 1),
        "composite_risk_score": round(composite, 1),
        "model_rating":        rating,
        "recommended_limit_mm": credit_limit,
    }


def run_credit_assessment(df: pd.DataFrame) -> pd.DataFrame:
    """
    Run full credit assessment for all counterparties.
    Returns enriched DataFrame with scoring components and recommendations.
    """
    scores = df.apply(score_counterparty, axis=1, result_type="expand")
    result = pd.concat([df[["counterparty", "current_ratio", "debt_to_equity",
                             "interest_coverage", "net_profit_margin",
                             "revenue_b", "credit_utilization", "years_relationship"]],
                        scores], axis=1)

    # Flag counterparties needing review
    result["review_flag"] = result["composite_risk_score"] > 60
    result["watchlist"] = result["composite_risk_score"] > 75

    return result.sort_values("composite_risk_score")


if __name__ == "__main__":
    df = pd.read_csv("../data/counterparties.csv")
    results = run_credit_assessment(df)
    print("\n=== Credit Assessment Results ===")
    print(results[["counterparty", "composite_risk_score", "model_rating",
                   "recommended_limit_mm", "review_flag"]].to_string(index=False))
    results.to_csv("../data/credit_assessment.csv", index=False)
    print("\nSaved to data/credit_assessment.csv")
