"""
risk_scorer.py
==============
Visa Risk Solutions — Chargeback Prevention Engine
---------------------------------------------------
Translates raw model output (chargeback probability 0–1) into:
  • A normalised 0–100 risk score
  • A human-readable risk tier (LOW / MEDIUM / HIGH / CRITICAL)
  • A recommended intervention (no_action / customer_confirmation /
    additional_auth / merchant_review)
  • A plain-English explanation of the top risk drivers

All business rules are isolated here so they can be updated by the
risk-ops team without touching the ML pipeline or the UI.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional
import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass
class RiskAssessment:
    """
    Fully self-contained risk verdict for a single transaction.

    Attributes
    ----------
    probability      : Raw chargeback probability from the model (0–1).
    risk_score       : Normalised 0–100 score (higher = riskier).
    risk_tier        : Categorical label: LOW | MEDIUM | HIGH | CRITICAL.
    intervention     : Recommended action code.
    intervention_label: Human-readable version of the action.
    explanation      : Plain-English list of the top risk drivers.
    feature_contributions: Dict mapping feature name → contribution magnitude.
    """
    probability: float
    risk_score: int
    risk_tier: str
    intervention: str
    intervention_label: str
    explanation: list[str]
    feature_contributions: dict[str, float]


# ---------------------------------------------------------------------------
# Threshold configuration
# ---------------------------------------------------------------------------

# Probability boundaries for risk tiers.
# Calibrated against the 6.6 % base rate in the training dataset.
TIER_THRESHOLDS = {
    "LOW":      (0.00, 0.15),
    "MEDIUM":   (0.15, 0.35),
    "HIGH":     (0.35, 0.60),
    "CRITICAL": (0.60, 1.01),
}

# Tier → intervention mapping (business rule, not ML).
TIER_INTERVENTIONS = {
    "LOW":      ("no_action",              "No Action Required"),
    "MEDIUM":   ("customer_confirmation",  "Customer Confirmation"),
    "HIGH":     ("additional_auth",        "Additional Authentication"),
    "CRITICAL": ("merchant_review",        "Merchant Review & Hold"),
}

# Tier display colours (Streamlit-compatible hex values).
TIER_COLOURS = {
    "LOW":      "#2ecc71",   # green
    "MEDIUM":   "#f39c12",   # amber
    "HIGH":     "#e74c3c",   # red
    "CRITICAL": "#8e44ad",   # purple
}


# ---------------------------------------------------------------------------
# Core scoring logic
# ---------------------------------------------------------------------------

def probability_to_score(probability: float) -> int:
    """
    Map a raw model probability (0–1) to a rounded 0–100 risk score.

    Uses a mild power transform (x^0.7) to spread scores across the
    range instead of clustering near 0 for most low-risk transactions.
    """
    transformed = probability ** 0.7
    return int(round(min(transformed * 100, 100)))


def classify_tier(probability: float) -> str:
    """Return the risk tier label for a given probability."""
    for tier, (lo, hi) in TIER_THRESHOLDS.items():
        if lo <= probability < hi:
            return tier
    return "CRITICAL"   # safety net for probability == 1.0


def get_intervention(tier: str) -> tuple[str, str]:
    """Return (intervention_code, intervention_label) for a risk tier."""
    return TIER_INTERVENTIONS.get(tier, ("merchant_review", "Merchant Review & Hold"))


# ---------------------------------------------------------------------------
# Feature-contribution explainability
# ---------------------------------------------------------------------------

# Human-readable labels for model features shown in the UI.
FEATURE_LABELS = {
    "prior_chargebacks":       "Prior chargebacks on account",
    "chargeback_rate":         "Historical chargeback rate",
    "has_prior_chargeback":    "Account has prior disputes",
    "repeat_offender":         "Repeat dispute pattern",
    "merchant_risk_score":     "Merchant risk level",
    "composite_risk":          "Overall composite risk signal",
    "category_risk_score":     "Merchant category risk",
    "device_risk_score":       "Device / session risk",
    "is_high_risk_category":   "High-risk merchant category",
    "amount_log":              "Transaction amount (scaled)",
    "is_high_value":           "High-value transaction",
    "is_new_customer":         "New customer (< 3 months)",
    "risk_amount_interaction":  "Risky merchant + large amount",
    "is_subscription":         "Subscription billing",
    "is_recent_txn":           "Very recent transaction",
}


def build_explanation(
    feature_row: pd.Series,
    feature_contributions: dict[str, float],
    top_n: int = 4,
) -> list[str]:
    """
    Generate plain-English risk driver sentences for the top N features.

    Parameters
    ----------
    feature_row          : A single row from the feature matrix.
    feature_contributions: Dict of {feature_name: SHAP/importance value}.
    top_n                : Number of drivers to surface.

    Returns
    -------
    List of strings, one per risk driver.
    """
    # Sort by absolute contribution magnitude, descending.
    sorted_feats = sorted(
        feature_contributions.items(),
        key=lambda kv: abs(kv[1]),
        reverse=True,
    )[:top_n]

    explanations: list[str] = []
    for feat, contrib in sorted_feats:
        label = FEATURE_LABELS.get(feat, feat.replace("_", " ").title())
        direction = "↑ increasing" if contrib > 0 else "↓ decreasing"
        explanations.append(f"{label} — {direction} risk")

    return explanations or ["No dominant risk drivers identified."]


# ---------------------------------------------------------------------------
# Rule-based feature-contribution fallback (no SHAP required)
# ---------------------------------------------------------------------------

def simple_feature_contributions(feature_row: pd.Series) -> dict[str, float]:
    """
    Rule-based proxy for SHAP values when the full explainer isn't loaded.

    Assigns contribution weights based on feature values and domain knowledge.
    Used by the Streamlit app for instant, dependency-free explanations.
    """
    contributions: dict[str, float] = {}

    def _add(name: str, value: float) -> None:
        if name in feature_row.index:
            contributions[name] = float(feature_row[name]) * value

    _add("prior_chargebacks",      3.0)
    _add("chargeback_rate",        2.5)
    _add("has_prior_chargeback",   2.0)
    _add("repeat_offender",        1.8)
    _add("merchant_risk_score",    1.5)
    _add("composite_risk",         1.4)
    _add("category_risk_score",    1.3)
    _add("device_risk_score",      1.2)
    _add("is_high_risk_category",  1.1)
    _add("amount_log",             1.0)
    _add("is_high_value",          0.9)
    _add("is_new_customer",        0.8)
    _add("risk_amount_interaction", 0.7)
    _add("is_subscription",        0.5)

    return contributions


# ---------------------------------------------------------------------------
# Main scorer
# ---------------------------------------------------------------------------

def score_transaction(
    probability: float,
    feature_row: pd.Series,
    feature_contributions: Optional[dict[str, float]] = None,
) -> RiskAssessment:
    """
    Produce a complete RiskAssessment for one transaction.

    Parameters
    ----------
    probability          : Chargeback probability from the XGBoost model.
    feature_row          : Single-row pd.Series of engineered features.
    feature_contributions: Optional pre-computed SHAP values; if None,
                           falls back to rule-based contributions.

    Returns
    -------
    RiskAssessment dataclass with all output fields populated.
    """
    risk_score = probability_to_score(probability)
    tier = classify_tier(probability)
    intervention_code, intervention_label = get_intervention(tier)

    if feature_contributions is None:
        feature_contributions = simple_feature_contributions(feature_row)

    explanation = build_explanation(feature_row, feature_contributions)

    return RiskAssessment(
        probability=round(probability, 4),
        risk_score=risk_score,
        risk_tier=tier,
        intervention=intervention_code,
        intervention_label=intervention_label,
        explanation=explanation,
        feature_contributions=feature_contributions,
    )
