"""
feature_engineering.py
=======================
Visa Risk Solutions — Chargeback Prevention Engine
---------------------------------------------------
Transforms raw transaction data into a rich feature matrix
suitable for gradient-boosted risk modelling.

Design principles
-----------------
* Pure functions only — no hidden state, easy to unit-test.
* All category encodings are derived from the CATEGORY_RISK_MAP
  constant so the Streamlit UI and the training pipeline stay in sync.
* Missing-value handling is explicit (no silent NaN propagation).
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from typing import Dict

# ---------------------------------------------------------------------------
# Domain constants
# ---------------------------------------------------------------------------

# Empirical chargeback rate by merchant category (derived from dataset EDA).
# Scale: 0.0 (low risk) → 1.0 (highest risk).
CATEGORY_RISK_MAP: Dict[str, float] = {
    "grocery": 0.05,
    "restaurants": 0.06,
    "utilities": 0.07,
    "healthcare": 0.07,
    "subscription": 0.08,
    "ecommerce_retail": 0.09,
    "digital_goods": 0.12,
    "travel": 0.14,
    "gambling": 0.18,
    "crypto_exchange": 0.22,
}

# Ordered list of categories used for one-hot encoding — ordering must match
# the training artefact so inference never shifts column indices.
CATEGORIES: list[str] = sorted(CATEGORY_RISK_MAP.keys())

# High-risk category set for binary flag.
HIGH_RISK_CATEGORIES: set[str] = {"crypto_exchange", "gambling", "travel", "digital_goods"}


# ---------------------------------------------------------------------------
# Core feature-engineering function
# ---------------------------------------------------------------------------

def build_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Convert a raw transactions DataFrame into a model-ready feature matrix.

    Parameters
    ----------
    df : pd.DataFrame
        Must contain the columns present in chargeback_transactions.xlsx.

    Returns
    -------
    pd.DataFrame
        Feature matrix with one row per transaction. Column order is stable
        across calls (sorted alphabetically within each feature group).
    """
    feat = pd.DataFrame(index=df.index)

    # ------------------------------------------------------------------
    # 1. Transaction amount features
    # ------------------------------------------------------------------
    feat["amount"] = df["transaction_amount"].clip(lower=0)
    feat["amount_log"] = np.log1p(feat["amount"])          # compress right tail
    feat["is_high_value"] = (feat["amount"] > 300).astype(int)
    feat["is_micro_txn"] = (feat["amount"] < 10).astype(int)

    # ------------------------------------------------------------------
    # 2. Merchant category features
    # ------------------------------------------------------------------
    cat = df["merchant_category"].str.lower().str.strip()
    feat["category_risk_score"] = cat.map(CATEGORY_RISK_MAP).fillna(0.10)
    feat["is_high_risk_category"] = cat.isin(HIGH_RISK_CATEGORIES).astype(int)

    # One-hot encode merchant category (stable column order)
    for c in CATEGORIES:
        feat[f"cat_{c}"] = (cat == c).astype(int)

    # ------------------------------------------------------------------
    # 3. Customer behaviour features
    # ------------------------------------------------------------------
    feat["customer_tenure_months"] = df["customer_tenure_months"].clip(lower=0)
    feat["is_new_customer"] = (feat["customer_tenure_months"] <= 3).astype(int)
    feat["tenure_log"] = np.log1p(feat["customer_tenure_months"])

    feat["prior_chargebacks"] = df["number_of_previous_chargebacks"].clip(lower=0)
    feat["has_prior_chargeback"] = (feat["prior_chargebacks"] > 0).astype(int)
    feat["repeat_offender"] = (feat["prior_chargebacks"] >= 2).astype(int)

    # Chargeback rate: ratio of disputes to total transactions
    prev_txns = df["number_of_previous_transactions"].clip(lower=1)
    feat["chargeback_rate"] = feat["prior_chargebacks"] / prev_txns

    # ------------------------------------------------------------------
    # 4. Subscription flag
    # ------------------------------------------------------------------
    feat["is_subscription"] = df["subscription_flag"].astype(int)

    # ------------------------------------------------------------------
    # 5. Merchant risk features
    # ------------------------------------------------------------------
    feat["merchant_risk_score"] = df["merchant_risk_score"].clip(0, 1)

    # Interaction: high-risk merchant + high-value transaction
    feat["risk_amount_interaction"] = (
        feat["merchant_risk_score"] * feat["amount_log"]
    )

    # ------------------------------------------------------------------
    # 6. Device risk features
    # ------------------------------------------------------------------
    feat["device_risk_score"] = df["device_risk_score"].clip(0, 1)
    feat["is_high_device_risk"] = (feat["device_risk_score"] > 0.6).astype(int)

    # ------------------------------------------------------------------
    # 7. Timing features
    # ------------------------------------------------------------------
    feat["days_since_purchase"] = df["days_since_purchase"].clip(lower=0)
    feat["is_recent_txn"] = (feat["days_since_purchase"] <= 3).astype(int)

    # ------------------------------------------------------------------
    # 8. Composite risk signal
    # ------------------------------------------------------------------
    feat["composite_risk"] = (
        0.30 * feat["category_risk_score"]
        + 0.25 * feat["merchant_risk_score"]
        + 0.20 * feat["device_risk_score"]
        + 0.15 * feat["chargeback_rate"].clip(0, 1)
        + 0.10 * feat["is_high_value"]
    )

    return feat.sort_index(axis=1)   # deterministic column order


# ---------------------------------------------------------------------------
# Single-transaction helper (used by the Streamlit UI)
# ---------------------------------------------------------------------------

def build_single_transaction_features(
    transaction_amount: float,
    merchant_category: str,
    customer_tenure_months: int,
    subscription_flag: int,
    number_of_previous_chargebacks: int,
    merchant_risk_score: float,
    device_risk_score: float = 0.2,
    days_since_purchase: int = 1,
    number_of_previous_transactions: int = 10,
) -> pd.DataFrame:
    """
    Build a single-row feature DataFrame for real-time inference.

    All parameters map directly to the UI input widgets.  Defaults for
    non-UI fields (device_risk_score, days_since_purchase, etc.) are set
    to representative values so the Streamlit app stays lean.
    """
    row = pd.DataFrame([{
        "transaction_amount": transaction_amount,
        "merchant_category": merchant_category,
        "customer_tenure_months": customer_tenure_months,
        "subscription_flag": subscription_flag,
        "number_of_previous_chargebacks": number_of_previous_chargebacks,
        "merchant_risk_score": merchant_risk_score,
        "device_risk_score": device_risk_score,
        "days_since_purchase": days_since_purchase,
        "number_of_previous_transactions": max(number_of_previous_transactions, 1),
        "refund_requested": 0,
    }])
    return build_features(row)
