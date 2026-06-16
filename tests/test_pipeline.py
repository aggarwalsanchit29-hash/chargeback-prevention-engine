"""
test_pipeline.py
================
Visa Risk Solutions — Chargeback Prevention Engine
---------------------------------------------------
Unit tests covering feature engineering and risk scoring logic.

Run with:
    python -m pytest tests/ -v
"""

from __future__ import annotations

import sys
from pathlib import Path
import numpy as np
import pandas as pd

# Make the project root importable
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.features.feature_engineering import (
    build_features,
    build_single_transaction_features,
    CATEGORIES,
    CATEGORY_RISK_MAP,
)
from src.scoring.risk_scorer import (
    probability_to_score,
    classify_tier,
    get_intervention,
    score_transaction,
    simple_feature_contributions,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

def _make_raw_row(**overrides) -> pd.DataFrame:
    """Return a minimal raw transaction DataFrame with sensible defaults."""
    defaults = {
        "transaction_amount": 100.0,
        "merchant_category": "ecommerce_retail",
        "customer_tenure_months": 12,
        "subscription_flag": 0,
        "number_of_previous_chargebacks": 0,
        "number_of_previous_transactions": 20,
        "merchant_risk_score": 0.5,
        "device_risk_score": 0.2,
        "days_since_purchase": 5,
        "refund_requested": 0,
    }
    defaults.update(overrides)
    return pd.DataFrame([defaults])


# ---------------------------------------------------------------------------
# Feature engineering tests
# ---------------------------------------------------------------------------

class TestFeatureEngineering:

    def test_output_is_dataframe(self):
        df = build_features(_make_raw_row())
        assert isinstance(df, pd.DataFrame)

    def test_columns_are_sorted(self):
        df = build_features(_make_raw_row())
        assert list(df.columns) == sorted(df.columns)

    def test_category_one_hot_sums_to_one(self):
        df = build_features(_make_raw_row(merchant_category="travel"))
        cat_cols = [c for c in df.columns if c.startswith("cat_")]
        assert df[cat_cols].sum(axis=1).iloc[0] == 1

    def test_unknown_category_risk_score_defaults(self):
        df = build_features(_make_raw_row(merchant_category="unknown_cat"))
        assert df["category_risk_score"].iloc[0] == 0.10

    def test_amount_log_is_log1p(self):
        amount = 99.0
        df = build_features(_make_raw_row(transaction_amount=amount))
        assert abs(df["amount_log"].iloc[0] - np.log1p(amount)) < 1e-6

    def test_is_new_customer_flag(self):
        new = build_features(_make_raw_row(customer_tenure_months=2))
        old = build_features(_make_raw_row(customer_tenure_months=24))
        assert new["is_new_customer"].iloc[0] == 1
        assert old["is_new_customer"].iloc[0] == 0

    def test_prior_chargeback_flags(self):
        none  = build_features(_make_raw_row(number_of_previous_chargebacks=0))
        one   = build_features(_make_raw_row(number_of_previous_chargebacks=1))
        multi = build_features(_make_raw_row(number_of_previous_chargebacks=3))
        assert none["has_prior_chargeback"].iloc[0] == 0
        assert one["has_prior_chargeback"].iloc[0] == 1
        assert multi["repeat_offender"].iloc[0] == 1

    def test_composite_risk_bounded(self):
        df = build_features(_make_raw_row())
        assert 0.0 <= df["composite_risk"].iloc[0] <= 1.0

    def test_build_single_transaction_features_shape(self):
        df = build_single_transaction_features(
            transaction_amount=200.0,
            merchant_category="travel",
            customer_tenure_months=6,
            subscription_flag=1,
            number_of_previous_chargebacks=1,
            merchant_risk_score=0.7,
        )
        assert df.shape[0] == 1
        assert "composite_risk" in df.columns

    def test_all_categories_produce_valid_features(self):
        for cat in CATEGORIES:
            df = build_features(_make_raw_row(merchant_category=cat))
            assert not df.isnull().any().any(), f"NaN found for category: {cat}"


# ---------------------------------------------------------------------------
# Risk scorer tests
# ---------------------------------------------------------------------------

class TestRiskScorer:

    def test_probability_to_score_zero(self):
        assert probability_to_score(0.0) == 0

    def test_probability_to_score_one(self):
        assert probability_to_score(1.0) == 100

    def test_probability_to_score_midpoint(self):
        score = probability_to_score(0.5)
        assert 0 <= score <= 100

    def test_classify_tier_low(self):
        assert classify_tier(0.05) == "LOW"

    def test_classify_tier_medium(self):
        assert classify_tier(0.25) == "MEDIUM"

    def test_classify_tier_high(self):
        assert classify_tier(0.50) == "HIGH"

    def test_classify_tier_critical(self):
        assert classify_tier(0.80) == "CRITICAL"

    def test_get_intervention_codes(self):
        assert get_intervention("LOW")[0] == "no_action"
        assert get_intervention("MEDIUM")[0] == "customer_confirmation"
        assert get_intervention("HIGH")[0] == "additional_auth"
        assert get_intervention("CRITICAL")[0] == "merchant_review"

    def test_score_transaction_returns_assessment(self):
        feat_df = build_features(_make_raw_row())
        feat_row = feat_df.iloc[0]
        contribs = simple_feature_contributions(feat_row)
        result = score_transaction(0.40, feat_row, contribs)
        assert result.risk_tier in ("LOW", "MEDIUM", "HIGH", "CRITICAL")
        assert 0 <= result.risk_score <= 100
        assert 0.0 <= result.probability <= 1.0
        assert isinstance(result.explanation, list)
        assert len(result.explanation) > 0

    def test_high_prob_gives_critical_tier(self):
        feat_df = build_features(_make_raw_row())
        feat_row = feat_df.iloc[0]
        result = score_transaction(0.90, feat_row)
        assert result.risk_tier == "CRITICAL"
        assert result.intervention == "merchant_review"

    def test_low_prob_gives_no_action(self):
        feat_df = build_features(_make_raw_row())
        feat_row = feat_df.iloc[0]
        result = score_transaction(0.05, feat_row)
        assert result.risk_tier == "LOW"
        assert result.intervention == "no_action"
