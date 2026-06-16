"""
app.py
======
Visa Risk Solutions — Chargeback Prevention Engine
---------------------------------------------------
Streamlit front-end for real-time chargeback risk scoring.

Layout
------
Sidebar  : Transaction input form
Main     : Risk score gauge, probability, intervention card,
           top risk drivers, feature contribution bar chart

Run with:
    streamlit run app.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

# ---------------------------------------------------------------------------
# Resolve project root so relative imports work when Streamlit is launched
# from any working directory.
# ---------------------------------------------------------------------------
BASE_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(BASE_DIR))

from src.features.feature_engineering import (
    build_single_transaction_features,
    CATEGORIES,
    CATEGORY_RISK_MAP,
)
from src.model.train_model import load_model
from src.scoring.risk_scorer import (
    score_transaction,
    TIER_COLOURS,
    FEATURE_LABELS,
)

# ---------------------------------------------------------------------------
# Page configuration
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="Chargeback Prevention Engine",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------------------------------------------------------------------------
# Custom CSS — minimal overrides for a professional dark-accent look
# ---------------------------------------------------------------------------
st.markdown(
    """
<style>
    /* Header banner */
    .visa-header {
        background: linear-gradient(135deg, #1a1f5e 0%, #2d3a8c 100%);
        padding: 1.2rem 2rem;
        border-radius: 10px;
        margin-bottom: 1.5rem;
        display: flex;
        align-items: center;
        gap: 1rem;
    }
    .visa-header h1 {
        color: #ffffff;
        font-size: 1.6rem;
        margin: 0;
        font-weight: 700;
        letter-spacing: 0.5px;
    }
    .visa-header p {
        color: #a8b4e8;
        margin: 0;
        font-size: 0.85rem;
    }

    /* Risk tier badge */
    .tier-badge {
        display: inline-block;
        padding: 0.4rem 1.2rem;
        border-radius: 20px;
        font-weight: 700;
        font-size: 1.1rem;
        letter-spacing: 1px;
        text-transform: uppercase;
        margin: 0.5rem 0;
    }

    /* Metric cards */
    .metric-card {
        background: #f8f9fe;
        border-left: 4px solid #2d3a8c;
        padding: 1rem 1.2rem;
        border-radius: 8px;
        margin-bottom: 0.8rem;
    }
    .metric-card h3 { margin: 0 0 0.3rem 0; font-size: 0.8rem; color: #666; text-transform: uppercase; letter-spacing: 0.5px; }
    .metric-card p  { margin: 0; font-size: 1.4rem; font-weight: 700; color: #1a1f5e; }

    /* Intervention card */
    .intervention-card {
        padding: 1.2rem 1.5rem;
        border-radius: 10px;
        margin: 1rem 0;
    }
    .intervention-card h3 { margin: 0 0 0.5rem 0; font-size: 0.85rem; text-transform: uppercase; letter-spacing: 0.5px; }
    .intervention-card p  { margin: 0; font-size: 1.3rem; font-weight: 700; }

    /* Driver list items */
    .driver-item {
        background: #f0f2fa;
        border-radius: 6px;
        padding: 0.5rem 0.8rem;
        margin-bottom: 0.4rem;
        font-size: 0.9rem;
        color: #333;
    }
    .driver-item span { font-weight: 600; color: #2d3a8c; }
</style>
""",
    unsafe_allow_html=True,
)


# ---------------------------------------------------------------------------
# Model loader — cached so it only runs once per session
# ---------------------------------------------------------------------------
@st.cache_resource(show_spinner="Loading risk model …")
def get_model() -> dict:
    """Load the persisted model artefact (model + feature_importances)."""
    return load_model()


# ---------------------------------------------------------------------------
# Helper: Plotly gauge chart
# ---------------------------------------------------------------------------
def render_gauge(risk_score: int, tier: str) -> go.Figure:
    """Render a half-donut gauge for the 0–100 risk score."""
    colour = TIER_COLOURS[tier]
    fig = go.Figure(
        go.Indicator(
            mode="gauge+number",
            value=risk_score,
            domain={"x": [0, 1], "y": [0, 1]},
            title={"text": "Risk Score", "font": {"size": 16, "color": "#555"}},
            gauge={
                "axis": {"range": [0, 100], "tickwidth": 1, "tickcolor": "#999"},
                "bar": {"color": colour, "thickness": 0.3},
                "bgcolor": "white",
                "borderwidth": 2,
                "bordercolor": "#ddd",
                "steps": [
                    {"range": [0, 15], "color": "#d5f5e3"},
                    {"range": [15, 35], "color": "#fef9e7"},
                    {"range": [35, 60], "color": "#fadbd8"},
                    {"range": [60, 100], "color": "#e8daef"},
                ],
                "threshold": {
                    "line": {"color": colour, "width": 4},
                    "thickness": 0.8,
                    "value": risk_score,
                },
            },
            number={"font": {"size": 40, "color": colour}, "suffix": ""},
        )
    )
    fig.update_layout(
        height=260,
        margin=dict(l=20, r=20, t=40, b=10),
        paper_bgcolor="white",
    )
    return fig


# ---------------------------------------------------------------------------
# Helper: Feature contribution bar chart
# ---------------------------------------------------------------------------
def render_contributions(contributions: dict[str, float], top_n: int = 8) -> go.Figure:
    """Horizontal bar chart of top feature contributions."""
    sorted_items = sorted(contributions.items(), key=lambda x: abs(x[1]), reverse=True)[
        :top_n
    ]
    labels = [
        FEATURE_LABELS.get(k, k.replace("_", " ").title()) for k, _ in sorted_items
    ]
    values = [v for _, v in sorted_items]
    colours = ["#e74c3c" if v > 0 else "#2ecc71" for v in values]

    fig = go.Figure(
        go.Bar(
            x=values,
            y=labels,
            orientation="h",
            marker_color=colours,
            text=[f"{v:+.3f}" for v in values],
            textposition="outside",
        )
    )
    fig.update_layout(
        title="Feature Contributions to Risk",
        xaxis_title="Contribution magnitude",
        height=320,
        margin=dict(l=10, r=80, t=40, b=20),
        paper_bgcolor="white",
        plot_bgcolor="#fafafa",
        font={"size": 12},
        yaxis={"autorange": "reversed"},
    )
    return fig


# ---------------------------------------------------------------------------
# Main app
# ---------------------------------------------------------------------------
def main() -> None:
    # ── Header ──────────────────────────────────────────────────────────────
    st.markdown(
        """
    <div class="visa-header">
        <div>
            <h1>🛡️ Chargeback Prevention Engine</h1>
            <p>Sanchit Risk Solutions · Real-Time Transaction Risk Scoring</p>
        </div>
    </div>
    """,
        unsafe_allow_html=True,
    )

    # ── Load model ───────────────────────────────────────────────────────────
    try:
        artefact = get_model()
        model = artefact["model"]
        feature_names = artefact["feature_names"]
        feat_importances = artefact.get("feature_importances", {})
    except FileNotFoundError as exc:
        st.error(str(exc))
        st.stop()

    # ── Sidebar inputs ───────────────────────────────────────────────────────
    with st.sidebar:
        st.markdown("## 📝 Transaction Details")
        st.markdown(
            "Enter the transaction parameters below to score the chargeback risk."
        )
        st.markdown("---")

        transaction_amount = st.number_input(
            "Transaction Amount ($)",
            min_value=0.01,
            max_value=50_000.0,
            value=150.0,
            step=1.0,
            help="Gross transaction value in USD",
        )

        merchant_category = st.selectbox(
            "Merchant Category",
            options=CATEGORIES,
            index=CATEGORIES.index("ecommerce_retail"),
            help="MCC-level category of the merchant",
        )

        customer_tenure_months = st.slider(
            "Customer Tenure (months)",
            min_value=0,
            max_value=120,
            value=12,
            help="How long this customer has held the card",
        )

        subscription_flag = st.radio(
            "Subscription Transaction?",
            options=[0, 1],
            format_func=lambda x: "Yes" if x == 1 else "No",
            horizontal=True,
            help="Is this a recurring subscription billing?",
        )

        prior_chargebacks = st.number_input(
            "Previous Chargebacks",
            min_value=0,
            max_value=20,
            value=0,
            step=1,
            help="Number of prior chargebacks on this account",
        )

        merchant_risk_score = st.slider(
            "Merchant Risk Score (0–1)",
            min_value=0.0,
            max_value=1.0,
            value=0.50,
            step=0.01,
            help="Merchant-level risk signal from Visa's merchant risk model",
        )

        device_risk_score = st.slider(
            "Device Risk Score (0–1)",
            min_value=0.0,
            max_value=1.0,
            value=0.20,
            step=0.01,
            help="Session/device risk score from fraud telemetry",
        )

        st.markdown("---")
        score_button = st.button(
            "🔍 Score Transaction", use_container_width=True, type="primary"
        )

    # ── Score on button click or on first load ───────────────────────────────
    if not score_button:
        st.info(
            "👈  Fill in the transaction details in the sidebar, then click **Score Transaction**."
        )
        st.markdown("#### How It Works")
        col1, col2, col3 = st.columns(3)
        with col1:
            st.markdown("""
            **1. Feature Engineering**
            Raw transaction fields are transformed into 31 risk signals
            including chargeback rate, composite risk index, and
            category-level risk.
            """)
        with col2:
            st.markdown("""
            **2. ML Scoring**
            A Gradient Boosting classifier trained on 10,000 transactions
            outputs a chargeback probability calibrated for the
            6.6 % class imbalance.
            """)
        with col3:
            st.markdown("""
            **3. Intervention Routing**
            Business rules map the probability to a risk tier and
            recommend the appropriate intervention: from no action
            to a full merchant hold.
            """)
        return

    # ── Build features ───────────────────────────────────────────────────────
    features_df = build_single_transaction_features(
        transaction_amount=transaction_amount,
        merchant_category=merchant_category,
        customer_tenure_months=customer_tenure_months,
        subscription_flag=int(subscription_flag),
        number_of_previous_chargebacks=int(prior_chargebacks),
        merchant_risk_score=merchant_risk_score,
        device_risk_score=device_risk_score,
    )

    # Align columns to training feature set (fills any missing cols with 0)
    features_aligned = features_df.reindex(columns=feature_names, fill_value=0)

    # ── Predict ──────────────────────────────────────────────────────────────
    probability = float(model.predict_proba(features_aligned)[0, 1])

    # Build feature contributions from model importances × feature values
    feat_row = features_aligned.iloc[0]
    feature_contributions = {
        feat: float(feat_row[feat]) * float(feat_importances.get(feat, 0))
        for feat in feature_names
    }

    # ── Score ────────────────────────────────────────────────────────────────
    assessment = score_transaction(probability, feat_row, feature_contributions)

    # ── Layout: top KPI row ──────────────────────────────────────────────────
    tier_colour = TIER_COLOURS[assessment.risk_tier]
    col_gauge, col_kpis = st.columns([1, 1], gap="large")

    with col_gauge:
        st.plotly_chart(
            render_gauge(assessment.risk_score, assessment.risk_tier),
            use_container_width=True,
        )

    with col_kpis:
        st.markdown("#### Risk Assessment Summary")

        # Tier badge
        st.markdown(
            f'<span class="tier-badge" style="background:{tier_colour}22; '
            f'color:{tier_colour}; border:2px solid {tier_colour};">'
            f"{assessment.risk_tier} RISK</span>",
            unsafe_allow_html=True,
        )

        k1, k2 = st.columns(2)
        with k1:
            st.markdown(
                f"""
            <div class="metric-card">
                <h3>Chargeback Probability</h3>
                <p>{assessment.probability * 100:.1f}%</p>
            </div>
            """,
                unsafe_allow_html=True,
            )
        with k2:
            st.markdown(
                f"""
            <div class="metric-card">
                <h3>Risk Score</h3>
                <p>{assessment.risk_score} / 100</p>
            </div>
            """,
                unsafe_allow_html=True,
            )

        # Intervention card
        intervention_icons = {
            "no_action": "✅",
            "customer_confirmation": "📱",
            "additional_auth": "🔐",
            "merchant_review": "🚨",
        }
        icon = intervention_icons.get(assessment.intervention, "ℹ️")
        st.markdown(
            f'<div class="intervention-card" '
            f'style="background:{tier_colour}18; border:2px solid {tier_colour};">'
            f'<h3 style="color:{tier_colour};">Recommended Intervention</h3>'
            f'<p style="color:{tier_colour};">{icon} {assessment.intervention_label}</p>'
            f"</div>",
            unsafe_allow_html=True,
        )

    st.markdown("---")

    # ── Layout: explanations + chart ─────────────────────────────────────────
    col_explain, col_chart = st.columns([1, 1.2], gap="large")

    with col_explain:
        st.markdown("#### 🔎 Top Risk Drivers")
        for driver in assessment.explanation:
            st.markdown(
                f'<div class="driver-item">▸ {driver}</div>', unsafe_allow_html=True
            )

        st.markdown("#### 📋 Intervention Guide")
        guide = {
            "no_action": "Transaction presents low chargeback risk. Process normally with standard monitoring.",
            "customer_confirmation": "Send an in-app or SMS push notification asking the cardholder to confirm the transaction.",
            "additional_auth": "Trigger a step-up authentication challenge (3DS2, OTP) before authorising.",
            "merchant_review": "Flag the transaction for the merchant risk team. Consider placing a temporary hold pending review.",
        }
        st.info(guide[assessment.intervention])

    with col_chart:
        non_zero = {k: v for k, v in feature_contributions.items() if abs(v) > 1e-6}
        if non_zero:
            st.plotly_chart(
                render_contributions(non_zero),
                use_container_width=True,
            )

    # ── Category risk reference table ─────────────────────────────────────────
    with st.expander("📊 Merchant Category Risk Reference"):
        cat_df = pd.DataFrame(
            [
                {
                    "Category": cat,
                    "Base Risk Rate": f"{rate * 100:.0f}%",
                    "Risk Level": "🔴 High"
                    if rate >= 0.14
                    else ("🟡 Medium" if rate >= 0.09 else "🟢 Low"),
                }
                for cat, rate in sorted(
                    CATEGORY_RISK_MAP.items(), key=lambda x: x[1], reverse=True
                )
            ]
        )
        st.dataframe(cat_df, use_container_width=True, hide_index=True)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    main()
