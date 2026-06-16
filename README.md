# 🛡️ Chargeback Prevention Engine
**Visa Risk Solutions · Portfolio Project**

Real-time chargeback risk scoring engine with an explainable ML model and Streamlit front-end.

---

## Project Overview

Predicts the likelihood of a transaction resulting in a chargeback using a Gradient Boosting
classifier trained on 10,000 transactions (6.6 % chargeback rate).

Outputs:
- **Risk Score** (0–100)
- **Chargeback Probability** (%)
- **Risk Tier** — LOW / MEDIUM / HIGH / CRITICAL
- **Recommended Intervention** — No Action / Customer Confirmation / Additional Auth / Merchant Review
- **Top Risk Drivers** — feature-contribution explanations

---

## Folder Structure

```
chargeback_engine/
├── app.py                          # Streamlit application (entry point)
├── requirements.txt                # Python dependencies
├── README.md
│
├── data/
│   └── chargeback_transactions.xlsx
│
├── src/
│   ├── features/
│   │   └── feature_engineering.py  # Feature transforms + category risk map
│   ├── model/
│   │   ├── train_model.py           # Training pipeline + evaluation
│   │   └── chargeback_model.joblib  # Saved model artefact (after training)
│   └── scoring/
│       └── risk_scorer.py           # Risk tiers, interventions, explanations
│
└── tests/
    └── test_pipeline.py             # 21 unit tests
```

---

## Quick Start

### 1. Install dependencies
```bash
pip install -r requirements.txt
```

### 2. Train the model
```bash
python -m src.model.train_model
```

### 3. Launch the Streamlit app
```bash
streamlit run app.py
```

---

## Model Details

| Property | Value |
|---|---|
| Algorithm | Gradient Boosting Classifier (sklearn) |
| Training samples | 8,000 |
| Test samples | 2,000 |
| Class imbalance handling | Sample weights (pos weight ≈ 14×) |
| ROC-AUC (test set) | ~0.63 |
| Features | 31 engineered features |

### Feature Groups

| Group | Features |
|---|---|
| Amount | Raw amount, log-amount, high-value flag, micro-txn flag |
| Category | One-hot encoding, category risk score, high-risk category flag |
| Customer | Tenure, new-customer flag, prior chargebacks, chargeback rate |
| Merchant | Merchant risk score, risk × amount interaction |
| Device | Device risk score, high-device-risk flag |
| Timing | Days since purchase, recent-transaction flag |
| Composite | Weighted combination of top risk signals |

---

## Risk Tiers & Interventions

| Tier | Probability Range | Intervention |
|---|---|---|
| 🟢 LOW | 0–15 % | No Action |
| 🟡 MEDIUM | 15–35 % | Customer Confirmation (push/SMS) |
| 🔴 HIGH | 35–60 % | Additional Authentication (3DS2 / OTP) |
| 🟣 CRITICAL | 60–100 % | Merchant Review & Hold |

---

## Running Tests

```bash
python -m pytest tests/ -v
```
or without pytest:
```bash
python tests/test_pipeline.py
```

21 tests cover feature engineering correctness and risk scoring logic.
