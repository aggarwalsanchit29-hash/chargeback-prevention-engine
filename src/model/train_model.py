"""
train_model.py
==============
Visa Risk Solutions — Chargeback Prevention Engine
---------------------------------------------------
Trains a Gradient Boosting classifier on the chargeback transaction dataset,
persists the trained artefact, and prints a classification report.

Run directly:
    python -m src.model.train_model

The model artefact is saved to:
    src/model/chargeback_model.joblib

Design notes
------------
* GradientBoostingClassifier is used (sklearn built-in, no extra deps).
* Class imbalance is handled via class_weight or sample_weight equivalents.
* Feature importances are stored alongside the model for fast UI explanations.
* All hyper-parameters are documented inline.
"""

from __future__ import annotations

import sys
import os
import joblib
import numpy as np
import pandas as pd

from pathlib import Path
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import (
    classification_report,
    roc_auc_score,
    average_precision_score,
)

# ---------------------------------------------------------------------------
# Resolve project root and make src importable regardless of working dir
# ---------------------------------------------------------------------------
BASE_DIR = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(BASE_DIR))

from src.features.feature_engineering import build_features

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
DATA_PATH  = BASE_DIR / "data" / "chargeback_transactions.xlsx"
MODEL_PATH = Path(__file__).resolve().parent / "chargeback_model.joblib"


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_data(path: Path = DATA_PATH) -> tuple[pd.DataFrame, pd.Series]:
    """
    Load the raw Excel file, engineer features, and return (X, y).

    Returns
    -------
    X : pd.DataFrame  — feature matrix
    y : pd.Series     — binary chargeback label (0 / 1)
    """
    df = pd.read_excel(path, sheet_name="Transactions")
    y  = df["chargeback_flag"].astype(int)
    X  = build_features(df)
    return X, y


# ---------------------------------------------------------------------------
# Model training
# ---------------------------------------------------------------------------

def train(X: pd.DataFrame, y: pd.Series) -> GradientBoostingClassifier:
    """
    Fit a Gradient Boosting classifier with class-imbalance correction.

    Hyper-parameter rationale
    -------------------------
    n_estimators=300     — enough trees for a 10k-row dataset without overfitting
    max_depth=4          — shallow trees reduce over-fit on tabular data
    learning_rate=0.05   — conservative LR pairs well with 300 estimators
    subsample=0.8        — stochastic GBM reduces variance
    min_samples_leaf=20  — prevents fitting noise in minority class
    """
    # Build sample weights to compensate for 6.6 % chargeback prevalence
    neg, pos = np.bincount(y)
    weight_for_pos = neg / pos   # ≈ 14

    sample_weights = np.where(y == 1, weight_for_pos, 1.0)

    model = GradientBoostingClassifier(
        n_estimators=300,
        max_depth=4,
        learning_rate=0.05,
        subsample=0.8,
        min_samples_leaf=20,
        random_state=42,
    )
    model.fit(X, y, sample_weight=sample_weights)
    return model


# ---------------------------------------------------------------------------
# Evaluation helpers
# ---------------------------------------------------------------------------

def evaluate(model: GradientBoostingClassifier, X: pd.DataFrame, y: pd.Series) -> None:
    """Print a full classification report plus ROC-AUC and PR-AUC."""
    y_prob = model.predict_proba(X)[:, 1]
    y_pred = (y_prob >= 0.40).astype(int)   # lower threshold for recall on minority class

    print("\n=== Classification Report ===")
    print(classification_report(y, y_pred, target_names=["No Chargeback", "Chargeback"]))
    print(f"ROC-AUC : {roc_auc_score(y, y_prob):.4f}")
    print(f"PR-AUC  : {average_precision_score(y, y_prob):.4f}")


# ---------------------------------------------------------------------------
# Persistence
# ---------------------------------------------------------------------------

def save_model(model: GradientBoostingClassifier, feature_names: list[str]) -> None:
    """
    Persist model + feature metadata so Streamlit can load a fully
    self-contained artefact without re-training.
    """
    importances = dict(zip(feature_names, model.feature_importances_))
    artefact = {
        "model":            model,
        "feature_names":    feature_names,
        "feature_importances": importances,
    }
    joblib.dump(artefact, MODEL_PATH)
    print(f"\nModel saved → {MODEL_PATH}")


def load_model() -> dict:
    """Load the persisted artefact. Raises FileNotFoundError if absent."""
    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"No trained model found at {MODEL_PATH}. "
            "Run `python -m src.model.train_model` first."
        )
    return joblib.load(MODEL_PATH)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    print("Loading data …")
    X, y = load_data()
    print(f"  Dataset shape    : {X.shape}")
    print(f"  Chargeback rate  : {y.mean()*100:.1f} %")
    print(f"  Feature columns  : {list(X.columns[:8])} …")

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, stratify=y, random_state=42
    )

    print("\nTraining GradientBoostingClassifier on 80 % split …")
    model = train(X_train, y_train)

    print("\nEvaluating on held-out 20 % …")
    evaluate(model, X_test, y_test)

    print("\nTop-10 feature importances:")
    importances = sorted(
        zip(X.columns, model.feature_importances_),
        key=lambda kv: kv[1], reverse=True
    )[:10]
    for feat, imp in importances:
        print(f"  {feat:<35} {imp:.4f}")

    save_model(model, list(X.columns))


if __name__ == "__main__":
    main()
