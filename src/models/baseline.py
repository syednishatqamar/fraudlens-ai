"""Logistic-regression baseline on raw columns only (no behavioural features).

Purpose: set the number every later model must beat. Deliberately simple.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, precision_recall_curve
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from src.data.loader import Schema


def make_raw_features(df: pd.DataFrame, schema: Schema = Schema()) -> pd.DataFrame:
    """Row-level features only: nothing here looks at other transactions."""
    return pd.DataFrame(
        {
            "log_amount": np.log1p(df[schema.amount].clip(lower=0)),
            "hour": df[schema.time].dt.hour,
            "category": df[schema.category].astype(str),
        },
        index=df.index,
    )


def build_baseline(class_weight: str | dict | None = "balanced") -> Pipeline:
    pre = ColumnTransformer(
        [
            ("num", StandardScaler(), ["log_amount", "hour"]),
            ("cat", OneHotEncoder(handle_unknown="ignore"), ["category"]),
        ]
    )
    return Pipeline([("pre", pre), ("clf", LogisticRegression(max_iter=1000, class_weight=class_weight))])


def recall_at_precision(y_true, scores, min_precision: float) -> float:
    """Best recall achievable while keeping precision >= min_precision (0.0 if never reached)."""
    precision, recall, _ = precision_recall_curve(y_true, scores)
    ok = precision >= min_precision
    return float(recall[ok].max()) if ok.any() else 0.0


def evaluate(y_true, scores, min_precision: float = 0.5) -> dict[str, float]:
    return {
        "pr_auc": float(average_precision_score(y_true, scores)),
        "positive_rate": float(np.mean(y_true)),  # PR-AUC of a random model equals this
        f"recall_at_precision_{min_precision:g}": recall_at_precision(y_true, scores, min_precision),
    }
