"""Gradient-boosted tree model on row features plus behaviour features.

Mirrors the experiment first run in the Kaggle EDA notebook. Needs the columns produced by
src.features.behaviour.add_behaviour_features.
"""
from __future__ import annotations

from typing import Sequence

import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.preprocessing import OrdinalEncoder

from src.data.loader import Schema

FEATURE_COLUMNS = ["amount", "hour", "category_code", "secs_since_prev", "amt_vs_avg"]  # default set
VELOCITY_COLUMNS = ["txn_count_1h", "txn_count_24h"]  # optional, opt in via `columns`
NOVELTY_COLUMNS = ["new_merchant"]
LOCATION_COLUMNS = ["distance_km"]
OPTIONAL_COLUMNS = VELOCITY_COLUMNS + NOVELTY_COLUMNS + LOCATION_COLUMNS


def fit_category_encoder(train_df: pd.DataFrame, schema: Schema = Schema()) -> OrdinalEncoder:
    enc = OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)
    return enc.fit(train_df[[schema.category]])


def make_tree_features(
    df: pd.DataFrame,
    encoder: OrdinalEncoder,
    schema: Schema = Schema(),
    columns: Sequence[str] = FEATURE_COLUMNS,
) -> pd.DataFrame:
    """Build the model matrix with the requested columns (default: the original five)."""
    full = pd.DataFrame(
        {
            "amount": df[schema.amount].to_numpy(),
            "hour": df[schema.time].dt.hour.to_numpy(),
            "category_code": encoder.transform(df[[schema.category]]).ravel(),
            "secs_since_prev": df["secs_since_prev"].to_numpy(),
            "amt_vs_avg": df["amt_vs_avg"].to_numpy(),
        },
        index=df.index,
    )
    for c in OPTIONAL_COLUMNS:
        if c in df.columns:
            full[c] = df[c].to_numpy()
    return full[list(columns)]


def build_tree_model(seed: int = 42) -> HistGradientBoostingClassifier:
    return HistGradientBoostingClassifier(
        max_iter=200, learning_rate=0.1, class_weight="balanced", random_state=seed
    )
