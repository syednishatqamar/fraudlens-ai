"""Chronological train/validation/test splitting.

Why not a random split: fraud data is a time series. A random split lets the model train on
transactions that happen *after* some of its test rows, so metrics look better than they would
in production. We cut by timestamp instead, and test that nothing overlaps.
"""
from __future__ import annotations

import pandas as pd


def chronological_split(
    df: pd.DataFrame,
    time_col: str,
    train_end: str | pd.Timestamp,
    val_end: str | pd.Timestamp,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Split into train (< train_end), validation [train_end, val_end), test (>= val_end).

    Choose the dates after looking at the time span and fraud counts per period in the EDA.
    """
    train_end, val_end = pd.Timestamp(train_end), pd.Timestamp(val_end)
    if not train_end < val_end:
        raise ValueError("train_end must be earlier than val_end")
    t = df[time_col]
    train = df[t < train_end]
    val = df[(t >= train_end) & (t < val_end)]
    test = df[t >= val_end]
    return train, val, test


def split_summary(parts: dict[str, pd.DataFrame], time_col: str, label_col: str) -> pd.DataFrame:
    """Rows, frauds, fraud rate and time range per split, to eyeball before modelling."""
    rows = []
    for name, p in parts.items():
        rows.append(
            {
                "split": name,
                "rows": len(p),
                "frauds": int(p[label_col].sum()),
                "fraud_rate": float(p[label_col].mean()) if len(p) else float("nan"),
                "start": p[time_col].min(),
                "end": p[time_col].max(),
            }
        )
    return pd.DataFrame(rows).set_index("split")
