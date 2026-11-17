"""Behaviour features built from a customer's *earlier* transactions only.

Point-in-time rule: the value for a row may use that customer's previous rows, never the row
itself (for averages) and never later rows. tests/test_features.py checks this.
"""
from __future__ import annotations

import pandas as pd

from src.data.loader import Schema


def add_behaviour_features(df: pd.DataFrame, schema: Schema = Schema()) -> pd.DataFrame:
    """Return a chronologically sorted copy with two extra columns.

    secs_since_prev: seconds since the same customer's previous transaction (NaN for the first).
    amt_vs_avg:      amount divided by the mean of the customer's earlier amounts (NaN for the first).

    `df[schema.time]` must already be a datetime column (load_transactions does this).
    """
    out = df.sort_values(schema.time, kind="stable").reset_index(drop=True)
    g = out.groupby(schema.customer)
    out["secs_since_prev"] = g[schema.time].diff().dt.total_seconds()
    prev_sum = g[schema.amount].cumsum() - out[schema.amount]  # earlier amounts only
    prev_cnt = g.cumcount()                                    # number of earlier rows
    out["amt_vs_avg"] = out[schema.amount] / (prev_sum / prev_cnt.where(prev_cnt > 0))
    return out
