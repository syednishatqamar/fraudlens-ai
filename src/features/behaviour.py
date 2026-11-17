"""Behaviour features built from a customer's *earlier* transactions only.

Point-in-time rule: the value for a row may use that customer's previous rows, never the row
itself (for averages) and never later rows. tests/test_features.py checks this.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from src.data.loader import Schema

# Window lengths in seconds for the recent-transaction counts.
WINDOWS = {"txn_count_1h": 3600.0, "txn_count_24h": 86400.0}


def _prior_count_in_window(seconds: np.ndarray, window: float) -> np.ndarray:
    """For time-sorted `seconds`, count EARLIER rows with time in [t - window, t].

    Row i counts only rows with a smaller position, so the row itself and anything after it never
    contribute. Rows sharing a timestamp are counted in sort order.
    """
    first_in_window = np.searchsorted(seconds, seconds - window, side="left")
    return np.arange(len(seconds)) - first_in_window


def add_behaviour_features(df: pd.DataFrame, schema: Schema = Schema()) -> pd.DataFrame:
    """Return a chronologically sorted copy with two extra columns.

    secs_since_prev: seconds since the same customer's previous transaction (NaN for the first).
    amt_vs_avg:      amount divided by the mean of the customer's earlier amounts (NaN for the first).
    txn_count_1h / txn_count_24h: number of the customer's EARLIER transactions in the last hour / 24 hours
                     (0 when there are none; the current transaction is not counted).

    `df[schema.time]` must already be a datetime column (load_transactions does this).
    """
    out = df.sort_values(schema.time, kind="stable").reset_index(drop=True)
    g = out.groupby(schema.customer)
    out["secs_since_prev"] = g[schema.time].diff().dt.total_seconds()
    prev_sum = g[schema.amount].cumsum() - out[schema.amount]  # earlier amounts only
    prev_cnt = g.cumcount()                                    # number of earlier rows
    out["amt_vs_avg"] = out[schema.amount] / (prev_sum / prev_cnt.where(prev_cnt > 0))

    seconds = (out[schema.time] - out[schema.time].min()).dt.total_seconds().to_numpy()
    groups = out.groupby(schema.customer).indices  # row positions per customer, already chronological
    for name, window in WINDOWS.items():
        counts = np.zeros(len(out), dtype="int64")
        for pos in groups.values():
            counts[pos] = _prior_count_in_window(seconds[pos], window)
        out[name] = counts
    return out
