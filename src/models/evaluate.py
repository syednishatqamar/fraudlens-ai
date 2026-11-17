"""Ranking-based evaluation helpers."""
from __future__ import annotations

import numpy as np
import pandas as pd


def top_k_caught(y_true, scores, k: int | None = None) -> dict[str, float]:
    """Flag the k highest-scoring rows (default: as many as there are real frauds).

    Ranking avoids choosing a probability cutoff. With k = number of frauds, caught/k is also
    the precision of the flagged set.
    """
    y = np.asarray(y_true)
    k = int(y.sum()) if k is None else int(k)
    order = pd.Series(scores).rank(method="first", ascending=False).to_numpy()
    flagged = order <= k
    caught = int((flagged & (y == 1)).sum())
    return {
        "k": k,
        "frauds_total": int(y.sum()),
        "caught": caught,
        "recall": caught / max(int(y.sum()), 1),
        "false_alarms": int(k - caught),
    }
