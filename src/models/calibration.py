"""Turn raw model margins into calibrated fraud probabilities, and check the calibration.

The tree model is trained with class_weight="balanced", so its scores run far higher than real fraud
rates. A calibrator learns the mapping from score to observed fraud rate on data the model did not train on.

Caveat: a calibrator learns the fraud rate of the period it was fitted on. If the base rate changes (here the
test file has a lower fraud rate than validation), calibrated probabilities will be too high or too low.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss, log_loss

DEFAULT_EDGES = (0.0, 0.001, 0.005, 0.02, 0.1, 0.3, 0.7, 1.0)


class PlattCalibrator:
    """Sigmoid calibration: p = sigmoid(a * margin + b). Two parameters, so it resists overfitting."""

    def fit(self, margins, y) -> "PlattCalibrator":
        self._lr = LogisticRegression(C=1e6, max_iter=1000).fit(np.asarray(margins, dtype=float).reshape(-1, 1), y)
        return self

    def predict_proba(self, margins) -> np.ndarray:
        return self._lr.predict_proba(np.asarray(margins, dtype=float).reshape(-1, 1))[:, 1]


class IsotonicCalibrator:
    """Monotone step-function calibration. More flexible than Platt, needs more frauds to be stable."""

    def fit(self, margins, y) -> "IsotonicCalibrator":
        self._iso = IsotonicRegression(out_of_bounds="clip", y_min=0.0, y_max=1.0).fit(np.asarray(margins, dtype=float), y)
        return self

    def predict_proba(self, margins) -> np.ndarray:
        return self._iso.predict(np.asarray(margins, dtype=float))


def make_calibrator(method: str):
    if method == "platt":
        return PlattCalibrator()
    if method == "isotonic":
        return IsotonicCalibrator()
    raise ValueError("method must be 'platt' or 'isotonic'")


def calibration_report(y_true, probs, edges=DEFAULT_EDGES) -> tuple[pd.DataFrame, dict[str, float]]:
    """Reliability table by probability band plus Brier score and log loss.

    In a well-calibrated model, mean_pred is close to observed_rate in every band.
    """
    y = np.asarray(y_true)
    p = np.clip(np.asarray(probs, dtype=float), 0.0, 1.0)
    band = pd.cut(p, bins=list(edges), include_lowest=True)
    df = pd.DataFrame({"band": band, "p": p, "y": y})
    table = (
        df.groupby("band", observed=True)
        .agg(n=("y", "size"), frauds=("y", "sum"), mean_pred=("p", "mean"), observed_rate=("y", "mean"))
        .reset_index()
    )
    summary = {
        "brier": float(brier_score_loss(y, p)),
        "log_loss": float(log_loss(y, np.clip(p, 1e-6, 1 - 1e-6), labels=[0, 1])),
    }
    return table, summary
