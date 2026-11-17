"""Thin wrapper around SHAP for the tree model.

Explains the model's *score* (class_weight="balanced" makes it an uncalibrated score, not a true
probability). With only a handful of features the exact algorithm is fast and gives pushes that
add up exactly: base_value + sum(pushes) = score.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


class ShapExplainer:
    def __init__(self, model, background: pd.DataFrame, max_background: int = 100, seed: int = 0):
        import shap  # imported here so the rest of the package works without shap installed

        self.columns = list(background.columns)
        self.model = model
        bg = background.sample(min(max_background, len(background)), random_state=seed)
        self._explainer = shap.Explainer(self._score, bg, algorithm="exact")

    def _score(self, data) -> np.ndarray:
        return self.model.predict_proba(pd.DataFrame(data, columns=self.columns))[:, 1]

    def explain(self, X: pd.DataFrame) -> tuple[pd.DataFrame, np.ndarray]:
        """Return (pushes as a DataFrame aligned with X, base value per row)."""
        sv = self._explainer(X[self.columns])
        return pd.DataFrame(sv.values, index=X.index, columns=self.columns), np.asarray(sv.base_values)
