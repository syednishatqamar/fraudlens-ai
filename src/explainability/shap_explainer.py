"""Thin wrapper around SHAP for the tree model.

Explains the model's output on one of two scales:

- "probability": the model score (class_weight="balanced" makes it uncalibrated, not a true probability).
  Near 1.0 this scale saturates, so a feature can have little room to show its effect.
- "logodds": the raw model margin, which does not saturate. score = 1 / (1 + exp(-margin)).

With only a handful of features the exact algorithm is fast and the pushes add up exactly:
base_value + sum(pushes) = model output on the chosen scale.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


class ShapExplainer:
    def __init__(
        self, model, background: pd.DataFrame, max_background: int = 100, seed: int = 0, scale: str = "probability"
    ):
        import shap  # imported here so the rest of the package works without shap installed

        if scale not in ("probability", "logodds"):
            raise ValueError("scale must be 'probability' or 'logodds'")
        self.scale = scale
        self.columns = list(background.columns)
        self.model = model
        bg = background.sample(min(max_background, len(background)), random_state=seed)
        self._explainer = shap.Explainer(self._score, bg, algorithm="exact")

    def _score(self, data) -> np.ndarray:
        frame = pd.DataFrame(data, columns=self.columns)
        if self.scale == "logodds":
            return self.model.decision_function(frame)
        return self.model.predict_proba(frame)[:, 1]

    def explain(self, X: pd.DataFrame) -> tuple[pd.DataFrame, np.ndarray]:
        """Return (pushes as a DataFrame aligned with X, base value per row)."""
        sv = self._explainer(X[self.columns])
        return pd.DataFrame(sv.values, index=X.index, columns=self.columns), np.asarray(sv.base_values)
