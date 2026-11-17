"""Turn per-feature SHAP pushes into short, factual reasons for an analyst.

Each reason states the feature's actual value and whether it raised or lowered the model's score.
It describes what the MODEL did, not what caused fraud. Wording is template-based on purpose: no LLM
is involved, so every number in a reason comes straight from the data.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Sequence

import numpy as np
import pandas as pd

RAISES, LOWERS = "raises risk", "lowers risk"


@dataclass(frozen=True)
class Reason:
    feature: str
    text: str
    push: float
    direction: str


@dataclass
class CaseExplanation:
    score: float
    base_value: float
    risk_reasons: list[Reason] = field(default_factory=list)
    protective_reasons: list[Reason] = field(default_factory=list)
    summary: str = ""


def humanize_seconds(sec: float) -> str:
    sec = float(sec)
    if sec < 120:
        return f"{sec:.0f} seconds"
    if sec < 2 * 3600:
        return f"{sec / 60:.0f} minutes"
    if sec < 2 * 86400:
        return f"{sec / 3600:.1f} hours"
    return f"{sec / 86400:.1f} days"


def describe_feature(feature: str, value: float, category_names: Sequence[str] | None = None) -> str:
    """Plain-language statement of one feature's value."""
    if feature == "amount":
        return f"Transaction amount is {value:,.2f}"
    if feature == "hour":
        return f"Transaction happened at hour {int(value):02d}:00"
    if feature == "category_code":
        code = int(value)
        if category_names is not None and 0 <= code < len(category_names):
            return f"Merchant category is '{category_names[code]}'"
        return "Merchant category is not one seen in training"
    if feature == "secs_since_prev":
        if pd.isna(value):
            return "No earlier transaction on record for this card"
        return f"Previous transaction on this card was {humanize_seconds(value)} earlier"
    if feature == "amt_vs_avg":
        if pd.isna(value):
            return "No earlier spending history to compare the amount with"
        return f"Amount is {value:.1f}x this customer's earlier average"
    return f"{feature} = {value}"


def explain_case(
    values: pd.Series,
    pushes: pd.Series,
    base_value: float,
    category_names: Sequence[str] | None = None,
    top_n: int = 3,
    min_abs_push: float = 0.01,
) -> CaseExplanation:
    """Build an explanation for one transaction.

    values: the feature values the model saw; pushes: SHAP value per feature (same index).
    Pushes smaller than `min_abs_push` (in score points) are treated as noise and omitted.
    """
    score = float(base_value + pushes.sum())
    reasons = []
    for feat in pushes.index:
        p = float(pushes[feat])
        if abs(p) < min_abs_push:
            continue
        reasons.append(
            Reason(feat, describe_feature(feat, values[feat], category_names), p, RAISES if p > 0 else LOWERS)
        )
    risk = sorted((r for r in reasons if r.push > 0), key=lambda r: -r.push)[:top_n]
    protect = sorted((r for r in reasons if r.push < 0), key=lambda r: r.push)[:top_n]

    if risk:
        parts = "; ".join(f"{r.text} ({r.push:+.2f})" for r in risk)
        summary = f"Model score {score:.2f} (average {base_value:.2f}). Main factors raising the score: {parts}."
    else:
        summary = f"Model score {score:.2f} (average {base_value:.2f}). No single factor raised the score noticeably."
    return CaseExplanation(score, float(base_value), risk, protect, summary)
