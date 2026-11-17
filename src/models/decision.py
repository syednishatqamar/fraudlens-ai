"""Cost-based CLEAR / REVIEW / BLOCK decisions from a calibrated fraud probability.

For each transaction with fraud probability p and amount A, pick the cheapest action:
  CLEAR  : lose the amount if it is fraud                      -> p * A
  REVIEW : an analyst reviews it (assumed to decide correctly) -> review_cost
  BLOCK  : block it; a legitimate customer is inconvenienced   -> (1 - p) * block_false_positive_cost

The default costs are PLACEHOLDERS, not measured values. Real costs must come from the business.
Because CLEAR depends on the amount, the same probability can be CLEAR for a small purchase and REVIEW for a large one.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

CLEAR, REVIEW, BLOCK = "CLEAR", "REVIEW", "BLOCK"


@dataclass(frozen=True)
class DecisionCosts:
    review_cost: float = 5.0
    block_false_positive_cost: float = 25.0


def decide(probs, amounts, costs: DecisionCosts = DecisionCosts()) -> np.ndarray:
    p = np.asarray(probs, dtype=float)
    a = np.asarray(amounts, dtype=float)
    cost_clear = p * a
    cost_review = np.full_like(p, costs.review_cost)
    cost_block = (1.0 - p) * costs.block_false_positive_cost
    pick = np.argmin(np.vstack([cost_clear, cost_review, cost_block]), axis=0)  # ties favour CLEAR, then REVIEW
    return np.array([CLEAR, REVIEW, BLOCK])[pick]


def realized_cost(decisions, y_true, amounts, costs: DecisionCosts = DecisionCosts()) -> float:
    """Total cost of the decisions given the true labels (same assumptions as decide)."""
    d = np.asarray(decisions)
    y = np.asarray(y_true, dtype=float)
    a = np.asarray(amounts, dtype=float)
    cost = np.where(d == CLEAR, y * a, np.where(d == REVIEW, costs.review_cost, (1.0 - y) * costs.block_false_positive_cost))
    return float(cost.sum())
