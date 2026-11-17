"""Fact-gathering tools for the investigation agent. Deterministic: no model calls, nothing invented.

All tools take `history`, the full chronologically sorted transaction table (with behaviour features) and `idx`, the
row position of the transaction being investigated. They only look at rows BEFORE that position, so nothing that
happens after the transaction can leak into the case.

Labels: in real systems a transaction's fraud label arrives days or weeks late. `similar_cases` therefore only
uses labels from transactions at least `label_delay_days` older than the one being investigated.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from src.data.loader import Schema

# Hours treated as "night". Dataset-specific: taken from the EDA (84.7% of train frauds at hours 22 to 3).
NIGHT_HOURS = (22, 23, 0, 1, 2, 3)


def mask_card(card_number) -> str:
    """Show only the last 4 digits of a card number."""
    return str(card_number)[-4:]


def card_prior(history: pd.DataFrame, idx: int, schema: Schema = Schema()) -> pd.DataFrame:
    """Earlier transactions of the same card as the transaction at position idx."""
    card = history.iloc[idx][schema.customer]
    past = history.iloc[:idx]
    return past[past[schema.customer] == card]


def recent_transactions(card_past: pd.DataFrame, n: int = 5, schema: Schema = Schema()) -> list[dict]:
    """The card's last n earlier transactions, most recent first."""
    rows = card_past.tail(n).iloc[::-1]
    return [
        {"time": str(r[schema.time]), "merchant": r[schema.merchant], "category": r[schema.category], "amount": float(r[schema.amount])}
        for _, r in rows.iterrows()
    ]


def spending_profile(card_past: pd.DataFrame, schema: Schema = Schema()) -> dict:
    if card_past.empty:
        return {"n_prior": 0}
    amt = card_past[schema.amount]
    return {
        "n_prior": int(len(card_past)),
        "mean_amount": float(amt.mean()),
        "median_amount": float(amt.median()),
        "max_amount": float(amt.max()),
        "first_seen": str(card_past[schema.time].iloc[0]),
    }


def merchant_history(card_past: pd.DataFrame, merchant: str, schema: Schema = Schema()) -> dict:
    visits = int((card_past[schema.merchant] == merchant).sum())
    return {"prior_visits": visits, "new_merchant": visits == 0}


def velocity_summary(txn: pd.Series) -> dict:
    def get(col):
        v = txn.get(col)
        return None if v is None or pd.isna(v) else float(v)

    return {"txn_count_1h": get("txn_count_1h"), "txn_count_24h": get("txn_count_24h"), "secs_since_prev": get("secs_since_prev")}


def location_summary(txn: pd.Series) -> dict | None:
    v = txn.get("distance_km")
    return None if v is None or pd.isna(v) else {"distance_km": float(v)}


def merchant_fraud_history(history: pd.DataFrame, idx: int, label_delay_days: int = 7, schema: Schema = Schema()) -> dict:
    """Labelled earlier transactions at the same merchant (any card), using only labels at least label_delay_days old."""
    txn = history.iloc[idx]
    past = history.iloc[:idx]
    past = past[past[schema.time] <= txn[schema.time] - pd.Timedelta(days=label_delay_days)]
    at_merchant = past[past[schema.merchant] == txn[schema.merchant]]
    return {"n_labelled": int(len(at_merchant)), "n_fraud": int(at_merchant[schema.label].sum()), "label_delay_days": label_delay_days}


def similar_cases(
    history: pd.DataFrame,
    idx: int,
    label_delay_days: int = 7,
    amount_band: tuple[float, float] = (0.5, 2.0),
    max_examples: int = 3,
    schema: Schema = Schema(),
) -> dict:
    """Comparable earlier transactions: same category, amount within the band, same night/day group.

    Only rows at least `label_delay_days` older than the transaction count, because newer labels would not be known yet.
    """
    txn = history.iloc[idx]
    cutoff = txn[schema.time] - pd.Timedelta(days=label_delay_days)
    past = history.iloc[:idx]
    past = past[past[schema.time] <= cutoff]
    amount = float(txn[schema.amount])
    is_night = txn[schema.time].hour in NIGHT_HOURS
    past_night = past[schema.time].dt.hour.isin(NIGHT_HOURS)
    mask = (
        (past[schema.category] == txn[schema.category])
        & past[schema.amount].between(amount * amount_band[0], amount * amount_band[1])
        & (past_night == is_night)
    )
    sim = past[mask]
    frauds = sim[sim[schema.label] == 1]
    closest = frauds.iloc[np.argsort(np.abs(np.log(frauds[schema.amount].to_numpy() / amount)))[:max_examples]] if len(frauds) else frauds
    return {
        "n_similar": int(len(sim)),
        "n_fraud": int(len(frauds)),
        "fraud_share": float(len(frauds) / len(sim)) if len(sim) else None,
        "label_delay_days": label_delay_days,
        "examples": [
            {"time": str(r[schema.time]), "merchant": r[schema.merchant], "amount": float(r[schema.amount])} for _, r in closest.iterrows()
        ],
    }
