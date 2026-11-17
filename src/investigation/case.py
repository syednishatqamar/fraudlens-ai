"""Build and render a case file for one flagged transaction using fixed rules (no LLM).

The case combines the model's output (calibrated probability, band, SHAP reasons) with facts from the card's own
history. Notes only compare the model's band with the fraud rate among comparable past transactions; the thresholds
below are heuristics, not tuned values.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import pandas as pd

from src.data.loader import Schema
from src.explainability.reasons import CaseExplanation, Reason, humanize_seconds
from src.investigation import tools
from src.models.decision import BLOCK, CLEAR, REVIEW, DecisionCosts

MIN_SIMILAR = 20            # fewer comparable past transactions than this: too few to compare
BLOCK_LOW_SHARE = 0.05      # BLOCK but fewer than 5% of comparable past transactions were fraud: flag for attention
CLEAR_HIGH_SHARE = 0.25     # CLEAR but more than 25% of comparable past transactions were fraud: flag for attention


@dataclass
class CaseFile:
    case_id: str
    timestamp: pd.Timestamp
    card_last4: str
    amount: float
    merchant: str
    category: str
    probability: float
    band: str
    summary: str
    risk_reasons: list[Reason]
    protective_reasons: list[Reason]
    recent_transactions: list[dict]
    profile: dict
    velocity: dict
    merchant_history: dict
    merchant_fraud: dict
    action_costs: dict
    location: dict | None
    similar: dict
    notes: list[str] = field(default_factory=list)


def build_case(
    history: pd.DataFrame,
    idx: int,
    probability: float,
    band: str,
    explanation: CaseExplanation,
    label_delay_days: int = 7,
    schema: Schema = Schema(),
    costs: DecisionCosts = DecisionCosts(),
) -> CaseFile:
    txn = history.iloc[idx]
    card_past = tools.card_prior(history, idx, schema)
    similar = tools.similar_cases(history, idx, label_delay_days, schema=schema)
    amount = float(txn[schema.amount])
    action_costs = {CLEAR: probability * amount, REVIEW: costs.review_cost, BLOCK: (1.0 - probability) * costs.block_false_positive_cost}
    cheapest = min(action_costs, key=action_costs.get)
    notes = []
    if cheapest != band:
        notes.append(f"CHECK: the band is {band} but {cheapest} is the cheapest action under the given costs.")
    if card_past.empty:
        notes.append("No earlier transactions on this card, so history-based facts are unavailable.")
    share, n = similar["fraud_share"], similar["n_similar"]
    if n < MIN_SIMILAR:
        notes.append(f"Only {n} comparable labelled past transactions, too few to compare against.")
    else:
        notes.append(
            f"{similar['n_fraud']} of {n} ({share:.0%}) comparable past transactions were fraud "
            f"(labels from at least {label_delay_days} days earlier only)."
        )
        if band == BLOCK and share < BLOCK_LOW_SHARE:
            notes.append("CHECK: the model band is BLOCK but few comparable past transactions were fraud.")
        if band == CLEAR and share > CLEAR_HIGH_SHARE:
            notes.append("CHECK: the model band is CLEAR but many comparable past transactions were fraud.")
    return CaseFile(
        case_id=str(idx),
        timestamp=txn[schema.time],
        card_last4=tools.mask_card(txn[schema.customer]),
        amount=float(txn[schema.amount]),
        merchant=txn[schema.merchant],
        category=txn[schema.category],
        probability=float(probability),
        band=band,
        summary=explanation.summary,
        risk_reasons=explanation.risk_reasons,
        protective_reasons=explanation.protective_reasons,
        recent_transactions=tools.recent_transactions(card_past, 5, schema),
        profile=tools.spending_profile(card_past, schema),
        velocity=tools.velocity_summary(txn),
        merchant_history=tools.merchant_history(card_past, txn[schema.merchant], schema),
        merchant_fraud=tools.merchant_fraud_history(history, idx, label_delay_days, schema),
        action_costs=action_costs,
        location=tools.location_summary(txn),
        similar=similar,
        notes=notes,
    )


def render_case_text(c: CaseFile) -> str:
    lines = [
        f"CASE {c.case_id}  |  card ending {c.card_last4}  |  {c.timestamp}",
        f"Transaction: {c.amount:,.2f} at {c.merchant} ({c.category})",
        f"Model: fraud probability {c.probability:.1%}  ->  recommended action {c.band}",
    ]
    if c.risk_reasons:
        factors = "; ".join(f"{r.text} ({r.push:+.2f})" for r in c.risk_reasons)
        lines.append(f"Why the model flagged it (log-odds pushes, compared with a typical transaction): {factors}")
    else:
        lines.append("Why the model flagged it: no single factor raised the score noticeably.")
    for r in c.protective_reasons:
        lines.append(f"  lowered the score vs a typical transaction: {r.text} ({r.push:+.2f})")
    ac = c.action_costs
    cheapest = min(ac, key=ac.get)
    lines.append(
        f"Why this action: expected cost if CLEARed {ac[CLEAR]:,.2f}, REVIEWed {ac[REVIEW]:,.2f}, BLOCKed {ac[BLOCK]:,.2f}; "
        f"cheapest is {cheapest} (placeholder costs)"
    )
    p = c.profile
    if p.get("n_prior"):
        lines.append(
            f"Card history: {p['n_prior']} earlier transactions since {p['first_seen']}; typical amount "
            f"{p['median_amount']:,.2f} (mean {p['mean_amount']:,.2f}, max {p['max_amount']:,.2f})"
        )
    v = c.velocity
    parts = []
    if v["secs_since_prev"] is not None:
        parts.append(f"{humanize_seconds(v['secs_since_prev'])} since the previous transaction")
    if v["txn_count_1h"] is not None:
        parts.append(f"{v['txn_count_1h']:.0f} earlier transactions in the last hour, {v['txn_count_24h']:.0f} in the last 24 hours")
    if parts:
        lines.append("Activity: " + "; ".join(parts))
    m = c.merchant_history
    lines.append("Merchant: first time this card used it" if m["new_merchant"] else f"Merchant: card used it {m['prior_visits']} time(s) before")
    mf = c.merchant_fraud
    lines.append(
        f"Merchant record: {mf['n_fraud']} of {mf['n_labelled']} labelled past transactions at this merchant were fraud"
        if mf["n_labelled"] else "Merchant record: no labelled past transactions at this merchant"
    )
    if c.location:
        lines.append(f"Distance between customer and merchant: {c.location['distance_km']:,.0f} km")
    if c.recent_transactions:
        lines.append("Most recent earlier transactions:")
        lines += [f"  {t['time']}  {t['amount']:>9,.2f}  {t['merchant']} ({t['category']})" for t in c.recent_transactions]
    for note in c.notes:
        lines.append(f"Note: {note}")
    if c.similar["examples"]:
        lines.append("Closest comparable past frauds:")
        lines += [f"  {e['time']}  {e['amount']:>9,.2f}  {e['merchant']}" for e in c.similar["examples"]]
    return "\n".join(lines)

