"""Calibrate the model's scores and evaluate CLEAR / REVIEW / BLOCK decisions.

Time layout (all from the training-period file; the final test file is not used):
    train      : before --train-end                -> trains the model
    calibration: --train-end to --calib-end        -> fits the calibrator
    evaluation : on or after --calib-end           -> reports calibration and cost, never seen by model or calibrator

Example:
    python scripts/calibrate_and_decide.py --files data/raw/fraudTrain.csv --train-end 2020-04-01 --calib-end 2020-05-15

Costs are placeholders (see src/models/decision.py); change them with --review-cost and --block-fp-cost.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd

from src.data.loader import Schema, load_transactions
from src.features.behaviour import add_behaviour_features
from src.models.calibration import calibration_report, make_calibrator
from src.models.decision import CLEAR, REVIEW, BLOCK, DecisionCosts, decide, realized_cost
from src.models.tree import V1_COLUMNS, build_tree_model, fit_category_encoder, make_tree_features


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("--files", nargs="+", required=True)
    ap.add_argument("--train-end", required=True)
    ap.add_argument("--calib-end", required=True)
    ap.add_argument("--method", choices=["platt", "isotonic"], default="platt")
    ap.add_argument("--review-cost", type=float, default=DecisionCosts.review_cost)
    ap.add_argument("--block-fp-cost", type=float, default=DecisionCosts.block_false_positive_cost)
    ap.add_argument("--seed", type=int, default=42)
    a = ap.parse_args()

    s = Schema()
    costs = DecisionCosts(a.review_cost, a.block_fp_cost)
    df = add_behaviour_features(load_transactions(a.files, s), s)
    t = df[s.time]
    parts = {"train": df[t < a.train_end], "calibration": df[(t >= a.train_end) & (t < a.calib_end)], "evaluation": df[t >= a.calib_end]}
    for name, d in parts.items():
        print(f"{name:12} rows={len(d):>9,}  frauds={int(d[s.label].sum()):>5,}  rate={100 * d[s.label].mean():.3f}%")
    if min(int(parts["calibration"][s.label].sum()), int(parts["evaluation"][s.label].sum())) < 300:
        print("WARNING: fewer than 300 frauds in calibration or evaluation; calibration results will be unstable.")

    enc = fit_category_encoder(parts["train"], s)
    X = {k: make_tree_features(v, enc, s, V1_COLUMNS) for k, v in parts.items()}
    model = build_tree_model(a.seed).fit(X["train"], parts["train"][s.label])
    margin = {k: model.decision_function(X[k]) for k in ("calibration", "evaluation")}
    raw = {k: 1 / (1 + np.exp(-margin[k])) for k in margin}

    cal = make_calibrator(a.method).fit(margin["calibration"], parts["calibration"][s.label])
    p_eval = cal.predict_proba(margin["evaluation"])
    y = parts["evaluation"][s.label].to_numpy()
    amt = parts["evaluation"][s.amount].to_numpy()

    print("\n=== Calibration on the evaluation slice ===")
    for label, probs in (("raw model score (uncalibrated)", raw["evaluation"]), (f"{a.method} calibrated", p_eval)):
        table, summ = calibration_report(y, probs)
        print(f"\n{label}: Brier {summ['brier']:.5f}, log loss {summ['log_loss']:.5f}")
        print(table.round(4).to_string(index=False))

    print(f"\n=== Decisions (review cost {costs.review_cost}, block false-positive cost {costs.block_false_positive_cost}; placeholders) ===")
    dec = decide(p_eval, amt, costs)
    rows = []
    for band in (CLEAR, REVIEW, BLOCK):
        m = dec == band
        rows.append({"band": band, "transactions": int(m.sum()), "share_%": 100 * m.mean(), "frauds": int(y[m].sum()),
                     "fraud_rate_%": 100 * y[m].mean() if m.any() else float("nan")})
    print(pd.DataFrame(rows).round(3).to_string(index=False))
    n = len(y)
    print(f"\nReviews per 1,000 transactions: {1000 * (dec == REVIEW).sum() / n:.1f}")
    print(f"Total cost with these decisions: {realized_cost(dec, y, amt, costs):,.0f}")
    print(f"Total cost if everything is CLEARed: {realized_cost(np.full(n, CLEAR), y, amt, costs):,.0f}")
    print(f"Total cost if everything is REVIEWed: {realized_cost(np.full(n, REVIEW), y, amt, costs):,.0f}")


if __name__ == "__main__":
    main()
