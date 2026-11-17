"""Print investigation case files for a few REVIEW and BLOCK transactions from the evaluation slice.

Pipeline: train model (v1 features) -> calibrate -> assign CLEAR/REVIEW/BLOCK -> explain with SHAP -> gather card history.
Everything is rule-based (no LLM). The true label is printed at the end of each case for YOUR learning only; the case
itself never sees the label of the transaction under investigation.

Example:
    python scripts/investigate_cases.py --files data/raw/fraudTrain.csv --train-end 2020-04-01 --calib-end 2020-05-15
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np

from src.data.loader import Schema, load_transactions
from src.explainability.reasons import explain_case
from src.explainability.shap_explainer import ShapExplainer
from src.features.behaviour import add_behaviour_features
from src.features.geo import add_location_features
from src.investigation.case import build_case, render_case_text
from src.models.calibration import make_calibrator
from src.models.decision import BLOCK, REVIEW, DecisionCosts, decide
from src.models.tree import V1_COLUMNS, build_tree_model, fit_category_encoder, make_tree_features


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("--files", nargs="+", required=True)
    ap.add_argument("--train-end", required=True)
    ap.add_argument("--calib-end", required=True)
    ap.add_argument("--n", type=int, default=2, help="cases to print per band")
    ap.add_argument("--label-delay-days", type=int, default=7)
    ap.add_argument("--seed", type=int, default=1)
    a = ap.parse_args()

    s = Schema()
    df = add_behaviour_features(load_transactions(a.files, s), s)
    if all(c in df.columns for c in s.geo()):
        df = add_location_features(df, s)
    t = df[s.time]
    train, calib, ev = df[t < a.train_end], df[(t >= a.train_end) & (t < a.calib_end)], df[t >= a.calib_end]

    enc = fit_category_encoder(train, s)
    X = {k: make_tree_features(v, enc, s, V1_COLUMNS) for k, v in (("train", train), ("calib", calib), ("eval", ev))}
    model = build_tree_model().fit(X["train"], train[s.label])
    cal = make_calibrator("platt").fit(model.decision_function(X["calib"]), calib[s.label])
    prob = cal.predict_proba(model.decision_function(X["eval"]))
    bands = decide(prob, ev[s.amount].to_numpy(), DecisionCosts())
    print(f"evaluation slice: {len(ev):,} transactions; REVIEW {np.sum(bands == REVIEW):,}, BLOCK {np.sum(bands == BLOCK):,}\n")

    explainer = ShapExplainer(model, X["train"], scale="logodds")
    names = list(enc.categories_[0])
    rng = np.random.default_rng(a.seed)
    for band in (BLOCK, REVIEW):
        pool = np.flatnonzero(bands == band)
        for pos in rng.choice(pool, size=min(a.n, len(pool)), replace=False):
            label_idx = ev.index[pos]
            pushes, base = explainer.explain(X["eval"].loc[[label_idx]])
            e = explain_case(X["eval"].loc[label_idx], pushes.iloc[0], float(base[0]), names, scale="logodds")
            case = build_case(df, df.index.get_loc(label_idx), float(prob[pos]), band, e, a.label_delay_days, s)
            print("=" * 100)
            print(render_case_text(case))
            print(f"[for your learning only] actual label: {'FRAUD' if ev.loc[label_idx, s.label] == 1 else 'legitimate'}\n")


if __name__ == "__main__":
    main()
