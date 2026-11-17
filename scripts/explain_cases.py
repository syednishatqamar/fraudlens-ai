"""Train the tree model and print plain-language reasons for a few flagged validation transactions.

Example:
    python scripts/explain_cases.py --files data/raw/fraudTrain.csv --train-end 2020-04-01

Picks cases at random (fixed seed), not the best-looking ones. Validation data only.
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
from src.models.tree import build_tree_model, fit_category_encoder, make_tree_features


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("--files", nargs="+", required=True)
    ap.add_argument("--train-end", required=True)
    ap.add_argument("--n", type=int, default=3, help="cases to print per group")
    ap.add_argument("--seed", type=int, default=1)
    a = ap.parse_args()

    s = Schema()
    df = add_behaviour_features(load_transactions(a.files, s), s)
    train, val = df[df[s.time] < a.train_end], df[df[s.time] >= a.train_end]
    enc = fit_category_encoder(train, s)
    Xtr, Xva = make_tree_features(train, enc, s), make_tree_features(val, enc, s)
    model = build_tree_model().fit(Xtr, train[s.label])

    score = model.predict_proba(Xva)[:, 1]
    k = int(val[s.label].sum())
    flagged = (np.argsort(np.argsort(-score)) < k)  # top-k by score
    val = val.assign(score=score, flagged=flagged)
    groups = {
        "flagged and really fraud": val[val["flagged"] & (val[s.label] == 1)],
        "flagged but NOT fraud (false alarm)": val[val["flagged"] & (val[s.label] == 0)],
    }

    explainer = ShapExplainer(model, Xtr)
    names = list(enc.categories_[0])
    for title, g in groups.items():
        print(f"\n######## {title}: {len(g)} cases ########")
        for idx in g.sample(min(a.n, len(g)), random_state=a.seed).index:
            pushes, base = explainer.explain(Xva.loc[[idx]])
            e = explain_case(Xva.loc[idx], pushes.iloc[0], float(base[0]), names)
            print(f"\n- {e.summary}")
            for r in e.protective_reasons:
                print(f"    lowering the score: {r.text} ({r.push:+.2f})")


if __name__ == "__main__":
    main()
