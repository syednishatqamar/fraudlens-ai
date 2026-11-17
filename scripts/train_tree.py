"""Train the tree model with behaviour features and report VALIDATION metrics only.

The final test file is deliberately not evaluated here; it is opened once, at the end.

Example:
    python scripts/train_tree.py --files data/raw/fraudTrain.csv --train-end 2020-04-01
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
from sklearn.metrics import average_precision_score

from src.data.loader import Schema, load_transactions
from src.data.split import split_summary
from src.features.behaviour import add_behaviour_features
from src.features.geo import add_location_features
from src.models.evaluate import top_k_caught
from src.models.tree import FEATURE_COLUMNS, OPTIONAL_COLUMNS, build_tree_model, fit_category_encoder, make_tree_features


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("--files", nargs="+", required=True, help="training-period file(s), not the final test file")
    ap.add_argument("--train-end", required=True, help="train = rows before this date; validation = rest")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--seeds", type=int, nargs="*", default=[],
                    help="train once per seed and report mean and std of validation PR-AUC (measures run-to-run noise)")
    ap.add_argument("--add", nargs="*", default=[], choices=OPTIONAL_COLUMNS,
                    help="optional features to include: txn_count_1h txn_count_24h new_merchant distance_km")
    ap.add_argument("--drop", nargs="*", default=[], choices=FEATURE_COLUMNS,
                    help="feature columns to leave out (ablation), e.g. --drop amt_vs_avg")
    a = ap.parse_args()

    s = Schema()
    df = add_behaviour_features(load_transactions(a.files, s), s)  # features use past rows only
    if all(c in df.columns for c in s.geo()):
        df = add_location_features(df, s)
    cut = a.train_end
    train, val = df[df[s.time] < cut], df[df[s.time] >= cut]
    print(split_summary({"train": train, "validation": val}, s.time, s.label), "\n")

    enc = fit_category_encoder(train, s)
    cols = [c for c in FEATURE_COLUMNS + a.add if c not in a.drop]
    xtr = make_tree_features(train, enc, s, cols)
    xva = make_tree_features(val, enc, s, cols)
    print("features used:", list(xtr.columns))
    y = val[s.label].to_numpy()
    seeds = a.seeds or [a.seed]
    aps = []
    for sd in seeds:
        model = build_tree_model(sd).fit(xtr, train[s.label])
        scores = model.predict_proba(xva)[:, 1]
        ap_score = float(average_precision_score(y, scores))
        aps.append(ap_score)
        tk = top_k_caught(y, scores)
        if len(seeds) > 1:
            print(f"seed {sd}: PR-AUC {ap_score:.4f}  caught {tk['caught']}/{tk['frauds_total']}")
        else:
            print("validation PR-AUC:", round(ap_score, 4))
            print("top-k:", tk)
    if len(seeds) > 1:
        print(f"PR-AUC over {len(seeds)} seeds: mean {np.mean(aps):.4f}, std {np.std(aps, ddof=1):.4f}")


if __name__ == "__main__":
    main()
