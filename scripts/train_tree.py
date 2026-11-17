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

from sklearn.metrics import average_precision_score

from src.data.loader import Schema, load_transactions
from src.data.split import split_summary
from src.features.behaviour import add_behaviour_features
from src.models.evaluate import top_k_caught
from src.models.tree import FEATURE_COLUMNS, VELOCITY_COLUMNS, build_tree_model, fit_category_encoder, make_tree_features


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("--files", nargs="+", required=True, help="training-period file(s), not the final test file")
    ap.add_argument("--train-end", required=True, help="train = rows before this date; validation = rest")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--add", nargs="*", default=[], choices=VELOCITY_COLUMNS,
                    help="optional velocity features to include, e.g. --add txn_count_1h txn_count_24h")
    ap.add_argument("--drop", nargs="*", default=[], choices=FEATURE_COLUMNS,
                    help="feature columns to leave out (ablation), e.g. --drop amt_vs_avg")
    a = ap.parse_args()

    s = Schema()
    df = add_behaviour_features(load_transactions(a.files, s), s)  # features use past rows only
    cut = a.train_end
    train, val = df[df[s.time] < cut], df[df[s.time] >= cut]
    print(split_summary({"train": train, "validation": val}, s.time, s.label), "\n")

    enc = fit_category_encoder(train, s)
    cols = [c for c in FEATURE_COLUMNS + a.add if c not in a.drop]
    xtr = make_tree_features(train, enc, s, cols)
    xva = make_tree_features(val, enc, s, cols)
    print("features used:", list(xtr.columns))
    model = build_tree_model(a.seed).fit(xtr, train[s.label])
    scores = model.predict_proba(xva)[:, 1]
    y = val[s.label].to_numpy()
    print("validation PR-AUC:", round(float(average_precision_score(y, scores)), 4))
    print("top-k:", top_k_caught(y, scores))


if __name__ == "__main__":
    main()
