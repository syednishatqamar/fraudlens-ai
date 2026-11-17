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
from src.models.tree import build_tree_model, fit_category_encoder, make_tree_features


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("--files", nargs="+", required=True, help="training-period file(s), not the final test file")
    ap.add_argument("--train-end", required=True, help="train = rows before this date; validation = rest")
    ap.add_argument("--seed", type=int, default=42)
    a = ap.parse_args()

    s = Schema()
    df = add_behaviour_features(load_transactions(a.files, s), s)  # features use past rows only
    cut = a.train_end
    train, val = df[df[s.time] < cut], df[df[s.time] >= cut]
    print(split_summary({"train": train, "validation": val}, s.time, s.label), "\n")

    enc = fit_category_encoder(train, s)
    model = build_tree_model(a.seed).fit(make_tree_features(train, enc, s), train[s.label])
    scores = model.predict_proba(make_tree_features(val, enc, s))[:, 1]
    y = val[s.label].to_numpy()
    print("validation PR-AUC:", round(float(average_precision_score(y, scores)), 4))
    print("top-k:", top_k_caught(y, scores))


if __name__ == "__main__":
    main()
