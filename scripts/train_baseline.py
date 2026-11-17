"""Train and evaluate the raw-feature baseline.

Example (after downloading data per data/README.md; choose dates from your EDA):
    python scripts/train_baseline.py --files data/raw/fraudTrain.csv data/raw/fraudTest.csv \
        --train-end 2020-01-01 --val-end 2020-03-01 --sample 150000

Prints metrics for validation and test. Nothing is hard-coded: results come from your data.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.data.loader import Schema, load_transactions
from src.data.split import chronological_split, split_summary
from src.models.baseline import build_baseline, evaluate, make_raw_features


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("--files", nargs="+", required=True)
    ap.add_argument("--train-end", required=True, help="train = rows before this date")
    ap.add_argument("--val-end", required=True, help="validation = [train-end, val-end); test = rest")
    ap.add_argument("--sample", type=int, default=None, help="subsample TRAIN rows only (keeps all frauds)")
    ap.add_argument("--seed", type=int, default=42)
    a = ap.parse_args()

    s = Schema()
    df = load_transactions(a.files, s)
    train, val, test = chronological_split(df, s.time, a.train_end, a.val_end)
    print(split_summary({"train": train, "val": val, "test": test}, s.time, s.label), "\n")

    if a.sample and len(train) > a.sample:
        frauds = train[train[s.label] == 1]
        legit = train[train[s.label] == 0].sample(max(a.sample - len(frauds), 0), random_state=a.seed)
        train = pd.concat([frauds, legit]).sort_values(s.time)
        print(f"Subsampled train to {len(train)} rows (all frauds kept; legit downsampled).")
        print("Note: this changes the training fraud rate, so scores are not calibrated probabilities.\n")

    model = build_baseline().fit(make_raw_features(train, s), train[s.label])
    for name, part in {"val": val, "test": test}.items():
        scores = model.predict_proba(make_raw_features(part, s))[:, 1]
        print(name, evaluate(part[s.label].to_numpy(), scores))


if __name__ == "__main__":
    main()
