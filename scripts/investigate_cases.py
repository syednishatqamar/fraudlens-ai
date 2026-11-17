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

from src.investigation.case import render_case_text
from src.investigation.runner import iter_cases


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("--files", nargs="+", required=True)
    ap.add_argument("--train-end", required=True)
    ap.add_argument("--calib-end", required=True)
    ap.add_argument("--n", type=int, default=2, help="cases to print per band")
    ap.add_argument("--label-delay-days", type=int, default=7)
    ap.add_argument("--seed", type=int, default=1)
    a = ap.parse_args()
    for case, is_fraud in iter_cases(a.files, a.train_end, a.calib_end, a.n, a.label_delay_days, a.seed):
        print("=" * 100)
        print(render_case_text(case))
        print(f"[for your learning only] actual label: {'FRAUD' if is_fraud else 'legitimate'}\n")


if __name__ == "__main__":
    main()
