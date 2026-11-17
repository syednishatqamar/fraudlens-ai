"""Fill the analyst queue: build REVIEW/BLOCK cases from the evaluation slice and POST them to a running API.

Example:
    python scripts/seed_cases.py --files data/raw/fraudTrain.csv --train-end 2020-04-01 --calib-end 2020-05-15 \
        --api-url http://localhost:8000 --n 20

Notes are template notes (no LLM). The true label is NOT sent to the API: the analyst decides without it.
If the API requires a key, set ANALYST_API_KEY in your environment.
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import httpx

from src.investigation.note import write_note
from src.investigation.runner import iter_cases
from src.investigation.serialize import case_to_dict


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("--files", nargs="+", required=True)
    ap.add_argument("--train-end", required=True)
    ap.add_argument("--calib-end", required=True)
    ap.add_argument("--api-url", default="http://localhost:8000")
    ap.add_argument("--n", type=int, default=10, help="cases per band")
    ap.add_argument("--seed", type=int, default=1)
    a = ap.parse_args()

    headers = {"X-API-Key": os.environ["ANALYST_API_KEY"]} if os.environ.get("ANALYST_API_KEY") else {}
    sent = skipped = 0
    with httpx.Client(base_url=a.api_url, headers=headers, timeout=30) as client:
        for case, _is_fraud in iter_cases(a.files, a.train_end, a.calib_end, a.n, seed=a.seed):
            note = write_note(case)
            body = {
                "case_ref": f"txn-{case.case_id}", "band": case.band, "probability": case.probability, "amount": case.amount,
                "card_last4": case.card_last4, "merchant": case.merchant, "category": case.category,
                "txn_time": str(case.timestamp).replace(" ", "T"), "case": case_to_dict(case),
                "note": note.text, "note_source": note.source,
            }
            r = client.post("/cases", json=body)
            if r.status_code == 201:
                sent += 1
            elif r.status_code == 409:
                skipped += 1                      # already in the queue
            else:
                sys.exit(f"API refused case {case.case_id}: {r.status_code} {r.text[:300]}")
    print(f"sent {sent} cases, skipped {skipped} already present")


if __name__ == "__main__":
    main()
