"""Write analyst notes for a few REVIEW and BLOCK cases, optionally with Gemini.

    --llm none   : fixed-template notes, no key or internet needed
    --llm gemini : ask Gemini; a note that breaks a guard rule is replaced by the template note and the reason is printed

The API key is read from the GEMINI_API_KEY environment variable or, on Kaggle, from a Kaggle Secret with that name.
Never paste the key into code or chat. Pass a current model name from Google AI Studio with --model.

Example:
    python scripts/write_notes.py --files data/raw/fraudTrain.csv --train-end 2020-04-01 --calib-end 2020-05-15 --llm none
"""
from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.investigation.case import render_case_text
from src.investigation.note import GeminiClient, write_note
from src.investigation.runner import iter_cases


def get_api_key() -> str | None:
    key = os.environ.get("GEMINI_API_KEY")
    if key:
        return key
    try:
        from kaggle_secrets import UserSecretsClient

        return UserSecretsClient().get_secret("GEMINI_API_KEY")
    except Exception:
        return None


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("--files", nargs="+", required=True)
    ap.add_argument("--train-end", required=True)
    ap.add_argument("--calib-end", required=True)
    ap.add_argument("--n", type=int, default=2, help="cases per band")
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--llm", choices=["none", "gemini"], default="none")
    ap.add_argument("--model", help="model name for --llm gemini (copy a current one from Google AI Studio)")
    ap.add_argument("--delay-seconds", type=float, default=5.0, help="pause between LLM calls (free-tier rate limits)")
    ap.add_argument("--show-case", action="store_true", help="also print the full case file")
    a = ap.parse_args()

    client = None
    if a.llm == "gemini":
        key = get_api_key()
        if not key or not a.model:
            sys.exit("--llm gemini needs --model and a GEMINI_API_KEY (environment variable or Kaggle Secret).")
        client = GeminiClient(key, a.model)

    for case, is_fraud in iter_cases(a.files, a.train_end, a.calib_end, a.n, seed=a.seed):
        if a.show_case:
            print("=" * 100)
            print(render_case_text(case))
        result = write_note(case, client)
        print("=" * 100)
        print(f"CASE {case.case_id} | card ending {case.card_last4} | {case.band} | probability {case.probability:.1%}")
        print(f"[note source: {result.source}]")
        for issue in result.issues:
            print(f"[LLM note rejected: {issue}]")
        print(result.text)
        print(f"[for your learning only] actual label: {'FRAUD' if is_fraud else 'legitimate'}\n")
        if client is not None:
            time.sleep(a.delay_seconds)


if __name__ == "__main__":
    main()
