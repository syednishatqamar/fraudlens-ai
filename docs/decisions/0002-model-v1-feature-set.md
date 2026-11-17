# 0002. Feature set for model v1

Date: 2026-11
Status: proposed (draft for the project owner to review)

## Context
Features were added one group at a time and judged on validation PR-AUC with 5 seeds (see docs/experiments.md).
Gains: amount/hour/category baseline 0.8240; + secs_since_prev and amt_vs_avg 0.8808 (mean); + txn_count_1h/24h 0.8897;
+ new_merchant 0.8925 (within noise of the previous step); + distance_km no gain.

## Decision
Model v1 uses: amount, hour, category_code, secs_since_prev, amt_vs_avg, txn_count_1h, txn_count_24h.
Command: `python scripts/train_tree.py --files <train file> --train-end 2020-04-01 --add txn_count_1h txn_count_24h`.
new_merchant and distance_km stay in the code as opt-in features but are not part of v1.

## Consequences
- Simpler model with fewer inputs to explain and monitor.
- new_merchant may be revisited; its measured gain is small and not clearly beyond noise.
- All numbers are from one validation period of a synthetic dataset; the sealed test file has not been evaluated.
- amt_vs_avg partly reflects a repeat-victim pattern of this dataset (see docs/experiments.md).
