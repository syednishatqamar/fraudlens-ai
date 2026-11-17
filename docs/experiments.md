# Experiment log (validation set only)

Run by the project owner in the Kaggle notebook `eda01_fraudlens`; the test file has not been evaluated.
Single runs, one random seed, one validation period. The data is synthetic, so these scores must not be
presented as real-world performance. The scripted version of #4 is `scripts/train_tree.py`.

Reproduction check (Kaggle, Nov 2026): `python scripts/train_tree.py --files .../fraudTrain.csv --train-end 2020-04-01`
printed train 1,097,693 rows / 6,343 frauds, validation 198,982 rows / 1,163 frauds, validation PR-AUC 0.8779 and
931 of 1,163 frauds caught (232 false alarms), matching the notebook result for #4. `scripts/train_baseline.py`
has not yet been reproduced against #1 and #2 on real data.

Metric: PR-AUC (a random model scores about the fraud rate, 0.0058 on validation).

| # | Model | Inputs | Validation PR-AUC |
|---|---|---|---|
| 1 | Logistic regression | log amount, hour as a number, category | 0.1429 |
| 2 | Logistic regression | log amount, night flag (hours 22 to 3), category | 0.2421 |
| 3 | Gradient-boosted trees | amount, hour, category | 0.8288 |
| 4 | Gradient-boosted trees | #3 + secs_since_prev + amt_vs_avg | 0.8779 |

Top-k check (flag as many transactions as there are frauds, 1,163):

| Model | Frauds caught | Night frauds caught | Daytime frauds caught |
|---|---|---|---|
| #3 | 881 | 815 of 989 (82.4%) | 66 of 174 (37.9%) |
| #4 | 931 | 84.7% | 53.4% |

Observations
- Missed frauds under #3 were mostly small amounts (median about 51, close to the median normal purchase).
- Behaviour features helped most on daytime frauds. Remaining uncertainty is not quantified yet.

## Diagnostic: what `amt_vs_avg` partly measures (train part, amounts 250 to 1,100)

Within large purchases the fraud rate is not monotonic in `amt_vs_avg`: ratio up to 2x is 93% fraud (356 of 382),
2x to 4x is 10.5%, 4x to 8x is 7.6%, 8x to 16x is 27.8%, above 16x is 42.3%. Among the 382 purchases with ratio up to 2x,
95.5% belong to customers who already had a fraud earlier (41.7% for all other large purchases).
Reading: a very low ratio mostly marks a customer whose earlier average was inflated by earlier large frauds, a
repeat-victim pattern that probably comes from how this dataset simulates fraud (streaks, no card blocking). Uses earlier
amounts only, so it is not a label leak, but part of the 0.8288 to 0.8779 gain may rest on this artifact.
This explains why SHAP shows a negative push for high ratios: pushes are relative to a reference of typical rows.
Label-based features such as "earlier frauds on this card" would be unavailable live because fraud labels arrive late.
Next: ablation with `scripts/train_tree.py --drop amt_vs_avg` and `--drop secs_since_prev`.
