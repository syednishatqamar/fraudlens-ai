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

## Ablation of behaviour features (validation, scripts/train_tree.py, seed 42, Kaggle, Nov 2026)

| Features | Validation PR-AUC | Frauds caught in top 1,163 | False alarms |
|---|---|---|---|
| amount, hour, category | 0.8240 | 891 | 272 |
| + secs_since_prev | 0.8685 | 920 | 243 |
| + amt_vs_avg | 0.8518 | 897 | 266 |
| + both | 0.8779 | 931 | 232 |

Reading
- Each feature helps; secs_since_prev adds more (+0.0445 PR-AUC) than amt_vs_avg (+0.0278). Together +0.0539, less than the
  sum of the two, so they overlap.
- amt_vs_avg alone adds only 6 caught frauds in the top-k, so the repeat-victim concern above affects the smaller share of the gain.
- SHAP ranked secs_since_prev as the least important feature on average, yet removing it hurts more than removing amt_vs_avg:
  average SHAP size is not the same as usefulness for the score.
- The no-behaviour run scores 0.8240 here versus 0.8288 in the earlier notebook run. The likely cause is row order: this script sorts
  by time, which changes the random 10% internal split HistGradientBoosting uses for early stopping. So an incidental choice moves PR-AUC by
  about 0.005; treat differences smaller than that as noise.
- Single seed, one validation period, synthetic data.

## Recent-transaction counts (validation, scripts/train_tree.py --add, seed 42, Kaggle, Nov 2026)

Baseline here is the five-feature model (0.8779, 931 caught, 232 false alarms).

| Added features | Validation PR-AUC | Caught (of 1,163) | False alarms |
|---|---|---|---|
| txn_count_1h | 0.8824 | 931 | 232 |
| txn_count_24h | 0.8786 | 934 | 229 |
| both | 0.8906 | 941 | 222 |

Reading: gain is small (+0.0127 for both, +10 frauds caught). Each count alone is within the roughly 0.005 noise level seen earlier;
the pair together is about 2.5x that, but this is one seed and one validation period. Diminishing returns versus the first two behaviour
features (+0.054). Measure noise properly with `--seeds`.

## Seed-to-seed noise check (validation, scripts/train_tree.py --seeds 1 2 3 4 5, Kaggle, Nov 2026)

| Features | PR-AUC per seed | Mean | Std | Frauds caught per seed (of 1,163) |
|---|---|---|---|---|
| five features | 0.8810, 0.8797, 0.8816, 0.8772, 0.8845 | 0.8808 | 0.0027 | 935, 934, 932, 933, 936 |
| + txn_count_1h, txn_count_24h | 0.8911, 0.8924, 0.8906, 0.8835, 0.8910 | 0.8897 | 0.0035 | 941, 939, 942, 929, 942 |

Reading
- Mean difference is +0.0089, about 4 standard errors of the difference (std/sqrt(5) per group), so the gain is unlikely to be seed noise.
- The seed only changes HistGradientBoosting's internal early-stopping split. It does not measure variation from using a
  different time period or a different draw of the 1,163 validation frauds, which is probably larger. Not quantified.
- Practical gain is small: about 5 more frauds caught in the top-k on average (934 to 939).
- Use the five-feature mean 0.8808 (std 0.0027) as the baseline reference, not the single-seed 0.8779.
