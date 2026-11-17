# Experiment log (validation set only)

Run by the project owner in the Kaggle notebook `eda01_fraudlens`; the test file has not been evaluated.
Single runs, one random seed, one validation period. The data is synthetic, so these scores must not be
presented as real-world performance. The scripted, repeatable versions are `scripts/train_baseline.py`
and `scripts/train_tree.py`; they have so far been checked only on synthetic rows, so reproducing these
exact numbers from the repo is still to be done.

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
