# Milestone 1 guide: data, split, baseline

Do these in order. For each, write your prediction in `LEARNING_LOG.md` *before* running.

1. **Download** the dataset locally into `data/raw/` (see `data/README.md`). Confirm file names.
2. **EDA:** run `notebooks/01_eda.ipynb` top to bottom. Fix the `EXPECTED` column map if needed, fill the
   observations, and copy verified facts into `data/README.md`. If real column names differ, also edit
   `Schema` in `src/data/loader.py`.
3. **Choose split dates** from the EDA (monthly fraud counts, time span). Record the reasoning as
   `docs/decisions/0001-time-split.md`.
4. **Run the baseline:**
   ```
   python scripts/train_baseline.py --files data/raw/<file>.csv [more files] \
       --train-end <date> --val-end <date> --sample 150000
   ```
   The script prints per-split fraud counts first. Check there are enough frauds in val and test
   before trusting any metric.
5. **Break it on purpose:** replace the chronological split with a random one and compare metrics.
   Write down the gap and why it appears.
6. **Metric sanity:** compare PR-AUC against `positive_rate` (what a random model scores) and compute
   the accuracy of "always predict legit". Explain why accuracy is useless here.

Tests: `python -m pytest -q` (uses synthetic rows only; no dataset needed).
