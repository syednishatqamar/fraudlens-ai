# data/

Datasets are **not** committed to git (see `.gitignore`).

| Folder | Purpose |
|---|---|
| `raw/` | Original downloads, never modified |
| `processed/` | Derived files (samples, splits, feature tables) |

## Planned primary dataset

Kaggle "Credit Card Transactions Fraud Detection Dataset" (Sparkov-simulated),
commonly published as `kartik2112/fraud-detection`.

**Status: not yet downloaded or verified.** Column names, row counts, fraud rate and
the train/test file layout in the planning notes came from memory. Confirm them
against the actual files in `notebooks/01_eda.ipynb` and record the verified facts
here, with the date, once known.

## Download

Requires a Kaggle account and API token (`~/.kaggle/kaggle.json`):

```
kaggle datasets download -d kartik2112/fraud-detection -p data/raw --unzip
```

Check the dataset's licence/terms on its Kaggle page before redistributing anything.

## Verified facts (fill in after EDA)

- Files:
- Rows / columns:
- Time span:
- Fraud rate:
