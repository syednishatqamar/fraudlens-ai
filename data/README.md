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

## Verified facts

Measured by the user in the Kaggle notebook (`eda01_fraudlens`) on the real files. Date of EDA: Nov 2026.

- Files: `fraudTrain.csv`, `fraudTest.csv` (Kaggle path `/kaggle/input/datasets/kartik2112/fraud-detection/`)
- Rows / columns: train 1,296,675 and test 555,719 rows (about 1.85M total), 23 columns each
- Columns: `Unnamed: 0` (row index, ignore), `trans_date_trans_time` (text, needs parsing), `cc_num`, `merchant`,
  `category`, `amt`, `first`, `last`, `gender`, `street`, `city`, `state`, `zip`, `lat`, `long`, `city_pop`, `job`,
  `dob`, `trans_num`, `unix_time`, `merch_lat`, `merch_long`, `is_fraud`
- Identifier / personal columns not used as features: `first`, `last`, `street`, `dob`, `trans_num`, `Unnamed: 0`
- Time span: train 2019-01-01 to 2020-06-21 12:13:37; test 2020-06-21 12:14:25 to 2020-12-31 (already a chronological split)
- Fraud rate: train 7,506 frauds = 0.579%; test 2,145 frauds = 0.386% (the rate drifts between periods)
- Always predicting "not fraud" would be 99.42% accurate on train
- Hour of day: in train, hours 22 to 3 hold 5,373 of 6,343 frauds (84.7%) in the train part used for modelling
- Amount: median fraud 395.30 vs median legit 47.23 in the train part used for modelling

Not yet verified: customer and merchant counts, geography (country coverage, home-location behaviour),
missing values and duplicates. The EDA notebook sections 2, 5, 8, 9 and 11 still cover these.

The dataset is simulated, so patterns here (for example the sharp night-time fraud window) may not match real data.
