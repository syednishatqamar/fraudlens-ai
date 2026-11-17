# 0001. Chronological split with a validation slice from the end of train

Date: 2026-11
Status: proposed (draft for the project owner to review and reword)

## Context
Fraud data is a time series. A random split would let the model see transactions that happen after some
of its evaluation rows. The dataset ships as two files that are already ordered in time: train covers
2019-01-01 to 2020-06-21 and test covers 2020-06-21 to 2020-12-31.

## Options considered
1. Random split of all rows: leaks future information, rejected.
2. Use the test file for model selection: test would no longer be an honest final exam, rejected.
3. Cut a validation slice from the end of train by date: chosen.

## Decision
- Train: before 2020-04-01 (1,097,693 rows, 6,343 frauds)
- Validation: 2020-04-01 to 2020-06-21 (198,982 rows, 1,163 frauds)
- Test: the provided test file, opened once at the end of the project (555,719 rows, 2,145 frauds)

Cut date chosen from monthly fraud counts (roughly 300 to 600 frauds a month), so every part has
more than 1,000 frauds.

## Consequences
- Validation fraud rate (0.584%) is close to train (0.578%), but test is lower (0.386%), so a drop from
  validation to test is expected and does not by itself mean a bug.
- Every look at validation makes it slightly less independent; model choices should be few and recorded.
