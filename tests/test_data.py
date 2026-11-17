import numpy as np
import pandas as pd
import pytest

from src.data.loader import Schema, SchemaError, load_transactions, validate_schema
from src.data.split import chronological_split


def synthetic(n=200, seed=0):
    """Synthetic rows that only mimic the expected column names. Not real data."""
    rng = np.random.default_rng(seed)
    return pd.DataFrame(
        {
            "trans_date_trans_time": pd.date_range("2020-01-01", periods=n, freq="6h")[rng.permutation(n)],
            "cc_num": rng.integers(1, 10, n),
            "merchant": rng.choice(["a", "b", "c"], n),
            "category": rng.choice(["food", "gas"], n),
            "amt": rng.uniform(1, 100, n),
            "is_fraud": rng.integers(0, 2, n),
        }
    )


def test_validate_schema_reports_missing_columns():
    with pytest.raises(SchemaError, match="amt"):
        validate_schema(synthetic().drop(columns=["amt"]))


def test_validate_schema_rejects_non_binary_label():
    df = synthetic()
    df.loc[0, "is_fraud"] = 2
    with pytest.raises(SchemaError):
        validate_schema(df)


def test_loader_sorts_chronologically(tmp_path):
    p = tmp_path / "x.csv"
    synthetic().to_csv(p, index=False)
    df = load_transactions([p])
    assert df[Schema().time].is_monotonic_increasing


def test_split_has_no_time_overlap_and_loses_no_rows():
    df = synthetic().sort_values("trans_date_trans_time")
    train, val, test = chronological_split(df, "trans_date_trans_time", "2020-01-20", "2020-02-10")
    assert len(train) + len(val) + len(test) == len(df)
    assert train["trans_date_trans_time"].max() < val["trans_date_trans_time"].min()
    assert val["trans_date_trans_time"].max() < test["trans_date_trans_time"].min()


def test_split_rejects_reversed_dates():
    with pytest.raises(ValueError):
        chronological_split(synthetic(), "trans_date_trans_time", "2020-02-01", "2020-01-01")
