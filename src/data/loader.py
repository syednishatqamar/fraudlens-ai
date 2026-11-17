"""Load raw transaction CSVs and check them against the schema we expect.

The column names below are *expected* (from planning notes), not verified. Verify them in
notebooks/01_eda.ipynb and edit `Schema` if the real files differ.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pandas as pd


@dataclass(frozen=True)
class Schema:
    label: str = "is_fraud"
    time: str = "trans_date_trans_time"
    customer: str = "cc_num"
    merchant: str = "merchant"
    category: str = "category"
    amount: str = "amt"

    def required(self) -> list[str]:
        return [self.label, self.time, self.customer, self.merchant, self.category, self.amount]


class SchemaError(ValueError):
    """Raised when the loaded data does not match the expected schema."""


def validate_schema(df: pd.DataFrame, schema: Schema = Schema()) -> None:
    missing = [c for c in schema.required() if c not in df.columns]
    if missing:
        raise SchemaError(f"Missing expected columns: {missing}. Found: {list(df.columns)}")
    if not set(df[schema.label].dropna().unique()) <= {0, 1}:
        raise SchemaError(f"Label column {schema.label!r} must contain only 0/1.")


def load_transactions(
    paths: list[Path | str],
    schema: Schema = Schema(),
    nrows: int | None = None,
) -> pd.DataFrame:
    """Read one or more CSVs, parse the timestamp, validate, and sort chronologically."""
    if not paths:
        raise FileNotFoundError("No input files given. See data/README.md for the download step.")
    frames = []
    for p in paths:
        f = pd.read_csv(p, nrows=nrows)
        f["source_file"] = Path(p).name
        frames.append(f)
    df = pd.concat(frames, ignore_index=True)
    validate_schema(df, schema)
    df[schema.time] = pd.to_datetime(df[schema.time])
    return df.sort_values(schema.time, kind="stable").reset_index(drop=True)
