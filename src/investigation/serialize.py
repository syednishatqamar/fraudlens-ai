"""Convert a CaseFile into plain JSON-safe data for storage and the API."""
from __future__ import annotations

from dataclasses import asdict, is_dataclass

import pandas as pd

from src.investigation.case import CaseFile


def _clean(value):
    if isinstance(value, pd.Timestamp):
        return str(value)
    if is_dataclass(value):
        return _clean(asdict(value))
    if isinstance(value, dict):
        return {k: _clean(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_clean(v) for v in value]
    if isinstance(value, float) and value != value:  # NaN is not valid JSON
        return None
    return value


def case_to_dict(case: CaseFile) -> dict:
    return _clean(case)
