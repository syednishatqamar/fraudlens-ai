"""Location features. Row-level, so they cannot leak across rows.

The dataset's lat/long are expected to be the customer's fixed home location and merch_lat/merch_long the
merchant's location; the EDA should confirm the home location is constant before relying on this.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from src.data.loader import Schema

EARTH_RADIUS_KM = 6371.0


def haversine_km(lat1, lon1, lat2, lon2) -> np.ndarray:
    lat1, lon1, lat2, lon2 = (np.radians(np.asarray(x, dtype="float64")) for x in (lat1, lon1, lat2, lon2))
    a = np.sin((lat2 - lat1) / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin((lon2 - lon1) / 2) ** 2
    return 2 * EARTH_RADIUS_KM * np.arcsin(np.sqrt(a))


def add_location_features(df: pd.DataFrame, schema: Schema = Schema()) -> pd.DataFrame:
    """Return a copy with distance_km: great-circle distance between customer and merchant locations."""
    missing = [c for c in schema.geo() if c not in df.columns]
    if missing:
        raise KeyError(f"Missing location columns: {missing}")
    out = df.copy()
    out["distance_km"] = haversine_km(
        out[schema.cust_lat], out[schema.cust_long], out[schema.merch_lat], out[schema.merch_long]
    )
    return out
