import numpy as np
import pandas as pd
import pytest

from src.features.behaviour import add_behaviour_features
from src.models.evaluate import top_k_caught
from src.models.tree import build_tree_model, fit_category_encoder, make_tree_features
from tests.test_data import synthetic


def prepared(n=400, seed=3):
    df = synthetic(n, seed)
    df["trans_date_trans_time"] = pd.to_datetime(df["trans_date_trans_time"])
    return add_behaviour_features(df)


def test_first_transaction_has_no_history():
    out = prepared()
    first = out.groupby("cc_num").head(1)
    assert first["secs_since_prev"].isna().all() and first["amt_vs_avg"].isna().all()


def test_second_transaction_uses_only_the_first():
    out = prepared()
    c = out[out["cc_num"] == out["cc_num"].iloc[0]].reset_index(drop=True)
    assert np.isclose(c.loc[1, "amt_vs_avg"], c.loc[1, "amt"] / c.loc[0, "amt"])
    assert np.isclose(
        c.loc[1, "secs_since_prev"],
        (c.loc[1, "trans_date_trans_time"] - c.loc[0, "trans_date_trans_time"]).total_seconds(),
    )


def test_features_ignore_future_rows():
    """Leakage test: changing a LATER transaction must not change EARLIER rows' features."""
    base = synthetic(400, 3)
    base["trans_date_trans_time"] = pd.to_datetime(base["trans_date_trans_time"])
    a = add_behaviour_features(base)
    changed = base.copy()
    last = changed["trans_date_trans_time"].idxmax()
    changed.loc[last, "amt"] = 1e6
    b = add_behaviour_features(changed)
    keep = a["trans_date_trans_time"] < base["trans_date_trans_time"].max()
    cols = ["secs_since_prev", "amt_vs_avg"]
    pd.testing.assert_frame_equal(a.loc[keep, cols], b.loc[keep, cols])


def test_top_k_caught_perfect_ranking():
    r = top_k_caught([0, 1, 0, 1], [0.1, 0.9, 0.2, 0.8])
    assert r["k"] == 2 and r["caught"] == 2 and r["false_alarms"] == 0


def test_tree_model_runs_on_synthetic_rows():
    out = prepared()
    enc = fit_category_encoder(out)
    X = make_tree_features(out, enc)
    scores = build_tree_model().fit(X, out["is_fraud"]).predict_proba(X)[:, 1]
    assert scores.shape == (len(out),)


def _one_customer(times_sec, amounts=None):
    base = pd.Timestamp("2020-01-01")
    n = len(times_sec)
    return pd.DataFrame({
        "trans_date_trans_time": [base + pd.Timedelta(seconds=t) for t in times_sec],
        "cc_num": [1] * n,
        "merchant": ["m"] * n,
        "category": ["food"] * n,
        "amt": amounts or [10.0] * n,
        "is_fraud": [0] * n,
    })


def test_recent_transaction_counts_hand_checked():
    out = add_behaviour_features(_one_customer([0, 10, 3500, 3600, 7300]))
    assert out["txn_count_1h"].tolist() == [0, 1, 2, 3, 0]
    assert out["txn_count_24h"].tolist() == [0, 1, 2, 3, 4]


def test_recent_counts_are_per_customer():
    a = _one_customer([0, 10, 20])
    b = _one_customer([5, 15]).assign(cc_num=2)
    out = add_behaviour_features(pd.concat([a, b], ignore_index=True))
    assert out.loc[out["cc_num"] == 1, "txn_count_1h"].tolist() == [0, 1, 2]
    assert out.loc[out["cc_num"] == 2, "txn_count_1h"].tolist() == [0, 1]


def test_recent_counts_ignore_future_rows():
    base = synthetic(400, 3)
    base["trans_date_trans_time"] = pd.to_datetime(base["trans_date_trans_time"])
    a = add_behaviour_features(base)
    extra = base.iloc[[0]].copy()
    extra["trans_date_trans_time"] = base["trans_date_trans_time"].max() + pd.Timedelta(minutes=1)
    b = add_behaviour_features(pd.concat([base, extra], ignore_index=True))
    cols = ["txn_count_1h", "txn_count_24h"]
    pd.testing.assert_frame_equal(a[cols], b.iloc[: len(a)][cols].reset_index(drop=True))


def test_velocity_columns_are_opt_in_for_the_tree_model():
    from src.models.tree import FEATURE_COLUMNS, VELOCITY_COLUMNS

    out = prepared()
    enc = fit_category_encoder(out)
    assert list(make_tree_features(out, enc).columns) == FEATURE_COLUMNS
    cols = FEATURE_COLUMNS + VELOCITY_COLUMNS
    assert list(make_tree_features(out, enc, columns=cols).columns) == cols


def test_haversine_known_distances():
    from src.features.geo import haversine_km

    assert haversine_km(0, 0, 0, 0) == pytest.approx(0.0)
    assert haversine_km(0, 0, 0, 1) == pytest.approx(111.19, rel=0.01)          # one degree of longitude at the equator
    assert haversine_km(40.7128, -74.0060, 34.0522, -118.2437) == pytest.approx(3936, rel=0.01)  # New York to Los Angeles


def test_location_features_need_location_columns():
    from src.features.geo import add_location_features

    with pytest.raises(KeyError, match="lat"):
        add_location_features(synthetic(10))


def test_location_features_add_distance_column():
    from src.features.geo import add_location_features

    df = synthetic(10).assign(lat=0.0, long=0.0, merch_lat=0.0, merch_long=1.0)
    out = add_location_features(df)
    assert out["distance_km"].between(111, 112).all() and len(out) == 10


def test_new_merchant_flag_hand_checked_and_per_customer():
    df = _one_customer([0, 10, 20, 30])
    df["merchant"] = ["a", "b", "a", "c"]
    other = _one_customer([5]).assign(cc_num=2, merchant="a")
    out = add_behaviour_features(pd.concat([df, other], ignore_index=True))
    assert out.loc[out["cc_num"] == 1, "new_merchant"].tolist() == [1, 1, 0, 1]
    assert out.loc[out["cc_num"] == 2, "new_merchant"].tolist() == [1]


def test_new_merchant_ignores_future_rows():
    base = synthetic(400, 3)
    base["trans_date_trans_time"] = pd.to_datetime(base["trans_date_trans_time"])
    a = add_behaviour_features(base)
    extra = base.iloc[[0]].copy()
    extra["trans_date_trans_time"] = base["trans_date_trans_time"].max() + pd.Timedelta(minutes=1)
    b = add_behaviour_features(pd.concat([base, extra], ignore_index=True))
    assert a["new_merchant"].tolist() == b["new_merchant"].iloc[: len(a)].tolist()


def test_all_optional_columns_reach_the_tree_matrix_when_present():
    from src.features.geo import add_location_features
    from src.models.tree import FEATURE_COLUMNS, OPTIONAL_COLUMNS

    df = synthetic(300, 3).assign(lat=1.0, long=2.0, merch_lat=1.5, merch_long=2.5)
    df["trans_date_trans_time"] = pd.to_datetime(df["trans_date_trans_time"])
    out = add_location_features(add_behaviour_features(df))
    enc = fit_category_encoder(out)
    X = make_tree_features(out, enc, columns=FEATURE_COLUMNS + OPTIONAL_COLUMNS)
    assert list(X.columns) == FEATURE_COLUMNS + OPTIONAL_COLUMNS and not X["new_merchant"].isna().any()
