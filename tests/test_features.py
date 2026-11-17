import numpy as np
import pandas as pd

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
