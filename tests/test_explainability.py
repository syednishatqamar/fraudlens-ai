import numpy as np
import pandas as pd
import pytest

from src.explainability.reasons import describe_feature, explain_case, humanize_seconds
from src.features.behaviour import add_behaviour_features
from src.models.tree import build_tree_model, fit_category_encoder, make_tree_features
from tests.test_data import synthetic

CATS = ["food", "gas"]


def test_humanize_seconds_units():
    assert humanize_seconds(45) == "45 seconds"
    assert humanize_seconds(600) == "10 minutes"
    assert humanize_seconds(3 * 3600) == "3.0 hours"
    assert humanize_seconds(5 * 86400) == "5.0 days"


def test_missing_history_is_described_not_crashing():
    assert "No earlier transaction" in describe_feature("secs_since_prev", np.nan)
    assert "No earlier spending" in describe_feature("amt_vs_avg", np.nan)


def test_unknown_category_code_is_handled():
    assert "not one seen" in describe_feature("category_code", -1, CATS)
    assert "'gas'" in describe_feature("category_code", 1, CATS)


def test_explain_case_orders_reasons_and_scores_add_up():
    values = pd.Series({"amount": 900.0, "hour": 23, "category_code": 0, "secs_since_prev": 30.0, "amt_vs_avg": 9.0})
    pushes = pd.Series({"amount": 0.50, "hour": 0.20, "category_code": -0.05, "secs_since_prev": 0.004, "amt_vs_avg": 0.10})
    e = explain_case(values, pushes, base_value=0.02, category_names=CATS, top_n=2)
    assert e.score == pytest.approx(0.02 + pushes.sum())
    assert [r.feature for r in e.risk_reasons] == ["amount", "hour"]          # largest first, top_n=2
    assert [r.feature for r in e.protective_reasons] == ["category_code"]
    assert "secs_since_prev" not in [r.feature for r in e.risk_reasons]        # below noise floor
    assert "900.00" in e.summary and "+0.50" in e.summary


def test_explain_case_with_no_risk_factors():
    values = pd.Series({"amount": 5.0, "hour": 12, "category_code": 0, "secs_since_prev": 9e5, "amt_vs_avg": 1.0})
    pushes = pd.Series({"amount": -0.1, "hour": 0.0, "category_code": 0.0, "secs_since_prev": 0.0, "amt_vs_avg": 0.0})
    e = explain_case(values, pushes, 0.02)
    assert e.risk_reasons == [] and "No single factor" in e.summary


def test_shap_pushes_add_up_to_model_score():
    pytest.importorskip("shap")
    from src.explainability.shap_explainer import ShapExplainer

    df = synthetic(600, seed=4)
    df["trans_date_trans_time"] = pd.to_datetime(df["trans_date_trans_time"])
    df = add_behaviour_features(df)
    enc = fit_category_encoder(df)
    X = make_tree_features(df, enc)
    model = build_tree_model().fit(X, df["is_fraud"])
    ex = ShapExplainer(model, X)
    rows = X.head(5)
    pushes, base = ex.explain(rows)
    np.testing.assert_allclose(base + pushes.sum(axis=1).to_numpy(), model.predict_proba(rows)[:, 1], atol=1e-6)
