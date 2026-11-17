import numpy as np

from src.models.baseline import build_baseline, evaluate, make_raw_features, recall_at_precision
from tests.test_data import synthetic


def test_recall_at_precision_perfect_scores():
    y = np.array([0, 0, 1, 1])
    assert recall_at_precision(y, np.array([0.1, 0.2, 0.8, 0.9]), 0.9) == 1.0


def test_recall_at_precision_unreachable_returns_zero():
    y = np.array([0, 0, 0, 1])
    assert recall_at_precision(y, np.array([0.9, 0.8, 0.7, 0.1]), 0.99) == 0.0


def test_baseline_pipeline_runs_end_to_end_on_synthetic_rows():
    df = synthetic(300)
    X = make_raw_features(df)
    model = build_baseline().fit(X, df["is_fraud"])
    scores = model.predict_proba(X)[:, 1]
    assert scores.shape == (300,) and set(evaluate(df["is_fraud"].to_numpy(), scores))
