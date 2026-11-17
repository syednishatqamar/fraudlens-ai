import numpy as np
import pytest

from src.models.calibration import IsotonicCalibrator, PlattCalibrator, calibration_report, make_calibrator
from src.models.decision import BLOCK, CLEAR, REVIEW, DecisionCosts, decide, realized_cost


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def synthetic_scores(n=200_000, seed=0):
    rng = np.random.default_rng(seed)
    m = rng.normal(0, 3, n)                       # raw margins from some model
    true_p = sigmoid(1.5 * m - 5.0)               # the real fraud probability is a different sigmoid
    return m, true_p, (rng.random(n) < true_p).astype(int)


@pytest.mark.parametrize("cal", [PlattCalibrator(), IsotonicCalibrator()])
def test_calibrators_recover_true_probabilities(cal):
    m, true_p, y = synthetic_scores()
    cal.fit(m, y)
    p = cal.predict_proba(m)
    assert p.min() >= 0 and p.max() <= 1
    assert abs(p.mean() - y.mean()) < 0.005
    order = np.argsort(m)
    assert np.all(np.diff(p[order]) >= -1e-12)    # monotone: higher margin never gives lower probability
    top = m > np.quantile(m, 0.9)
    assert abs(p[top].mean() - true_p[top].mean()) < 0.03


def test_make_calibrator_rejects_unknown_method():
    with pytest.raises(ValueError):
        make_calibrator("nope")


def test_calibration_report_columns_and_perfect_case():
    _, true_p, y = synthetic_scores()
    table, summ = calibration_report(y, true_p)
    assert list(table.columns) == ["band", "n", "frauds", "mean_pred", "observed_rate"]
    big = table[table["n"] > 1000]
    assert np.allclose(big["mean_pred"], big["observed_rate"], atol=0.03)
    assert summ["brier"] >= 0 and summ["log_loss"] >= 0


def test_decide_hand_checked_cases():
    c = DecisionCosts(review_cost=5.0, block_false_positive_cost=25.0)
    d = decide([0.001, 0.5, 0.95, 0.5], [100, 100, 100, 2], c)
    assert d.tolist() == [CLEAR, REVIEW, BLOCK, CLEAR]   # last: tiny amount, expected loss 1 < review cost 5


def test_decisions_never_step_down_as_probability_rises():
    order = {CLEAR: 0, REVIEW: 1, BLOCK: 2}
    p = np.linspace(0, 1, 501)
    levels = [order[x] for x in decide(p, np.full_like(p, 200.0))]
    assert levels == sorted(levels) and levels[0] == 0 and levels[-1] == 2


def test_realized_cost_hand_checked():
    c = DecisionCosts(review_cost=5.0, block_false_positive_cost=25.0)
    dec = np.array([CLEAR, CLEAR, REVIEW, BLOCK, BLOCK])
    y = np.array([1, 0, 1, 1, 0])
    amt = np.array([100.0, 100.0, 300.0, 400.0, 50.0])
    assert realized_cost(dec, y, amt, c) == pytest.approx(100 + 0 + 5 + 0 + 25)
