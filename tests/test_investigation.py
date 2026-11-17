import numpy as np
import pandas as pd
import pytest

from src.explainability.reasons import CaseExplanation
from src.features.behaviour import add_behaviour_features
from src.investigation import tools
from src.investigation.case import build_case, render_case_text
from src.models.decision import BLOCK, CLEAR, REVIEW

CARD = 1234567890123456
BASE = pd.Timestamp("2020-03-01 12:00")


def make_history(rows):
    """rows: list of (days_offset, merchant, category, amount, is_fraud, card). Returns sorted table with features."""
    df = pd.DataFrame(
        {
            "trans_date_trans_time": [BASE + pd.Timedelta(days=d) for d, *_ in rows],
            "cc_num": [r[5] for r in rows],
            "merchant": [r[1] for r in rows],
            "category": [r[2] for r in rows],
            "amt": [r[3] for r in rows],
            "is_fraud": [r[4] for r in rows],
        }
    )
    return add_behaviour_features(df)


def explanation():
    return CaseExplanation(score=0.9, base_value=0.02, risk_reasons=[], protective_reasons=[], summary="Model score 0.90 (average 0.02).")


def card_rows():
    # six earlier transactions on the card, then the investigated one (offset 10)
    return [
        (0, "A", "food", 20.0, 0, CARD),
        (1, "B", "gas", 40.0, 0, CARD),
        (2, "A", "food", 30.0, 0, CARD),
        (3, "C", "food", 50.0, 0, CARD),
        (4, "A", "gas", 60.0, 0, CARD),
        (5, "B", "food", 70.0, 0, CARD),
        (10, "A", "food", 500.0, 1, CARD),
    ]


def test_mask_card_shows_only_last_four():
    assert tools.mask_card(CARD) == "3456"


def test_recent_transactions_most_recent_first_and_limited():
    h = make_history(card_rows())
    past = tools.card_prior(h, 6)
    rec = tools.recent_transactions(past, n=3)
    assert [r["amount"] for r in rec] == [70.0, 60.0, 50.0]


def test_profile_and_merchant_history_hand_checked():
    h = make_history(card_rows())
    past = tools.card_prior(h, 6)
    prof = tools.spending_profile(past)
    assert prof["n_prior"] == 6 and prof["median_amount"] == pytest.approx(45.0) and prof["max_amount"] == 70.0
    assert tools.merchant_history(past, "A") == {"prior_visits": 3, "new_merchant": False}
    assert tools.merchant_history(past, "Z") == {"prior_visits": 0, "new_merchant": True}


def test_only_earlier_rows_are_used():
    rows = card_rows()
    h1 = make_history(rows)
    h2 = make_history(rows + [(11, "A", "food", 99999.0, 0, CARD)])   # a LATER transaction on the same card
    c1 = build_case(h1, 6, 0.9, REVIEW, explanation())
    c2 = build_case(h2, 6, 0.9, REVIEW, explanation())
    assert c1.profile == c2.profile and c1.recent_transactions == c2.recent_transactions
    assert c1.velocity == c2.velocity and c1.similar == c2.similar


def test_other_cards_do_not_leak_into_the_case():
    rows = card_rows() + [(8, "A", "food", 500.0, 1, 999)]
    h = make_history(rows)
    idx = int(h.index[(h["cc_num"] == CARD) & (h["amt"] == 500.0)][0])
    c = build_case(h, idx, 0.9, REVIEW, explanation())
    assert c.profile["n_prior"] == 6


def test_similar_cases_respect_label_delay():
    # comparable fraud 3 days before (labels not yet known with a 7-day delay) and one 10 days before
    rows = [
        (0, "A", "food", 480.0, 1, 7), (7, "A", "food", 510.0, 1, 7), (8, "A", "food", 450.0, 1, 7),   # fraud rows on other cards
        (10, "A", "food", 500.0, 0, CARD),
    ]
    h = make_history(rows)
    idx = len(h) - 1
    s7 = tools.similar_cases(h, idx, label_delay_days=7)
    assert s7["n_similar"] == 1 and s7["n_fraud"] == 1          # only the day-0 row is at least 7 days older (10 - 0 = 10; 10 - 7 = 3 too new)
    s0 = tools.similar_cases(h, idx, label_delay_days=0)
    assert s0["n_similar"] == 3 and s0["n_fraud"] == 3


def test_similar_cases_filters_category_amount_and_night():
    rows = [
        (0, "A", "food", 500.0, 1, 7),      # comparable
        (1, "A", "gas", 500.0, 1, 7),       # wrong category
        (2, "A", "food", 5000.0, 1, 7),     # amount far outside the band
        (10, "A", "food", 500.0, 0, CARD),  # investigated (12:00, daytime)
    ]
    h = make_history(rows)
    s = tools.similar_cases(h, len(h) - 1, label_delay_days=1)
    assert s["n_similar"] == 1
    night_rows = [(0, "A", "food", 500.0, 1, 7), (10, "A", "food", 500.0, 0, CARD)]
    hn = make_history(night_rows)
    hn["trans_date_trans_time"] = hn["trans_date_trans_time"].where(hn.index != 0, hn["trans_date_trans_time"] + pd.Timedelta(hours=11))  # 23:00 vs 12:00
    assert tools.similar_cases(hn, 1, label_delay_days=1)["n_similar"] == 0


def test_notes_flag_disagreement_between_band_and_similar_cases():
    legit = [(i * 0.1, "A", "food", 500.0, 0, 7) for i in range(30)]   # 30 comparable legit transactions
    h = make_history(legit + [(10, "A", "food", 500.0, 0, CARD)])
    c = build_case(h, len(h) - 1, 0.95, BLOCK, explanation())
    assert any("CHECK" in n and "BLOCK" in n for n in c.notes)
    fraud = [(i * 0.1, "A", "food", 500.0, 1, 7) for i in range(30)]
    h2 = make_history(fraud + [(10, "A", "food", 500.0, 0, CARD)])
    c2 = build_case(h2, len(h2) - 1, 0.001, CLEAR, explanation())
    assert any("CHECK" in n and "CLEAR" in n for n in c2.notes)


def test_notes_say_too_few_cases_and_no_history():
    h = make_history([(0, "A", "food", 500.0, 0, CARD)])
    c = build_case(h, 0, 0.5, REVIEW, explanation())
    assert any("too few" in n for n in c.notes) and any("No earlier transactions" in n for n in c.notes)


def test_rendered_case_masks_the_card_number():
    h = make_history(card_rows())
    text = render_case_text(build_case(h, 6, 0.93, BLOCK, explanation()))
    assert "3456" in text and str(CARD) not in text
    assert "BLOCK" in text and "93.0%" in text and "Most recent earlier transactions" in text


def test_rendered_case_shows_one_probability_and_the_reasons():
    from src.explainability.reasons import Reason

    e = CaseExplanation(0.9, 0.02, [Reason("amount", "Transaction amount is 500.00", 2.5, "raises risk")], [], "Model score 0.90 (average 0.02).")
    h = make_history(card_rows())
    text = render_case_text(build_case(h, 6, 0.42, REVIEW, e))
    assert "42.0%" in text and "Transaction amount is 500.00 (+2.50)" in text
    assert "Model score" not in text        # the raw, uncalibrated score must not appear next to the calibrated probability
