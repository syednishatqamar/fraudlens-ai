"""Shared pipeline for the case scripts: train, calibrate, band, explain, build case files for a sample of REVIEW/BLOCK rows."""
from __future__ import annotations

from typing import Iterator

import numpy as np

from src.data.loader import Schema, load_transactions
from src.explainability.reasons import explain_case
from src.explainability.shap_explainer import ShapExplainer
from src.features.behaviour import add_behaviour_features
from src.features.geo import add_location_features
from src.investigation.case import CaseFile, build_case
from src.models.calibration import make_calibrator
from src.models.decision import BLOCK, REVIEW, DecisionCosts, decide
from src.models.tree import V1_COLUMNS, build_tree_model, fit_category_encoder, make_tree_features


def iter_cases(
    files,
    train_end: str,
    calib_end: str,
    n_per_band: int = 2,
    label_delay_days: int = 7,
    seed: int = 1,
    costs: DecisionCosts = DecisionCosts(),
) -> Iterator[tuple[CaseFile, bool]]:
    """Yield (case, actual_label_is_fraud) for random BLOCK and REVIEW transactions in the evaluation slice.

    The label is returned for the user's learning only; it is never given to the case builder for the investigated row.
    """
    s = Schema()
    df = add_behaviour_features(load_transactions(list(files), s), s)
    if all(c in df.columns for c in s.geo()):
        df = add_location_features(df, s)
    t = df[s.time]
    train, calib, ev = df[t < train_end], df[(t >= train_end) & (t < calib_end)], df[t >= calib_end]

    enc = fit_category_encoder(train, s)
    X = {k: make_tree_features(v, enc, s, V1_COLUMNS) for k, v in (("train", train), ("calib", calib), ("eval", ev))}
    model = build_tree_model().fit(X["train"], train[s.label])
    cal = make_calibrator("platt").fit(model.decision_function(X["calib"]), calib[s.label])
    prob = cal.predict_proba(model.decision_function(X["eval"]))
    bands = decide(prob, ev[s.amount].to_numpy(), costs)
    print(f"evaluation slice: {len(ev):,} transactions; REVIEW {np.sum(bands == REVIEW):,}, BLOCK {np.sum(bands == BLOCK):,}\n")

    explainer = ShapExplainer(model, X["train"], scale="logodds")
    names = list(enc.categories_[0])
    rng = np.random.default_rng(seed)
    for band in (BLOCK, REVIEW):
        pool = np.flatnonzero(bands == band)
        for pos in rng.choice(pool, size=min(n_per_band, len(pool)), replace=False):
            label_idx = ev.index[pos]
            pushes, base = explainer.explain(X["eval"].loc[[label_idx]])
            e = explain_case(X["eval"].loc[label_idx], pushes.iloc[0], float(base[0]), names, scale="logodds")
            case = build_case(df, df.index.get_loc(label_idx), float(prob[pos]), band, e, label_delay_days, s, costs)
            yield case, bool(ev.loc[label_idx, s.label] == 1)
