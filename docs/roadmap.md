# Roadmap

Time estimates are deliberately omitted; pace depends on learning depth. Each milestone has an
exit criterion and a learning goal. Tick boxes only when the criterion is demonstrably met.

## M0: Scaffold
- [x] Repo layout, README, learning log, EDA notebook skeleton

## M1: Data and baseline
- Do: download dataset, run EDA, verify schema, stratified dev sample, chronological split, logistic-regression baseline.
- Exit: documented dataset facts in `data/README.md`; a baseline metric with its reproducing command; split-leakage test.
- Learn: imbalance, why accuracy misleads, PR-AUC, time-based validation.

## M2: Feature engineering
- Candidate features: transactions in last 5 min / 1 h, amount vs. user average, merchant frequency, distance from previous transaction, new merchant / new state flags, night flag, velocity score.
- Exit: feature pipeline with tests proving no feature uses future data.
- Learn: point-in-time correctness, rolling windows, train/serve skew.

## M3: Models
- Random Forest, XGBoost or LightGBM, Isolation Forest or autoencoder. Track every run in MLflow.
- Exit: comparison table against the baseline, produced by a script.
- Learn: gradient boosting, imbalance handling (class weights, resampling), anomaly detection vs. supervised.

## M4: Evaluation and thresholds
- PR curves, recall at fixed precision, cost-based threshold, calibration.
- Exit: a justified operating threshold written up as an ADR.

## M5: Explainability
- SHAP values per flagged transaction; mapping to readable reason codes.
- Exit: explanation for any flagged case; sanity checks that reasons match feature values.
- Learn: what SHAP does and does not tell you; pitfalls with correlated features.

## M6: Investigation agent
- LangGraph graph with tools: transaction history, velocity analysis, location patterns, similar past cases, SHAP reader.
- Exit: report + CLEAR/REVIEW/BLOCK on sample cases; tool functions unit-tested; LLM calls mockable.
- Learn: agent state machines, tool design, grounding output in retrieved facts.

## M7: API and persistence
- FastAPI endpoints for scoring, cases, decisions; PostgreSQL schema; optional Redis.
- Exit: API tests passing; analyst decisions persisted.

## M8: Analyst dashboard
- React + Recharts: case queue, explanation panel, recommendation, Approve/Reject/Escalate.
- Exit: end-to-end flow from a replayed transaction to a stored decision.

## M9: Productionise
- Docker Compose for all services, GitHub Actions (lint, tests, image build), monitoring, n8n notifications, optional free-tier deployment.
- Exit: one-command local startup; CI green.

## Stretch: graph fraud rings
- Build an entity graph (customer / card / device / IP / merchant) and detect shared-infrastructure clusters.
- Only after M1–M9. The Sparkov data has no device/IP fields, so this likely needs PaySim or synthetic augmentation (decide via ADR).
