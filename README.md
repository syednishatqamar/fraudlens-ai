# FraudLens

**An explainable fraud investigation and risk decision engine.**
A miniature AI fraud analyst: a trained model *detects*, an agent *investigates*, a human *decides*.

> **Status: Milestone 0 (scaffolding).** No model, agent, API or UI exists yet.
> Everything below describes the *plan*. Nothing in this README reports results; metrics
> will be added only after they have been measured and are reproducible.

## Why this project

Most fraud demos stop at "train XGBoost, show SHAP". FraudLens aims at the full decision-support loop that real financial systems need:

- Behavioural features that model a customer's history, not isolated rows.
- Honest evaluation: chronological splits, imbalance-aware metrics, comparison against baselines.
- Explanations that are turned into analyst-readable reasons.
- An investigation agent that gathers context; it does **not** make the fraud prediction.
- Human-in-the-loop: the analyst's final decision is stored and auditable.
- Production practices: tests, containers, experiment tracking, CI, monitoring.

## Architecture (planned)

```
Transaction
   │
   ▼
Feature engineering  (velocity, amount vs. user baseline, geo, merchant, time)
   │
   ▼
Fraud model  (LogReg / RF baselines → XGBoost or LightGBM, + anomaly detector)
   │
   ├── score below threshold ──► CLEAR (logged)
   │
   └── suspicious
          │
          ▼
       SHAP explanation
          │
          ▼
       Investigation agent (LangGraph)
         tools: transaction history · velocity · location patterns · similar cases
          │
          ▼
       Recommendation: CLEAR | REVIEW | BLOCK  + written report
          │
          ▼
       Analyst dashboard: APPROVE | REJECT | ESCALATE  → decision stored
```

Key principle: **the ML model predicts; the agent investigates; the human decides.**
See [`docs/architecture.md`](docs/architecture.md) for components and boundaries.

## Planned stack

| Layer | Technology |
|---|---|
| Data / ML | pandas, scikit-learn, XGBoost/LightGBM, PyTorch (only if an autoencoder is used) |
| Explainability | SHAP |
| Agent | LangGraph; LLM provider to be chosen (free tier for development) |
| Integrations | n8n for notifications/webhooks, not core reasoning |
| Backend | FastAPI, PostgreSQL, Redis |
| Frontend | React, Recharts |
| MLOps | MLflow, GitHub Actions, model versioning |
| Infra | Docker, Docker Compose |
| Monitoring | Prometheus/Grafana or a lightweight alternative |

Choices are provisional and are recorded as ADRs in [`docs/decisions/`](docs/decisions/) when finalised.

## Repository layout

```
configs/         configuration files
data/            raw/ and processed/ (git-ignored contents) + data README
docker/          Dockerfiles and compose (later milestone)
docs/            architecture, roadmap, decision records
notebooks/       exploration (01_eda.ipynb)
scripts/         thin CLI entry points
src/
  data/          loading, validation, time-based splits
  features/      behavioural feature engineering
  models/        baselines, primary, anomaly models, evaluation
  explainability/ SHAP and reason codes
  investigation/ LangGraph agent
  api/           FastAPI service
tests/           pytest tests mirroring src/
LEARNING_LOG.md  what was learned and why decisions were made
```

## Dataset

Planned: the Sparkov-simulated "Credit Card Transactions Fraud Detection" dataset on Kaggle
(`kartik2112/fraud-detection`). It is **not yet downloaded or verified**; see
[`data/README.md`](data/README.md). Reasons it was preferred over the popular ULB dataset
(PCA-anonymised features, so behavioural features are impossible) will be validated in the EDA.

## Roadmap

Each milestone ends with something demoable and a short write-up in `LEARNING_LOG.md`.
Full detail in [`docs/roadmap.md`](docs/roadmap.md).

- [x] **M0 Scaffold:** repo structure, docs, EDA notebook skeleton
- [ ] **M1 Data and baseline:** download, verify, EDA, chronological split, logistic-regression baseline
- [ ] **M2 Feature engineering:** velocity, user baselines, geo, merchant, time features (leakage-safe)
- [ ] **M3 Models:** Random Forest, XGBoost/LightGBM, anomaly detector; compared and tracked in MLflow
- [ ] **M4 Evaluation and thresholds:** PR-AUC, cost-aware threshold selection, calibration
- [ ] **M5 Explainability:** SHAP output turned into reason codes
- [ ] **M6 Investigation agent:** LangGraph agent with tools, report, CLEAR/REVIEW/BLOCK recommendation
- [ ] **M7 API and persistence:** FastAPI + PostgreSQL (+ Redis), cases and analyst decisions
- [ ] **M8 Dashboard:** React analyst UI with Approve/Reject/Escalate
- [ ] **M9 Productionise:** Docker Compose, GitHub Actions CI, monitoring, n8n notifications
- [ ] **Stretch: graph-based fraud-ring detection** (only after M1–M9; likely needs another dataset such as PaySim)

## Getting started

Nothing runnable yet beyond the notebook:

```
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
# download the dataset per data/README.md, then
jupyter notebook notebooks/01_eda.ipynb
```

## Working principles

1. Split by time, never randomly.
2. No claim without a measurement; no metric without the command that produced it.
3. Baselines first; every complex model must beat them.
4. Record decisions and what was learned in `LEARNING_LOG.md` and `docs/decisions/`.

## Cost

Intended to run at $0 locally using open-source tools and free tiers. Free-tier limits change; verify before relying on them.
