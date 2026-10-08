# Architecture

Status: **design intent, not implemented.** Update this file as components are built.

## Responsibilities

| Component | Responsibility | Must NOT |
|---|---|---|
| `src/data` | Load, validate schema, split chronologically | Compute features |
| `src/features` | Turn raw transactions into behavioural features, using only past data per transaction | Peek at future rows |
| `src/models` | Train, evaluate, version, and serve scores | Explain or decide |
| `src/explainability` | SHAP values → structured reason codes | Change the score |
| `src/investigation` | Agent gathers context via tools, writes report, recommends CLEAR/REVIEW/BLOCK | Predict fraud probability |
| `src/api` | Expose scoring, cases, decisions; persist to PostgreSQL | Contain ML logic |
| Frontend | Analyst review and decision UI | Hold business logic |
| n8n | Notifications, webhooks | Core reasoning |

## Data flow

1. A transaction arrives (batch replay initially; streaming/API later).
2. Features are computed from the transaction plus that customer's *prior* history.
3. The model returns a fraud probability; a configurable threshold decides suspicious or not.
4. Suspicious cases get SHAP explanations, then are handed to the investigation agent.
5. The agent's report and recommendation are stored as a case.
6. An analyst approves, rejects or escalates; the decision is stored for audit and future retraining.

## Cross-cutting concerns

- **Leakage:** features and splits are time-aware; tests will enforce it.
- **Reproducibility:** seeds, configs and MLflow runs for every reported number.
- **Auditability:** every case stores model version, features, explanation, recommendation, analyst decision.
- **Observability:** structured logs; metrics/monitoring in M9.

## Open questions (to resolve via ADRs)

- Online feature computation: how to compute velocity features at scoring time (Redis vs. DB query)?
- Anomaly detector: Isolation Forest vs. autoencoder, and how its score combines with the supervised model.
- Which LLM provider and how to keep the agent deterministic enough to test.
