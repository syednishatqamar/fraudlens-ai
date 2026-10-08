# Learning Log

Purpose: turn working code into understanding. Fill an entry after each session. If you can't
explain something here in your own words, that is the thing to study next.

Method for each step: **predict → run → compare → explain → break it on purpose.**

---

## Entry template (copy for each session)

### YYYY-MM-DD: Session title

**Goal:**

**Predictions before running** (what did I expect to see?):

**What actually happened:**

**What surprised me / what I got wrong:**

**Concepts learned** (own words):
-

**Decisions made and why** (link ADRs in `docs/decisions/`):
-

**Alternatives I considered and rejected:**
-

**Break-it experiment** (what I changed on purpose, what happened):

**Interview explanation** (3 sentences, as if explaining to a recruiter):

**Open questions / next steps:**
-

---

## Concept checklist

Tick when you can explain it without notes.

- [ ] Why accuracy is misleading on imbalanced data
- [ ] Precision, recall, F1, PR-AUC vs. ROC-AUC, and when each is appropriate
- [ ] Why fraud data needs a chronological split; what temporal leakage looks like
- [ ] Rolling-window features and point-in-time correctness
- [ ] Class weights vs. resampling (and why resampling must happen only on training data)
- [ ] How gradient boosting differs from a random forest
- [ ] Supervised vs. anomaly detection for fraud
- [ ] What a SHAP value is, and its limits
- [ ] Probability calibration and threshold selection with asymmetric costs
- [ ] Why the agent investigates but does not predict
- [ ] Experiment tracking: what an MLflow run records and why
- [ ] Train/serve skew

## Sessions

<!-- Add entries below, newest first. -->
