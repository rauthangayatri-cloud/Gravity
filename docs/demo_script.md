# Demo script

# Timed 2-minute demo script

Use the highest-priority student shown in the current worklist. Do not add a
student-specific number unless it appears in the precomputed screen you are
showing.

## 0:00–0:20 — Unified view

“This platform is entirely synthetic. The unified output contains `8,400`
rows for `1,400` students, with `1,400` outcome rows held separately from the
features. The dashboard keeps the current-state Success Score, model risk,
and worklist priority separate.”

## 0:20–0:40 — One student

“I’ll open the highest-priority student. This view combines the score and its
confidence, both calibrated risk probabilities, trajectory, and the
precomputed SHAP drivers. The drivers explain the model output; they are not
causal explanations.”

## 0:40–1:00 — What-if sensitivity

“The what-if table changes one indicator at a time and recomputes the
recommended model’s risk offline. The label is deliberate: model-based
sensitivity on synthetic data, not a causal estimate. It is a way to inspect
model response, not to promise an intervention result.”

## 1:00–1:20 — Segments

“Next, the Segments page groups students by their precomputed profiles. These
are descriptive groups for navigation and playbook context, not diagnoses.”

## 1:20–1:40 — Priority worklist

“The worklist orders review by risk, severity, and trend. A faculty member
can filter it, inspect the driver and action context, and decide whether to
start a follow-up.”

## 1:40–2:00 — Evaluation and close

“In this synthetic evaluation, academic XGBoost has PR-AUC `0.7488` and
top-ten-percent lift `2.4407`. Placement logistic regression has PR-AUC
`0.4055` and top-ten-percent lift `1.9792`. These are synthetic ranking
results, not real-world accuracy, fairness, or causal claims.”

<!-- Superseded demo script retained only for repository history.

Before presenting, run the offline build and choose one student ID from the top row of `data/outputs/priority.csv`. Replace `[STUDENT_ID]` below with that exact value. Do not quote a metric unless it appears in `data/outputs/`.

## 0:00–0:20 — Set the context

“This is a Smart Campus Student Success prototype built entirely on synthetic data. It separates a transparent Success Score, two ML risk probabilities, and an intervention priority score. Nothing trains in the dashboard.”

## 0:20–0:45 — Overview

“The Overview shows the current score distribution, tier counts, and department comparison. The visible synthetic-data badge is deliberate: these are demonstration outputs, not real student records.”

## 0:45–1:10 — Worklist

“The worklist orders students by the separate priority formula, which combines projected risk, severity, and trajectory. I can filter by department, tier, or segment. I’ll open `[STUDENT_ID]`, selected from the generated priority file.”

## 1:10–1:55 — Student drill-down

“Here is the Success Score and its confidence. If confidence is below 50%, the page says Insufficient data rather than presenting a score. The academic and placement risks are separate from that score. The trajectory shows the precomputed months-two-to-five fit and projection band; the SHAP chart shows the exported top model drivers.”

## 1:55–2:20 — Action

“The recommended action is rule-based. It maps a relevant top driver to an intervention, owner, and timeline. Starting an intervention records a simple follow-up entry; it does not retrain a model or change a student score.”

## 2:20–2:40 — Simulated follow-up

“For a small demo subset, this timeline is explicitly labelled simulated. It shows illustrative before, 30-day, and 60-day snapshots. It is not an observed outcome and it does not imply an intervention caused any change.”

## 2:40–3:00 — Close

“The Admin page makes the implementation auditable: weights can be scenario-tested from precomputed components, and the quality, synthetic model metrics, and fairness tables are visible. The next step for real deployment would be governed data integration, human review, privacy controls, and real-world validation.”
-->
