# Slide outline

## Slide 1 — Problem and campus context

- A synthetic unified view keeps the Success Score, risk probabilities, and
  worklist priority distinct.
- Screenshot placeholder: Overview with the visible synthetic-data badge.

## Slide 2 — Data pipeline and cleaning

- The cleaning log reports `8,400` unified rows, `1,400` students, and
  `1,400` outcome rows.
- LMS processing recorded `8,379` input rows, `399` duplicate rows removed,
  `79` unmatched joins, and `106` out-of-range values fixed.
- Diagram placeholder: raw sources → crosswalk and cleaning → precomputed
  outputs → API and dashboard.

## Slide 3 — Score, risk, and priority logic

- The Success Score is a transparent current-state formula; risk probabilities
  are separate model outputs; priority ranks support review.
- Screenshot placeholder: score components, calibrated risks, and priority.

## Slide 4 — Dashboard walkthrough

- Screenshot placeholders: Overview, priority worklist, and student detail.
- Call out precomputed SHAP drivers and the what-if panel label: “Model-based
  sensitivity on synthetic data, not a causal estimate.”

## Slide 5 — Generated model findings

- Academic risk: recommended XGBoost ROC-AUC `0.8279`, PR-AUC `0.7488`, and
  top-ten-percent lift `2.4407`.
- Placement risk: recommended logistic regression ROC-AUC `0.7102`, PR-AUC
  `0.4055`, and top-ten-percent lift `1.9792`.
- Academic XGBoost PR-AUC is `0.7488` versus logistic regression at `0.7082`;
  placement logistic regression PR-AUC is `0.4055` versus XGBoost at `0.3975`.

## Slide 6 — Segments and priority worklist

- Show segment profiles as descriptive groups, then open the precomputed
  priority worklist to illustrate review flow.
- Screenshot placeholder: filtered worklist and one student’s recommended
  action, SHAP drivers, and what-if table.

## Slide 7 — Limitations

- All inputs, labels, and outputs are synthetic.
- XGBoost is recommended for academic risk while logistic regression is
  recommended for placement risk in this synthetic evaluation.
- Lift is a ranking comparison against the synthetic cohort base rate, not a
  claim of operational benefit.
- SHAP, playbooks, and what-if changes are not causal estimates.

## Slide 8 — Typical dashboards vs ours

| Typical dashboard | This prototype |
| --- | --- |
| Static aggregate reporting | Current-state Score, separate risks, and worklist priority. |
| Unexplained risk | Precomputed SHAP drivers and rule-based actions. |
| Live model work | Offline precomputed artifacts served by API. |
| Causal implication from changes | Synthetic labels and explicit non-causal sensitivity wording. |

<!-- Superseded slide outline retained only for repository history.

There are six core slides plus one comparison slide. Do not add figures until the offline pipeline has generated `data/outputs/`; this repository currently contains no output values beyond `.gitkeep`.

## 1. Problem and campus context

- Fragmented academic, attendance, LMS, engagement, placement, skills, and feedback data make early support difficult.
- Goal: a unified synthetic student view that keeps Score, risk, and priority distinct.

## 2. Data pipeline and cleaning

- Synthetic raw records have six monthly snapshots and deliberately realistic quality issues.
- Show: raw sources → cleaning/unification → offline features/models/SHAP → precomputed outputs → API/dashboard.
- Screenshot placeholder: cleaning log.

## 3. Score and risk logic

- Show the six Score weights and confidence rule.
- Show the two next-period risk labels, historical-to-current cohort evaluation, and priority formula.

## 4. Dashboard screenshots

- Screenshot placeholder: Overview page.
- Screenshot placeholder: Worklist and Student drill-down page.
- Call out the synthetic-data label and the simulated follow-up disclaimer.

## 5. Three key findings from generated outputs

Populate only after the pipeline completes. Cite the exact source file and value beside each finding.

- Finding 1: `[exact value from data/outputs/metrics.json]` — model evaluation or recall-at-k result.
- Finding 2: `[exact value from data/outputs/priority.csv or trajectories.csv]` — worklist or trend result.
- Finding 3: `[exact value from data/outputs/segments.csv or score_validation.json]` — segment/profile or score-validation result.

No numerical finding is stated here because `data/outputs/` currently has no generated values.

## 6. Impact and limitations

- Potential workflow: review precomputed priorities, inspect drivers, select a rule-based action, and log follow-up.
- Explicit limitations: synthetic data and labels, illustrative simulated snapshots, no causal claims, and no real-world accuracy claim.

## 7. Typical dashboards vs ours

| Typical dashboard | This prototype |
| --- | --- |
| Static aggregate reporting | Per-student current-state Score, separate risks, and intervention priority. |
| Risk shown without explanation | Offline SHAP top drivers and rule-based playbook mapping. |
| One undifferentiated metric | Transparent Score, ML probabilities, and priority kept separate. |
| Live model work in the UI | Precomputed outputs; API and web read files only. |
| Outcome claims from visual changes | Explicit synthetic and simulated labels; no causal claims. |
-->
