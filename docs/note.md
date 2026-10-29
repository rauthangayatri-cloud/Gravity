# Smart Campus Student Success Platform — project note

## Current evidence and method note

The Success Score is a transparent current-state measure. Risk probabilities
and worklist priority remain separate concepts. All records and all evaluation
figures below are synthetic.

The cleaning log reports `8,400` unified rows, `1,400` unified students, and
`1,400` outcome rows. For the LMS source it records `8,379` rows in, `399`
duplicate rows removed, `79` unmatched crosswalk rows, and `106` out-of-range
values fixed.

For academic risk, the recommended XGBoost model has ROC-AUC `0.8279` and
PR-AUC `0.7488`; logistic regression has PR-AUC `0.7082`. The XGBoost
top-ten-percent lift is `2.4407`, and its ROC-AUC confidence interval is
`0.7933` to `0.8562`.

For placement risk, the recommended logistic regression has ROC-AUC `0.7102`
and PR-AUC `0.4055`; XGBoost has PR-AUC `0.3975`. Logistic regression's
top-ten-percent lift is `1.9792`.

## Limitations

The data, labels, metrics, drivers, segments, playbooks, and follow-up
snapshots are synthetic. The academic result favours XGBoost while the
placement result favours logistic regression; this is a prototype comparison,
not evidence that either model will transfer to a campus setting. The reported
lift values are ranking comparisons against the relevant synthetic cohort base
rate, not operational impact. SHAP and what-if sensitivity show model
associations only; they are not causal estimates.

<!-- Superseded project note retained only for repository history.

## Indicators and Success Score

The Success Score is a transparent current-state measure, not a risk probability. It uses six components:

| Component | Weight | Rule |
| --- | ---: | --- |
| Academic | 35 | `0.7 × CGPA/10 + 0.3 × (1 - min(backlogs, 5)/5)` |
| Attendance | 20 | Attendance percentage normalized from 0 to 100. |
| LMS | 15 | Mean of LMS logins and assignment completion, each normalized from 0 to 100. |
| Placement readiness | 15 | Mean of aptitude, coding, and mock-interview scores, each normalized from 0 to 100. |
| Engagement + skills | 10 | Mean of engagement, communication, hackathons, and certifications; the latter two are capped at 5 before normalization. |
| Feedback | 5 | Feedback score normalized from 0 to 100. |

Each component is normalized to a 0–1 scale using fixed ranges, not fitted percentiles. The score is the weighted mean of available components, multiplied by 100. `confidence` is the available component weight divided by the configured total. When confidence is below 50%, the score is null and the status is **Insufficient data**. Tiers are Green (70+), Amber (50–69), and Red (below 50).

## Risk, trajectory, and priority

Academic risk is a next-period label when next-semester CGPA drop is at least 0.5 or a new backlog occurs. Placement risk is a non-placement result or an offer below the configured threshold. These are separate from Success Score.

Monthly risk features use only information available at that snapshot: current values, trailing three-month slopes and volatility, missingness flags, and department. Risk trajectories fit a simple line to months 2–5, project one step ahead, and show a residual-spread band. Rising risk is labelled `declining`; falling risk is labelled `improving`.

```text
priority = min(100, 100 × risk × severity × trend_multiplier)
```

Severity is 1.0 for attendance below 75% or at least two backlogs; 0.8 for attendance from 75% to 80% or one backlog; otherwise 0.6. Trend multipliers are 1.2 for declining, 1.0 for stable, and 0.8 for improving.

## Models and evaluation

For each risk target, the prototype compares a baseline rule, class-balanced logistic regression, and class-weighted XGBoost. Historical-cohort records train the models and current-cohort records evaluate them; there is no random split across cohorts. Reported outputs include ROC-AUC, PR-AUC, precision, recall, confusion matrix, the threshold nearest 80% recall, recall at the top 10% of students, SHAP top drivers, and department-level recall and false-positive rate.

## Limitations

All records, labels, metrics, SHAP values, segments, and playbooks are synthetic. The models are demonstrations, not validated decision tools. Associations in the data are engineered for the prototype and do not establish causation. Simulated 30- and 60-day follow-ups are illustrative snapshots, not observed outcomes or estimates of an intervention’s effect. Human review, governance, privacy controls, bias evaluation, and real-world validation would be required before any operational use.
-->
