# Methodology note

All outputs are computed from synthetic data and are demonstration artifacts,
not evidence of real-world model performance, fairness, or causal effects.

## New single-snapshot contract

New bundles use 15 current indicators, 15 missingness flags and four fixed
department indicators. No history-dependent slope or volatility is silently
imputed for a single-snapshot upload. Training, dashboard risk, inference and
what-if calculations share the feature builder and selected predictor.

Historical students are stratified with seed 42 into 60% fitting, 20% calibration,
and 20% model/threshold selection. Imputation/scaling fit only on the fitting
subset. Both balanced logistic regression and weighted XGBoost receive logistic
recalibration on the calibration subset. Selection prefers logistic regression
unless XGBoost improves selection-set average precision (reported as PR-AUC) by
more than 0.02. The threshold nearest 80% recall is chosen on selection students;
this is an objective, not a guarantee on unseen students. Current labels are
evaluated only after selection and thresholds are frozen.

Both cohorts cover the same generated period: this is an unseen synthetic student
holdout, NOT temporal validation. Metrics include frozen-threshold errors, AUCs,
top-share lift/recall, Brier scores, calibration bins, department class counts and
Wilson recall/FPR intervals, and missingness slices. Bootstrap AUC intervals use
200 seeded resamples and condition on fitted models; they do not cover model-
selection uncertainty. Cross-validation and real-world validation remain unperformed.
Synthetic department differences are engineered; no fairness guarantee follows.

## Scores, explanations and priority

Success Score retains fixed normalization and weights 35/20/15/15/10/5. Missing
components are reweighted; available weight below 50% suppresses the score.
Coverage is component-weight availability, not statistical confidence. Partially
populated components can count as available. Canonical fields use row-wise non-null
alias fallback. Existing saved scores are never automatically recomputed.

Risk probabilities use the selected estimator AND its serialized calibrator.
Explanations describe that selected BASE model in log-odds, not the nonlinear
calibrated probability. XGBoost uses TreeSHAP; logistic regression uses independent-
feature linear SHAP about the standardized training mean. The UI identifies target,
estimator and units. Signs refer to the base model. Correlated features affect
interpretation. Only top-three contributions per target are shown; these do not
sum to the complete model output. Neither SHAP nor what-if is causal.

What-if makes a bounded one-at-a-time change to a current indicator; missing inputs
remain missing. Its baseline uses the same predictor as dashboard risk. Combined
risk is max of two risks, not the probability of either outcome. Trajectory fits
and residual-spread bands remain heuristics, NOT confidence/prediction intervals.
Runtime single snapshots receive no uncertainty band or forecast. Priority remains
the separate risk × severity × trend formula. Runtime segments use fixed descriptive
rules without fitting; offline descriptive KMeans remains available. Simulated
follow-ups are illustrative, not observed intervention effects.

## Preserved legacy demonstration

With no selected new bundle, existing artifacts remain in labelled legacy mode.
Offline legacy risks use calibrated XGBoost; legacy uploads use uncalibrated models
trained with history absent from uploads. They are not equivalent prediction paths.
Legacy what-if comparisons are withheld because their estimator may differ from
dashboard risk. Old metric numbers refer only to preserved legacy artifacts, not
new builds. No existing artifact or stored prediction is silently replaced.
