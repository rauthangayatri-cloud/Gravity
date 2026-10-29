"""Develop synthetic snapshot models with a separate untouched cohort evaluation."""

from __future__ import annotations

import numpy as np
import pandas as pd
import shap
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    confusion_matrix,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier

from src.features import SNAPSHOT_SCHEMA, assert_no_outcome_features, build_snapshot_features
from src.prediction import RiskPredictor

SEED = 42
OFFER_THRESHOLD = 500_000
META_COLUMNS = {"student_id", "snapshot_month", "cohort", "department"}
BOOTSTRAP_RESAMPLES = 200


def build_targets(outcomes: pd.DataFrame, offer_threshold: float = OFFER_THRESHOLD) -> pd.DataFrame:
    """Derive the two binary next-period targets from outcome labels only."""
    required = {"student_id", "cohort", "next_semester_cgpa_drop", "new_backlog", "placement_result"}
    missing = required.difference(outcomes.columns)
    if missing:
        raise ValueError(f"outcomes.csv is missing: {sorted(missing)}")
    identifiers = outcomes.student_id.astype("string").str.strip().str.casefold()
    if identifiers.isna().any() or identifiers.eq("").any() or identifiers.duplicated().any():
        raise ValueError("Outcome student IDs must be present and unique across cohorts")
    numeric_drop = pd.to_numeric(outcomes["next_semester_cgpa_drop"], errors="coerce")
    numeric_backlog = pd.to_numeric(outcomes["new_backlog"], errors="coerce")
    placement = outcomes["placement_result"].astype("string").str.strip().str.lower()
    if (
        not np.isfinite(numeric_drop).all()
        or not numeric_backlog.isin([0, 1]).all()
        or not placement.isin(["placed", "not_placed"]).all()
        or not outcomes.cohort.isin(["historical", "current"]).all()
    ):
        raise ValueError("Missing or invalid outcome labels; unlabeled students cannot become negatives")
    result = outcomes[["student_id", "cohort"]].copy()
    drop = pd.to_numeric(outcomes["next_semester_cgpa_drop"], errors="coerce").ge(0.5)
    backlog = pd.to_numeric(outcomes["new_backlog"], errors="coerce").eq(1)
    result["academic_risk"] = (drop | backlog).astype(int)
    below_offer = pd.Series(False, index=outcomes.index)
    for column in ("offer_amount", "offer_value", "annual_offer"):
        if column in outcomes:
            offers = pd.to_numeric(outcomes[column], errors="coerce")
            if not np.isfinite(offers).all() or offers.lt(0).any():
                raise ValueError("Supplied offer labels must be finite and non-negative")
            below_offer = offers.lt(offer_threshold)
            break
    result["placement_risk"] = (placement.ne("placed") | below_offer).astype(int)
    return result


def _latest_snapshots(unified: pd.DataFrame) -> pd.DataFrame:
    """Select each student's latest available prediction snapshot."""
    return (
        unified.assign(snapshot_month=pd.to_datetime(unified["snapshot_month"]))
        .sort_values(["student_id", "snapshot_month"], kind="stable")
        .drop_duplicates("student_id", keep="last")
    )


def _model_matrix(features: pd.DataFrame) -> pd.DataFrame:
    """Remove split metadata and assert only valid model columns remain."""
    matrix = features.drop(columns=[column for column in META_COLUMNS if column in features], errors="ignore")
    assert_no_outcome_features(matrix)
    if matrix.empty:
        raise ValueError("No model feature columns were supplied")
    return matrix


def _metrics(y_true: pd.Series, probability: np.ndarray, threshold: float = 0.5) -> dict[str, object]:
    """Return standard binary metrics, safely handling a one-class evaluation set."""
    prediction = probability >= threshold
    roc = float(roc_auc_score(y_true, probability)) if y_true.nunique() == 2 else None
    pr = float(average_precision_score(y_true, probability)) if y_true.nunique() == 2 else None
    return {
        "roc_auc": round(roc, 4) if roc is not None else None,
        "pr_auc": round(pr, 4) if pr is not None else None,
        "threshold": float(threshold),
        "precision": round(float(precision_score(y_true, prediction, zero_division=0)), 4),
        "recall": round(float(recall_score(y_true, prediction, zero_division=0)), 4),
        "confusion_matrix": confusion_matrix(y_true, prediction, labels=[0, 1]).tolist(),
    }


def _recall_threshold(y_true: pd.Series, probability: np.ndarray) -> tuple[float, float]:
    """Choose the probability threshold whose recall is closest to 80%."""
    precision, recall, thresholds = precision_recall_curve(y_true, probability)
    if not len(thresholds):
        return 0.5, 0.0
    index = int(np.argmin(np.abs(recall[:-1] - 0.8)))
    return float(thresholds[index]), float(precision[index])


def _recall_at_k(y_true: pd.Series, probability: np.ndarray, share: float = 0.10) -> float:
    """Compute recall after taking the highest-risk share of students."""
    count = max(1, int(np.ceil(len(y_true) * share)))
    selected = y_true.iloc[np.argsort(probability)[-count:]]
    return round(float(selected.sum() / max(1, y_true.sum())), 4)


def _lift_at_k(y_true: pd.Series, probability: np.ndarray, share: float) -> float | None:
    """Return top-share event rate divided by the current-cohort base rate."""
    base_rate = float(y_true.mean())
    if base_rate == 0:
        return None
    count = max(1, int(np.ceil(len(y_true) * share)))
    top_rate = float(y_true.iloc[np.argsort(probability)[-count:]].mean())
    return round(top_rate / base_rate, 4)


def _bootstrap_auc_ci(
    y_true: pd.Series, probability: np.ndarray, resamples: int = BOOTSTRAP_RESAMPLES
) -> dict[str, list[float] | None]:
    """Bootstrap current-cohort ROC-AUC and PR-AUC confidence intervals."""
    truth = np.asarray(y_true, dtype=int)
    scores = np.asarray(probability, dtype=float)
    rng = np.random.default_rng(SEED)
    roc_values, pr_values = [], []
    for _ in range(resamples):
        sample = rng.integers(0, len(truth), len(truth))
        if np.unique(truth[sample]).size < 2:
            continue
        roc_values.append(roc_auc_score(truth[sample], scores[sample]))
        pr_values.append(average_precision_score(truth[sample], scores[sample]))

    def interval(values: list[float]) -> list[float] | None:
        return [round(float(value), 4) for value in np.quantile(values, [0.025, 0.975])] if values else None

    return {"roc_auc_95ci": interval(roc_values), "pr_auc_95ci": interval(pr_values)}


def _calibration_table(y_true: pd.Series, probability: np.ndarray, target_name: str, bins: int = 10) -> pd.DataFrame:
    """Summarise current-cohort calibrated probabilities by fixed bins."""
    frame = pd.DataFrame({"actual": np.asarray(y_true, dtype=int), "probability": np.asarray(probability, dtype=float)})
    frame["bin"] = pd.cut(frame["probability"], bins=np.linspace(0, 1, bins + 1), include_lowest=True)
    summary = (
        frame.groupby("bin", observed=True)
        .agg(
            student_count=("actual", "size"),
            mean_predicted_probability=("probability", "mean"),
            observed_rate=("actual", "mean"),
        )
        .reset_index()
    )
    summary.insert(0, "target", target_name)
    summary["bin"] = summary["bin"].astype(str)
    return summary


def _fit_probability_calibrator(probability: np.ndarray, target: pd.Series) -> LogisticRegression:
    """Fit a lightweight Platt calibrator to held-out historical predictions."""
    calibrator = LogisticRegression(random_state=SEED, max_iter=1000)
    calibrator.fit(np.asarray(probability, dtype=float).reshape(-1, 1), target.astype(int))
    return calibrator


def _fairness(
    y_true: pd.Series, probability: np.ndarray, department: pd.Series, threshold: float
) -> list[dict[str, object]]:
    """Calculate evaluation recall and false-positive rate by department."""
    records = []
    evaluation = pd.DataFrame(
        {"truth": y_true.to_numpy(), "probability": probability, "department": department.fillna("Unknown").to_numpy()}
    )
    for name, group in evaluation.groupby("department", dropna=False):
        predicted = group.probability.to_numpy() >= threshold
        truth = group.truth.to_numpy()
        positives, negatives = (truth == 1).sum(), (truth == 0).sum()
        records.append(
            {
                "department": str(name),
                "students": int(len(group)),
                "positives": int(positives),
                "negatives": int(negatives),
                "threshold": float(threshold),
                "brier_score": round(float(brier_score_loss(truth, group.probability)), 4),
                "recall_95ci": _wilson(int(((predicted == 1) & (truth == 1)).sum()), int(positives)),
                "false_positive_rate_95ci": _wilson(int(((predicted == 1) & (truth == 0)).sum()), int(negatives)),
                "recall": round(float(((predicted == 1) & (truth == 1)).sum() / positives), 4) if positives else None,
                "false_positive_rate": round(float(((predicted == 1) & (truth == 0)).sum() / negatives), 4)
                if negatives
                else None,
            }
        )
    return records


def _wilson(events: int, count: int) -> list[float] | None:
    """Describe binomial sampling uncertainty without asserting fairness."""
    if not count:
        return None
    rate, z = events / count, 1.96
    denominator = 1 + z * z / count
    center = (rate + z * z / (2 * count)) / denominator
    half = z * np.sqrt(rate * (1 - rate) / count + z * z / (4 * count * count)) / denominator
    return [round(max(0, center - half), 4), round(min(1, center + half), 4)]


def _shap_drivers(model: Pipeline, features: pd.DataFrame, student_ids: pd.Series, target_name: str) -> pd.DataFrame:
    """Return the three strongest local XGBoost SHAP drivers per student."""
    imputer = model.named_steps["imputer"]
    transformed = imputer.transform(features)
    feature_names = imputer.get_feature_names_out(features.columns)
    values = shap.TreeExplainer(model.named_steps["model"])(transformed).values
    if values.ndim == 3:
        values = values[:, :, -1]
    rows = []
    for row_index, student_id in enumerate(student_ids):
        top = np.argsort(np.abs(values[row_index]))[-3:][::-1]
        for rank, column_index in enumerate(top, start=1):
            shap_value = float(values[row_index, column_index])
            rows.append(
                {
                    "student_id": student_id,
                    "target": target_name,
                    "rank": rank,
                    "feature": feature_names[column_index],
                    "direction": "increases risk" if shap_value > 0 else "reduces risk",
                    "magnitude": round(abs(shap_value), 5),
                }
            )
    return pd.DataFrame(rows)


def train_risk_model(features: pd.DataFrame, target: pd.Series) -> dict[str, object]:
    """Fit/calibrate/select on disjoint historical students; evaluate current once."""
    for column in ("cohort", "student_id", "department"):
        if column not in features:
            raise ValueError(f"features must include {column} metadata for the cohort split")
    matrix = _model_matrix(features)
    ids = features.student_id.astype("string").str.strip().str.casefold()
    if ids.isna().any() or ids.eq("").any() or ids.duplicated().any():
        raise ValueError("Training requires one row per distinct student across cohorts")
    if not features.index.is_unique or not features.index.equals(target.index) or not target.isin([0, 1]).all():
        raise ValueError("Target alignment or binary labels are invalid")
    if not features.cohort.isin(["historical", "current"]).all():
        raise ValueError("Unknown training cohort")
    historical = features.cohort.eq("historical")
    current = features.cohort.eq("current")
    if not historical.any() or not current.any():
        raise ValueError("Both historical development and current evaluation cohorts are required")
    x_historical, x_test = matrix.loc[historical], matrix.loc[current]
    y_historical, y_test = target.loc[historical].astype(int), target.loc[current].astype(int)
    if y_historical.nunique() < 2 or y_historical.value_counts().min() < 5:
        raise ValueError("Historical development requires at least five students in each class")
    train_index, development_index = train_test_split(
        x_historical.index,
        test_size=0.4,
        random_state=SEED,
        stratify=y_historical,
    )
    calibration_index, selection_index = train_test_split(
        development_index, test_size=0.5, random_state=SEED, stratify=y_historical.loc[development_index]
    )
    x_train, y_train = x_historical.loc[train_index], y_historical.loc[train_index]
    x_calibration, y_calibration = x_historical.loc[calibration_index], y_historical.loc[calibration_index]
    x_selection, y_selection = x_historical.loc[selection_index], y_historical.loc[selection_index]
    baseline = (
        (
            (x_test.get("cgpa_current", pd.Series(np.nan, index=x_test.index)) < 6)
            | (x_test.get("attendance_pct_current", pd.Series(np.nan, index=x_test.index)) < 75)
        )
        .astype(float)
        .to_numpy()
    )
    imbalance = max(1.0, float((y_train == 0).sum() / max(1, (y_train == 1).sum())))
    logistic = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="median", keep_empty_features=True)),
            ("scale", StandardScaler()),
            ("model", LogisticRegression(class_weight="balanced", max_iter=1000, random_state=SEED)),
        ]
    )
    xgb = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="median", keep_empty_features=True)),
            (
                "model",
                XGBClassifier(
                    n_estimators=120,
                    max_depth=3,
                    learning_rate=0.05,
                    subsample=0.85,
                    colsample_bytree=0.85,
                    scale_pos_weight=imbalance,
                    random_state=SEED,
                    n_jobs=1,
                    eval_metric="logloss",
                ),
            ),
        ]
    )
    logistic.fit(x_train, y_train)
    xgb.fit(x_train, y_train)
    models = {"logistic_regression": logistic, "xgboost": xgb}
    calibrators, predictors, selection = {}, {}, {}
    for name, model in models.items():
        calibrators[name] = _fit_probability_calibrator(model.predict_proba(x_calibration)[:, 1], y_calibration)
        predictor = RiskPredictor(target.name or "risk_target", name, model, calibrators[name], 0.5)
        validation_probability = predictor.predict_proba(x_selection)[:, 1]
        predictor.threshold, _ = _recall_threshold(y_selection, validation_probability)
        predictors[name] = predictor
        selection[name] = float(average_precision_score(y_selection, validation_probability))
    recommended = "xgboost" if selection["xgboost"] - selection["logistic_regression"] > 0.02 else "logistic_regression"
    # Only after selection/thresholds are frozen are current labels evaluated.
    probabilities = {
        "baseline_rule": baseline,
        **{name: p.predict_proba(x_test)[:, 1] for name, p in predictors.items()},
    }
    metric_report = {
        name: _metrics(y_test, values, predictors[name].threshold if name in predictors else 0.5)
        for name, values in probabilities.items()
    }
    for name, values in probabilities.items():
        metric_report[name].update(
            {
                "lift_at_top_10pct": _lift_at_k(y_test, values, 0.10),
                "lift_at_top_20pct": _lift_at_k(y_test, values, 0.20),
                **_bootstrap_auc_ci(y_test, values),
            }
        )
    for name, predictor in predictors.items():
        raw = models[name].predict_proba(x_test)[:, 1]
        values = probabilities[name]
        metric_report[name].update(
            {
                "current_positive_base_rate": round(float(y_test.mean()), 4),
                "threshold_for_approximately_80pct_recall": predictor.threshold,
                "threshold_source": "historical selection students; 80% recall is a development objective, not a guarantee",
                "precision_at_that_threshold": metric_report[name]["precision"],
                "recall_at_top_10pct": _recall_at_k(y_test, values),
                "fairness_by_department": _fairness(
                    y_test, values, features.loc[current, "department"], predictor.threshold
                ),
                "calibration": {
                    "method": "Logistic recalibration of raw probabilities on held-out historical students only",
                    "historical_calibration_students": int(len(y_calibration)),
                    "brier_score_before": round(float(brier_score_loss(y_test, raw)), 4),
                    "brier_score_after": round(float(brier_score_loss(y_test, values)), 4),
                },
            }
        )
    metric_report["recommended_model"] = recommended
    metric_report["selection_pr_auc"] = selection
    metric_report["feature_schema"] = SNAPSHOT_SCHEMA
    metric_report["evaluation_note"] = (
        "Unseen synthetic students, same generated period; not temporal or real-world validation"
    )
    metric_report["cross_validation"] = (
        "Not performed; one seeded development split. Bootstrap CIs condition on fitted models."
    )
    missing = x_test.filter(like="_missing").sum(axis=1)
    metric_report["missingness_slices"] = [
        {
            "slice": label,
            "students": int(mask.sum()),
            "positives": int(y_test.loc[mask].sum()),
            **_metrics(
                y_test.loc[mask], probabilities[recommended][mask.to_numpy()], predictors[recommended].threshold
            ),
        }
        for label, mask in [("complete", missing.eq(0)), ("missing_indicators", missing.gt(0))]
        if mask.any()
    ]
    predictions = pd.DataFrame(
        {
            "student_id": features.loc[current, "student_id"].values,
            "department": features.loc[current, "department"].values,
            "actual": y_test.values,
        }
    )
    for name, values in probabilities.items():
        predictions[f"{name}_probability"] = values
    predictions["xgboost_uncalibrated_probability"] = xgb.predict_proba(x_test)[:, 1]
    predictions["selected_probability"] = probabilities[recommended]
    predictions["selected_model"] = recommended
    predictions["decision_threshold"] = predictors[recommended].threshold
    return {
        "models": models,
        "calibrators": calibrators,
        "predictor": predictors[recommended],
        "split_students": {
            name: features.loc[index, "student_id"].astype(str).tolist()
            for name, index in [
                ("fit", train_index),
                ("calibration", calibration_index),
                ("selection", selection_index),
                ("evaluation", features.index[current]),
            ]
        },
        "metrics": metric_report,
        "predictions": predictions,
        "calibration": pd.concat(
            [
                _calibration_table(y_test, probabilities[name], target.name or "risk_target").assign(model=name)
                for name in predictors
            ],
            ignore_index=True,
        ),
        "shap": predictors[recommended].drivers(x_test, features.loc[current, "student_id"]),
    }


def train_from_files(unified: pd.DataFrame, outcomes: pd.DataFrame) -> dict[str, object]:
    """Build latest-snapshot cohort features and train both requested risk targets."""
    keys = unified.assign(snapshot_month=pd.to_datetime(unified.snapshot_month))
    if (
        keys[["student_id", "snapshot_month", "cohort"]].isna().any().any()
        or keys.duplicated(["student_id", "snapshot_month"]).any()
    ):
        raise ValueError("Missing or duplicate snapshot keys")
    if keys.groupby("student_id").cohort.nunique().gt(1).any():
        raise ValueError("A student cannot cross development/evaluation cohorts")
    latest = _latest_snapshots(unified)
    feature_frame = build_snapshot_features(latest)
    metadata = latest[
        [column for column in ("student_id", "snapshot_month", "cohort", "department") if column in latest]
    ].reset_index(drop=True)
    feature_frame = feature_frame.reset_index(drop=True)
    targets = build_targets(outcomes)
    training = pd.concat([metadata, feature_frame], axis=1).merge(
        targets, on=["student_id", "cohort"], how="outer", validate="one_to_one", indicator=True
    )
    if not training.pop("_merge").eq("both").all():
        raise ValueError("Every student must have exactly one matching outcome/cohort")
    results = {}
    for target_name in ("academic_risk", "placement_risk"):
        target = training[target_name].rename(target_name)
        results[target_name] = train_risk_model(training.drop(columns=["academic_risk", "placement_risk"]), target)
    return results


def main() -> None:
    """Use the single explicit, non-overwriting bundle publisher for both CLIs."""
    from src.build_outputs import main as build_main

    build_main()


if __name__ == "__main__":
    main()
