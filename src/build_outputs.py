"""Precompute students, risks, trajectories, priorities, segments, and actions."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from src.features import build_snapshot_features
from src.priority import calculate_trajectories, ranked_worklist
from src.score import score_students
from src.segment import segment_students
from src.train import train_from_files

OUTPUT_DIR = Path("data/outputs")


def _records(frame: pd.DataFrame) -> list[dict]:
    """Convert data safely to JSON dashboard records."""
    return json.loads(frame.to_json(orient="records"))


PLAYBOOK_RULES = (
    {
        "driver": "attendance_pct",
        "field": "attendance_pct",
        "limit": 65,
        "intervention": "Attendance recovery plan",
        "owner": "Faculty",
        "timeline": "7d",
    },
    {
        "driver": "cgpa",
        "field": "cgpa",
        "limit": 6.5,
        "intervention": "Academic mentor check-in",
        "owner": "Faculty",
        "timeline": "7d",
    },
    {
        "driver": "lms_logins",
        "field": "lms_logins",
        "limit": 25,
        "intervention": "LMS engagement plan",
        "owner": "Mentor",
        "timeline": "7d",
    },
    {
        "driver": "coding_score",
        "field": "coding_score",
        "limit": 50,
        "intervention": "Coding improvement program",
        "owner": "Placement cell",
        "timeline": "14d",
    },
    {
        "driver": "mock_interview_score",
        "field": "mock_interview_score",
        "limit": 50,
        "intervention": "Mock interview practice",
        "owner": "Placement cell",
        "timeline": "14d",
    },
)


SIMULATION_NOTE = "SIMULATED illustrative follow-up snapshot. It is not an observed outcome or a causal estimate."
WHAT_IF_NOTE = "Model-based sensitivity on synthetic data, not a causal estimate."
WHAT_IF_ACTIONS = (
    ("attendance_pct_current", 10, "Attendance +10 points"),
    ("assignment_completion_pct_current", 15, "Assignment completion +15 points"),
    ("coding_score_current", 10, "Coding score +10 points"),
    ("mock_interview_score_current", 10, "Mock interview score +10 points"),
)


def generate_simulated_followups(students: pd.DataFrame, seed: int = 42, demo_size: int = 20) -> pd.DataFrame:
    """Create deterministic 30/60-day illustrative snapshots for demo students."""
    rng = np.random.default_rng(seed)
    selected = students.sort_values("priority_score", ascending=False).head(demo_size)
    rows = []
    for _, student in selected.iterrows():
        baseline_score = pd.to_numeric(pd.Series([student.get("success_score")]), errors="coerce").iloc[0]
        baseline_risk = pd.to_numeric(
            pd.Series([student.get("risk", student.get("projected_risk"))]), errors="coerce"
        ).iloc[0]
        if pd.isna(baseline_score):
            baseline_score = 50.0
        if pd.isna(baseline_risk):
            baseline_risk = 0.5
        rows.append(
            {
                "student_id": student.student_id,
                "day_after_start": 0,
                "timepoint": "Before start",
                "success_score": round(float(baseline_score), 1),
                "risk": round(float(baseline_risk), 4),
                "simulation_label": "SIMULATED baseline",
                "note": SIMULATION_NOTE,
            }
        )
        score_30 = np.clip(baseline_score + rng.normal(1.5, 3.5), 0, 100)
        risk_30 = np.clip(baseline_risk + rng.normal(-0.025, 0.055), 0, 1)
        score_60 = np.clip(score_30 + rng.normal(1.0, 3.5), 0, 100)
        risk_60 = np.clip(risk_30 + rng.normal(-0.02, 0.055), 0, 1)
        rows.extend(
            [
                {
                    "student_id": student.student_id,
                    "day_after_start": 30,
                    "timepoint": "30 days",
                    "success_score": round(float(score_30), 1),
                    "risk": round(float(risk_30), 4),
                    "simulation_label": "SIMULATED follow-up",
                    "note": SIMULATION_NOTE,
                },
                {
                    "student_id": student.student_id,
                    "day_after_start": 60,
                    "timepoint": "60 days",
                    "success_score": round(float(score_60), 1),
                    "risk": round(float(risk_60), 4),
                    "simulation_label": "SIMULATED follow-up",
                    "note": SIMULATION_NOTE,
                },
            ]
        )
    return pd.DataFrame(rows)


def _latest(frame: pd.DataFrame) -> pd.DataFrame:
    """Return each student's latest snapshot without using future observations."""
    return (
        frame.assign(snapshot_month=pd.to_datetime(frame["snapshot_month"]))
        .sort_values(["student_id", "snapshot_month"], kind="stable")
        .drop_duplicates("student_id", keep="last")
    )


def _risk_history(unified: pd.DataFrame, results: dict[str, object]) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Score every snapshot with models trained on historical students only."""
    feature_values = build_snapshot_features(unified)
    history = unified[["student_id", "snapshot_month", "cohort"]].copy()
    drivers = []
    for target_name, result in results.items():
        predictor = result["predictor"]
        history[f"{target_name}_probability"] = predictor.predict_proba(feature_values)[:, 1]
        history[f"{target_name}_threshold"] = predictor.threshold
        drivers.append(result["shap"])
    history["risk"] = history[["academic_risk_probability", "placement_risk_probability"]].max(axis=1)
    return history, pd.concat(drivers, ignore_index=True)


def _recommended_probability(result: dict[str, object], matrix: pd.DataFrame) -> np.ndarray:
    """Return a recommended model's offline probability, calibrated when needed."""
    if "predictor" in result:
        return result["predictor"].predict_proba(matrix)[:, 1]
    model_name = result["metrics"]["recommended_model"]
    model = result["models"][model_name]
    probability = model.predict_proba(matrix)[:, 1]
    if model_name == "xgboost":
        probability = result["calibrators"]["xgboost"].predict_proba(probability.reshape(-1, 1))[:, 1]
    return probability


def generate_what_if(
    unified: pd.DataFrame, priority: pd.DataFrame, results: dict[str, object], limit: int = 200
) -> pd.DataFrame:
    """Precompute one-at-a-time model sensitivity rows for priority students."""
    columns = ["student_id", "target", "action", "risk_before", "risk_after", "delta"]
    selected = priority.sort_values("priority_score", ascending=False).head(limit)
    latest = _latest(unified)
    feature_history = build_snapshot_features(unified)
    latest_features = feature_history.loc[latest.index].copy()
    latest_features["student_id"] = latest["student_id"].to_numpy()
    features_by_student = latest_features.drop_duplicates("student_id").set_index("student_id")
    student_ids = [student_id for student_id in selected.student_id if student_id in features_by_student.index]
    rows: list[dict[str, object]] = []
    for target, result in results.items():
        model_name = result["metrics"]["recommended_model"]
        model = result["models"][model_name]
        matrix = features_by_student.loc[student_ids].reindex(columns=list(model.feature_names_in_), fill_value=np.nan)
        before = _recommended_probability(result, matrix)
        for feature, increase, action in WHAT_IF_ACTIONS:
            changed = matrix.copy()
            if feature in changed:
                original = pd.to_numeric(changed[feature], errors="coerce")
                changed.loc[original.notna(), feature] = np.minimum(original[original.notna()] + increase, 100)
            after = _recommended_probability(result, changed)
            rows.extend(
                {
                    "student_id": student_id,
                    "target": target,
                    "action": action,
                    "risk_before": round(float(before[index]), 6),
                    "risk_after": round(float(after[index]), 6),
                    "delta": round(float(after[index] - before[index]), 6),
                }
                for index, student_id in enumerate(student_ids)
            )
    return pd.DataFrame(rows, columns=columns)


def map_playbook_actions(students: pd.DataFrame, drivers: pd.DataFrame) -> pd.DataFrame:
    """Map a student's highest SHAP drivers to deterministic rule-table actions."""
    driver_groups = {
        student_id: group.sort_values("rank") for student_id, group in drivers.groupby("student_id", sort=False)
    }
    actions = []
    for _, student in students.iterrows():
        selected = None
        for _, driver in driver_groups.get(student.student_id, pd.DataFrame()).iterrows():
            feature = str(driver.get("feature", ""))
            for rule in PLAYBOOK_RULES:
                value = pd.to_numeric(pd.Series([student.get(rule["field"])]), errors="coerce").iloc[0]
                if rule["driver"] in feature and pd.notna(value) and value < rule["limit"]:
                    selected = {**rule, "driver_feature": feature}
                    break
            if selected:
                break
        if selected is None:
            selected = {
                "intervention": "Continue monitoring",
                "owner": "Mentor",
                "timeline": "30d",
                "driver_feature": "No actionable top driver",
            }
        score = student.get("priority_score", np.nan)
        trend = student.get("trend", "stable")
        summary = (
            f"{student.student_id} has {trend} risk with priority {score:.1f}; {selected['intervention']} is recommended based on {selected['driver_feature']}."
            if pd.notna(score)
            else f"{student.student_id} has {trend} risk; {selected['intervention']} is recommended based on {selected['driver_feature']}."
        )
        actions.append(
            {
                "student_id": student.student_id,
                "recommended_action": selected["intervention"],
                "owner": selected["owner"],
                "timeline": selected["timeline"],
                "driver_feature": selected["driver_feature"],
                "summary": summary,
            }
        )
    return pd.DataFrame(actions)


def build_pipeline_outputs(unified: pd.DataFrame, outcomes: pd.DataFrame) -> dict[str, object]:
    """Run all offline stages after unification and return precomputed artifacts."""
    training_results = train_from_files(unified, outcomes)
    scores = score_students(unified)
    risks, drivers = _risk_history(unified, training_results)
    trajectories = calculate_trajectories(risks)
    latest = _latest(unified)
    latest_scores = _latest(scores).drop(columns="snapshot_month")
    latest_risks = _latest(risks).drop(columns=["snapshot_month", "cohort"])
    students = (
        latest[latest.cohort.eq("current")]
        .merge(latest_scores, on="student_id", how="left")
        .merge(latest_risks, on="student_id", how="left")
        .merge(trajectories, on="student_id", how="left")
    )
    worklist = ranked_worklist(students)
    assignments, profiles = segment_students(worklist)
    worklist = pd.concat([worklist.reset_index(drop=True), assignments.reset_index(drop=True)], axis=1)
    actions = map_playbook_actions(worklist, drivers)
    worklist = worklist.merge(actions, on="student_id", how="left")
    simulated_followups = generate_simulated_followups(worklist)
    what_if = generate_what_if(unified, worklist, training_results)
    metrics = {
        "data_note": "All model metrics and outputs are computed on SYNTHETIC data.",
        "evaluation_scope": "Untouched current synthetic cohort, not a later period or real-world validation",
        "risk_note": "Combined risk is max of two risks, not an either-outcome probability. Projection bands are descriptive residual spread, not confidence intervals.",
        "targets": {name: result["metrics"] for name, result in training_results.items()},
    }
    calibration = pd.concat([result["calibration"] for result in training_results.values()], ignore_index=True)
    predictions = pd.concat(
        [result["predictions"].assign(target=target_name) for target_name, result in training_results.items()],
        ignore_index=True,
    )
    return {
        "students": worklist,
        "scores": scores,
        "risks": risks,
        "drivers": drivers,
        "trajectories": trajectories,
        "priority": worklist,
        "segments": profiles,
        "actions": actions,
        "simulated_followups": simulated_followups,
        "what_if": what_if,
        "calibration": calibration,
        "predictions": predictions,
        "metrics": metrics,
        "models": training_results,
    }


def main() -> None:
    """Explicitly create a new immutable bundle without altering legacy artifacts."""
    import argparse
    import io

    from src.artifacts import digest, publish_bundle

    parser = argparse.ArgumentParser(description="Build all precomputed synthetic dashboard outputs.")
    parser.add_argument("--unified", default="data/clean/unified.csv")
    parser.add_argument("--outcomes", default="data/clean/outcomes.csv")
    parser.add_argument("--bundle-dir", required=True, type=Path, help="New, nonexistent private output directory")
    args = parser.parse_args()
    if args.bundle_dir.exists():
        parser.error("Bundle destination already exists; choose a new directory")
    inputs = {"unified": Path(args.unified).read_bytes(), "outcomes": Path(args.outcomes).read_bytes()}
    artifacts = build_pipeline_outputs(*(pd.read_csv(io.BytesIO(inputs[name])) for name in ("unified", "outcomes")))
    destination = publish_bundle(
        args.bundle_dir, artifacts, {"input_sha256": {k: digest(v) for k, v in inputs.items()}}
    )
    print(
        f"Published SYNTHETIC bundle at {destination}. Activation requires GRAVITY_ML_BUNDLE; no active data changed."
    )


if __name__ == "__main__":
    main()
