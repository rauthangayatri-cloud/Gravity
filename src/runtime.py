"""Validate, persist, and score user-supplied students with saved models."""

from __future__ import annotations

import json
import sqlite3
import uuid
from contextlib import closing
from functools import lru_cache
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
import shap

from src.artifacts import configured_bundle, digest, load_bundle
from src.build_outputs import map_playbook_actions
from src.clean import DEPARTMENT_NAMES, RANGES
from src.features import INDICATORS, build_features, build_snapshot_features
from src.priority import ranked_worklist
from src.score import score_students
from src.segment import _legacy_assign_segments

MODEL_DIR = Path("models")
DEFAULT_SNAPSHOT_MONTH = "2026-06-30"
SUPPORTED_DEPARTMENTS = frozenset(DEPARTMENT_NAMES.values())
REQUIRED_COLUMNS = (
    "student_id",
    "department",
    "cgpa",
    "active_backlogs",
    "attendance_pct",
    "lms_logins",
    "assignment_completion_pct",
    "coding_score",
    "aptitude_score",
    "mock_interview_score",
)
ALIASES = {
    "attendance": "attendance_pct",
    "attendance_percentage": "attendance_pct",
    "backlogs": "active_backlogs",
    "assignment_score": "assignment_completion_pct",
    "assignment_completion": "assignment_completion_pct",
    "lms_activity": "lms_logins",
    "logins": "lms_logins",
    "placement_test_score": "aptitude_score",
    "placement_test": "aptitude_score",
    "mock_interview": "mock_interview_score",
    "coding": "coding_score",
    "communication": "communication_score",
    "feedback": "feedback_score",
}
RUNTIME_MODEL_NOTE = (
    "Runtime predictions use the saved XGBoost pipelines. Historical probability calibrators were not serialized, "
    "so uploaded-student XGBoost probabilities are uncalibrated."
    " LEGACY DEMO ONLY: training used historical features absent from uploads; runtime accuracy is not validated."
)


class RuntimeValidationError(ValueError):
    """Carry clear, row-level validation messages for API responses."""

    def __init__(self, errors: list[str]):
        super().__init__("; ".join(errors))
        self.errors = errors


def _header(value: object) -> str:
    """Normalise a user-provided CSV header into a canonical field name."""
    return "_".join(str(value).strip().casefold().replace("-", " ").split())


def _number(values: pd.Series) -> pd.Series:
    """Parse numeric input while accepting percentage strings such as ``78%``."""
    return pd.to_numeric(values.astype(str).str.strip().str.replace("%", "", regex=False), errors="coerce")


def _records(frame: pd.DataFrame) -> list[dict[str, Any]]:
    """Convert a frame to JSON-safe records with null rather than NaN values."""
    return json.loads(frame.where(pd.notna(frame), None).to_json(orient="records"))


def _frame(records: str | None) -> pd.DataFrame:
    """Restore a stored JSON record list to a DataFrame."""
    return pd.DataFrame(json.loads(records or "[]"))


def _canonical_department(values: pd.Series) -> pd.Series:
    """Map accepted department spellings to the four model departments."""
    text = values.astype(str).str.strip()
    return text.str.casefold().map(DEPARTMENT_NAMES).fillna(text)


def validate_student_data(frame: pd.DataFrame) -> pd.DataFrame:
    """Validate and normalise uploaded or form-entered student records.

    The returned records contain only known model/score fields. Bad values are
    rejected with explicit messages; no input value is silently fabricated.
    """
    if frame.empty or not len(frame.columns):
        raise RuntimeValidationError(["Upload failed: no student rows were supplied."])
    data = frame.copy()
    empty_rows = data.apply(lambda row: row.isna() | row.astype(str).str.strip().eq(""), axis=1).all(axis=1)
    errors = [f"Row {index + 2}: empty row." for index in data.index[empty_rows]]
    if empty_rows.any():
        raise RuntimeValidationError(errors)
    renamed = {column: ALIASES.get(_header(column), _header(column)) for column in data.columns}
    if len(set(renamed.values())) != len(renamed):
        raise RuntimeValidationError(["Upload failed: two supplied columns map to the same field."])
    data = data.rename(columns=renamed)
    missing = [column for column in REQUIRED_COLUMNS if column not in data]
    if missing:
        raise RuntimeValidationError([f"Upload failed: missing required columns: {', '.join(missing)}."])
    for column in REQUIRED_COLUMNS:
        blank = data[column].isna() | data[column].astype(str).str.strip().eq("")
        errors.extend(f"Row {index + 2}: {column} is required." for index in data.index[blank])
    student_ids = data["student_id"].astype(str).str.strip()
    duplicate = student_ids.str.casefold().duplicated(keep=False) & student_ids.ne("")
    errors.extend(
        f"Row {index + 2}: duplicate student_id '{student_ids.loc[index]}'." for index in data.index[duplicate]
    )
    departments = _canonical_department(data["department"])
    invalid_departments = ~departments.isin(SUPPORTED_DEPARTMENTS)
    errors.extend(
        f"Row {index + 2}: department must be one of {', '.join(sorted(SUPPORTED_DEPARTMENTS))}."
        for index in data.index[invalid_departments]
    )
    numeric_columns = [column for column in data.columns if column in RANGES]
    for column in numeric_columns:
        original = data[column]
        parsed = _number(original)
        supplied = original.notna() & original.astype(str).str.strip().ne("")
        invalid_number = supplied & parsed.isna()
        errors.extend(f"Row {index + 2}: {column} must be numeric." for index in data.index[invalid_number])
        low, high = RANGES[column]
        invalid_range = parsed.notna() & ~parsed.between(low, high)
        errors.extend(
            f"Row {index + 2}: {column} must be between {low} and {high}." for index in data.index[invalid_range]
        )
        data[column] = parsed
    if "snapshot_month" in data:
        parsed_dates = pd.to_datetime(data["snapshot_month"], errors="coerce")
        invalid_dates = data["snapshot_month"].notna() & parsed_dates.isna()
        errors.extend(f"Row {index + 2}: snapshot_month must be a valid date." for index in data.index[invalid_dates])
        data["snapshot_month"] = parsed_dates.dt.strftime("%Y-%m-%d")
    if errors:
        raise RuntimeValidationError(errors)
    data["student_id"] = student_ids
    data["department"] = departments
    if "snapshot_month" not in data:
        data["snapshot_month"] = DEFAULT_SNAPSHOT_MONTH
    else:
        data["snapshot_month"] = data["snapshot_month"].fillna(DEFAULT_SNAPSHOT_MONTH)
    data["cohort"] = "runtime"
    for indicator in INDICATORS:
        if indicator not in data:
            data[indicator] = np.nan
    for column in ("year", "section"):
        if column not in data:
            data[column] = "Not supplied"
    selected = ["student_id", "snapshot_month", "department", "year", "section", "cohort", *INDICATORS]
    extras = [
        column
        for column in ("average_mark", "subject_marks_std", "subject_attendance_min", "weakest_subject")
        if column in data
    ]
    return data[selected + extras].reset_index(drop=True)


@lru_cache(maxsize=4)
def _load_legacy_content(content: bytes):
    """Cache trusted legacy model contents, not potentially replaced paths."""
    import io

    return joblib.load(io.BytesIO(content))


def _load_model(path: str):
    """Re-read content identity before using the legacy model cache."""
    return _load_legacy_content(Path(path).read_bytes())


_load_model.cache_clear = _load_legacy_content.cache_clear


def _model(target: str):
    """Load one saved XGBoost pipeline for lightweight inference."""
    path = (MODEL_DIR / f"{target}_xgboost.joblib").resolve()
    if not path.exists():
        raise FileNotFoundError(f"Saved model is unavailable: {path}")
    return _load_model(str(path))


def _model_matrix(model: Any, feature_frame: pd.DataFrame) -> pd.DataFrame:
    """Recreate the trained model's exact ordered feature matrix."""
    return feature_frame.reindex(columns=list(model.feature_names_in_), fill_value=np.nan)


def _drivers(model: Any, matrix: pd.DataFrame, student_ids: pd.Series, target: str) -> pd.DataFrame:
    """Compute local top-three SHAP drivers from the saved XGBoost estimator."""
    imputer = model.named_steps["imputer"]
    transformed = imputer.transform(matrix)
    names = imputer.get_feature_names_out(matrix.columns)
    values = np.asarray(shap.TreeExplainer(model.named_steps["model"])(transformed).values)
    if values.ndim == 3:
        values = values[:, :, -1]
    rows: list[dict[str, Any]] = []
    for position, student_id in enumerate(student_ids):
        top = np.argsort(np.abs(values[position]))[-3:][::-1]
        for rank, feature_index in enumerate(top, start=1):
            value = float(values[position, feature_index])
            rows.append(
                {
                    "student_id": student_id,
                    "target": target,
                    "rank": rank,
                    "feature": names[feature_index],
                    "direction": "increases risk" if value > 0 else "reduces risk",
                    "magnitude": round(abs(value), 5),
                }
            )
    return pd.DataFrame(rows)


def _runtime_segments(students: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Use fixed descriptive rules; never fit clustering/scaling on uploaded data."""
    labels = _legacy_assign_segments(students)
    names = ["Urgent support", "Watchlist", "On track", "Developing"]
    assignments = pd.DataFrame(
        {"segment_id": labels.map({n: i for i, n in enumerate(names)}), "segment": labels}, index=students.index
    )
    profile_columns = [
        column
        for column in (
            "cgpa",
            "attendance_pct",
            "lms_logins",
            "engagement_score",
            "placement_readiness",
            "coding_score",
        )
        if column in students
    ]
    profiles = students[profile_columns].apply(pd.to_numeric, errors="coerce").join(assignments)
    grouped = profiles.groupby(["segment_id", "segment"], sort=True)
    result = grouped[profile_columns].mean().join(grouped.size().rename("size")).reset_index()
    result["method"] = "Fixed descriptive rules, not KMeans or a validated decision policy"
    return assignments, result


def process_student_data(frame: pd.DataFrame) -> dict[str, Any]:
    """Validate, score, infer, explain, prioritise, and recommend for new students."""
    raw = validate_student_data(frame)
    bundle = configured_bundle()
    predictors, bundle_id = load_bundle(bundle) if bundle is not None else ({}, "legacy")
    features = build_snapshot_features(raw) if bundle is not None else build_features(raw)
    students = raw.reset_index(drop=True).copy()
    scored = score_students(raw).drop(columns=["student_id", "snapshot_month"], errors="ignore")
    students = pd.concat([students, scored.reset_index(drop=True)], axis=1)
    all_drivers = []
    for target in ("academic_risk", "placement_risk"):
        if bundle is not None:
            predictor = predictors[target]
            students[f"{target}_probability"] = predictor.predict_proba(features)[:, 1]
            students[f"{target}_threshold"] = predictor.threshold
            all_drivers.append(predictor.drivers(features, students["student_id"]))
        else:
            model = _model(target)
            matrix = _model_matrix(model, features)
            students[f"{target}_probability"] = model.predict_proba(matrix)[:, 1]
            drivers = _drivers(model, matrix, students["student_id"], target)
            drivers["model"] = "legacy_xgboost"
            drivers["units"] = "base-model log-odds"
            all_drivers.append(drivers)
            students[f"{target}_artifact_sha256"] = digest((MODEL_DIR / f"{target}_xgboost.joblib").read_bytes())
    students["risk"] = students[["academic_risk_probability", "placement_risk_probability"]].max(axis=1)
    students["risk_slope"] = 0.0
    students["projected_risk"] = students["risk"].round(4)
    students["risk_band_low"] = np.nan
    students["risk_band_high"] = np.nan
    students["trend"] = "stable"
    students["trajectory_points"] = 1
    students["model_bundle_id"] = bundle_id
    students["runtime_model_note"] = (
        "Selected, calibrated single-snapshot models; evaluated only on synthetic students. "
        "Combined risk is a max-risk index, not an either-outcome probability. No forecast or uncertainty interval."
        if bundle is not None
        else RUNTIME_MODEL_NOTE
    )
    worklist = ranked_worklist(students)
    assignments, profiles = _runtime_segments(worklist)
    worklist = pd.concat([worklist.reset_index(drop=True), assignments.reset_index(drop=True)], axis=1)
    drivers = pd.concat(all_drivers, ignore_index=True)
    actions = map_playbook_actions(worklist, drivers)
    worklist = worklist.merge(actions, on="student_id", how="left")
    quality = {
        "available": True,
        "source": "runtime upload",
        "student_count": int(len(worklist)),
        "model_bundle_id": bundle_id,
        "validation": "Values were validated before runtime inference; no missing or out-of-range values were fabricated.",
    }
    return {
        "raw": raw,
        "students": worklist,
        "drivers": drivers,
        "actions": actions,
        "segments": profiles,
        "quality": quality,
    }


class RuntimeDatasetStore:
    """Persist active user datasets separately from immutable demo CSV artifacts."""

    def __init__(self, database: Path):
        self.database = database

    def _connect(self) -> sqlite3.Connection:
        self.database.parent.mkdir(parents=True, exist_ok=True)
        return sqlite3.connect(self.database)

    def _initialise(self) -> None:
        with closing(self._connect()) as connection:
            with connection:
                connection.execute(
                    "CREATE TABLE IF NOT EXISTS runtime_datasets (dataset_id TEXT PRIMARY KEY, kind TEXT NOT NULL, label TEXT NOT NULL, raw_json TEXT NOT NULL, students_json TEXT NOT NULL, drivers_json TEXT NOT NULL, actions_json TEXT NOT NULL, segments_json TEXT NOT NULL, quality_json TEXT NOT NULL)"
                )
                connection.execute(
                    "CREATE TABLE IF NOT EXISTS runtime_state (key TEXT PRIMARY KEY, value TEXT NOT NULL)"
                )

    def active(self) -> dict[str, Any] | None:
        """Return the selected user dataset, or ``None`` when demo is active."""
        self._initialise()
        with closing(self._connect()) as connection:
            selected = connection.execute("SELECT value FROM runtime_state WHERE key = 'active_dataset_id'").fetchone()
            if selected is None or selected[0] == "demo":
                return None
            row = connection.execute(
                "SELECT dataset_id, kind, label, raw_json, students_json, drivers_json, actions_json, segments_json, quality_json FROM runtime_datasets WHERE dataset_id = ?",
                (selected[0],),
            ).fetchone()
        if row is None:
            return None
        return {
            "dataset_id": row[0],
            "kind": row[1],
            "label": row[2],
            "raw": _frame(row[3]),
            "students": _frame(row[4]),
            "drivers": _frame(row[5]),
            "actions": _frame(row[6]),
            "segments": _frame(row[7]),
            "quality": json.loads(row[8]),
        }

    def save(self, processed: dict[str, Any], kind: str, label: str) -> dict[str, Any]:
        """Store processed runtime results and make them the active dataset."""
        self._initialise()
        dataset_id = f"runtime-{uuid.uuid4().hex}"
        payload = (
            dataset_id,
            kind,
            label,
            json.dumps(_records(processed["raw"])),
            json.dumps(_records(processed["students"])),
            json.dumps(_records(processed["drivers"])),
            json.dumps(_records(processed["actions"])),
            json.dumps(_records(processed["segments"])),
            json.dumps(processed["quality"]),
        )
        with closing(self._connect()) as connection:
            with connection:
                connection.execute(
                    "INSERT INTO runtime_datasets (dataset_id, kind, label, raw_json, students_json, drivers_json, actions_json, segments_json, quality_json) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
                    payload,
                )
                connection.execute(
                    "INSERT OR REPLACE INTO runtime_state (key, value) VALUES ('active_dataset_id', ?)", (dataset_id,)
                )
        return {
            "dataset_id": dataset_id,
            "kind": kind,
            "label": label,
            "student_count": int(len(processed["students"])),
        }

    def reset_demo(self) -> None:
        """Switch active reads back to the untouched precomputed demo dataset."""
        self._initialise()
        with closing(self._connect()) as connection:
            with connection:
                connection.execute(
                    "INSERT OR REPLACE INTO runtime_state (key, value) VALUES ('active_dataset_id', 'demo')"
                )
