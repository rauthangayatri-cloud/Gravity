"""End-to-end tests for persisted runtime student analysis."""

from __future__ import annotations

from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import pytest
import shap
from fastapi.testclient import TestClient
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from xgboost import XGBClassifier

import api.main as api
import src.runtime as runtime
from src.features import build_features
from src.runtime import (
    RuntimeValidationError,
    _drivers,
    _load_model,
    _model,
    _model_matrix,
    process_student_data,
    validate_student_data,
)


def runtime_students() -> pd.DataFrame:
    """Return unseen, deliberately varied students for runtime tests."""
    return pd.read_csv(Path(__file__).parent / "fixtures" / "runtime_students.csv")


@pytest.fixture(scope="module")
def legacy_models(tmp_path_factory):
    """Tiny seeded test-only models; no dependency on ignored developer artifacts."""
    directory = tmp_path_factory.mktemp("synthetic-legacy-models")
    source = pd.concat([runtime_students().iloc[[0]]] * 80, ignore_index=True)
    source["student_id"] = [f"synthetic-{i}" for i in range(len(source))]
    source["attendance_pct"] = np.linspace(30, 100, len(source))
    source["department"] = np.resize(sorted(runtime.SUPPORTED_DEPARTMENTS), len(source))
    features = build_features(validate_student_data(source))
    model = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="median", keep_empty_features=True)),
            ("model", XGBClassifier(n_estimators=20, max_depth=2, random_state=42, n_jobs=1)),
        ]
    )
    model.fit(features, source.attendance_pct.lt(65).astype(int))
    for target in ("academic_risk", "placement_risk"):
        joblib.dump(model, directory / f"{target}_xgboost.joblib")
    return directory


@pytest.fixture(autouse=True)
def isolated_runtime(legacy_models, tmp_path, monkeypatch):
    """All workflow writes and demo reads belong to this test, never the checkout."""
    monkeypatch.delenv("GRAVITY_ML_BUNDLE", raising=False)
    monkeypatch.setattr(runtime, "MODEL_DIR", legacy_models)
    monkeypatch.setattr(api, "INTERVENTION_DB", tmp_path / "runtime.sqlite3")
    monkeypatch.setattr(api, "OUTPUT_DIR", tmp_path)
    source = runtime_students().iloc[:2].copy()
    source["student_id"] = ["DEMO-1", "DEMO-2"]
    demo = process_student_data(source)
    for name in ("students", "drivers", "actions", "segments"):
        demo[name].to_csv(tmp_path / f"{name}.csv", index=False)
    return demo


def test_runtime_processing_uses_unseen_student_values():
    """Scores, risks, drivers, and actions are produced from new records."""
    processed = process_student_data(runtime_students())
    students = processed["students"].set_index("student_id")
    drivers = processed["drivers"]
    actions = processed["actions"].set_index("student_id")
    assert set(students.index) == set(runtime_students().student_id)
    assert students.loc["U9001", "success_score"] < students.loc["U9002", "success_score"]
    assert students.loc["U9001", "academic_risk_probability"] > students.loc["U9002", "academic_risk_probability"]
    assert len(drivers[drivers.student_id.eq("U9001")]) == 6
    assert actions.loc["U9001", "recommended_action"] == "Attendance recovery plan"


def test_legacy_readiness_is_read_only(tmp_path):
    """Readiness can load synthetic artifacts without creating runtime state."""
    assert TestClient(api.app).get("/ready").json()["mode"] == "legacy_demo"
    assert not (tmp_path / "runtime.sqlite3").exists()


def test_runtime_inference_preserves_feature_order_and_shap_driver_alignment():
    """Runtime prediction and local SHAP use the saved pipeline's exact matrix."""
    raw = validate_student_data(runtime_students())
    features = build_features(raw)
    model = _model("academic_risk")
    matrix = _model_matrix(model, features)
    assert list(matrix.columns) == list(model.feature_names_in_)
    processed = process_student_data(raw)
    expected_probability = model.predict_proba(matrix)[0, 1]
    actual_probability = processed["students"].iloc[0]["academic_risk_probability"]
    assert np.isclose(actual_probability, expected_probability)

    transformed = model.named_steps["imputer"].transform(matrix)
    names = model.named_steps["imputer"].get_feature_names_out(matrix.columns)
    values = np.asarray(shap.TreeExplainer(model.named_steps["model"])(transformed).values)
    if values.ndim == 3:
        values = values[:, :, -1]
    expected_feature = names[np.abs(values[0]).argmax()]
    drivers = _drivers(model, matrix, raw["student_id"], "academic_risk")
    first_driver = drivers.query("student_id == 'U9001' and rank == 1").iloc[0]
    assert first_driver["feature"] == expected_feature


def test_runtime_model_cache_is_scoped_to_the_resolved_artifact_path(tmp_path, monkeypatch):
    """A model cached for one isolated build cannot leak into another one."""
    first_dir = tmp_path / "first-models"
    second_dir = tmp_path / "second-models"
    for directory in (first_dir, second_dir):
        directory.mkdir()
        (directory / "academic_risk_xgboost.joblib").write_bytes(directory.name.encode())
    _load_model.cache_clear()
    monkeypatch.setattr(runtime.joblib, "load", lambda stream: stream.getvalue().decode())
    monkeypatch.setattr(runtime, "MODEL_DIR", first_dir)
    first = runtime._model("academic_risk")
    monkeypatch.setattr(runtime, "MODEL_DIR", second_dir)
    second = runtime._model("academic_risk")
    assert first != second
    assert first == "first-models"
    assert second == "second-models"
    (second_dir / "academic_risk_xgboost.joblib").write_bytes(b"updated")
    assert runtime._model("academic_risk") == "updated"
    _load_model.cache_clear()


def test_runtime_validation_rejects_bad_values_and_duplicates():
    """Validation reports invalid ranges and duplicate unseen IDs clearly."""
    invalid = runtime_students().iloc[:2].copy()
    invalid.loc[0, "attendance_pct"] = 120
    invalid.loc[1, "student_id"] = "U9001"
    try:
        validate_student_data(invalid)
    except RuntimeValidationError as error:
        message = " ".join(error.errors)
        assert "attendance_pct must be between 0 and 100" in message
        assert "duplicate student_id 'U9001'" in message
    else:
        raise AssertionError("Invalid runtime input should not be accepted")
    missing = runtime_students().drop(columns="mock_interview_score")
    try:
        validate_student_data(missing)
    except RuntimeValidationError as error:
        assert "missing required columns: mock_interview_score" in " ".join(error.errors)
    else:
        raise AssertionError("A missing model input should not be accepted")
    invalid_department = runtime_students().iloc[:1].copy()
    invalid_department.loc[0, "department"] = "Unknown Department"
    try:
        validate_student_data(invalid_department)
    except RuntimeValidationError as error:
        assert "department must be one of" in " ".join(error.errors)
    else:
        raise AssertionError("An unsupported department should not be accepted")


def test_upload_add_reset_and_dashboard_switching(tmp_path, monkeypatch, isolated_runtime):
    """Upload, add, persistence, validation, changed inputs, and reset use one active dataset flow."""
    monkeypatch.setattr(api, "INTERVENTION_DB", tmp_path / "runtime.sqlite3")
    client = TestClient(api.app)
    source = runtime_students()
    uploaded = client.post(
        "/data/upload",
        files={"file": ("judge_students.csv", source.to_csv(index=False), "text/csv")},
    )
    assert uploaded.status_code == 200
    assert uploaded.json()["student_count"] == 6
    assert client.get("/summary").json()["kpis"]["student_count"] == 6
    first = client.get("/students/U9001").json()
    assert first["score"]["success_score"] is not None
    assert first["risks"]["academic_risk_probability"] is not None
    assert first["shap_drivers"]
    assert first["actions"][0]["recommended_action"] == "Attendance recovery plan"
    assert first["trajectory"]["trajectory_points"] == 1
    assert client.get("/students/U9001/what_if").json()["items"] == []
    active_worklist = client.get("/worklist?top_n=25").json()
    assert {row["student_id"] for row in active_worklist} == set(source.student_id)
    reloaded = TestClient(api.app)
    assert reloaded.get("/data/active").json()["mode"] == "user"
    added = source.iloc[1].to_dict() | {"student_id": "U9007", "attendance_pct": 48}
    assert client.post("/data/students", json=added).status_code == 201
    assert client.get("/summary").json()["kpis"]["student_count"] == 7
    added_detail = client.get("/students/U9007").json()
    assert added_detail["score"]["success_score"] is not None
    assert added_detail["shap_drivers"]
    changed = source.copy()
    changed.loc[0, "attendance_pct"] = 96
    changed_upload = client.post(
        "/data/upload",
        files={"file": ("changed_students.csv", changed.to_csv(index=False), "text/csv")},
    )
    assert changed_upload.status_code == 200
    changed_student = client.get("/students/U9001").json()
    assert changed_student["score"]["success_score"] > first["score"]["success_score"]
    assert changed_student["risks"]["academic_risk_probability"] < first["risks"]["academic_risk_probability"]
    invalid = source.iloc[:1].copy()
    invalid.loc[0, "attendance_pct"] = 101
    rejected = client.post("/data/upload", files={"file": ("invalid.csv", invalid.to_csv(index=False), "text/csv")})
    assert rejected.status_code == 422
    assert "attendance_pct must be between 0 and 100" in str(rejected.json())
    reset = client.post("/data/reset")
    assert reset.status_code == 200
    assert reset.json()["mode"] == "demo"
    assert client.get("/summary").json()["kpis"]["student_count"] == len(isolated_runtime["students"])
    assert "U9001" not in {row["student_id"] for row in client.get("/worklist?top_n=25").json()}


def test_web_runtime_detail_does_not_fabricate_uploaded_history():
    """The detail page distinguishes an uploaded snapshot from a real time series."""
    source = (Path(__file__).parents[1] / "web" / "src" / "main.jsx").read_text(encoding="utf-8")
    assert "function sparkPoints" not in source
    assert "One uploaded snapshot is available; no trajectory is shown." in source
    assert "Current score components; no historical lines are fabricated." in source
    assert "const controller = new AbortController();" in source
    assert "faculty-department" in source
    assert "FACULTY_DEPARTMENT" not in source
    assert "datasetKey={activeDatasetKey}" in source
    assert "}, [id, datasetKey]);" in source
