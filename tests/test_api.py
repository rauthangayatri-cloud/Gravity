import json

import pandas as pd
import pytest
from fastapi.testclient import TestClient

import api.main as api


@pytest.fixture(autouse=True)
def isolated_configuration(monkeypatch):
    """Developer environment choices must not redirect test artifact reads."""
    monkeypatch.delenv("GRAVITY_ML_BUNDLE", raising=False)


def test_cors_allowlist_accepts_explicit_deployment_origins():
    """Render-style frontend origins are parsed without weakening local defaults."""
    assert api.cors_allow_origins(" https://campus-ui.onrender.com/ , https://preview.example ") == [
        "https://campus-ui.onrender.com",
        "https://preview.example",
    ]
    assert api.cors_allow_origins("") == list(api.DEFAULT_CORS_ALLOW_ORIGINS)


def test_api_health_is_read_only_ready_check():
    assert api.health() == {"status": "ok"}


def test_students_summary_and_weight_scenario_use_precomputed_files(tmp_path, monkeypatch):
    students = pd.DataFrame(
        {
            "student_id": ["C0001", "C0002"],
            "department": ["Business", "Computer Science"],
            "segment": ["Steady performers", "Silent strugglers"],
            "success_score": [72.0, 46.0],
            "score_confidence": [1.0, 1.0],
            "score_tier": ["Green", "Red"],
            "priority_score": [30.0, 80.0],
            "academic_risk_probability": [0.2, 0.8],
            "placement_risk_probability": [0.3, 0.7],
            "academic_component": [0.8, 0.4],
            "attendance_component": [0.7, 0.5],
            "lms_component": [0.7, 0.4],
            "placement_readiness_component": [0.8, 0.3],
            "engagement_skills_component": [0.7, 0.5],
            "feedback_component": [0.8, 0.5],
        }
    )
    students.to_csv(tmp_path / "students.csv", index=False)
    pd.DataFrame({"student_id": ["C0001"], "summary": ["Synthetic summary"]}).to_csv(
        tmp_path / "actions.csv", index=False
    )
    pd.DataFrame(
        {
            "student_id": ["C0001"],
            "target": ["academic_risk"],
            "action": ["Attendance +10 points"],
            "risk_before": [0.2],
            "risk_after": [0.15],
            "delta": [-0.05],
        }
    ).to_csv(tmp_path / "what_if.csv", index=False)
    (tmp_path / "metrics.json").write_text(json.dumps({"data_note": "SYNTHETIC"}), encoding="utf-8")
    monkeypatch.setattr(api, "OUTPUT_DIR", tmp_path)
    monkeypatch.setattr(api, "INTERVENTION_DB", tmp_path / "interventions.sqlite3")
    client = TestClient(api.app)
    assert client.get("/summary").status_code == 200
    assert client.get("/students", params={"department": "Business"}).json()["total"] == 1
    response = client.post(
        "/weights",
        json={
            "weights": {
                "academic": 35,
                "attendance": 20,
                "lms": 15,
                "placement_readiness": 15,
                "engagement_skills": 10,
                "feedback": 5,
            }
        },
    )
    assert response.status_code == 200
    assert response.json()["rank_change"]["spearman_correlation"] is not None
    what_if = client.get("/students/C0001/what_if")
    assert what_if.status_code == 200
    assert "estimator can differ" in what_if.json()["data_note"]
    assert what_if.json()["items"] == []


def test_readiness_failure_does_not_create_a_database(tmp_path, monkeypatch):
    """Absent artifacts cannot report ready or initialize legacy state."""
    monkeypatch.setenv("GRAVITY_ML_BUNDLE", str(tmp_path / "missing-bundle"))
    monkeypatch.setattr(api, "INTERVENTION_DB", tmp_path / "must-not-exist.sqlite3")
    client = TestClient(api.app)
    assert client.get("/health").status_code == 200
    assert client.get("/ready").status_code == 503
    assert not api.INTERVENTION_DB.exists()


def test_interventions_are_persisted(tmp_path, monkeypatch):
    monkeypatch.setattr(api, "INTERVENTION_DB", tmp_path / "interventions.sqlite3")
    client = TestClient(api.app)
    created = client.post(
        "/interventions",
        json={
            "student_id": "C0001",
            "type": "Mentoring",
            "owner": "Faculty",
            "start_date": "2026-10-08",
            "before_snapshot": "2026-05",
        },
    )
    assert created.status_code == 201
    assert client.get(f"/interventions/{created.json()['id']}").json()["owner"] == "Faculty"
