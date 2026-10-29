"""Small end-to-end check for the offline pipeline and read-only API."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd
from fastapi.testclient import TestClient

import api.main as api
from src import build_outputs
from src.clean import TABLES, clean_raw_tables
from src.generate_data import generate_data


def test_small_pipeline_and_all_api_endpoints(tmp_path, monkeypatch):
    """Build fresh runtime artifacts, then serve and infer from them."""
    raw_dir = tmp_path / "data" / "raw"
    clean_dir = tmp_path / "data" / "clean"
    output_dir = tmp_path / "data" / "outputs"
    raw_dir.mkdir(parents=True)
    clean_dir.mkdir(parents=True)
    output_dir.mkdir(parents=True)
    tables, historical, current = generate_data(n_historical=60, n_current=60)
    for name in TABLES:
        tables[name].to_csv(raw_dir / f"{name}.csv", index=False)
    historical.to_csv(raw_dir / "outcomes_historical.csv", index=False)
    current.to_csv(raw_dir / "outcomes_current_holdout.csv", index=False)
    unified, outcomes, cleaning_log = clean_raw_tables(raw_dir)
    unified.to_csv(clean_dir / "unified.csv", index=False)
    outcomes.to_csv(clean_dir / "outcomes.csv", index=False)
    (output_dir / "cleaning_log.json").write_text(json.dumps(cleaning_log), encoding="utf-8")
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "build_outputs",
            "--unified",
            str(clean_dir / "unified.csv"),
            "--outcomes",
            str(clean_dir / "outcomes.csv"),
            "--bundle-dir",
            str(tmp_path / "bundle"),
        ],
    )
    build_outputs.main()
    assert (tmp_path / "bundle" / "manifest.json").is_file()
    assert (tmp_path / "bundle" / "bundle.joblib").is_file()
    monkeypatch.setenv("GRAVITY_ML_BUNDLE", str(tmp_path / "bundle"))

    monkeypatch.setattr(api, "OUTPUT_DIR", output_dir)
    monkeypatch.setattr(api, "INTERVENTION_DB", tmp_path / "interventions.sqlite3")
    client = TestClient(api.app)
    student_id = pd.read_csv(tmp_path / "bundle" / "students.csv").iloc[0].student_id
    assert client.get("/health").status_code == 200
    assert client.get("/ready").json()["mode"] == "bundle"
    assert client.get("/summary").status_code == 200
    assert client.get("/students").status_code == 200
    assert client.get(f"/students/{student_id}").status_code == 200
    assert client.get(f"/students/{student_id}/what_if").status_code == 200
    assert client.get("/worklist").status_code == 200
    assert client.get("/segments").status_code == 200
    assert client.get("/quality").status_code == 200
    assert client.get("/metrics").status_code == 200
    weights = {
        "academic": 35,
        "attendance": 20,
        "lms": 15,
        "placement_readiness": 15,
        "engagement_skills": 10,
        "feedback": 5,
    }
    assert client.post("/weights", json={"weights": weights}).status_code == 200
    fixture = Path(__file__).parent / "fixtures" / "runtime_students.csv"
    upload = client.post("/data/upload", files={"file": (fixture.name, fixture.read_bytes(), "text/csv")})
    assert upload.status_code == 200
    assert client.get("/data/active").json()["mode"] == "user"
    runtime_detail = client.get("/students/U9001")
    assert runtime_detail.status_code == 200
    runtime_profile = runtime_detail.json()
    assert runtime_profile["score"]["success_score"] is not None
    assert runtime_profile["risks"]["academic_risk_probability"] is not None
    assert runtime_profile["risks"]["placement_risk_probability"] is not None
    assert runtime_profile["shap_drivers"]
    assert runtime_profile["actions"]
    action = runtime_profile["actions"][0]
    created = client.post(
        "/interventions",
        json={
            "student_id": "U9001",
            "type": action["recommended_action"],
            "owner": action["owner"],
            "start_date": "2026-10-08",
            "before_snapshot": "2026-06",
        },
    )
    assert created.status_code == 201
    assert client.get(f"/interventions/{created.json()['id']}").status_code == 200
