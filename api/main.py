"""Serve demo artifacts plus lightweight saved-model runtime inference."""

from __future__ import annotations

import io
import json
import math
import os
import sqlite3
from contextlib import closing
from datetime import date
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from fastapi import FastAPI, File, HTTPException, Query, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from starlette.responses import JSONResponse

from src.artifacts import ArtifactError, configured_bundle, load_bundle, read_manifest, verified_bytes
from src.runtime import RUNTIME_MODEL_NOTE, RuntimeDatasetStore, RuntimeValidationError, _model, process_student_data

DEFAULT_CORS_ALLOW_ORIGINS = ("http://localhost:5173", "http://127.0.0.1:5173")


def cors_allow_origins(value: str | None = None) -> list[str]:
    """Return explicit CORS origins from deployment configuration or local defaults."""
    raw = os.getenv("CORS_ALLOW_ORIGINS", "") if value is None else value
    configured = [origin.strip().rstrip("/") for origin in raw.split(",") if origin.strip()]
    return configured or list(DEFAULT_CORS_ALLOW_ORIGINS)


OUTPUT_DIR = Path("data/outputs")
INTERVENTION_DB = Path(os.getenv("RUNTIME_STATE_DB", "api/interventions.sqlite3"))
CORS_ALLOW_ORIGINS = cors_allow_origins()
# Demo output is precomputed. Upload inference uses fitted artifacts only;
# no training, calibration fitting, or clustering fitting runs in a request.
app = FastAPI(title="Smart Campus Student Success API", version="0.2.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=CORS_ALLOW_ORIGINS,
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)


@app.exception_handler(ArtifactError)
async def artifact_error_handler(request, error):
    """Fail closed on incompatible artifacts without disclosing server paths."""
    return JSONResponse(
        status_code=503,
        content={"detail": "Configured ML bundle is unavailable or incompatible. Review server configuration."},
    )


class StudentsPage(BaseModel):
    items: list[dict[str, Any]]
    total: int
    page: int
    page_size: int


class StudentDetail(BaseModel):
    student_id: str
    score: dict[str, Any]
    risks: dict[str, Any]
    trajectory: dict[str, Any]
    shap_drivers: list[dict[str, Any]]
    segment: str | None = None
    actions: list[dict[str, Any]]
    summary_text: str | None = None
    simulated_followups: list[dict[str, Any]] = []


class WhatIfResponse(BaseModel):
    data_note: str
    items: list[dict[str, Any]]


class WeightRequest(BaseModel):
    weights: dict[str, float] = Field(description="Six Success Score component weights")


class InterventionCreate(BaseModel):
    student_id: str
    type: str
    owner: str
    start_date: date
    before_snapshot: str


class InterventionResponse(InterventionCreate):
    id: int


class ActiveDatasetResponse(BaseModel):
    mode: str
    dataset_id: str | None = None
    label: str
    student_count: int


class RuntimeDatasetResponse(ActiveDatasetResponse):
    message: str


def _csv(name: str, required: bool = True) -> pd.DataFrame:
    """Read one immutable offline demo artifact."""
    bundle = configured_bundle()
    if bundle is not None:
        manifest = read_manifest(bundle)
        if not required and f"{name}.csv" not in manifest["files"]:
            return pd.DataFrame()
        return pd.read_csv(io.BytesIO(verified_bytes(bundle, f"{name}.csv", manifest)))
    path = OUTPUT_DIR / f"{name}.csv"
    if not path.exists():
        if required:
            raise HTTPException(503, f"{path.name} is unavailable. Run the offline output build first.")
        return pd.DataFrame()
    return pd.read_csv(path)


def _json(name: str, required: bool = True) -> dict[str, Any]:
    """Read an offline JSON artifact."""
    bundle = configured_bundle()
    if bundle is not None:
        manifest = read_manifest(bundle)
        if not required and f"{name}.json" not in manifest["files"]:
            return {"available": False, "message": "Not included in this bundle"}
        return json.loads(verified_bytes(bundle, f"{name}.json", manifest))
    path = OUTPUT_DIR / f"{name}.json"
    if not path.exists():
        if required:
            raise HTTPException(503, f"{path.name} is unavailable. Run the offline output build first.")
        return {"available": False, "message": f"{path.name} was not produced by the offline pipeline."}
    return json.loads(path.read_text(encoding="utf-8"))


def _records(frame: pd.DataFrame) -> list[dict[str, Any]]:
    """Return JSON-compatible records with nulls instead of NaN."""
    return json.loads(frame.where(pd.notna(frame), None).to_json(orient="records"))


def _students() -> pd.DataFrame:
    """Read active runtime records, falling back to the immutable demo output."""
    active = _runtime_store().active()
    if active is not None:
        return active["students"]
    frame = _csv("students")
    if configured_bundle() is None:
        frame["runtime_model_note"] = (
            "Preserved legacy synthetic demo: offline risks are calibrated XGBoost, but legacy uploads are not. "
            "These outputs are not the new single-snapshot contract. " + RUNTIME_MODEL_NOTE
        )
    return frame


def _runtime_store() -> RuntimeDatasetStore:
    """Use the existing lightweight SQLite file for runtime dataset state."""
    return RuntimeDatasetStore(INTERVENTION_DB)


def _active_frame(name: str, required: bool = True) -> pd.DataFrame:
    """Read an active runtime artifact or the matching demo artifact."""
    active = _runtime_store().active()
    if active is not None:
        return active.get(name, pd.DataFrame()).copy()
    return _csv(name, required=required)


def _active_dataset_response() -> ActiveDatasetResponse:
    """Describe which isolated dataset current dashboard reads use."""
    active = _runtime_store().active()
    if active is None:
        return ActiveDatasetResponse(mode="demo", label="Demo data", student_count=int(len(_csv("students"))))
    return ActiveDatasetResponse(
        mode="user",
        dataset_id=active["dataset_id"],
        label=active["label"],
        student_count=int(len(active["students"])),
    )


def _runtime_validation_error(error: RuntimeValidationError) -> HTTPException:
    """Translate validation messages into a clear unprocessable-upload response."""
    return HTTPException(status_code=422, detail={"message": "Upload failed", "errors": error.errors})


def _normalise_tier(frame: pd.DataFrame) -> pd.Series:
    """Use the precomputed tier field when available."""
    return frame.get("score_tier", pd.Series("Unknown", index=frame.index)).fillna("Unknown")


def _initialise_store() -> None:
    """Create the lightweight intervention store if it does not exist."""
    INTERVENTION_DB.parent.mkdir(parents=True, exist_ok=True)
    with closing(sqlite3.connect(INTERVENTION_DB)) as connection:
        with connection:
            connection.execute(
                "CREATE TABLE IF NOT EXISTS interventions (id INTEGER PRIMARY KEY AUTOINCREMENT, student_id TEXT NOT NULL, type TEXT NOT NULL, owner TEXT NOT NULL, start_date TEXT NOT NULL, before_snapshot TEXT NOT NULL)"
            )


def _intervention_row(row: tuple[Any, ...]) -> InterventionResponse:
    """Convert a SQLite row into the public intervention response model."""
    return InterventionResponse(
        id=row[0],
        student_id=row[1],
        type=row[2],
        owner=row[3],
        start_date=date.fromisoformat(row[4]),
        before_snapshot=row[5],
    )


@app.get("/health", response_model=dict[str, str])
def health() -> dict[str, str]:
    """Report service readiness without inspecting or training models."""
    return {"status": "ok"}


@app.get("/ready", response_model=dict[str, str])
def ready() -> dict[str, str]:
    """Check demo/model availability without opening or creating any database."""
    try:
        bundle = configured_bundle()
        if bundle is not None:
            load_bundle(bundle)
        else:
            for target in ("academic_risk", "placement_risk"):
                _model(target)
        frame = _csv("students")
        if frame.empty or not {
            "student_id",
            "success_score",
            "academic_risk_probability",
            "placement_risk_probability",
        }.issubset(frame):
            raise ValueError("Missing demo fields")
    except (ArtifactError, OSError, ValueError, TypeError) as error:
        raise HTTPException(
            503, "Demo/model artifacts are unavailable or incompatible; no data was changed."
        ) from error
    return {
        "status": "ready",
        "mode": "bundle" if bundle is not None else "legacy_demo",
        "scope": "local synthetic demonstration only; no authentication or tenant isolation",
    }


@app.get("/data/active", response_model=ActiveDatasetResponse)
def active_dataset() -> ActiveDatasetResponse:
    """Report whether dashboard reads use demo or persisted user data."""
    return _active_dataset_response()


@app.post("/data/upload", response_model=RuntimeDatasetResponse)
async def upload_student_data(file: UploadFile = File(...)) -> RuntimeDatasetResponse:
    """Validate a CSV and make its scored student records the active dataset."""
    if not file.filename or not file.filename.casefold().endswith(".csv"):
        raise HTTPException(422, "Upload failed: provide a CSV file.")
    content = await file.read()
    if not content.strip():
        raise HTTPException(422, "Upload failed: the CSV file is empty.")
    try:
        uploaded = pd.read_csv(io.StringIO(content.decode("utf-8-sig")), skip_blank_lines=False)
    except UnicodeDecodeError as error:
        raise HTTPException(422, "Upload failed: CSV must be UTF-8 encoded.") from error
    except pd.errors.ParserError as error:
        raise HTTPException(422, f"Upload failed: CSV could not be parsed ({error}).") from error
    try:
        processed = process_student_data(uploaded)
    except RuntimeValidationError as error:
        raise _runtime_validation_error(error) from error
    except FileNotFoundError as error:
        raise HTTPException(503, str(error)) from error
    saved = _runtime_store().save(processed, kind="upload", label=file.filename)
    return RuntimeDatasetResponse(**saved, mode="user", message="Upload processed and set as the active dataset.")


@app.post("/data/students", response_model=RuntimeDatasetResponse, status_code=201)
def add_student(student: dict[str, Any]) -> RuntimeDatasetResponse:
    """Add one validated student through the same active runtime inference flow."""
    store = _runtime_store()
    active = store.active()
    incoming = pd.DataFrame([student])
    source = incoming if active is None else pd.concat([active["raw"], incoming], ignore_index=True, sort=False)
    try:
        processed = process_student_data(source)
    except RuntimeValidationError as error:
        raise _runtime_validation_error(error) from error
    except FileNotFoundError as error:
        raise HTTPException(503, str(error)) from error
    label = "Individually added students" if active is None else active["label"]
    kind = "individual" if active is None else active["kind"]
    saved = store.save(processed, kind=kind, label=label)
    return RuntimeDatasetResponse(**saved, mode="user", message="Student processed and added to the active dataset.")


@app.post("/data/reset", response_model=ActiveDatasetResponse)
def reset_to_demo() -> ActiveDatasetResponse:
    """Switch dashboard reads back to the untouched precomputed demo dataset."""
    _runtime_store().reset_demo()
    return _active_dataset_response()


@app.get("/summary", response_model=dict[str, Any])
def summary() -> dict[str, Any]:
    """Return KPI, tier, score-distribution, and department comparison summaries."""
    students = _students()
    scores = pd.to_numeric(students.get("success_score"), errors="coerce")
    risk_columns = [
        column for column in ("academic_risk_probability", "placement_risk_probability", "risk") if column in students
    ]
    mean_risk = (
        students[risk_columns].apply(pd.to_numeric, errors="coerce").max(axis=1).mean()
        if risk_columns
        else float("nan")
    )
    distribution = (
        pd.cut(
            scores, [-math.inf, 40, 50, 60, 70, 80, math.inf], labels=["<40", "40-49", "50-59", "60-69", "70-79", "80+"]
        )
        .value_counts(sort=False)
        .to_dict()
    )
    comparison_columns = [
        column
        for column in ("success_score", "priority_score", "academic_risk_probability", "placement_risk_probability")
        if column in students
    ]
    department = (
        students.groupby("department", dropna=False)[comparison_columns].mean(numeric_only=True).reset_index()
        if "department" in students
        else pd.DataFrame()
    )
    return {
        "kpis": {
            "student_count": int(len(students)),
            "mean_success_score": round(float(scores.mean()), 1) if scores.notna().any() else None,
            "mean_risk": round(float(mean_risk), 4) if pd.notna(mean_risk) else None,
            "high_priority_count": int(pd.to_numeric(students.get("priority_score"), errors="coerce").ge(60).sum()),
        },
        "tier_counts": {str(key): int(value) for key, value in _normalise_tier(students).value_counts().items()},
        "score_distribution": {str(key): int(value) for key, value in distribution.items()},
        "department_comparison": _records(department),
        "data_note": "All KPIs are computed from precomputed SYNTHETIC data.",
    }


@app.get("/students", response_model=StudentsPage)
def students(
    department: str | None = None,
    tier: str | None = None,
    segment: str | None = None,
    search: str | None = None,
    page: int = Query(1, ge=1),
    page_size: int = Query(25, ge=1, le=100),
) -> StudentsPage:
    """List filtered, paginated current students from precomputed outputs."""
    frame = _students().copy()
    if department and "department" in frame:
        frame = frame[frame.department.astype(str).str.casefold() == department.casefold()]
    if tier:
        frame = frame[_normalise_tier(frame).str.casefold() == tier.casefold()]
    if segment and "segment" in frame:
        frame = frame[frame.segment.astype(str).str.casefold() == segment.casefold()]
    if search:
        needle = search.casefold()
        searchable = (
            frame.get("student_id", pd.Series("", index=frame.index))
            .astype(str)
            .str.casefold()
            .str.contains(needle, regex=False)
        )
        if "department" in frame:
            searchable |= frame.department.astype(str).str.casefold().str.contains(needle, regex=False)
        frame = frame[searchable]
    total = len(frame)
    start = (page - 1) * page_size
    return StudentsPage(
        items=_records(frame.iloc[start : start + page_size]), total=total, page=page, page_size=page_size
    )


@app.get("/students/{student_id}", response_model=StudentDetail)
def student(student_id: str) -> StudentDetail:
    """Return the complete detail view for the selected active student."""
    students_frame = _students()
    found = students_frame[students_frame.student_id.astype(str) == student_id]
    if found.empty:
        raise HTTPException(404, "Student not found")
    record = _records(found.iloc[[0]])[0]
    drivers = _active_frame("drivers", required=False)
    trajectories = _active_frame("trajectories", required=False)
    actions = _active_frame("actions", required=False)
    followups = _active_frame("simulated_followups", required=False)
    contributions = {
        key: value for key, value in record.items() if key.endswith("_component") or key.endswith("_contribution")
    }
    risks = {
        key: record.get(key)
        for key in (
            "academic_risk_probability",
            "placement_risk_probability",
            "risk",
            "projected_risk",
            "runtime_model_note",
            "model_bundle_id",
            "academic_risk_threshold",
            "placement_risk_threshold",
        )
        if key in record
    }
    runtime_trajectory = {
        key: record.get(key)
        for key in ("risk_slope", "projected_risk", "risk_band_low", "risk_band_high", "trend", "trajectory_points")
        if key in record
    }
    trajectory = (
        _records(trajectories[trajectories.student_id.astype(str) == student_id])[0]
        if not trajectories.empty and (trajectories.student_id.astype(str) == student_id).any()
        else runtime_trajectory
    )
    student_actions = _records(actions[actions.student_id.astype(str) == student_id]) if not actions.empty else []
    summary_text = record.get("summary") or (student_actions[0].get("summary") if student_actions else None)
    return StudentDetail(
        student_id=student_id,
        score={
            "success_score": record.get("success_score"),
            "confidence": record.get("score_confidence"),
            "tier": record.get("score_tier"),
            "status": record.get("score_status"),
            "components": contributions,
        },
        risks=risks,
        trajectory=trajectory,
        shap_drivers=_records(drivers[drivers.student_id.astype(str) == student_id]) if not drivers.empty else [],
        segment=record.get("segment"),
        actions=student_actions,
        summary_text=summary_text,
        simulated_followups=_records(followups[followups.student_id.astype(str) == student_id])
        if not followups.empty
        else [],
    )


@app.get("/students/{student_id}/what_if", response_model=WhatIfResponse)
def student_what_if(student_id: str) -> WhatIfResponse:
    """Return active-dataset sensitivity rows when that artifact is available."""
    if not (_students().student_id.astype(str) == student_id).any():
        raise HTTPException(404, "Student not found")
    if configured_bundle() is None:
        return WhatIfResponse(
            data_note="Legacy what-if comparisons are unavailable: their estimator can differ from dashboard risk. Select a reviewed bundle for consistent, non-causal sensitivity analysis.",
            items=[],
        )
    frame = _active_frame("what_if", required=False)
    rows = _records(frame[frame.student_id.astype(str) == student_id]) if not frame.empty else []
    return WhatIfResponse(
        data_note="Model-based sensitivity on synthetic data, not a causal estimate.",
        items=rows,
    )


@app.get("/worklist", response_model=list[dict[str, Any]])
def worklist(top_n: int = Query(50, ge=1, le=500)) -> list[dict[str, Any]]:
    """Return the highest priority rows from the active dataset."""
    frame = _active_frame("priority", required=False)
    if frame.empty:
        frame = _students()
    return _records(frame.sort_values("priority_score", ascending=False).head(top_n))


@app.get("/segments", response_model=list[dict[str, Any]])
def segments() -> list[dict[str, Any]]:
    """Return segment profiles for the active dataset."""
    return _records(_active_frame("segments"))


@app.get("/quality", response_model=dict[str, Any])
def quality() -> dict[str, Any]:
    """Return the precomputed cleaning log when the cleaning stage provides it."""
    active = _runtime_store().active()
    if active is not None:
        return active["quality"]
    for name in ("quality", "cleaning_log"):
        value = _json(name, required=False)
        if value.get("available", True):
            return value
    return {"available": False, "message": "No precomputed cleaning log is available yet."}


@app.get("/metrics", response_model=dict[str, Any])
def metrics() -> dict[str, Any]:
    """Never attribute another bundle's evaluation to persisted runtime results."""
    active = _runtime_store().active()
    bundle = configured_bundle()
    bundle_id = load_bundle(bundle)[1] if bundle is not None else "legacy"
    if active is not None and (bundle is None or active["quality"].get("model_bundle_id") != bundle_id):
        return {
            "available": False,
            "targets": {},
            "data_note": "No matching evaluation for this persisted dataset's model version. Legacy/unmatched runtime metrics are unavailable.",
        }
    report = _json("metrics")
    report["model_bundle_id"] = bundle_id
    report["evaluation_scope"] = "Offline synthetic evaluation, not measured performance on uploaded students"
    if bundle is None:
        report["data_note"] += (
            " Legacy demo artifacts: runtime calibration and history differ; not an approved runtime evaluation."
        )
    return report


@app.post("/weights", response_model=dict[str, Any])
def weights(request: WeightRequest) -> dict[str, Any]:
    """Recompute a score scenario solely from exported component values."""
    required = {"academic", "attendance", "lms", "placement_readiness", "engagement_skills", "feedback"}
    if (
        set(request.weights) != required
        or any(value < 0 for value in request.weights.values())
        or sum(request.weights.values()) <= 0
    ):
        raise HTTPException(422, "weights must contain six non-negative component weights with a positive total")
    frame = _students().copy()
    total = sum(request.weights.values())
    missing_components = [name for name in required if f"{name}_component" not in frame]
    if missing_components:
        raise HTTPException(503, f"Precomputed component columns are missing: {missing_components}")
    values = pd.DataFrame({name: pd.to_numeric(frame[f"{name}_component"], errors="coerce") for name in required})
    available = values.notna().mul(pd.Series(request.weights), axis=1).sum(axis=1)
    score = values.mul(pd.Series(request.weights), axis=1).sum(axis=1).div(available.replace(0, np.nan)).mul(100)
    valid = available.div(total).ge(0.5)
    score = score.where(valid)
    prior = pd.to_numeric(frame.get("success_score"), errors="coerce")
    paired = pd.DataFrame({"prior": prior, "scenario": score}).dropna()
    rank_correlation = (
        float(paired.prior.rank().corr(paired.scenario.rank(), method="spearman")) if len(paired) > 1 else None
    )
    rank_change = (paired.prior.rank() - paired.scenario.rank()).abs()
    tiers = pd.cut(score, [-math.inf, 50, 70, math.inf], labels=["Red", "Amber", "Green"], right=False)
    return {
        "weights": request.weights,
        "score_distribution": {
            str(key): int(value)
            for key, value in pd.cut(
                score, [-math.inf, 40, 60, 80, math.inf], labels=["<40", "40-60", "60-80", "80+"], right=False
            )
            .value_counts(sort=False)
            .items()
        },
        "tier_counts": {str(key): int(value) for key, value in tiers.value_counts().items()},
        "rank_change": {
            "spearman_correlation": round(rank_correlation, 4) if rank_correlation is not None else None,
            "mean_absolute_rank_change": round(float(rank_change.mean()), 2) if len(rank_change) else None,
            "max_absolute_rank_change": int(rank_change.max()) if len(rank_change) else None,
        },
        "note": "Scenario only: recomputed from precomputed components; no model training or inference occurred.",
    }


@app.post("/interventions", response_model=InterventionResponse, status_code=201)
def create_intervention(intervention: InterventionCreate) -> InterventionResponse:
    """Persist a manual intervention record without modifying scores or models."""
    _initialise_store()
    with closing(sqlite3.connect(INTERVENTION_DB)) as connection:
        with connection:
            cursor = connection.execute(
                "INSERT INTO interventions (student_id, type, owner, start_date, before_snapshot) VALUES (?, ?, ?, ?, ?)",
                (
                    intervention.student_id,
                    intervention.type,
                    intervention.owner,
                    intervention.start_date.isoformat(),
                    intervention.before_snapshot,
                ),
            )
            identifier = int(cursor.lastrowid)
    return InterventionResponse(id=identifier, **intervention.model_dump())


@app.get("/interventions/{intervention_id}", response_model=InterventionResponse)
def intervention(intervention_id: int) -> InterventionResponse:
    """Fetch one manually recorded intervention."""
    _initialise_store()
    with closing(sqlite3.connect(INTERVENTION_DB)) as connection:
        row = connection.execute(
            "SELECT id, student_id, type, owner, start_date, before_snapshot FROM interventions WHERE id = ?",
            (intervention_id,),
        ).fetchone()
    if row is None:
        raise HTTPException(404, "Intervention not found")
    return _intervention_row(row)
