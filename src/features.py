"""Build time-safe risk-model features from a unified snapshot table."""

from __future__ import annotations

import re

import numpy as np
import pandas as pd

INDICATORS = (
    "cgpa",
    "active_backlogs",
    "attendance_pct",
    "lms_logins",
    "assignment_completion_pct",
    "engagement_score",
    "placement_readiness",
    "mock_interview_score",
    "aptitude_score",
    "coding_score",
    "communication_score",
    "hackathons",
    "certifications",
    "feedback_score",
    "support_rating",
)
OUTCOME_TOKENS = (
    "outcome",
    "next_semester",
    "new_backlog",
    "placement_result",
    "offer_amount",
    "risk_label",
    "academic_risk",
    "placement_risk",
    "target",
)

SNAPSHOT_SCHEMA = "single-snapshot-v1"
MODEL_DEPARTMENTS = ("Business", "Computer Science", "Digital Media", "Mechanical Engineering")


def build_snapshot_features(frame: pd.DataFrame) -> pd.DataFrame:
    """Use one fixed current-state schema offline and online; no history is imputed."""
    if "department" not in frame or not frame.department.isin(MODEL_DEPARTMENTS).all():
        raise ValueError("Unsupported or missing model department")
    result = pd.DataFrame(index=frame.index)
    for indicator in INDICATORS:
        values = _number(frame[indicator]) if indicator in frame else pd.Series(np.nan, index=frame.index)
        if np.isinf(values.to_numpy(dtype=float)).any():
            raise ValueError(f"Non-finite model input: {indicator}")
        result[f"{indicator}_current"] = values
        result[f"{indicator}_missing"] = values.isna().astype("int8")
    for department in MODEL_DEPARTMENTS:
        result[f"department_{_safe_name(department)}"] = frame.department.eq(department).astype("int8")
    assert_no_outcome_features(result)
    return result


def assert_no_outcome_features(features: pd.DataFrame) -> None:
    """Fail if an outcome-derived column has reached the feature matrix."""
    leaked = [column for column in features.columns if any(token in column.lower() for token in OUTCOME_TOKENS)]
    if leaked:
        raise AssertionError(f"Outcome-derived feature columns are forbidden: {leaked}")


def _number(values: pd.Series) -> pd.Series:
    """Parse numeric fields, including raw percentage strings."""
    return pd.to_numeric(values.astype(str).str.replace("%", "", regex=False), errors="coerce")


def _three_month_slope(values: pd.Series) -> pd.Series:
    """Compute a trailing three-observation linear slope without future rows."""

    def slope(window: np.ndarray) -> float:
        valid = ~np.isnan(window)
        if valid.sum() < 2:
            return np.nan
        positions = np.arange(len(window))[valid]
        return float(np.polyfit(positions, window[valid], 1)[0])

    return values.rolling(3, min_periods=2).apply(slope, raw=True)


def _safe_name(value: str) -> str:
    """Produce a stable column-safe department suffix."""
    return re.sub(r"[^a-z0-9]+", "_", value.lower()).strip("_")


def build_features(frame: pd.DataFrame) -> pd.DataFrame:
    """Build current, trailing-slope, volatility, missingness, and department features.

    Input must contain one row per student/snapshot and `snapshot_month`. Every
    slope and volatility is calculated with the current row and its two prior
    rows only. Outcome fields in the input are ignored rather than propagated.
    """
    required = {"student_id", "snapshot_month", "department"}
    missing = required.difference(frame.columns)
    if missing:
        raise ValueError(f"Unified data is missing required columns: {sorted(missing)}")
    ordered = frame.copy()
    ordered["_row_order"] = np.arange(len(ordered))
    ordered["snapshot_month"] = pd.to_datetime(ordered["snapshot_month"])
    ordered = ordered.sort_values(["student_id", "snapshot_month", "_row_order"], kind="stable")
    result = pd.DataFrame(index=ordered.index)
    for indicator in INDICATORS:
        values = _number(ordered[indicator]) if indicator in ordered else pd.Series(np.nan, index=ordered.index)
        grouped = values.groupby(ordered["student_id"], sort=False)
        result[f"{indicator}_current"] = values
        result[f"{indicator}_slope_3m"] = grouped.transform(_three_month_slope)
        result[f"{indicator}_volatility_3m"] = grouped.transform(
            lambda series: series.rolling(3, min_periods=2).std(ddof=0)
        )
        result[f"{indicator}_missing"] = values.isna().astype("int8")
    departments = ordered["department"].fillna("Unknown").astype(str).map(_safe_name)
    department_features = pd.get_dummies(departments, prefix="department", dtype="int8")
    result = pd.concat([result, department_features.set_axis(result.index)], axis=1)
    result = result.reindex(frame.index)
    assert_no_outcome_features(result)
    return result
