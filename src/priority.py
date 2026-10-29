"""Derive risk trajectories and a separate intervention-priority worklist."""

from __future__ import annotations

import numpy as np
import pandas as pd

TREND_MULTIPLIERS = {"declining": 1.2, "stable": 1.0, "improving": 0.8}


def _legacy_priority(frame: pd.DataFrame) -> pd.DataFrame:
    """Calculate priority = risk × severity × trend multiplier."""
    return calculate_priority(frame)


def calculate_trajectories(risk_history: pd.DataFrame) -> pd.DataFrame:
    """Fit risk lines through months 2–5 and project one snapshot ahead.

    A rising risk is labelled `declining` because the student's outlook is
    worsening; a falling risk is `improving`. The projection band is ± one
    residual standard deviation around the one-step-ahead fitted value.
    """
    required = {"student_id", "snapshot_month", "risk"}
    if missing := required.difference(risk_history.columns):
        raise ValueError(f"risk history is missing {sorted(missing)}")
    data = risk_history.copy()
    data["snapshot_month"] = pd.to_datetime(data["snapshot_month"])
    selected_months = sorted(data.snapshot_month.dropna().unique())[1:5]
    records = []
    for student_id, group in data[data.snapshot_month.isin(selected_months)].groupby("student_id", sort=False):
        values = group.sort_values("snapshot_month").risk.astype(float).dropna().to_numpy()
        if len(values) < 2:
            slope, projected, spread, trend = 0.0, (values[0] if len(values) else np.nan), np.nan, "stable"
        else:
            x = np.arange(len(values), dtype=float)
            slope, intercept = np.polyfit(x, values, 1)
            spread = float(np.std(values - (slope * x + intercept), ddof=0))
            projected = float(np.clip(slope * len(values) + intercept, 0, 1))
            trend = "declining" if slope > 0.02 else "improving" if slope < -0.02 else "stable"
        records.append(
            {
                "student_id": student_id,
                "risk_slope": round(float(slope), 5),
                "projected_risk": round(float(projected), 4),
                "risk_band_low": round(float(max(0, projected - (0 if np.isnan(spread) else spread))), 4),
                "risk_band_high": round(float(min(1, projected + (0 if np.isnan(spread) else spread))), 4),
                "trend": trend,
                "trajectory_points": int(len(values)),
            }
        )
    return pd.DataFrame(records)


def calculate_priority(frame: pd.DataFrame) -> pd.DataFrame:
    """Calculate unrounded priority = 100 × risk × severity × trend multiplier."""
    risk_source = frame["projected_risk"] if "projected_risk" in frame else frame["risk"]
    risk = pd.to_numeric(risk_source, errors="coerce").clip(0, 1)
    attendance_source = frame["attendance_pct"] if "attendance_pct" in frame else pd.Series(np.nan, index=frame.index)
    backlog_source = (
        frame["active_backlogs"]
        if "active_backlogs" in frame
        else frame["backlogs"]
        if "backlogs" in frame
        else pd.Series(np.nan, index=frame.index)
    )
    attendance = pd.to_numeric(attendance_source, errors="coerce")
    backlogs = pd.to_numeric(backlog_source, errors="coerce")
    severe = attendance.lt(75) | backlogs.ge(2)
    borderline = attendance.between(75, 80, inclusive="both") | backlogs.eq(1)
    severity = np.select([severe, borderline], [1.0, 0.8], default=0.6)
    multiplier = frame["trend"].map(TREND_MULTIPLIERS).fillna(1.0)
    priority = 100 * risk * severity * multiplier
    return pd.DataFrame(
        {
            "risk": risk.round(4),
            "severity": severity,
            "trend_multiplier": multiplier,
            "priority_score": priority.round(4),
        },
        index=frame.index,
    )


def ranked_worklist(frame: pd.DataFrame) -> pd.DataFrame:
    """Attach priority fields and return highest-priority students first."""
    priority = calculate_priority(frame)
    priority = priority.rename(columns={"risk": "priority_risk"})
    return (
        frame.join(priority)
        .sort_values(["priority_score", "priority_risk"], ascending=False, kind="stable")
        .reset_index(drop=True)
    )
