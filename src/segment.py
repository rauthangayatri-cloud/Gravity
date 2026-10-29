"""Create descriptive KMeans segments from current student profiles."""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.impute import SimpleImputer
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler


def _legacy_assign_segments(frame: pd.DataFrame) -> pd.Series:
    """Return mutually exclusive dashboard segments."""
    risk = frame[["academic_risk_probability", "placement_risk_probability"]].max(axis=1)
    return pd.Series(
        np.select(
            [risk >= 0.65, (risk >= 0.4) | (frame["success_score"] < 60), frame["success_score"] >= 80],
            ["Urgent support", "Watchlist", "On track"],
            default="Developing",
        ),
        index=frame.index,
        name="segment",
    )


def _profile_columns(frame: pd.DataFrame) -> list[str]:
    """Select stable profile measures present in the latest unified snapshot."""
    candidates = [
        "cgpa",
        "attendance_pct",
        "lms_logins",
        "engagement_score",
        "placement_readiness",
        "coding_score",
        "mock_interview_score",
        "feedback_score",
    ]
    return [column for column in candidates if column in frame]


def _cluster_name(profile: pd.Series, medians: pd.Series) -> str:
    """Give a transparent descriptive name to a cluster profile."""
    cgpa = profile.get("cgpa", medians.get("cgpa", 0))
    attendance = profile.get("attendance_pct", medians.get("attendance_pct", 100))
    readiness = profile.get("placement_readiness", medians.get("placement_readiness", 100))
    engagement = profile.get("engagement_score", medians.get("engagement_score", 100))
    if cgpa >= medians.get("cgpa", cgpa) and readiness < medians.get("placement_readiness", readiness):
        return "High CGPA / low placement readiness"
    if cgpa >= medians.get("cgpa", cgpa) and attendance < medians.get("attendance_pct", attendance):
        return "High potential / low attendance"
    if cgpa < medians.get("cgpa", cgpa) and engagement < medians.get("engagement_score", engagement):
        return "Silent strugglers"
    return "Steady performers"


def segment_students(frame: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Choose k=3–6 by silhouette score and return labels plus profiles."""
    columns = _profile_columns(frame)
    if len(columns) < 2 or len(frame) < 4:
        raise ValueError("At least four students and two profile fields are required for segmentation")
    numeric = frame[columns].apply(pd.to_numeric, errors="coerce")
    values = StandardScaler().fit_transform(SimpleImputer(strategy="median").fit_transform(numeric))
    choices = range(3, min(6, len(frame) - 1) + 1)
    candidates = []
    for k in choices:
        labels = KMeans(n_clusters=k, random_state=42, n_init=10).fit_predict(values)
        candidates.append((silhouette_score(values, labels), labels, k))
    _, labels, chosen_k = max(candidates, key=lambda item: item[0])
    assignment = pd.DataFrame({"segment_id": labels}, index=frame.index)
    profiles = frame.assign(segment_id=labels).groupby("segment_id")[columns].mean()
    profiles["size"] = pd.Series(labels).value_counts().sort_index()
    medians = numeric.median()
    name_counts: dict[str, int] = {}
    names = {}
    for segment_id, profile in profiles.iterrows():
        base = _cluster_name(profile, medians)
        name_counts[base] = name_counts.get(base, 0) + 1
        names[segment_id] = base if name_counts[base] == 1 else f"{base} {name_counts[base]}"
    assignment["segment"] = assignment.segment_id.map(names)
    profiles["segment"] = profiles.index.map(names)
    profiles["chosen_k"] = chosen_k
    return assignment, profiles.reset_index()


def assign_segments(frame: pd.DataFrame) -> pd.Series:
    """Compatibility wrapper returning descriptive KMeans segment labels."""
    return segment_students(frame)[0]["segment"]
