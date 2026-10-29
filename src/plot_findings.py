"""Plot the five engineered synthetic-data findings from the raw exports."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

RAW_DIR = Path("data/raw")


def _latest(filename: str) -> pd.DataFrame:
    """Read an intentionally messy source and retain its latest unique row."""
    data = pd.read_csv(RAW_DIR / filename).drop_duplicates()
    return data.sort_values("snapshot_month").drop_duplicates("student_id", keep="last")


def _number(values: pd.Series) -> pd.Series:
    """Parse numeric values, including intentionally messy percentages."""
    return pd.to_numeric(values.astype(str).str.replace("%", "", regex=False), errors="coerce")


def _department(values: pd.Series) -> pd.Series:
    """Canonicalize display labels only; raw exports stay intentionally messy."""
    return values.replace(
        {
            "Comp. Science": "Computer Science",
            "computer science": "Computer Science",
            "Business School": "Business",
            "business": "Business",
            "Mechanical Engg.": "Mechanical Engineering",
            "Mech Engineering": "Mechanical Engineering",
            "digital media": "Digital Media",
            "Digital-Media": "Digital Media",
        }
    )


def main() -> None:
    """Save a compact five-panel visual check to the raw-data directory."""
    academic, attendance, lms = _latest("academic.csv"), _latest("attendance.csv"), _latest("lms.csv")
    placement, skills = _latest("placement.csv"), _latest("skills.csv")
    outcomes = pd.read_csv(RAW_DIR / "outcomes_historical.csv")
    latest = (
        academic.merge(attendance[["student_id", "attendance_pct"]], on="student_id")
        .merge(lms[["student_id", "lms_logins"]], on="student_id")
        .merge(placement[["student_id", "placement_readiness"]], on="student_id")
        .merge(skills[["student_id", "hackathons", "certifications", "coding_score", "department"]], on="student_id")
        .merge(outcomes, on="student_id")
    )
    for column in [
        "cgpa",
        "attendance_pct",
        "lms_logins",
        "placement_readiness",
        "hackathons",
        "certifications",
        "coding_score",
    ]:
        latest[column] = _number(latest[column])
    low_group = (latest.attendance_pct < 75) & (latest.lms_logins < 45)
    figure, axes = plt.subplots(1, 5, figsize=(22, 4.5))
    backlog_rates = [latest.loc[low_group, "new_backlog"].mean(), latest.loc[~low_group, "new_backlog"].mean()]
    axes[0].bar(["Low attendance\n+ low LMS", "All others"], backlog_rates, color=["#d95f02", "#1b9e77"])
    axes[0].set(title="1. New backlog rate", ylim=(0, max(0.35, max(backlog_rates) * 1.2)))
    axes[1].scatter(latest.cgpa, latest.placement_readiness, s=9, alpha=0.35)
    axes[1].axvline(7.7, ls="--", c="grey")
    axes[1].axhline(50, ls="--", c="grey")
    axes[1].set(xlabel="CGPA", ylabel="Readiness", title="2. High CGPA, low readiness")
    activity = (latest.hackathons.fillna(0) + latest.certifications.fillna(0)).clip(upper=8)
    rate = (
        latest.assign(activity=activity, placed=latest.placement_result.eq("placed"))
        .groupby("activity", observed=True)
        .placed.mean()
    )
    axes[2].plot(rate.index, rate.values, marker="o")
    axes[2].set(xlabel="Hackathons + certifications", ylabel="Placement rate", title="3. Activities and placement")
    feedback = pd.read_csv(RAW_DIR / "feedback.csv").drop_duplicates()
    monthly_attendance = pd.read_csv(RAW_DIR / "attendance.csv").drop_duplicates()
    feedback.feedback_score, monthly_attendance.attendance_pct = (
        _number(feedback.feedback_score),
        _number(monthly_attendance.attendance_pct),
    )
    trend = feedback.merge(
        monthly_attendance[["student_id", "snapshot_month", "attendance_pct"]], on=["student_id", "snapshot_month"]
    ).sort_values(["student_id", "snapshot_month"])
    trend["next_attendance_change"] = trend.groupby("student_id").attendance_pct.shift(-1) - trend.attendance_pct
    trend["feedback_dip"] = trend.groupby("student_id").feedback_score.diff() < -8
    follow_up = trend.groupby("feedback_dip", observed=True).next_attendance_change.mean()
    axes[3].bar(
        ["No prior dip", "Prior feedback dip"],
        [follow_up.get(False, 0), follow_up.get(True, 0)],
        color=["#7570b3", "#e7298a"],
    )
    axes[3].axhline(0, c="grey", lw=0.8)
    axes[3].set(title="4. Next-month attendance change")
    department = (
        latest.assign(department=_department(latest.department))
        .groupby("department", observed=True)[["lms_logins", "coding_score"]]
        .mean()
        .sort_values("lms_logins")
    )
    department.plot.bar(ax=axes[4], secondary_y="coding_score", color=["#66c2a5", "#fc8d62"])
    axes[4].set(title="5. LMS high, coding weak dept", xlabel="")
    axes[4].tick_params(axis="x", rotation=30)
    figure.suptitle("Synthetic-data verification findings", y=1.04)
    figure.tight_layout()
    output = RAW_DIR / "synthetic_findings.png"
    figure.savefig(output, dpi=160, bbox_inches="tight")
    print(f"Wrote {output}")


if __name__ == "__main__":
    main()
