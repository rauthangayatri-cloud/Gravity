"""Clean seven synthetic source tables into one snapshot-level unified table."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

RAW_DIR = Path("data/raw")
CLEAN_DIR = Path("data/clean")
OUTPUT_DIR = Path("data/outputs")
SOURCE_TABLES = (
    "academic",
    "attendance",
    "lms",
    "engagement",
    "placement",
    "skills",
    "feedback",
    "subject_marks",
    "subject_attendance",
)
TABLES = ("student_master",) + SOURCE_TABLES
KEYS = ["student_id", "snapshot_month"]
IDENTIFIER_COLUMNS = {"academic": "roll_no", "lms": "email", "placement": "placement_id"}
EVENT_TIME_COLUMNS = {"academic": "release_date", "placement": "assessment_date"}
DEPARTMENT_NAMES = {
    "comp. science": "Computer Science",
    "computer science": "Computer Science",
    "business school": "Business",
    "business": "Business",
    "mechanical engg.": "Mechanical Engineering",
    "mech engineering": "Mechanical Engineering",
    "digital media": "Digital Media",
    "digital-media": "Digital Media",
}
RANGES = {
    "cgpa": (0, 10),
    "average_mark": (0, 100),
    "active_backlogs": (0, 20),
    "attendance_pct": (0, 100),
    "classes_attended": (0, 100),
    "classes_scheduled": (1, 200),
    "lms_logins": (0, 130),
    "assignment_completion_pct": (0, 100),
    "learning_minutes": (0, 1500),
    "engagement_score": (0, 100),
    "event_participation": (0, 50),
    "mentoring_sessions": (0, 50),
    "placement_readiness": (0, 100),
    "mock_interview_score": (0, 100),
    "aptitude_score": (0, 100),
    "coding_score": (0, 100),
    "communication_score": (0, 100),
    "hackathons": (0, 50),
    "certifications": (0, 50),
    "feedback_score": (0, 100),
    "support_rating": (0, 100),
    "response_count": (0, 100),
    "internal_marks": (0, 40),
    "exam_marks": (0, 60),
}
META_COLUMNS = {
    "student_id",
    "roll_no",
    "email",
    "placement_id",
    "cohort",
    "department",
    "snapshot_month",
    "semester",
    "release_date",
    "assessment_date",
    "subject",
}


def _canonical_department(values: pd.Series) -> pd.Series:
    """Map deliberately inconsistent department strings to canonical labels."""
    text = values.fillna("Unknown").astype(str).str.strip()
    return text.str.casefold().map(DEPARTMENT_NAMES).fillna(text)


def _numeric(values: pd.Series) -> pd.Series:
    """Parse numeric raw values including attendance strings such as 78%."""
    return pd.to_numeric(values.astype(str).str.replace("%", "", regex=False), errors="coerce")


def _identifier_key(values: pd.Series, identifier: str) -> pd.Series:
    """Normalise a source-system identifier for crosswalk matching."""
    text = values.fillna("").astype(str).str.strip().str.casefold()
    return text.str.replace(r"\s+", "", regex=True) if identifier == "email" else text


def _resolve_identifiers(
    frame: pd.DataFrame, table_name: str, master: pd.DataFrame
) -> tuple[pd.DataFrame, dict[str, int]]:
    """Join one source table to student master and report match outcomes."""
    identifier = IDENTIFIER_COLUMNS.get(table_name, "student_id")
    if identifier not in frame:
        raise ValueError(f"{table_name}.csv is missing {identifier}")
    master_columns = (
        ["student_id", "department"] if identifier == "student_id" else ["student_id", identifier, "department"]
    )
    master_keys = master[master_columns].copy()
    master_keys["_key"] = _identifier_key(
        master_keys["student_id"] if identifier == "student_id" else master_keys[identifier], identifier
    )
    ambiguous_keys = set(master_keys.loc[master_keys["_key"].duplicated(keep=False), "_key"])
    lookup = master_keys.drop_duplicates("_key", keep=False).set_index("_key")
    keys = _identifier_key(frame[identifier], identifier)
    ambiguous = keys.isin(ambiguous_keys)
    matched_ids = keys.map(lookup["student_id"])
    matched = matched_ids.notna() & ~ambiguous
    resolved = frame.loc[matched].copy()
    resolved["student_id"] = matched_ids.loc[matched].to_numpy()
    resolved["department"] = resolved["student_id"].map(lookup["department"])
    resolved["cohort"] = resolved["student_id"].str[0].map({"H": "historical", "C": "current"})
    return resolved, {
        "matched": int(matched.sum()),
        "unmatched": int((~matched & ~ambiguous).sum()),
        "ambiguous": int(ambiguous.sum()),
    }


def clean_table(frame: pd.DataFrame, table_name: str) -> tuple[pd.DataFrame, dict[str, int]]:
    """Deduplicate, canonicalize, validate, and flag one raw source table."""
    required = {"student_id", "cohort", "department"}
    if missing := required.difference(frame.columns):
        raise ValueError(f"{table_name}.csv is missing {sorted(missing)}")
    log: dict[str, int] = {"rows_in": int(len(frame)), "input_rows": int(len(frame))}
    result = frame.drop_duplicates().copy()
    duplicate_rows = int(len(frame) - len(result))
    log["duplicates_removed"] = duplicate_rows
    log["duplicate_rows_removed"] = duplicate_rows
    result["department"] = _canonical_department(result["department"])
    metric_columns = [column for column in result.columns if column not in META_COLUMNS]
    invalid = 0
    for column in metric_columns:
        result[column] = _numeric(result[column])
        if column in RANGES:
            low, high = RANGES[column]
            bad = result[column].notna() & ~result[column].between(low, high)
            invalid += int(bad.sum())
            result.loc[bad, column] = pd.NA
        result[f"{column}_was_missing"] = result[column].isna().astype("int8")
    missing_values = int(result[[f"{column}_was_missing" for column in metric_columns]].sum().sum())
    log["out_of_range_fixed"] = invalid
    log["out_of_range_values_nulled"] = invalid
    # Missingness remains explicit for downstream score confidence and model
    # imputation. The cleaner never fabricates feature values.
    log["values_imputed"] = 0
    log["values_flagged_missing"] = missing_values
    log["metric_missing_values"] = missing_values
    log["output_rows"] = int(len(result))
    return result, log


def _asof_monthly(table: pd.DataFrame, grid: pd.DataFrame, date_column: str) -> pd.DataFrame:
    """Carry a released/event value forward only into its own and later months."""
    payload_columns = [column for column in table if column not in META_COLUMNS]
    events = table[["student_id", date_column] + payload_columns].copy()
    events["_time"] = pd.to_datetime(events.pop(date_column))
    events["_is_snapshot"] = False
    snapshots = grid[["student_id", "snapshot_month"]].copy()
    snapshots["_time"] = pd.to_datetime(snapshots["snapshot_month"]) + pd.offsets.MonthEnd(0)
    snapshots["_is_snapshot"] = True
    combined = pd.concat([events, snapshots], ignore_index=True, sort=False).sort_values(
        ["student_id", "_time", "_is_snapshot"], kind="stable"
    )
    combined[payload_columns] = combined.groupby("student_id", sort=False)[payload_columns].ffill()
    return combined.loc[combined["_is_snapshot"], ["student_id", "snapshot_month"] + payload_columns].reset_index(
        drop=True
    )


def _subject_features(marks: pd.DataFrame, attendance: pd.DataFrame, grid: pd.DataFrame) -> pd.DataFrame:
    """Aggregate subject records into optional semester features before alignment."""
    marks = marks.copy()
    marks["_total"] = marks["internal_marks"] + marks["exam_marks"]
    valid = marks.dropna(subset=["_total"])
    weakest = (
        valid.loc[
            valid.groupby(["student_id", "release_date"])["_total"].idxmin(), ["student_id", "release_date", "subject"]
        ].rename(columns={"subject": "weakest_subject"})
        if not valid.empty
        else pd.DataFrame(columns=["student_id", "release_date", "weakest_subject"])
    )
    spread = (
        marks.groupby(["student_id", "release_date"], as_index=False)["_total"]
        .std(ddof=0)
        .rename(columns={"_total": "subject_marks_std"})
    )
    minimum = (
        attendance.groupby(["student_id", "release_date"], as_index=False)["attendance_pct"]
        .min()
        .rename(columns={"attendance_pct": "subject_attendance_min"})
    )
    summary = spread.merge(minimum, on=["student_id", "release_date"], how="outer").merge(
        weakest, on=["student_id", "release_date"], how="left"
    )
    for column in ("subject_marks_std", "subject_attendance_min"):
        summary[f"{column}_was_missing"] = summary[column].isna().astype("int8")
    return _asof_monthly(summary, grid, "release_date")


def clean_raw_tables(raw_dir: Path = RAW_DIR) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, object]]:
    """Crosswalk, clean, and time-align the nine raw source tables."""
    master = pd.read_csv(raw_dir / "student_master.csv")
    required_master = {"student_id", "roll_no", "email", "placement_id", "department", "year", "section"}
    if missing := required_master.difference(master):
        raise ValueError(f"student_master.csv is missing {sorted(missing)}")
    master["department"] = _canonical_department(master["department"])
    tables: dict[str, pd.DataFrame] = {}
    log: dict[str, object] = {"tables": {}}
    for table_name in SOURCE_TABLES:
        raw = pd.read_csv(raw_dir / f"{table_name}.csv")
        deduplicated = raw.drop_duplicates().copy()
        resolved, join_log = _resolve_identifiers(deduplicated, table_name, master)
        cleaned, table_log = clean_table(resolved, table_name)
        table_log.update(
            {
                "rows_in": int(len(raw)),
                "input_rows": int(len(raw)),
                "duplicates_removed": int(len(raw) - len(deduplicated)),
                "duplicate_rows_removed": int(len(raw) - len(deduplicated)),
                "join": join_log,
                "students_missing": int(master.student_id.nunique() - cleaned.student_id.nunique()),
            }
        )
        tables[table_name] = cleaned
        log["tables"][table_name] = table_log
    months = sorted(tables["attendance"]["snapshot_month"].dropna().astype(str).unique())
    identity = pd.MultiIndex.from_product([master.student_id, months], names=KEYS).to_frame(index=False)
    identity = identity.merge(master[["student_id", "department", "year", "section"]], on="student_id", how="left")
    identity["cohort"] = identity["student_id"].str[0].map({"H": "historical", "C": "current"})
    unified = identity
    for table_name in ("attendance", "lms", "engagement", "skills", "feedback"):
        payload = tables[table_name].drop(columns=["cohort", "department"], errors="ignore")
        unified = unified.merge(payload, on=KEYS, how="left", validate="one_to_one")
    for table_name, date_column in EVENT_TIME_COLUMNS.items():
        payload = _asof_monthly(tables[table_name], unified, date_column)
        unified = unified.merge(payload, on=KEYS, how="left", validate="one_to_one")
    subject_payload = _subject_features(tables["subject_marks"], tables["subject_attendance"], unified)
    unified = unified.merge(subject_payload, on=KEYS, how="left", validate="one_to_one")
    historical = pd.read_csv(raw_dir / "outcomes_historical.csv")
    current = pd.read_csv(raw_dir / "outcomes_current_holdout.csv")
    outcomes = pd.concat([historical, current], ignore_index=True).drop_duplicates("student_id", keep="last")
    log["unified_rows"] = int(len(unified))
    log["unified_students"] = int(unified.student_id.nunique())
    log["outcome_rows"] = int(len(outcomes))
    log["note"] = "Counts describe deterministic cleaning of synthetic raw data."
    return unified, outcomes, log


def main() -> None:
    """Write unified snapshots, combined outcomes, and the cleaning log."""
    CLEAN_DIR.mkdir(parents=True, exist_ok=True)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    unified, outcomes, log = clean_raw_tables()
    unified.to_csv(CLEAN_DIR / "unified.csv", index=False)
    outcomes.to_csv(CLEAN_DIR / "outcomes.csv", index=False)
    (OUTPUT_DIR / "cleaning_log.json").write_text(json.dumps(log, indent=2), encoding="utf-8")
    print("Wrote unified.csv, outcomes.csv, and cleaning_log.json")


if __name__ == "__main__":
    main()
