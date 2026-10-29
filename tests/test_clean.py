import pandas as pd

from src.clean import _resolve_identifiers, clean_table


def test_clean_table_dedupes_canonicalises_parses_and_flags_missing():
    frame = pd.DataFrame(
        {
            "student_id": ["C0001", "C0001"],
            "snapshot_month": ["2026-01", "2026-01"],
            "cohort": ["current", "current"],
            "department": ["digital media", "digital media"],
            "attendance_pct": ["78%", "78%"],
        }
    )
    result, log = clean_table(frame, "attendance")
    assert len(result) == 1
    assert result.loc[0, "department"] == "Digital Media"
    assert result.loc[0, "attendance_pct"] == 78
    assert result.loc[0, "attendance_pct_was_missing"] == 0
    assert log["duplicate_rows_removed"] == 1


def test_clean_table_nulls_out_of_range_values():
    frame = pd.DataFrame(
        {
            "student_id": ["C0001"],
            "snapshot_month": ["2026-01"],
            "cohort": ["current"],
            "department": ["Business"],
            "attendance_pct": ["118%"],
        }
    )
    result, log = clean_table(frame, "attendance")
    assert pd.isna(result.loc[0, "attendance_pct"])
    assert result.loc[0, "attendance_pct_was_missing"] == 1
    assert log["out_of_range_values_nulled"] == 1


def test_crosswalk_normalises_email_and_reports_unmatched_rows():
    master = pd.DataFrame(
        {
            "student_id": ["C0001"],
            "roll_no": ["21CS045"],
            "email": ["21cs045@college.edu"],
            "placement_id": ["PLC-C0001"],
            "department": ["Computer Science"],
        }
    )
    raw = pd.DataFrame(
        {"email": [" 21CS045@COLLEGE.EDU ", "21cs045@colleg.edu"], "snapshot_month": ["2026-01", "2026-01"]}
    )
    resolved, log = _resolve_identifiers(raw, "lms", master)
    assert resolved.student_id.tolist() == ["C0001"]
    assert log == {"matched": 1, "unmatched": 1, "ambiguous": 0}
