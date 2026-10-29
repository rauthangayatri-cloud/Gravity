import pandas as pd
import pytest

from src.features import INDICATORS, assert_no_outcome_features, build_features, build_snapshot_features


def test_features_use_only_current_and_prior_months_and_exclude_outcomes():
    frame = pd.DataFrame(
        {
            "student_id": ["H0001"] * 3,
            "snapshot_month": ["2026-01", "2026-02", "2026-03"],
            "department": ["Business"] * 3,
            "cgpa": [6.0, 7.0, 8.0],
            "attendance_pct": [80, 81, 82],
            "new_backlog": [0, 0, 1],
        }
    )
    features = build_features(frame)
    assert features.loc[2, "cgpa_current"] == 8
    assert features.loc[2, "cgpa_slope_3m"] == pytest.approx(1)
    assert not any("new_backlog" in column for column in features)
    assert "department_business" in features


def test_outcome_feature_assertion_rejects_leakage():
    with pytest.raises(AssertionError):
        assert_no_outcome_features(pd.DataFrame({"next_semester_cgpa_drop": [0.5]}))


def test_full_history_has_nonempty_slope_and_volatility_features():
    rows = []
    for month, value in enumerate([60, 65, 70], start=1):
        row = {"student_id": "H0001", "snapshot_month": f"2026-0{month}", "department": "Business"}
        row.update({indicator: value for indicator in INDICATORS})
        rows.append(row)
    features = build_features(pd.DataFrame(rows))
    history_columns = [
        column for column in features if column.endswith("_slope_3m") or column.endswith("_volatility_3m")
    ]
    assert not [column for column in history_columns if features[column].isna().all()]


def test_snapshot_contract_is_independent_of_history_batch_and_outcomes():
    """Serving a single row must equal extracting it from the offline batch."""
    frame = pd.DataFrame({"department": ["Business", "Digital Media"], "cgpa": [6, 8], "new_backlog": [1, 0]})
    features = build_snapshot_features(frame)
    pd.testing.assert_frame_equal(features.iloc[[0]], build_snapshot_features(frame.iloc[[0]]))
    assert features.shape[1] == 34
    assert not any("slope" in c or "volatility" in c or "backlog" == c for c in features)
    assert features.department_digital_media.tolist() == [0, 1]
    with pytest.raises(ValueError, match="department"):
        build_snapshot_features(pd.DataFrame({"department": ["unknown"]}))
