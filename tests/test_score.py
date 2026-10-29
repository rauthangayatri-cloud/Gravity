import pandas as pd
import pytest

from src.score import normalize, score_students


def test_normalize_clips_to_documented_range():
    values = normalize(pd.Series([-2, 5, 13]), 0, 10)
    assert values.tolist() == [0, 0.5, 1]


def test_score_reweights_over_available_components():
    frame = pd.DataFrame({"cgpa": [10.0], "active_backlogs": [0], "attendance_pct": [50.0]})
    result = score_students(frame)
    assert result.loc[0, "score_confidence"] == 0.55
    assert result.loc[0, "success_score"] == 81.8
    assert result.loc[0, "academic_contribution"] == 35
    assert result.loc[0, "attendance_contribution"] == 10


def test_score_flags_insufficient_data_below_half_confidence():
    frame = pd.DataFrame({"cgpa": [8.0], "active_backlogs": [0]})
    result = score_students(frame)
    assert result.loc[0, "score_status"] == "Insufficient data"
    assert pd.isna(result.loc[0, "success_score"])
    assert result.loc[0, "score_tier"] == "Insufficient data"


def test_score_alias_fallback_and_coverage_are_rowwise():
    """Null primary fields may use aliases; coverage remains component-weight share."""
    frame = pd.DataFrame(
        {
            "cgpa": [8.0],
            "active_backlogs": [0],
            "attendance_pct": [80],
            "feedback_score": [None],
            "support_rating": [100],
        }
    )
    result = score_students(frame)
    assert result.score_confidence.iloc[0] == 0.6
    assert result.feedback_component.iloc[0] == 1
    pd.testing.assert_frame_equal(result, score_students(frame.drop(columns="feedback_score")))


@pytest.mark.parametrize("bad", [float("nan"), float("inf"), -1])
def test_score_rejects_nonfinite_or_negative_weights(bad):
    """No NaN/Infinity can contaminate weighted scenarios."""
    from src.score import DEFAULT_WEIGHTS

    with pytest.raises(ValueError):
        score_students(pd.DataFrame({"cgpa": [8]}), DEFAULT_WEIGHTS | {"academic": bad})
