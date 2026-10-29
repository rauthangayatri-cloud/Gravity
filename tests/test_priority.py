import pandas as pd

from src.priority import calculate_priority


def test_priority_uses_risk_severity_and_trend():
    frame = pd.DataFrame(
        {
            "projected_risk": [0.9, 0.5],
            "attendance_pct": [70, 77],
            "active_backlogs": [0, 0],
            "trend": ["declining", "stable"],
        }
    )
    result = calculate_priority(frame)
    assert result.loc[0, "severity"] == 1.0
    assert result.loc[0, "priority_score"] == 108
    assert result.loc[1, "severity"] == 0.8
    assert result.loc[1, "priority_score"] == 40
