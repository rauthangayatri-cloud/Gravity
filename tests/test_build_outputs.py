import numpy as np
import pandas as pd

from src.build_outputs import generate_simulated_followups, generate_what_if, map_playbook_actions


def test_playbook_maps_top_attendance_driver_to_recovery_plan():
    students = pd.DataFrame(
        {"student_id": ["C0001"], "attendance_pct": [60], "priority_score": [72.0], "trend": ["declining"]}
    )
    drivers = pd.DataFrame({"student_id": ["C0001"], "rank": [1], "feature": ["attendance_pct_current"]})
    actions = map_playbook_actions(students, drivers)
    assert actions.loc[0, "recommended_action"] == "Attendance recovery plan"
    assert actions.loc[0, "owner"] == "Faculty"


def test_simulated_followups_have_before_30_and_60_day_snapshots():
    students = pd.DataFrame(
        {"student_id": ["C0001", "C0002"], "priority_score": [80, 60], "success_score": [50, 70], "risk": [0.7, 0.3]}
    )
    followups = generate_simulated_followups(students, demo_size=2)
    assert len(followups) == 6
    assert set(followups.day_after_start) == {0, 30, 60}
    assert followups.note.str.contains("not an observed outcome or a causal estimate", regex=False).all()


def test_what_if_changes_one_indicator_and_records_risk_delta():
    class Model:
        feature_names_in_ = np.array(
            [
                "attendance_pct_current",
                "assignment_completion_pct_current",
                "coding_score_current",
                "mock_interview_score_current",
            ]
        )

        def predict_proba(self, frame):
            probability = pd.to_numeric(frame["attendance_pct_current"], errors="coerce").fillna(0).to_numpy() / 100
            return np.column_stack([1 - probability, probability])

    unified = pd.DataFrame(
        {
            "student_id": ["C0001"],
            "snapshot_month": ["2026-06"],
            "department": ["Business"],
            "attendance_pct": [50],
            "assignment_completion_pct": [60],
            "coding_score": [40],
            "mock_interview_score": [45],
        }
    )
    priority = pd.DataFrame({"student_id": ["C0001"], "priority_score": [90.0]})
    results = {
        "academic_risk": {
            "metrics": {"recommended_model": "logistic_regression"},
            "models": {"logistic_regression": Model()},
        }
    }

    what_if = generate_what_if(unified, priority, results)

    attendance = what_if[what_if.action == "Attendance +10 points"].iloc[0]
    assert len(what_if) == 4
    assert attendance.risk_before == 0.5
    assert attendance.risk_after == 0.6
    assert attendance.delta == 0.1
