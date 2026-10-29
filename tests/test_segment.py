import pandas as pd

from src.segment import segment_students


def test_segment_chooses_a_valid_kmeans_profile_count():
    frame = pd.DataFrame(
        {
            "cgpa": [5, 5.2, 6.2, 6.4, 8.1, 8.3, 7.5, 7.7],
            "attendance_pct": [55, 58, 75, 77, 92, 90, 68, 70],
            "placement_readiness": [40, 43, 55, 57, 80, 82, 45, 47],
            "engagement_score": [35, 38, 55, 58, 82, 84, 42, 45],
        }
    )
    assignments, profiles = segment_students(frame)
    assert len(assignments) == len(frame)
    assert profiles.chosen_k.iloc[0] in range(3, 7)
