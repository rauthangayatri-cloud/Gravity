from src.generate_data import TABLES, generate_data


def test_generation_is_reproducible_and_keeps_latent_traits_private():
    first_tables, first_historical, first_current = generate_data()
    second_tables, second_historical, second_current = generate_data()
    assert set(first_tables) == set(TABLES)
    assert first_historical.equals(second_historical)
    assert first_current.equals(second_current)
    assert len(first_historical) == 600
    assert len(first_current) == 800
    assert {"ability", "effort", "consistency", "career_interest"}.isdisjoint(first_tables["academic"].columns)
    assert first_tables["attendance"].attendance_pct.astype(str).str.endswith("%").any()


def test_generation_uses_crosswalk_identifiers_and_subject_sources():
    tables, _, _ = generate_data()
    master = tables["student_master"]
    assert {"student_id", "roll_no", "email", "placement_id", "department", "year", "section"}.issubset(master.columns)
    assert "student_id" not in tables["academic"] and "roll_no" in tables["academic"]
    assert "student_id" not in tables["lms"] and "email" in tables["lms"]
    assert "student_id" not in tables["placement"] and "placement_id" in tables["placement"]
    assert {"subject_marks", "subject_attendance"}.issubset(tables)
    assert tables["academic"]["semester"].nunique() == 2
    assert tables["placement"]["assessment_date"].nunique() == 3
