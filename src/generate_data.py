"""Generate deterministic synthetic Smart Campus raw source tables.

Latent traits exist only while this script runs; no exported table contains them.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

SEED = 42
N_HISTORICAL = 600
N_CURRENT = 800
RAW_DIR = Path("data/raw")
MONTHS = pd.date_range("2026-01-01", periods=6, freq="MS").strftime("%Y-%m").tolist()
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
SUBJECTS = ("Algorithms", "Databases", "Statistics", "Communication", "Domain Studies")


def _clip(values: np.ndarray, low: float, high: float) -> np.ndarray:
    """Constrain a synthetic measure to its clean range."""
    return np.clip(values, low, high)


def _sigmoid(values: np.ndarray) -> np.ndarray:
    """Return logistic probabilities."""
    return 1 / (1 + np.exp(-np.clip(values, -30, 30)))


def _population(rng: np.random.Generator, n_historical: int = N_HISTORICAL, n_current: int = N_CURRENT) -> pd.DataFrame:
    """Create output-safe identifiers plus temporary, private latent traits."""
    cohort = np.array(["historical"] * n_historical + ["current"] * n_current)
    size = len(cohort)
    department = rng.choice(
        ["Computer Science", "Business", "Mechanical Engineering", "Digital Media"], size, p=[0.34, 0.27, 0.23, 0.16]
    )
    interest = rng.normal(0, 0.85, size)
    low_interest = rng.random(size) < 0.32
    interest[low_interest] = rng.normal(-1.05, 0.35, low_interest.sum())
    population = pd.DataFrame(
        {
            "student_id": [f"H{i:04d}" for i in range(1, n_historical + 1)]
            + [f"C{i:04d}" for i in range(1, n_current + 1)],
            "cohort": cohort,
            "department": department,
            "ability": rng.normal(0, 1, size),
            "effort": rng.normal(0, 1, size),
            "consistency": rng.beta(4, 2, size),
            "career_interest": interest,
        }
    )
    department_code = {
        "Computer Science": "CS",
        "Business": "BU",
        "Mechanical Engineering": "ME",
        "Digital Media": "DM",
    }
    sequence = population.groupby("department", sort=False).cumcount().add(1)
    population["roll_no"] = [f"21{department_code[dept]}{number:03d}" for dept, number in zip(department, sequence)]
    population["email"] = population["roll_no"].str.lower() + "@college.edu"
    population["placement_id"] = "PLC-" + population["student_id"]
    return population


def _student_master(population: pd.DataFrame, rng: np.random.Generator) -> pd.DataFrame:
    """Build the unmessy identifier crosswalk used by the cleaner."""
    return population[["student_id", "roll_no", "email", "placement_id", "department"]].assign(
        year=rng.choice([2, 3, 4], len(population), p=[0.2, 0.45, 0.35]),
        section=rng.choice(["A", "B", "C"], len(population)),
    )


def _states(population: pd.DataFrame, rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Create correlated six-month ability, effort, and support states."""
    size, months = len(population), len(MONTHS)

    def evolve(baseline: np.ndarray) -> np.ndarray:
        values = np.empty((size, months))
        values[:, 0] = baseline + rng.normal(0, 0.18, size)
        for index in range(1, months):
            values[:, index] = (
                population.consistency.to_numpy() * values[:, index - 1]
                + (1 - population.consistency.to_numpy()) * baseline
                + rng.normal(0, 0.30, size)
            )
        return values

    return evolve(population.ability.to_numpy()), evolve(population.effort.to_numpy()), rng.normal(0, 1, (size, months))


def _table(population: pd.DataFrame, **metrics: np.ndarray) -> pd.DataFrame:
    """Flatten six monthly arrays into one source table."""
    result = population[["student_id", "department"]].loc[population.index.repeat(6)].reset_index(drop=True)
    result["snapshot_month"] = np.tile(MONTHS, len(population))
    for name, values in metrics.items():
        result[name] = values.reshape(-1).round(2) if np.issubdtype(values.dtype, np.floating) else values.reshape(-1)
    return result


def _academic_table(population: pd.DataFrame, cgpa: np.ndarray, marks: np.ndarray) -> pd.DataFrame:
    """Create released semester records instead of monthly academic records."""
    selected = [0, 5]
    result = population[["student_id", "department"]].loc[population.index.repeat(2)].reset_index(drop=True)
    result["semester"] = np.tile(["2025-S2", "2026-S1"], len(population))
    result["release_date"] = np.tile(["2025-12-20", "2026-06-20"], len(population))
    result["cgpa"] = cgpa[:, selected].reshape(-1).round(2)
    result["average_mark"] = marks[:, selected].reshape(-1).round(2)
    result["active_backlogs"] = np.maximum(0, np.round((68 - marks[:, selected]) / 18)).astype(int).reshape(-1)
    return result


def _placement_events(
    population: pd.DataFrame, readiness: np.ndarray, aptitude: np.ndarray, rng: np.random.Generator
) -> pd.DataFrame:
    """Create placement assessment events that are later aligned as-of month."""
    selected = [0, 2, 4]
    result = population[["student_id", "department"]].loc[population.index.repeat(3)].reset_index(drop=True)
    result["assessment_date"] = np.tile(["2026-01-20", "2026-03-20", "2026-05-20"], len(population))
    result["placement_readiness"] = readiness[:, selected].reshape(-1).round(2)
    result["mock_interview_score"] = (
        _clip(readiness[:, selected] + rng.normal(0, 8, (len(population), 3)), 0, 100).reshape(-1).round(2)
    )
    result["aptitude_score"] = aptitude[:, selected].reshape(-1).round(2)
    return result


def _subject_tables(
    population: pd.DataFrame, marks: np.ndarray, attendance: np.ndarray, rng: np.random.Generator
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Create subject-level semester records derived from generated indicators."""
    mark_rows: list[dict[str, object]] = []
    attendance_rows: list[dict[str, object]] = []
    offsets = np.linspace(-7, 7, len(SUBJECTS))
    for row, student in population.iterrows():
        for semester, released_on, month_index in (("2025-S2", "2025-12-20", 0), ("2026-S1", "2026-06-20", 5)):
            for subject, offset in zip(SUBJECTS, offsets):
                total = float(_clip(marks[row, month_index] + offset + rng.normal(0, 4), 0, 100))
                internal = float(_clip(0.4 * total + rng.normal(0, 2), 0, 40))
                exam = float(_clip(total - internal + rng.normal(0, 2), 0, 60))
                base = {
                    "student_id": student.student_id,
                    "department": student.department,
                    "semester": semester,
                    "release_date": released_on,
                    "subject": subject,
                }
                mark_rows.append({**base, "internal_marks": round(internal, 2), "exam_marks": round(exam, 2)})
                attendance_rows.append(
                    {
                        **base,
                        "attendance_pct": round(
                            float(_clip(attendance[row, month_index] + offset / 2 + rng.normal(0, 4), 0, 100)), 2
                        ),
                    }
                )
    return pd.DataFrame(mark_rows), pd.DataFrame(attendance_rows)


def _clean_ground_truth(
    population: pd.DataFrame, rng: np.random.Generator
) -> tuple[dict[str, pd.DataFrame], pd.DataFrame]:
    """Generate clean source data and outcomes before raw-data issues are added."""
    ability_state, effort_state, feedback_shock = _states(population, rng)
    size, months = ability_state.shape
    ability, effort, interest = (
        population.ability.to_numpy()[:, None],
        population.effort.to_numpy()[:, None],
        population.career_interest.to_numpy()[:, None],
    )
    digital_media = (population.department.to_numpy()[:, None] == "Digital Media").astype(float)
    feedback = _clip(
        71 + 7 * effort_state + 3 * ability - 13 * feedback_shock + rng.normal(0, 5, (size, months)), 0, 100
    )
    lagged_shock = np.column_stack([np.zeros(size), feedback_shock[:, :-1]])
    attendance = _clip(
        80 + 8 * effort_state + 2 * ability - 12 * lagged_shock + rng.normal(0, 6, (size, months)), 35, 100
    )
    lms = _clip(31 + 13 * effort_state + 3 * ability + 25 * digital_media + rng.normal(0, 8, (size, months)), 0, 130)
    completion = _clip(68 + 14 * effort_state + 5 * ability + rng.normal(0, 7, (size, months)), 0, 100)
    cgpa = _clip(6.7 + 0.76 * ability_state + 0.16 * effort_state + rng.normal(0, 0.22, (size, months)), 4, 10)
    marks = _clip(64 + 10 * ability_state + 3 * effort_state + rng.normal(0, 6, (size, months)), 25, 100)
    engagement = _clip(58 + 13 * effort_state + 3 * interest + rng.normal(0, 8, (size, months)), 0, 100)
    aptitude = _clip(58 + 13 * ability + 4 * effort_state + rng.normal(0, 8, (size, months)), 0, 100)
    coding = _clip(55 + 16 * ability + 3 * effort_state - 22 * digital_media + rng.normal(0, 9, (size, months)), 0, 100)
    communication = _clip(62 + 6 * ability + 7 * effort_state + 9 * interest + rng.normal(0, 8, (size, months)), 0, 100)
    certifications = np.maximum(
        0,
        np.floor(
            np.maximum(0, 0.25 + 0.42 * effort + 0.36 * interest + rng.normal(0, 0.45, (size, months))).cumsum(axis=1)
        ),
    ).astype(int)
    hackathons = np.maximum(
        0,
        np.floor(
            np.maximum(0, 0.12 + 0.32 * effort + 0.43 * interest + rng.normal(0, 0.42, (size, months))).cumsum(axis=1)
        ),
    ).astype(int)
    readiness = _clip(
        45
        + 7 * ability
        + 9 * effort_state
        + 18 * interest
        + 2 * certifications
        + 3 * hackathons
        + rng.normal(0, 7, (size, months)),
        0,
        100,
    )
    subject_marks, subject_attendance = _subject_tables(population, marks, attendance, rng)
    tables = {
        "academic": _academic_table(population, cgpa, marks),
        "attendance": _table(
            population,
            attendance_pct=attendance,
            classes_attended=np.round(attendance).astype(int),
            classes_scheduled=np.full((size, months), 100),
        ),
        "lms": _table(
            population,
            lms_logins=lms,
            assignment_completion_pct=completion,
            learning_minutes=_clip(lms * 9 + rng.normal(0, 35, (size, months)), 0, 1500),
        ),
        "engagement": _table(
            population,
            engagement_score=engagement,
            event_participation=np.maximum(
                0, np.round(1 + 0.35 * effort_state + rng.normal(0, 0.8, (size, months)))
            ).astype(int),
            mentoring_sessions=np.maximum(
                0, np.round(0.6 + 0.25 * effort_state + rng.normal(0, 0.6, (size, months)))
            ).astype(int),
        ),
        "placement": _placement_events(population, readiness, aptitude, rng),
        "skills": _table(
            population,
            coding_score=coding,
            communication_score=communication,
            hackathons=hackathons,
            certifications=certifications,
        ),
        "feedback": _table(
            population,
            feedback_score=feedback,
            support_rating=_clip(feedback + rng.normal(0, 6, (size, months)), 0, 100),
            response_count=np.maximum(1, np.round(4 + effort_state + rng.normal(0, 1, (size, months)))).astype(int),
        ),
        "subject_marks": subject_marks,
        "subject_attendance": subject_attendance,
    }
    low_group = (attendance[:, -1] < 75) & (lms[:, -1] < 45)
    backlog_probability = _sigmoid(
        -2.65
        - 0.48 * population.ability.to_numpy()
        - 0.30 * population.effort.to_numpy()
        + 1.15 * low_group
        + rng.normal(0, 0.15, size)
    )
    new_backlog = rng.binomial(1, backlog_probability)
    outcomes = population[["student_id", "cohort"]].copy()
    outcomes["next_semester_cgpa_drop"] = _clip(
        0.20
        - 0.24 * population.ability
        - 0.12 * population.effort
        + 0.34 * low_group
        + 0.18 * new_backlog
        + rng.normal(0, 0.22, size),
        -0.6,
        2.5,
    ).round(2)
    outcomes["new_backlog"] = new_backlog
    placed_probability = _sigmoid(
        1.00
        + 0.40 * population.ability
        + 0.20 * population.effort
        + 0.62 * population.career_interest
        + 0.34 * hackathons[:, -1]
        + 0.27 * certifications[:, -1]
        + rng.normal(0, 0.25, size)
    )
    outcomes["placement_result"] = np.where(rng.binomial(1, placed_probability), "placed", "not_placed")
    return tables, outcomes


def _messy_departments(series: pd.Series, rng: np.random.Generator) -> pd.Series:
    """Apply spelling variants after clean generation."""
    variants = {
        "Computer Science": ["Comp. Science", "computer science"],
        "Business": ["Business School", "business"],
        "Mechanical Engineering": ["Mechanical Engg.", "Mech Engineering"],
        "Digital Media": ["digital media", "Digital-Media"],
    }
    result = series.copy()
    change = rng.random(len(result)) < 0.15
    for canonical, choices in variants.items():
        mask = change & series.eq(canonical)
        result.loc[mask] = rng.choice(choices, mask.sum())
    return result


def _source_identifier_layout(frame: pd.DataFrame, table: str, population: pd.DataFrame) -> pd.DataFrame:
    """Apply the external identifier used by each source system."""
    result = frame.copy()
    mapping = {"academic": "roll_no", "lms": "email", "placement": "placement_id"}
    identifier = mapping.get(table)
    if identifier:
        result = result.merge(population[["student_id", identifier]], on="student_id", how="left").drop(
            columns="student_id"
        )
    return result


def _mess_email(values: pd.Series, rng: np.random.Generator) -> pd.Series:
    """Apply case, whitespace, and typo issues to three percent of LMS emails."""
    result = values.copy()
    for offset, position in enumerate(rng.choice(len(result), max(1, round(len(result) * 0.03)), replace=False)):
        value = str(result.iloc[position])
        result.iloc[position] = (
            value.upper()
            if offset % 3 == 0
            else f" {value} "
            if offset % 3 == 1
            else value.replace("@college.edu", "@colleg.edu")
        )
    return result


def _messify(frame: pd.DataFrame, table: str, rng: np.random.Generator) -> pd.DataFrame:
    """Add duplicates, gaps, outliers, name variants, and source-specific mess."""
    whole_student_gaps = {
        "academic": 0.02,
        "attendance": 0.04,
        "lms": 0.05,
        "engagement": 0.06,
        "placement": 0.04,
        "skills": 0.03,
        "feedback": 0.06,
        "subject_marks": 0.03,
        "subject_attendance": 0.03,
    }[table]
    source_id = {"academic": "roll_no", "lms": "email", "placement": "placement_id"}.get(table, "student_id")
    retained = (
        frame[source_id]
        .drop_duplicates()
        .sample(frac=1 - whole_student_gaps, random_state=int(rng.integers(1_000_000)))
    )
    result = frame[frame[source_id].isin(retained)].copy()
    result["department"] = _messy_departments(result.department, rng)
    metrics = [
        column
        for column in result
        if column
        not in {source_id, "department", "snapshot_month", "semester", "release_date", "assessment_date", "subject"}
    ]
    missing_rate = {
        "academic": 0.08,
        "attendance": 0.10,
        "lms": 0.12,
        "engagement": 0.09,
        "placement": 0.11,
        "skills": 0.08,
        "feedback": 0.12,
        "subject_marks": 0.08,
        "subject_attendance": 0.10,
    }[table]
    for column in metrics:
        result.loc[rng.random(len(result)) < missing_rate, column] = np.nan
    for column in [
        column
        for column in metrics
        if "score" in column
        or "pct" in column
        or column in {"cgpa", "average_mark", "lms_logins", "internal_marks", "exam_marks"}
    ]:
        bad = rng.random(len(result)) < 0.01
        result.loc[bad, column] = rng.choice([-12, 118, 145], bad.sum())
    if table in {"attendance", "subject_attendance"}:
        numeric = pd.to_numeric(result.attendance_pct, errors="coerce").round().astype("Int64").astype(str)
        result["attendance_pct"] = np.where(result.attendance_pct.notna(), numeric + "%", np.nan)
    if table == "lms":
        result["email"] = _mess_email(result["email"].reset_index(drop=True), rng).to_numpy()
    duplicates = result.iloc[rng.choice(len(result), max(1, round(len(result) * 0.05)), replace=False)]
    return pd.concat([result, duplicates], ignore_index=True).sample(frac=1, random_state=SEED).reset_index(drop=True)


def generate_data(
    seed: int = SEED, n_historical: int = N_HISTORICAL, n_current: int = N_CURRENT
) -> tuple[dict[str, pd.DataFrame], pd.DataFrame, pd.DataFrame]:
    """Return a crosswalk, messy source tables, and separate future labels."""
    rng = np.random.default_rng(seed)
    population = _population(rng, n_historical, n_current)
    sources, outcomes = _clean_ground_truth(population, rng)
    tables = {"student_master": _student_master(population, rng)}
    for name, frame in sources.items():
        tables[name] = _messify(_source_identifier_layout(frame, name, population), name, rng)
    return (
        tables,
        outcomes[outcomes.cohort.eq("historical")].reset_index(drop=True),
        outcomes[outcomes.cohort.eq("current")].reset_index(drop=True),
    )


def main() -> None:
    """Write the source tables, crosswalk, and known/held-back labels."""
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    tables, historical, current_holdout = generate_data()
    for name, frame in tables.items():
        frame.to_csv(RAW_DIR / f"{name}.csv", index=False)
    historical.to_csv(RAW_DIR / "outcomes_historical.csv", index=False)
    current_holdout.to_csv(RAW_DIR / "outcomes_current_holdout.csv", index=False)
    print("Wrote student master, nine raw source tables, historical outcomes, and current holdout outcomes.")


if __name__ == "__main__":
    main()
