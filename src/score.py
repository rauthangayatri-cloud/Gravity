"""Calculate the transparent current-state Success Score."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

DEFAULT_WEIGHTS = {
    "academic": 35,
    "attendance": 20,
    "lms": 15,
    "placement_readiness": 15,
    "engagement_skills": 10,
    "feedback": 5,
}
# Kept as an import-compatible name for callers that previously used WEIGHTS.
WEIGHTS = DEFAULT_WEIGHTS

# All component inputs use fixed ranges rather than fitted percentiles so scores
# are reproducible and comparable between snapshots. Values outside a range are
# clipped before scoring. Hackathons and certifications are capped at five.
NORMALISATION_RULES = {
    "cgpa": "0–10",
    "backlogs": "0–5, lower is better",
    "attendance_pct": "0–100",
    "lms_logins": "0–100",
    "assignment_completion_pct": "0–100",
    "aptitude_score": "0–100",
    "coding_score": "0–100",
    "mock_interview_score": "0–100",
    "engagement_score": "0–100",
    "communication_score": "0–100",
    "hackathons": "0–5",
    "certifications": "0–5",
    "feedback_score": "0–100",
}
COMPONENTS = tuple(DEFAULT_WEIGHTS)


def _numeric(frame: pd.DataFrame, *names: str) -> pd.Series:
    """Prefer canonical numeric values, with row-wise fallback to valid aliases."""
    result = pd.Series(np.nan, index=frame.index, dtype=float)
    for name in names:
        if name in frame:
            parsed = pd.to_numeric(frame[name].astype(str).str.replace("%", "", regex=False), errors="coerce")
            result = result.fillna(parsed.where(np.isfinite(parsed)))
    return result


def normalize(values: pd.Series, minimum: float, maximum: float) -> pd.Series:
    """Linearly normalize a measure to 0–1 after clipping to fixed bounds."""
    return _numeric(pd.DataFrame({"value": values}), "value").clip(minimum, maximum).sub(minimum).div(maximum - minimum)


def _mean_available(*values: pd.Series) -> pd.Series:
    """Average indicator values while retaining null for a wholly absent component."""
    return pd.concat(values, axis=1).mean(axis=1)


def component_values(frame: pd.DataFrame) -> pd.DataFrame:
    """Return six transparent 0–1 components from a unified snapshot table.

    Current unified names are preferred; legacy aliases allow existing callers
    to use the scorer while source-table unification is still being built.
    """
    academic = 0.7 * normalize(_numeric(frame, "cgpa", "current_gpa"), 0, 10) + 0.3 * (
        1 - normalize(_numeric(frame, "active_backlogs", "backlogs", "missed_assessments"), 0, 5)
    )
    # Academic needs both CGPA and backlog inputs because its published formula
    # deliberately specifies their 70/30 split.
    academic = academic.where(
        _numeric(frame, "cgpa", "current_gpa").notna()
        & _numeric(frame, "active_backlogs", "backlogs", "missed_assessments").notna()
    )
    lms = _mean_available(
        normalize(_numeric(frame, "lms_logins", "lms_activity_score"), 0, 100),
        normalize(_numeric(frame, "assignment_completion_pct"), 0, 100),
    )
    placement = _mean_available(
        normalize(_numeric(frame, "aptitude_score"), 0, 100),
        normalize(_numeric(frame, "coding_score", "skills_score"), 0, 100),
        normalize(_numeric(frame, "mock_interview_score", "placement_readiness"), 0, 100),
    )
    engagement_skills = _mean_available(
        normalize(_numeric(frame, "engagement_score"), 0, 100),
        normalize(_numeric(frame, "communication_score"), 0, 100),
        normalize(_numeric(frame, "hackathons"), 0, 5),
        normalize(_numeric(frame, "certifications"), 0, 5),
    )
    return pd.DataFrame(
        {
            "academic": academic,
            "attendance": normalize(_numeric(frame, "attendance_pct"), 0, 100),
            "lms": lms,
            "placement_readiness": placement,
            "engagement_skills": engagement_skills,
            "feedback": normalize(_numeric(frame, "feedback_score", "support_rating"), 0, 100),
        },
        index=frame.index,
    )


def score_students(frame: pd.DataFrame, weights: dict[str, float] | None = None) -> pd.DataFrame:
    """Score each student snapshot using supplied component weights.

    Missing components are reweighted over the available total. Confidence is
    the available share of the configured total; scores below 50% confidence
    are intentionally hidden rather than inferred.
    """
    active_weights = DEFAULT_WEIGHTS if weights is None else weights
    if set(active_weights) != set(COMPONENTS) or any(
        not np.isfinite(value) or value < 0 for value in active_weights.values()
    ):
        raise ValueError(f"weights must contain non-negative values for {COMPONENTS}")
    total_weight = sum(active_weights.values())
    if total_weight <= 0:
        raise ValueError("weights must sum to more than zero")
    components = component_values(frame)
    weight_series = pd.Series(active_weights, dtype=float)
    available_weight = components.notna().mul(weight_series, axis=1).sum(axis=1)
    numerator = components.mul(weight_series, axis=1).sum(axis=1, skipna=True)
    confidence = available_weight / total_weight
    score = numerator.div(available_weight.replace(0, np.nan)).mul(100)
    result = pd.DataFrame(index=frame.index)
    if "student_id" in frame:
        result["student_id"] = frame["student_id"].values
    if "snapshot_month" in frame:
        result["snapshot_month"] = frame["snapshot_month"].values
    result["success_score"] = score.round(1)
    result["score_confidence"] = confidence.round(2)
    result["score_status"] = np.where(confidence < 0.5, "Insufficient data", "Available")
    result["score_tier"] = pd.cut(
        score, bins=[-np.inf, 50, 70, np.inf], labels=["Red", "Amber", "Green"], right=False
    ).astype("object")
    result.loc[result.score_status.eq("Insufficient data"), ["success_score", "score_tier"]] = [
        np.nan,
        "Insufficient data",
    ]
    for name in COMPONENTS:
        result[f"{name}_component"] = components[name].round(4)
        result[f"{name}_contribution"] = (components[name] * active_weights[name]).round(4)
    return result


def _score_band(score: pd.Series) -> pd.Categorical:
    """Place scores in the requested validation bands."""
    return pd.cut(score, bins=[-np.inf, 40, 60, 80, np.inf], labels=["<40", "40-60", "60-80", "80+"], right=False)


def rank_stability(frame: pd.DataFrame, weights: dict[str, float] | None = None) -> dict[str, float]:
    """Return Spearman rank correlations after each component shifts ±5 points."""
    baseline_weights = DEFAULT_WEIGHTS if weights is None else weights
    baseline = score_students(frame, baseline_weights).success_score
    correlations: dict[str, float] = {}
    for component in COMPONENTS:
        for shift in (-5, 5):
            changed = dict(baseline_weights)
            changed[component] = max(0, changed[component] + shift)
            candidate = score_students(frame, changed).success_score
            paired = pd.DataFrame({"base": baseline, "candidate": candidate}).dropna()
            correlations[f"{component}_{shift:+d}"] = (
                round(float(paired["base"].corr(paired["candidate"], method="spearman")), 4)
                if len(paired) > 1
                else float("nan")
            )
    return correlations


def write_validation(
    scored: pd.DataFrame, outcomes: pd.DataFrame, source_frame: pd.DataFrame, output_dir: str = "data/outputs"
) -> dict[str, object]:
    """Write score-band outcome checks, rank stability, and a validation plot."""
    import json
    from pathlib import Path

    import matplotlib.pyplot as plt

    required = {"student_id", "new_backlog", "placement_result"}
    if not required.issubset(outcomes):
        raise ValueError("outcomes must include student_id, new_backlog, and placement_result")
    validation_scores = scored
    if "snapshot_month" in scored:
        validation_scores = scored.sort_values("snapshot_month").drop_duplicates("student_id", keep="last")
    merged = validation_scores.merge(outcomes[list(required)], on="student_id", how="inner")
    merged["score_band"] = _score_band(merged.success_score)
    rates = (
        merged.assign(not_placed=merged.placement_result.ne("placed"))
        .groupby("score_band", observed=False)
        .agg(
            backlog_rate=("new_backlog", "mean"),
            non_placement_rate=("not_placed", "mean"),
            student_count=("student_id", "size"),
        )
        .reset_index()
    )
    observed = rates.dropna(subset=["backlog_rate", "non_placement_rate"])
    gradient_clear = bool(
        len(observed) >= 3
        and observed.backlog_rate.is_monotonic_decreasing
        and observed.non_placement_rate.is_monotonic_decreasing
    )
    report = (
        "Clear decreasing outcome gradient across score bands."
        if gradient_clear
        else "Outcome gradient is not clearly decreasing; revisit score inputs or cutoffs."
    )
    distribution = scored.success_score.dropna()
    tier_counts = scored.score_tier.value_counts().to_dict()
    lopsided = bool(
        len(distribution)
        and max(tier_counts.get(tier, 0) for tier in ("Green", "Amber", "Red")) / len(distribution) > 0.70
    )
    suggestion = (
        "Consider percentile cutoffs at the 33rd and 67th score percentiles."
        if lopsided
        else "Current tier split is not very lopsided."
    )
    payload = {
        "score_band_rates": rates.to_dict(orient="records"),
        "gradient_clear": gradient_clear,
        "gradient_report": report,
        "rank_stability_spearman": rank_stability(source_frame),
        "tier_distribution": tier_counts,
        "tier_cutoff_suggestion": suggestion,
        "normalisation_rules": NORMALISATION_RULES,
    }
    target = Path(output_dir)
    target.mkdir(parents=True, exist_ok=True)
    (target / "score_validation.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    plot = rates.set_index("score_band")[["backlog_rate", "non_placement_rate"]]
    axis = plot.plot.bar(rot=0, ylim=(0, 1), title="Outcome rates by Success Score band")
    axis.set_ylabel("Rate")
    axis.figure.tight_layout()
    axis.figure.savefig(target / "score_validation.png", dpi=160)
    plt.close(axis.figure)
    print(f"Tier distribution: {tier_counts}")
    print(f"Tier cutoff guidance: {suggestion}")
    print(report)
    return payload


def main() -> None:
    """Score a prepared unified table and validate it against outcomes.csv."""
    import argparse

    parser = argparse.ArgumentParser(description="Compute and validate Success Scores.")
    parser.add_argument("--unified", default="data/clean/unified.csv", help="Prepared unified snapshot CSV")
    parser.add_argument("--outcomes", default="data/clean/outcomes.csv", help="Historical outcomes.csv")
    parser.add_argument("--output-dir", default="data/outputs")
    args = parser.parse_args()
    unified, outcomes = pd.read_csv(args.unified), pd.read_csv(args.outcomes)
    scored = score_students(unified)
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    scored.to_csv(Path(args.output_dir) / "success_scores.csv", index=False)
    write_validation(scored, outcomes, unified, args.output_dir)


if __name__ == "__main__":
    main()
