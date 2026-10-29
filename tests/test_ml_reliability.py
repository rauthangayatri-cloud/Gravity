"""Synthetic, temporary-bundle reliability regressions; never touch project data."""

import json
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import pandas as pd
import pytest
from fastapi.testclient import TestClient

import api.main as api
from src.artifacts import ArtifactError, digest, load_bundle, publish_bundle, read_manifest, verified_bytes
from src.build_outputs import _risk_history, generate_what_if
from src.features import INDICATORS, build_snapshot_features
from src.prediction import RiskPredictor
from src.runtime import _runtime_segments, process_student_data
from src.train import build_targets, train_from_files, train_risk_model


@pytest.fixture(scope="module")
def experiment():
    """Fit tiny models only on in-memory synthetic rows, never saved legacy models."""
    rng = np.random.default_rng(42)
    size = 140
    frame = pd.DataFrame(
        {
            "student_id": [f"H{i:03d}" if i < 100 else f"C{i:03d}" for i in range(size)],
            "snapshot_month": "2026-06-30",
            "cohort": ["historical"] * 100 + ["current"] * 40,
            "department": np.resize(["Business", "Computer Science", "Digital Media", "Mechanical Engineering"], size),
        }
    )
    for name in INDICATORS:
        frame[name] = rng.uniform(20, 95, size)
    frame["cgpa"] = rng.uniform(4, 10, size)
    frame["active_backlogs"] = rng.integers(0, 4, size)
    frame["hackathons"] = rng.integers(0, 6, size)
    frame["certifications"] = rng.integers(0, 6, size)
    outcomes = frame[["student_id", "cohort"]].copy()
    outcomes["next_semester_cgpa_drop"] = np.where(frame.cgpa < 7, 0.7, 0.1)
    outcomes["new_backlog"] = 0
    outcomes["placement_result"] = np.where(frame.coding_score < 55, "not_placed", "placed")
    results = train_from_files(frame, outcomes)
    return frame, outcomes, results


def artifacts_for(results):
    """Minimal complete publication payload for focused loader tests."""
    return {
        "models": results,
        "metrics": {"data_note": "SYNTHETIC", "targets": {k: r["metrics"] for k, r in results.items()}},
    }


def test_selected_predictions_match_offline_runtime_and_what_if(experiment, tmp_path, monkeypatch):
    """One bundle controls every probability, including persisted reloads."""
    frame, _, results = experiment
    current = frame[frame.cohort.eq("current")].reset_index(drop=True)
    bundle = publish_bundle(tmp_path / "bundle", artifacts_for(results), {"test": "synthetic"})
    monkeypatch.setenv("GRAVITY_ML_BUNDLE", str(bundle))
    runtime = process_student_data(current)["students"].set_index("student_id")
    history, _ = _risk_history(current, results)
    what_if = generate_what_if(current, current.assign(priority_score=50), results)
    loaded, identity = load_bundle(bundle)
    for target, result in results.items():
        expected = result["predictor"].predict_proba(build_snapshot_features(current))[:, 1]
        np.testing.assert_allclose(history[f"{target}_probability"], expected)
        np.testing.assert_allclose(runtime.loc[current.student_id, f"{target}_probability"], expected)
        np.testing.assert_allclose(loaded[target].predict_proba(build_snapshot_features(current))[:, 1], expected)
        before = (
            what_if[what_if.target.eq(target)]
            .drop_duplicates("student_id")
            .set_index("student_id")
            .loc[current.student_id]
        )
        np.testing.assert_allclose(before.risk_before, expected, atol=0.000001)
    assert runtime.model_bundle_id.eq(identity).all()
    assert runtime.risk_band_low.isna().all()


def test_selection_and_thresholds_do_not_use_current_labels(experiment, monkeypatch):
    """Flipping evaluation labels cannot change model selection or predictions."""
    import src.train as train

    frame, outcomes, results = experiment
    features = pd.concat([frame[["student_id", "cohort", "department"]], build_snapshot_features(frame)], axis=1)
    target = build_targets(outcomes).academic_risk.rename("academic_risk")
    target.loc[frame.cohort.eq("current")] = 1 - target.loc[frame.cohort.eq("current")]
    monkeypatch.setattr(train, "_bootstrap_auc_ci", lambda *args: {})
    changed = train_risk_model(features, target)
    original = results["academic_risk"]
    assert changed["metrics"]["recommended_model"] == original["metrics"]["recommended_model"]
    assert changed["predictor"].threshold == original["predictor"].threshold
    np.testing.assert_array_equal(
        changed["predictions"].selected_probability, original["predictions"].selected_probability
    )
    groups = [set(v) for v in original["split_students"].values()]
    assert all(not a & b for i, a in enumerate(groups) for b in groups[i + 1 :])


@pytest.mark.parametrize(
    "column,value", [("new_backlog", None), ("placement_result", "unknown"), ("next_semester_cgpa_drop", np.nan)]
)
def test_invalid_outcomes_cannot_become_labels(experiment, column, value):
    """Absent labels are not silently converted into positive/negative examples."""
    _, outcomes, _ = experiment
    bad = outcomes.copy()
    bad.loc[0, column] = value
    with pytest.raises(ValueError, match="outcome labels"):
        build_targets(bad)


def test_snapshot_and_cohort_duplicates_are_rejected(experiment):
    """A duplicate ID cannot contaminate the holdout or multiply merged targets."""
    frame, outcomes, _ = experiment
    with pytest.raises(ValueError, match="duplicate snapshot"):
        train_from_files(pd.concat([frame, frame.iloc[[0]]]), outcomes)
    changed = frame.copy()
    changed.loc[100, "student_id"] = changed.loc[0, "student_id"]
    with pytest.raises(ValueError):
        train_from_files(changed, outcomes)


def test_bundle_refuses_overwrite_tampering_and_environment_drift(experiment, tmp_path):
    """Neither an existing build nor a cached predictor can hide corruption."""
    _, _, results = experiment
    bundle = publish_bundle(tmp_path / "bundle", artifacts_for(results), {})
    first, _ = load_bundle(bundle)
    with pytest.raises(FileExistsError):
        publish_bundle(bundle, artifacts_for(results), {})
    model_path = bundle / "bundle.joblib"
    model_path.write_bytes(model_path.read_bytes() + b"tampered-test-data")
    with pytest.raises(ArtifactError, match="integrity"):
        load_bundle(bundle)
    assert set(first) == {"academic_risk", "placement_risk"}
    other = publish_bundle(tmp_path / "other", artifacts_for(results), {})
    manifest = read_manifest(other)
    manifest["environment"]["packages"]["numpy"] = "not-the-installed-version"
    (other / "manifest.json").write_text(json.dumps(manifest))
    with pytest.raises(ArtifactError, match="environment"):
        load_bundle(other)
    with pytest.raises(ArtifactError):
        verified_bytes(other, "../outside")


def test_interrupted_and_concurrent_publications_are_safe(experiment, tmp_path, monkeypatch):
    """Incomplete new directories are never loadable; two writers cannot overwrite."""
    import src.artifacts as artifacts

    _, _, results = experiment
    original = artifacts.joblib.dump

    def interrupted(*args, **kwargs):
        raise RuntimeError("synthetic interruption")

    monkeypatch.setattr(artifacts.joblib, "dump", interrupted)
    with pytest.raises(RuntimeError):
        publish_bundle(tmp_path / "interrupted", artifacts_for(results), {})
    with pytest.raises(ArtifactError):
        load_bundle(tmp_path / "interrupted")
    monkeypatch.setattr(artifacts.joblib, "dump", original)

    def attempt(_):
        try:
            publish_bundle(tmp_path / "concurrent", artifacts_for(results), {})
            return "published"
        except FileExistsError:
            return "refused"

    with ThreadPoolExecutor(max_workers=2) as executor:
        assert sorted(executor.map(attempt, [0, 1])) == ["published", "refused"]
    load_bundle(tmp_path / "concurrent")


def test_explanations_describe_selected_model_margin(experiment):
    """The stated base model and raw margin must match its estimator output."""
    frame, _, results = experiment
    features = build_snapshot_features(frame.iloc[:4])
    predictors = [
        RiskPredictor(target, name, model, result["calibrators"][name], 0.5)
        for target, result in results.items()
        for name, model in result["models"].items()
    ]
    for predictor in predictors:
        drivers = predictor.drivers(features, frame.student_id.iloc[:4])
        transformed = predictor.model.named_steps["imputer"].transform(features)
        if predictor.model_name == "logistic_regression":
            transformed = predictor.model.named_steps["scale"].transform(transformed)
            margin = predictor.model.named_steps["model"].decision_function(transformed)
        else:
            margin = predictor.model.named_steps["model"].predict(transformed, output_margin=True)
        np.testing.assert_allclose(drivers.query("rank == 1").raw_margin, margin, atol=1e-5)
        assert drivers.model.eq(predictor.model_name).all()
        assert drivers.units.eq("base-model log-odds").all()


def test_inference_never_fits_and_rejects_duplicate_features(experiment, tmp_path, monkeypatch):
    """Every deployed model variant must infer without fitting any preprocessing."""
    from sklearn.cluster import KMeans
    from sklearn.impute import SimpleImputer
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler
    from xgboost import XGBClassifier

    frame, _, results = experiment
    bundle = publish_bundle(tmp_path / "no-fit", artifacts_for(results), {})
    monkeypatch.setenv("GRAVITY_ML_BUNDLE", str(bundle))

    def forbidden(*args, **kwargs):
        raise AssertionError("Fitting is forbidden at runtime")

    for kind in (Pipeline, SimpleImputer, StandardScaler, LogisticRegression, XGBClassifier, KMeans):
        monkeypatch.setattr(kind, "fit", forbidden)
    assert len(process_student_data(frame.iloc[:4])["students"]) == 4
    features = build_snapshot_features(frame.iloc[:4])
    with pytest.raises(ValueError, match="schema mismatch"):
        results["academic_risk"]["predictor"].predict_proba(pd.concat([features, features.iloc[:, [0]]], axis=1))


@pytest.mark.parametrize("malformed", [[], None, {"format": 1, "targets": [], "files": {}}])
def test_malformed_manifests_fail_with_artifact_error(tmp_path, malformed):
    """Damaged operator metadata must not leak raw type errors to the API."""
    (tmp_path / "manifest.json").write_text(json.dumps(malformed))
    with pytest.raises(ArtifactError):
        read_manifest(tmp_path)


def test_runtime_segmentation_never_fits_and_is_batch_independent(monkeypatch):
    """Identical/small cohorts are safe and adding a peer cannot move an existing label."""
    from sklearn.cluster import KMeans

    def forbidden(*args, **kwargs):
        raise AssertionError("Runtime must not fit")

    monkeypatch.setattr(KMeans, "fit_predict", forbidden)
    student = pd.DataFrame(
        {"academic_risk_probability": [0.7], "placement_risk_probability": [0.2], "success_score": [50], "cgpa": [6]}
    )
    one, _ = _runtime_segments(student)
    many, _ = _runtime_segments(pd.concat([student] * 10, ignore_index=True))
    assert many.segment.eq(one.segment.iloc[0]).all()


def test_metrics_never_claim_unmatched_saved_dataset_is_evaluated(experiment, tmp_path, monkeypatch):
    """Persisted records are not automatically rescored when a bundle changes."""
    _, _, results = experiment
    bundle = publish_bundle(tmp_path / "bundle", artifacts_for(results), {})
    monkeypatch.setenv("GRAVITY_ML_BUNDLE", str(bundle))

    class Store:
        def active(self):
            return {"quality": {"model_bundle_id": "different-bundle"}}

    monkeypatch.setattr(api, "_runtime_store", lambda: Store())
    response = TestClient(api.app).get("/metrics")
    assert response.status_code == 200
    assert response.json()["available"] is False
    manifest = read_manifest(bundle)
    assert manifest["files"]["bundle.joblib"] == digest((bundle / "bundle.joblib").read_bytes())
