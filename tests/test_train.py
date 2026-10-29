from types import SimpleNamespace

import numpy as np
import pandas as pd

from src.train import _bootstrap_auc_ci, _calibration_table, _lift_at_k, _shap_drivers, build_targets


def test_build_targets_matches_documented_rules():
    outcomes = pd.DataFrame(
        {
            "student_id": ["H0001", "C0001"],
            "cohort": ["historical", "current"],
            "next_semester_cgpa_drop": [0.6, 0.1],
            "new_backlog": [0, 0],
            "placement_result": ["placed", "not_placed"],
            "offer_amount": [600_000, 600_000],
        }
    )
    targets = build_targets(outcomes)
    assert targets.academic_risk.tolist() == [1, 0]
    assert targets.placement_risk.tolist() == [0, 1]


def test_shap_driver_name_uses_imputer_output_columns(monkeypatch):
    """The selected driver name must match the largest absolute SHAP value."""

    class Imputer:
        def transform(self, features):
            return features[["feature_b", "feature_a"]].to_numpy()

        def get_feature_names_out(self, input_features):
            return np.array(["feature_b", "feature_a"])

    class Explainer:
        def __call__(self, transformed):
            return SimpleNamespace(values=np.array([[0.2, -0.9]]))

    model = SimpleNamespace(named_steps={"imputer": Imputer(), "model": object()})
    monkeypatch.setattr("src.train.shap.TreeExplainer", lambda _: Explainer())
    features = pd.DataFrame({"feature_a": [1.0], "feature_b": [2.0], "all_missing": [np.nan]})

    drivers = _shap_drivers(model, features, pd.Series(["C0001"]), "academic_risk")

    assert drivers.iloc[0]["feature"] == "feature_a"
    assert drivers.iloc[0]["magnitude"] == 0.9


def test_lift_bootstrap_and_calibration_summary_are_deterministic():
    truth = pd.Series([0, 0, 0, 1, 1, 1])
    probability = np.array([0.05, 0.10, 0.20, 0.70, 0.80, 0.95])
    assert _lift_at_k(truth, probability, 0.20) == 2.0
    assert _bootstrap_auc_ci(truth, probability) == _bootstrap_auc_ci(truth, probability)
    table = _calibration_table(truth, probability, "academic_risk", bins=3)
    assert set(table) == {"target", "bin", "student_count", "mean_predicted_probability", "observed_rate"}
    assert table.student_count.sum() == len(truth)
