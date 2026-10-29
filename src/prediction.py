"""Shared selected-model prediction and explicitly scoped raw-model explanations."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
import shap

from src.features import SNAPSHOT_SCHEMA


@dataclass
class RiskPredictor:
    """An offline-fitted estimator, calibrator and development-selected threshold."""

    target: str
    model_name: str
    model: Any
    calibrator: Any
    threshold: float
    schema: str = SNAPSHOT_SCHEMA

    @property
    def feature_names_in_(self):
        """Expose the exact trained input order."""
        return self.model.feature_names_in_

    def matrix(self, features: pd.DataFrame) -> pd.DataFrame:
        """Reject schema drift rather than silently fabricating model columns."""
        if not features.columns.is_unique or set(features.columns) != set(self.feature_names_in_):
            raise ValueError("Model feature schema mismatch")
        return features.loc[:, list(self.feature_names_in_)]

    def predict_proba(self, features: pd.DataFrame) -> np.ndarray:
        """Apply the same saved calibration for every prediction consumer."""
        raw = self.model.predict_proba(self.matrix(features))[:, 1]
        result = self.calibrator.predict_proba(raw.reshape(-1, 1))
        if not np.isfinite(result).all() or (result < 0).any() or (result > 1).any():
            raise ValueError("Invalid model probabilities")
        return result

    def drivers(self, features: pd.DataFrame, student_ids: pd.Series) -> pd.DataFrame:
        """Explain the selected base model's log-odds, not calibrated probability."""
        matrix = self.matrix(features)
        imputer = self.model.named_steps["imputer"]
        transformed = imputer.transform(matrix)
        names = imputer.get_feature_names_out(matrix.columns)
        estimator = self.model.named_steps["model"]
        if self.model_name == "logistic_regression":
            # Independent-feature linear SHAP about the training mean. The fitted
            # StandardScaler centers the training columns at zero; no fit here.
            transformed = self.model.named_steps["scale"].transform(transformed)
            values = transformed * estimator.coef_[0]
            base = np.full(len(matrix), float(estimator.intercept_[0]))
        else:
            explanation = shap.TreeExplainer(estimator)(transformed)
            values = np.asarray(explanation.values)
            base = np.asarray(explanation.base_values)
        rows = []
        for position, student_id in enumerate(student_ids):
            for rank, column in enumerate(np.argsort(np.abs(values[position]))[-3:][::-1], start=1):
                value = float(values[position, column])
                rows.append(
                    {
                        "student_id": student_id,
                        "target": self.target,
                        "model": self.model_name,
                        "rank": rank,
                        "feature": names[column],
                        "direction": "increases risk" if value > 0 else "reduces risk",
                        "magnitude": round(abs(value), 5),
                        "units": "base-model log-odds",
                        "scope": "Signs describe the selected base model only, not calibrated probability or causal effects",
                        "base_value": float(base[position]),
                        "raw_margin": float(base[position] + values[position].sum()),
                    }
                )
        return pd.DataFrame(rows)
