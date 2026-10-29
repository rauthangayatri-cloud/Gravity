"""Explicit fresh-bundle publication and integrity-checked, read-only loading."""

from __future__ import annotations

import hashlib
import io
import json
import math
import os
import platform
from functools import lru_cache
from importlib.metadata import version
from pathlib import Path
from typing import Any

import joblib
import pandas as pd

from src.features import SNAPSHOT_SCHEMA

FORMAT_VERSION = 1
PACKAGES = ("numpy", "pandas", "scikit-learn", "scipy", "xgboost", "shap", "joblib", "numba", "llvmlite")
TARGETS = {"academic_risk", "placement_risk"}


class ArtifactError(ValueError):
    """A configured bundle is incomplete, incompatible, or has changed."""


def digest(content: bytes) -> str:
    """Hash content, not timestamps, for artifact identity."""
    return hashlib.sha256(content).hexdigest()


def environment() -> dict[str, Any]:
    """Record the actual numerical environment, including transitive dependencies."""
    return {"python": platform.python_version(), "packages": {name: version(name) for name in PACKAGES}}


def configured_bundle() -> Path | None:
    """Never select a new bundle implicitly or mutate process configuration."""
    value = os.getenv("GRAVITY_ML_BUNDLE", "").strip()
    return Path(value).resolve() if value else None


def publish_bundle(destination: Path, artifacts: dict[str, Any], provenance: dict[str, Any]) -> Path:
    """Reserve a new directory; the manifest is the final atomic commit marker.

    A failed publication leaves an incomplete new directory for inspection. No
    old directory/file is overwritten or deleted, and no active pointer changes.
    """
    destination = destination.resolve()
    destination.mkdir(parents=True, exist_ok=False)
    predictors = {target: result["predictor"] for target, result in artifacts["models"].items()}
    if set(predictors) != TARGETS:
        raise ArtifactError("Both risk targets are required")
    payload = io.BytesIO()
    joblib.dump(predictors, payload)
    files = {"bundle.joblib": payload.getvalue()}
    for name, value in artifacts.items():
        if isinstance(value, pd.DataFrame):
            files[f"{name}.csv"] = value.to_csv(index=False).encode("utf-8")
    files["metrics.json"] = json.dumps(artifacts["metrics"], indent=2, allow_nan=False).encode("utf-8")
    for name, content in files.items():
        with (destination / name).open("xb") as stream:
            stream.write(content)
    manifest = {
        "format": FORMAT_VERSION,
        "feature_schema": SNAPSHOT_SCHEMA,
        "seed": 42,
        "synthetic_only": True,
        "environment": environment(),
        "provenance": provenance,
        "source_hashes": {p.name: digest(p.read_bytes()) for p in sorted(Path(__file__).parent.glob("*.py"))},
        "files": {name: digest(content) for name, content in files.items()},
        "targets": {
            target: {"model": p.model_name, "threshold": p.threshold, "features": list(p.feature_names_in_)}
            for target, p in predictors.items()
        },
        "splits": {
            target: {
                name: {"count": len(ids), "sha256": digest(json.dumps(sorted(ids)).encode())}
                for name, ids in result["split_students"].items()
            }
            for target, result in artifacts["models"].items()
        },
    }
    marker = destination / ".manifest.incomplete"
    with marker.open("x", encoding="utf-8") as stream:
        json.dump(manifest, stream, indent=2, allow_nan=False)
        stream.flush()
        os.fsync(stream.fileno())
    marker.rename(destination / "manifest.json")
    return destination


def read_manifest(directory: Path) -> dict[str, Any]:
    """Refuse incomplete/unknown schemas; never fall back to legacy on failure."""
    try:
        manifest = json.loads((directory / "manifest.json").read_text(encoding="utf-8"))
        if (
            not isinstance(manifest, dict)
            or not isinstance(manifest.get("files"), dict)
            or not isinstance(manifest.get("targets"), dict)
        ):
            raise ArtifactError("Invalid ML bundle manifest structure")
        if manifest["format"] != FORMAT_VERSION or manifest["feature_schema"] != SNAPSHOT_SCHEMA:
            raise ArtifactError("Unsupported ML bundle schema")
        if set(manifest["targets"]) != TARGETS or "bundle.joblib" not in manifest["files"]:
            raise ArtifactError("Incomplete ML bundle")
        for contract in manifest["targets"].values():
            if (
                not isinstance(contract, dict)
                or contract.get("model") not in {"logistic_regression", "xgboost"}
                or not isinstance(contract.get("features"), list)
                or not all(isinstance(name, str) for name in contract["features"])
                or len(set(contract["features"])) != len(contract["features"])
                or not isinstance(contract.get("threshold"), (int, float))
                or not math.isfinite(contract["threshold"])
                or not 0 <= contract["threshold"] <= 1
            ):
                raise ArtifactError("Invalid predictor contract")
        saved = manifest.get("environment")
        if (
            not isinstance(saved, dict)
            or not isinstance(saved.get("python"), str)
            or not isinstance(saved.get("packages"), dict)
        ):
            raise ArtifactError("Invalid ML environment metadata")
        return manifest
    except (OSError, KeyError, TypeError, json.JSONDecodeError) as error:
        raise ArtifactError("ML bundle manifest is unavailable or invalid") from error


def verified_bytes(directory: Path, name: str, manifest: dict[str, Any] | None = None) -> bytes:
    """Read only named, checksummed bundle members; reject traversal and symlinks."""
    manifest = read_manifest(directory) if manifest is None else manifest
    if not name or any(c in name for c in ("/", "\\", ":")) or name not in manifest["files"]:
        raise ArtifactError("Unknown ML bundle member")
    member = directory / name
    if member.is_symlink() or member.resolve().parent != directory.resolve():
        raise ArtifactError("ML bundle member is outside the bundle")
    try:
        content = member.read_bytes()
    except OSError as error:
        raise ArtifactError("ML bundle member is unavailable") from error
    if digest(content) != manifest["files"][name]:
        raise ArtifactError("ML bundle integrity check failed")
    return content


@lru_cache(maxsize=2)
def _deserialize(content: bytes) -> dict:
    """Cache by verified content, never mutable path; trusted local pickle only."""
    return joblib.load(io.BytesIO(content))


def load_bundle(directory: Path) -> tuple[dict, str]:
    """Load matching predictors; hashes detect corruption, not malicious publishers."""
    manifest = read_manifest(directory)
    current = environment()
    saved = manifest.get("environment", {})
    if (
        saved.get("packages") != current["packages"]
        or saved.get("python", "").split(".")[:2] != current["python"].split(".")[:2]
    ):
        raise ArtifactError("ML bundle numerical environment differs; use its recorded environment")
    content = verified_bytes(directory, "bundle.joblib", manifest)
    predictors = _deserialize(content)
    from src.prediction import RiskPredictor

    if not isinstance(predictors, dict) or set(predictors) != TARGETS:
        raise ArtifactError("Invalid predictor payload")
    for target, predictor in predictors.items():
        contract = manifest["targets"][target]
        if (
            not isinstance(predictor, RiskPredictor)
            or predictor.target != target
            or predictor.schema != SNAPSHOT_SCHEMA
            or predictor.model_name != contract["model"]
            or predictor.threshold != contract["threshold"]
            or list(predictor.feature_names_in_) != contract["features"]
        ):
            raise ArtifactError("Predictor does not match its manifest")
    return predictors, manifest["files"]["bundle.joblib"]
