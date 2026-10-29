"""Regression checks for the clean Docker deployment contract."""

from pathlib import Path

ROOT = Path(__file__).parents[1]


def test_docker_build_recreates_runtime_artifacts_without_host_state():
    """The image rebuilds models and outputs instead of copying local artifacts."""
    dockerfile = (ROOT / "Dockerfile").read_text(encoding="utf-8")
    dockerignore = (ROOT / ".dockerignore").read_text(encoding="utf-8")

    for command in (
        "python -m src.generate_data",
        "python -m src.clean",
        "python -m src.build_outputs --unified data/clean/unified.csv --outcomes data/clean/outcomes.csv",
        '--bundle-dir "$GRAVITY_ML_BUNDLE"',
        'test -s "$GRAVITY_ML_BUNDLE/manifest.json"',
        'test -s "$GRAVITY_ML_BUNDLE/bundle.joblib"',
        "${PORT:-8000}",
    ):
        assert command in dockerfile

    for ignored_path in ("models/", "data/raw/", "data/clean/", "data/outputs/", "api/*.sqlite3", ".venv/"):
        assert ignored_path in dockerignore


def test_runtime_requirements_are_exactly_pinned_for_image_rebuilds():
    """Avoid model drift from resolving a newer direct dependency during image builds."""
    requirements = (ROOT / "requirements.txt").read_text(encoding="utf-8").splitlines()
    active = [line for line in requirements if line and not line.startswith("#")]
    assert active
    assert all("==" in line for line in active)


def test_deployment_python_matches_supported_local_numerical_environment():
    """NumPy/SHAP require 3.12+; use the verified 3.13 interpreter consistently."""
    assert "FROM python:3.13.7-slim" in (ROOT / "Dockerfile").read_text()
    assert 'python-version: "3.13.7"' in (ROOT / ".github/workflows/ci.yml").read_text()
    assert (ROOT / ".python-version").read_text().strip() == "3.13.7"
    assert 'python_version = "3.13"' in (ROOT / "pyproject.toml").read_text()


def test_local_compose_and_image_preserve_host_state():
    """Bindings are loopback-only; no writable repository or legacy DB is mounted."""
    compose = (ROOT / "docker-compose.yml").read_text()
    docker = (ROOT / "Dockerfile").read_text()
    assert '"127.0.0.1:8000:8000"' in compose
    assert '"127.0.0.1:5173:5173"' in compose
    assert "./web:/app/web:ro" in compose
    assert "demo-state:/state" in compose
    assert "RUNTIME_STATE_DB: /state/demo.sqlite3" in compose
    assert "USER gravity" in docker
    assert "HEALTHCHECK" in docker
    assert "exec uvicorn" in docker
