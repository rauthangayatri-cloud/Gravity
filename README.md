# Smart Campus Student Success Platform

## Windows 11 setup and run path

This is a **single-operator, localhost-only demonstration using synthetic data**.
It has no authentication or faculty/institution isolation and must not be published
or used for real student decisions. Frontend role/department controls are filters,
not authorization. The database foundation is not integrated into these routes.

Use Python **3.13.7**, as recorded in `.python-version`, Docker and CI. The pinned
NumPy and SHAP dependencies do not support Python 3.11. In PowerShell at the repository root:

```powershell
# Only if creating a NEW environment; do not overwrite an existing environment.
py -3.13 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

If PowerShell blocks activation, run `Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass` and activate the environment again.

Generation, training, SHAP evaluation, and output building are offline. The
demo dashboard reads precomputed files. Uploaded and individually added
students are validated, persisted locally, and passed through the saved model
pipelines for lightweight inference only; the API never retrains a model.

### Existing demo: preserve artifacts and use new session state

Do not run generation/cleaning/training over this checkout's existing data. Existing
`data/outputs/` and `models/` support the labelled legacy demonstration. The offline
pipeline expects these cleaned inputs, but normal startup does not modify them:

```text
data/clean/unified.csv
data/clean/outcomes.csv
```

Choose a fresh local SQLite session path so uploads/interventions cannot change
an existing session. These commands do not initialize/migrate the Phase 1 database:

```powershell
$demoSession = Join-Path $env:TEMP ("gravity-demo-" + [guid]::NewGuid().ToString("N"))
New-Item -ItemType Directory -Path $demoSession -ErrorAction Stop | Out-Null
$env:RUNTIME_STATE_DB = Join-Path $demoSession "session.sqlite3"
$env:GRAVITY_ML_BUNDLE = "" # Explicit legacy demonstration; see limitations below.
uvicorn api.main:app --host 127.0.0.1 --port 8000
```

`GET /health` checks the running process. `GET /ready` checks model/demo artifact
availability without opening or creating a database; it is not a production or
database-readiness guarantee. Stop with Ctrl+C. Keep the session directory if you
want its intervention history; no automatic cleanup or migration is performed.

### Optional new ML bundle — separate explicit approval/build

For consistent calibrated single-snapshot inference, review and explicitly authorize
a build from the existing synthetic clean inputs into a **nonexistent** directory:

```powershell
python -m src.build_outputs --unified data/clean/unified.csv --outcomes data/clean/outcomes.csv --bundle-dir data/outputs/bundles/reviewed-demo-v1
# After reviewing its manifest/metrics, select it for a NEW demo session:
$env:GRAVITY_ML_BUNDLE = (Resolve-Path data/outputs/bundles/reviewed-demo-v1).Path
uvicorn api.main:app --host 127.0.0.1 --port 8000
```

In a second PowerShell window, start the React app:

```powershell
cd web
npm ci
npm run dev -- --host 127.0.0.1
```

Open the Vite address shown in the terminal, normally `http://localhost:5173`. Set `VITE_API_URL` if the API runs elsewhere.

## Runtime student data

With a fresh session, the app opens in **Demo data** mode. Use **Upload student data** to submit a
CSV, or **Add student** to enter one record. The active user dataset is stored
at `RUNTIME_STATE_DB`, separately from the immutable demo CSVs. Without that setting,
the legacy default is `api/interventions.sqlite3`; avoid it when preserving old state.
**Reset to Demo** switches dashboard reads back to the original precomputed
demo without deleting the stored user dataset.

CSV uploads require these columns (common aliases such as `attendance`,
`backlogs`, and `assignment_score` are accepted):

```text
student_id, department, cgpa, active_backlogs, attendance_pct, lms_logins,
assignment_completion_pct, coding_score, aptitude_score, mock_interview_score
```

Department must be one of Business, Computer Science, Digital Media, or
Mechanical Engineering. Numeric values are validated before inference. New
students in **bundle mode** use the same selected estimator, saved calibrator,
single-snapshot features and threshold as dashboard risks and what-if baselines.
Runtime segmentation uses fixed descriptive rules; it does not fit clustering.

In **legacy mode**, uploads use uncalibrated XGBoost with history features unavailable
at upload time. Their accuracy is unvalidated; old offline metrics do not evaluate
this runtime path. Legacy what-if comparisons are intentionally unavailable because
their estimator can differ from the displayed risk. Files are retained unchanged.
Saved user datasets are not automatically rescored when configuration changes.

Uploads still lack secure previews, configured resource limits and multi-user
authorization. Use only small, trusted synthetic inputs. Do not expose these endpoints.

## Linux/macOS setup and run path

From the repository root:

```bash
python3.13 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
export RUNTIME_STATE_DB="$(mktemp -d)/session.sqlite3"
export GRAVITY_ML_BUNDLE=""
uvicorn api.main:app --host 127.0.0.1 --port 8000
```

In a second terminal:

```bash
cd web
npm ci
npm run dev -- --host 127.0.0.1
```

## Optional local Docker demo, not public deployment

The API image is self-contained: it intentionally excludes generated data,
models, local SQLite state, caches, and virtual environments from the Docker
context. During `docker build`, it generates synthetic raw data, cleans it,
trains new synthetic models through `build_outputs` inside the image, and verifies
the immutable bundle. This is an explicit build operation requiring approval;
it is never part of application startup. Do not commit model bundles or generated
data files.

Build and run the API locally:

```powershell
# Only after approving an image build that creates new synthetic artifacts:
docker compose up --build
```

Compose publishes only to 127.0.0.1. The API runs as a non-root user and stores
its separate demo database in the `demo-state` named volume. The frontend mounts
source read-only and uses a separate dependency volume. Do not use `down -v` if
you need that state. No host legacy database is mounted or migrated. This remains
a development server, not a hardened deployment or a tested backup system.

`VITE_API_URL` is public build-time configuration; never put secrets in `VITE_*`.
The default localhost URL works for a browser on the same computer. CORS is not
authentication. Environment files are optional; Uvicorn only loads one when
explicitly passed `--env-file`. Do not modify an existing local `.env` blindly.

## Quality checks

Run these commands from an activated Python environment at the repository root:

```powershell
python -m pip install -r requirements-dev.txt
python -m ruff check .
python -m ruff format --check .
python -m mypy
python -B -m pytest -o addopts= -p no:cacheprovider -q -ra --ignore=tests/test_database.py --ignore=tests/test_migrations.py
python -m bandit -r src api
python -m pip_audit
cd web
npm ci
npm run lint
node --test src/analytics.test.js
npm run build
```

On Linux or macOS, use the same commands after `source .venv/bin/activate`.

The demo/ML tests fit tiny **test-only** synthetic models and write only temporary
fixtures, never ignored developer models or existing outputs. Database/migration
tests are separate explicit checks in `docs/auth-data-contract.md`; do not run live
migrations as a demo setup step. For preservation checks, build the frontend into
a fresh temporary output directory instead of overwriting an existing `web/dist`.

See [the prediction contract and remaining security work](docs/ml-reliability.md).

An intentionally lightweight, reproducible hackathon prototype. It uses **synthetic data only** and is not evidence of real-world accuracy, fairness, causation, or student outcomes.

The dashboard keeps three concepts separate:

- Success Score (0–100): transparent current-state weighted formula.
- Academic and placement risk: separate offline ML probabilities.
- Priority score: `risk × severity × trend multiplier` for intervention ordering.

See [the data dictionary](docs/data_dictionary.md), [project note](docs/note.md), [slide outline](docs/slides_outline.md), and [demo script](docs/demo_script.md).
