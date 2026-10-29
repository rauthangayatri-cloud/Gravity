# Gravity authentication and data contract

Approved architecture amendment to Prompt 3: Gravity issues its own JWTs and
uses database-backed accounts as its only identity and permission store. There
is no authoritative `users.json` and no external identity provider.

## Authentication and provisioning

- Accounts are explicitly provisioned by an institution administrator; there is
  no public registration, default password, or anonymous demo identity.
- Store password hashes, never plaintext passwords. Signing secrets come from
  environment configuration and must never enter source control.
- JWT verification must enforce the configured application issuer, audience,
  expiry, and fixed allowed algorithm. The subject is an internal user UUID.
- Resolve current account/membership status and permissions server-side on every
  request. Disabling access preserves historical records.
- Initial administrator bootstrap is a separate, explicit provisioning command;
  schema migration never creates an account or credential.

## Authorization

- Administrators may access/manage data only in their active institution.
- Faculty must own the dataset AND have permission for every department in its
  declared scope. Department membership does not grant access to peers' data.
- Viewers require an explicit dataset grant AND all required department
  permissions. They may read authorized data/history and change their own
  selection, but cannot change datasets or create interventions.
- All roles require an active account, institution, and membership. Dataset
  versions, summaries, counts, explanations, and interventions inherit the same
  access checks. Client role switches, filters, and submitted owner IDs confer
  no permissions. An inaccessible resource must not disclose its existence.
- Records must belong to a department declared by their dataset. Mixed-
  department datasets are denied as a whole when department coverage is missing.
- Intervention creation must reference an authorized student record/version;
  capture its snapshot and authenticated creator on the server. Viewer access
  to intervention history follows dataset access, not intervention authorship.
- The weights endpoint is a read-authorized, non-persistent scenario operation.

## Identity, versions, and selection

- External student IDs are text (including leading zeros). Trim and case-fold
  their lookup keys. Identity is dataset-local, not institution-wide or global.
- A student's internal identity is stable across versions of the same dataset.
  Matching IDs in different datasets do not imply the same identity.
- Published dataset versions are immutable. Preparation happens outside the
  publication transaction; insert all records and conditionally update selection
  in one transaction. Failed imports retain the previous valid selection.
- Selection is keyed by user/institution and references one authorized version.
  Selection never grants access. Revision checks prevent stale publication from
  overwriting a newer selection/reset. Resolve one version per dashboard view.
- Reset selects only that user's authorized private demo dataset (or clears
  selection if none exists). It deletes no data or intervention history.
- Archive rather than cascade-delete referenced datasets, versions, or accounts.

## Storage and migration boundaries

PostgreSQL is required for shared deployment. SQLite is only for authenticated,
single-user local development/demo, with foreign keys enabled on every
connection. No external identity provider or PostgreSQL row-level security is
required for the minimum implementation; application authorization is mandatory.

Use a separately configured new database. Never modify, migrate, overwrite,
delete, or regenerate existing databases, datasets, models, or outputs as a side
effect of startup or schema setup. Legacy migration needs separate approval,
backup, dry run, explicit ownership/history mappings, reconciliation, and a
rollback/cutover plan. Unresolved history must be retained, not guessed away.

## Phase 1 operational boundary

Phase 1 supplies schema/configuration/storage foundations only. Existing API
routes are deliberately not switched to this database and remain unsuitable for
shared deployment until authentication and endpoint authorization are completed.
No `create_all`, account creation, migration, or seed runs at application import.

`DATABASE_URL` is required by the new layer; `GRAVITY_MODE=shared` only accepts
PostgreSQL. `RUNTIME_STATE_DB` remains solely the old API's legacy setting.
The configured SQLite parent directory must already exist; setup never guesses
or creates a broad directory tree. The known legacy path is rejected, and
database inspection also rejects legacy tables at other paths.

After separately choosing a **new** database target, explicit schema setup is:

```powershell
# Set DATABASE_URL and GRAVITY_MODE in this shell; do not use the legacy file.
python -m alembic -x confirm_schema_change=yes upgrade head
```

Alembic does not automatically load `.env`. No live migration command is executed
as part of this phase. Destructive downgrades are intentionally disabled; use a
reviewed backup/restore plan. New tables contain no accounts or student data.

The Phase 1 database tests run against temporary SQLite files. To additionally
run the same checks against PostgreSQL, explicitly set
`GRAVITY_TEST_POSTGRES_URL` to a disposable local database whose name starts with
`gravity_test_`. Tests create a unique schema and remove only that schema. They
never fall back to `DATABASE_URL`. An absent test URL produces visible skips,
not a PostgreSQL pass. Concurrent application-operation tests belong to Phase 3.

### Foundation implementation details

The Phase 1 implementation uses SQLAlchemy Core tables and explicit connection
transactions, not ORM relationship cascades. Its thirteen tables are:
`institutions`, `users`, `memberships`, `departments`,
`membership_departments`, `datasets`, `dataset_departments`, `dataset_grants`,
`dataset_versions`, `dataset_students`, `student_records`, `interventions`, and
`user_dataset_selections`. Alembic additionally maintains `alembic_version`.

UUIDs identify records. PostgreSQL uses native UUID/JSONB; SQLite uses SQLAlchemy's
portable UUID/JSON types. The frozen initial revision has no dependency on mutable
application metadata. Checks enforce valid role/kind values, score/probability
ranges, nonnegative revisions/counts, and complete-or-empty selections. Unique
constraints enforce dataset-local student keys, version numbers, and one record
per student/version. Composite restrictive foreign keys prevent mismatched
institutions, datasets, versions, departments, students, and interventions.

`DatasetRepository` requires an explicit institution UUID. It provides scoped
dataset, version, selection, student-identity, student-record, and intervention
lookups, plus normalized student-identity insertion. Insertions do not commit;
`transaction(engine)` owns the whole unit of work and rolls it back on failure.
External IDs remain text, including leading zeros; lookup keys use trim/casefold.

These primitives are **not authorization**. Owner/department/viewer decisions,
current membership checks, immutable-publication enforcement, cardinality checks
against `student_count`, optimistic revision updates, and upload/reset workflows
remain deferred. No route uses these primitives yet. Direct SQL writes are not
an alternative authorized interface. Role/department tables are schema only;
there is no account provisioning, JWT implementation, or login in this phase.

### Safe local initialization (optional, not run against project data)

Run from the repository root after reviewing the target. The following creates
a uniquely named **new** directory/database; it never reuses the legacy file:

```powershell
$phase1Directory = Join-Path (Get-Location).Path ("data/runtime/phase1-" + [guid]::NewGuid().ToString("N"))
New-Item -ItemType Directory -Path $phase1Directory -ErrorAction Stop | Out-Null
$env:GRAVITY_MODE = "local"
$env:DATABASE_URL = "sqlite:///" + ((Join-Path $phase1Directory "gravity.sqlite3") -replace '\\', '/')
.\.venv\Scripts\python.exe -B -m alembic -x confirm_schema_change=yes upgrade head
```

This initializes empty schema only, not a demo dataset or user. An explicit repeat
of `upgrade head` on the recognized complete schema is a no-op. Existing legacy,
unversioned, unrelated, partial, or unknown-revision databases are refused. No
database is repaired automatically. Offline migration execution is disabled so
target safety checks cannot be skipped through that CLI mode.

For PostgreSQL, provision a separate empty database/schema yourself, set
`GRAVITY_MODE=shared` and a private `DATABASE_URL` using the `postgresql+psycopg`
driver, then use the same explicit Alembic command. The application does not
create a PostgreSQL database or configure roles. Keep runtime and migration role
separation for the deployment phase; no credentials belong in examples or Git.

Focused checks (no coverage/cache files written into the repository):

```powershell
.\.venv\Scripts\python.exe -B -m pytest tests/test_database.py tests/test_migrations.py -o addopts= -p no:cacheprovider -q -ra
```

Without `GRAVITY_TEST_POSTGRES_URL`, PostgreSQL execution cases are explicitly
skipped. SQL compilation is checked separately and is not live-server validation.
Fixtures never read `DATABASE_URL` as a fallback. Each PostgreSQL case creates
and removes only its own randomly named schema inside the explicitly opted-in,
local `gravity_test_*` database; existing schemas and databases are not dropped.
