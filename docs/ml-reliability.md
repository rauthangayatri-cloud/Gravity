# Local demonstration contract and preservation

## What is complete

Python 3.13.7 is the demo target. New ML bundles contain the selected per-target
estimator, calibrator, threshold, fixed single-snapshot schema, metrics and outputs.
New dashboard/runtime/what-if calculations share that predictor. See `methodology.md`
for evaluation limits and the unchanged Success Score formula. Legacy files remain
supported with explicit limitations, not retroactively validated.

`GRAVITY_ML_BUNDLE` is a model/demo-artifact setting, not a faculty dataset selection
or authorization boundary. The source supports new bundles; activating one requires
a separately approved offline build. Tests only fit tiny seeded synthetic fixtures
in temporary directories. No existing data/model is regenerated during verification.

## Explicit publication, activation and rollback

Read the existing synthetic clean inputs and choose a nonexistent private directory:

```powershell
# Separate approval required; this is not a normal startup or verification command.
python -m src.build_outputs --unified data/clean/unified.csv --outcomes data/clean/outcomes.csv --bundle-dir data/outputs/bundles/reviewed-demo-v1
```

`python -m src.train` uses the same publisher and required `--bundle-dir`. Existing
destinations are refused. The publisher reserves a fresh directory exclusively,
writes members, then publishes a manifest completion marker. An interrupted build
can leave a new incomplete directory for manual inspection; loaders refuse it.
No old files are deleted and no active pointer is changed. Keep completed bundles
immutable and do not commit them.

The manifest records input/source hashes, ordered features, model selection,
thresholds, split counts and hashed ID sets, seed and installed numerical package
versions. Loaders verify member hashes and require matching numerical packages and
Python major/minor. The cache keys models by verified content, not mutable paths.
Source hashes are provenance, not an enforced code signature: deploy matching source
and maintain the feature-schema version. Cross-platform bitwise reproduction is
not promised by direct dependency pins alone.

After reviewing the metrics, explicitly set `GRAVITY_ML_BUNDLE` to the completed
bundle and start a new session as shown in README. Saved runtime rows retain their
bundle ID and are not rescored; `/metrics` refuses to attribute another bundle's
evaluation to them. Invalid configured bundles fail without legacy fallback.

Rollback means explicitly selecting a retained reviewed bundle and restarting,
not replacing files or migrating data. Unsetting the variable selects labelled
legacy mode. Back up the entire immutable bundle plus its recorded environment;
verify all manifest hashes after restoration. Joblib/pickle can execute code:
checksums detect corruption/mixing, not malicious publishers. Only operator-trusted
bundles may be loaded. Model uploads are not supported.

## Security and database work deliberately deferred

Phase 1 supplies new schema/configuration, restrictive foreign keys, transactions,
and institution-scoped storage primitives. It is not yet formally approved or
integrated into API routes. Existing database test results are historical evidence,
not evidence that authentication/isolation exists. Legacy SQLite is not migrated.

The remaining dependency order is:

1. Review Phase 1 and isolated PostgreSQL behavior.
2. Database-backed custom JWT accounts and explicit administrator provisioning;
   no default passwords, users.json or external identity provider.
3. Central authorization: institution-limited admins; owner-only faculty with
   department permission; explicitly granted read-only viewers.
4. Versioned dataset/student/intervention integration and per-user selection,
   transactional publication, stale-version checks and independent resets.
5. Bounded content-validated CSV preparation, owned expiring previews, explicit
   confirmation, all-or-nothing publication and import-specific cleanup.
6. Tested persistence, backups/restoration and shared-deployment verification.

Current routes still use a global legacy dataset pointer and unauthenticated
interventions/uploads. File-size/row/column limits, secure previews and multi-user
concurrency controls remain missing. UI roles/filters do not protect data. Restrict
the demo to one trusted operator, small synthetic inputs and localhost. No public
tunnel, real student data, or shared deployment is approved.

## Safe verification

The non-foundation suite can run without ignored local model/data files:

```powershell
python -B -m pytest -o addopts= -p no:cacheprovider -q -ra --ignore=tests/test_database.py --ignore=tests/test_migrations.py
node --test web/src/analytics.test.js
```

Database/migration tests are separately documented in `auth-data-contract.md` and
must use disposable targets. No live migration is part of startup. A missing
PostgreSQL test URL is a skip, never a server-validation pass.

Readiness checks artifacts without creating a database; liveness only checks the
process. Neither is a production-readiness guarantee. Docker builds fit new
synthetic artifacts inside their image and therefore require separate approval;
they do not copy or overwrite host data. Compose uses loopback ports and separate
named volumes. A volume is persistence, not a tested backup/restore strategy.

Still unproven: real-world validity, temporal/institution generalization, broader
source-outage/subgroup robustness, repeated-split selection stability, forecast
coverage, intervention effectiveness and production load/recovery behavior.
