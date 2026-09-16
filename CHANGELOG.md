# v3.2.0 (16-09-2026)

## Added
- Algorithm benchmarking API: `POST /api/v0/benchmark-algorithm` starts a benchmark using an algorithm's synthetic-data `benchmark()` implementation, and `GET /api/v0/benchmarks/{benchmark_id}` returns its persisted status, log, and result data.
- Benchmark execution support for both FastAPI background tasks and Celery, including stop/failure handling and emergency-record fallback. Server startup initializes a `benchmark-store` collection.
- `benchmark_outputs` algorithm metadata and a `BaseRunner.benchmark()` extension point; the algorithm template and workflow documentation include the benchmark contract.
- Algorithm storage metrics in algorithm records and responses: logical byte size, module/asset/checkpoint size breakdowns and counts, plus the recomputation timestamp. Metrics account for unique referenced module and asset objects within an algorithm.
- `get_object_sizes()` storage-backend support and implementations for S3, in-memory, and temporary-file connections.
- `Image2MultiRegionSegmentationRunner`, `MultiRegionSegmentationSchema`, shared HDF5 helpers, and a multi-region segmentation algorithm template.

## Changed
- API failures are now normalized through typed Compox exceptions. Error responses consistently include a machine-readable `code` and `retryable` flag, while retaining `detail`; task and emergency records retain structured failure metadata.
- Algorithm deployment, export, bundle validation/import, execution, training, checkpoints, files, and storage errors now expose specific typed errors with stable codes and contextual details. Server-side errors continue to avoid exposing their internal cause in API responses.
- `inference.algorithm_cache_maxsize` configures the runner cache capacity. Cache keys now use a tuple of argument representations rather than a concatenated string.
- Algorithm deployment, training, checkpoint, and algorithm-record timestamps are now stored as UTC ISO 8601 timestamps.
- S3 lifecycle configuration requests now include `Content-MD5` for MinIO compatibility. Retried object uploads use exponential backoff, and S3 object sizes are retrieved without downloading object bodies.
- Deployment tolerates transient file locks when renaming staged module directories.
- Algorithm records with missing or stale storage metrics refresh them lazily when read; checkpoint changes mark the parent algorithm's metrics stale and algorithm-version deletion recomputes them.

## Fixed
- Corrected the bundle build/import CLI commands so they honor the CLI configuration path and overrides.
- Checkpoint deletion no longer makes a redundant existence check before loading the checkpoint record.
- The built-in MinIO downloader now uses pinned GitHub release assets and fails on unsuccessful HTTP responses.
- Removed unsuitable thresholding options from the multi-region segmentation template.

## Compatibility Notes
- Custom `BaseConnection` implementations should add `get_object_sizes(collection_name, object_names)` to support algorithm storage metrics.
- Clients that parse error bodies should accept the additional `code` and `retryable` fields. The existing `detail` field is preserved.
- Existing algorithm records without metrics are refreshed lazily by the algorithm-list APIs; installations will also gain the `benchmark-store` collection at startup.

# v3.1.0 (19-05-2026)
## Added
- Compox algorithm bundle support, including bundle building, bundle-backed storage access, builtin bundle import, and persistent zip-based runtime importing.
- `AlgorithmRecordRegistrar` for shared semantic algorithm record lookup, versioning, and storage.
- Emergency fallback record storage plus storage exception classification for execution, training, and deployment failures.
- CLI support for bundle-related deployment and packaging workflows.
- Expanded test coverage for bundles, emergency record fallback, storage exceptions, and CLI flows.

## Changed
- Algorithm deployment was refactored around the new registrar and runtime zip importer flow.
- Builtin algorithm import is now wired into API startup through the bundle importer path.
- Server configuration was expanded for bundle paths, bundle keys, and runtime module cache behavior.
- Algorithm API responses now include the backward-compatible `algorithm_minor_version` field alongside `latest_algorithm_minor_version`.
- ParticleSeg3D tutorial packaging/docs were updated, including a `numpy<2` constraint for that tutorial flow.

## Fixed
- OpenAPI documentation for binary file download responses now declares the correct response type.
- Template algorithm README snippets were updated to match current runner usage.

## Compatibility Notes
- Builtin bundle import remains opt-in through bundle path and key configuration.
- `AlgorithmRegisteredResponse` continues to expose the legacy `algorithm_minor_version` field for backward compatibility while `latest_algorithm_minor_version` remains the canonical field.

# v3.0.2 (27-04-2026)
## Fixed
- Fixed the ddocumentation formatting error.

# v3.0.1 (27-04-2026)
## Fixed
- Fixed the documentation formatting and structure.

# v3.0.0 (27-04-2026)
## Added
- Algorithm training workflow: samples, training jobs, checkpoints, training state/progress/logging, and stop support.
- New API routes for `/api/v0/sample`, `/api/v0/train-algorithm`, `/api/v0/training/{id}`, `/api/v0/checkpoint/*`, and deploy management.
- Algorithm export support via zip streaming, including optional minor-version and checkpoint-based asset override.
- Local and async algorithm deployment endpoints, removable/exportable algorithm flags, and deletion of removable algorithms.
- `BaseRunner` training helpers: `train`, `run_training`, checkpoint saving, training dataset loading, temp-store file helpers, state handling, and progress/log methods.
- New `compox.training` package with `TrainingHandler`, `TrainingSample`, `TrainingDataset`, `TempStore`, checkpoint manifests, and training task runners.
- CLI commands for local deploy, algorithm delete, algorithm export, selective `deploy-algorithms`, config overrides, and improved test execution.
- Stop-request infrastructure for executions and training, with `STOPPED` status.
- Additional storage collections: `training-store`, `sample-store`, `algorithm-checkpoint-store`, `stop-requests`, and `deploy-store`.
- Logging configuration with separate console/file levels and reduced console noise for polling/file-transfer logs.
- Documentation for execution client workflow, training client workflow, algorithm training, updated deployment, and security notes.
- Test coverage for training, samples, checkpoints, deployment endpoints, export, versioning, stop execution, logging, CLI, and server utilities.

## Changed
- Algorithm metadata now tracks multiple minor versions plus `latest_algorithm_minor_version`.
- Algorithm config supports `training_parameters`, `removable`, `exportable`, `displayed_name`, and float `decimal_precision`.
- Execution requests and records now support `checkpoint_id`, `algorithm_minor_version`, and `resolved_execution_device`.
- Algorithm deployment now deduplicates modules/assets by content and supports zip deployment.
- Algorithm manager deletion now cleans related modules, assets, checkpoints, and minor versions more precisely.
- S3 lifecycle handling now applies separate expiration policies for data, execution, training, deploy, and stop-request stores.
- Default documented service ports changed to Compox `5481`, MinIO console `5482`, and MinIO API `5483`.
- Dependencies were relaxed/updated: `numpy>=1.26,<3`, `requests>=2.32.3`, added `tomli` for Python <3.11, removed `natsort`.

## Fixed
- Hotfix commits removed accidental merge-conflict markers and restored accidentally commented code.
- Improved systray/server shutdown handling, especially for frozen Windows builds.
- Improved task failure/stopped handling so records are updated with terminal status, completion time, logs, and file transfer stats.
- Improved algorithm import/export path handling with traversal checks and safer reconstruction.

## Compatibility Notes
- `AlgorithmRegisteredResponse` changed from a single `algorithm_minor_version` to `algorithm_minor_versions` plus `latest_algorithm_minor_version`.
- New API surfaces expose local-path deployment and algorithm deletion; these should be treated as administrative/trusted operations.

# v2.1.2 (16-02-2026)

- Updated links in README.md

# v2.1.1 (13-02-2026)

- Added ParticleSeg3D tutorial
- Critical bugfixes

# v2.1.0 (11-12-2025)

- Improved docs and tutorials section
- General data schema

# v2.0.0 (24-11-2025)

- Initial release
