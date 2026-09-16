## Training workflow

This doc covers the client-facing flow for training in Compox: upload data, create training samples, start training, and retrieve results. All endpoints and payloads are derived from the current server code under `compox/src/compox/routers`.

---

### Upload data files

Endpoint:
- `POST /api/v0/files`

Behavior (from `file_controller.py`):
- The request body is treated as raw bytes and must be a valid HDF5 file.
- On success, returns `{ "file_id": "<uuid>" }`.

Minimal example:
```http
POST /api/v0/files
Content-Type: application/octet-stream

<HDF5 bytes>
```

Response:
```json
{ "file_id": "..." }
```

Notes:
- The server validates that the uploaded bytes open as HDF5.
- Files are stored in the `data-store` bucket/collection.
- Files by default expire after 1 day (configurable in `S3Connection`).

---

### Create training samples

Endpoint:
- `POST /api/v0/sample`

Payload model: `IncomingSampleRequest` (see `pydantic_models.py`)
- `files`: list of dicts mapping arbitrary keys to lists of file IDs
- `tags`: list of strings (optional)

Example:
```json
{
  "files": [
    { "input": ["<file_id_1>", "<file_id_2>"], "target": ["<file_id_3>"] }
  ],
  "tags": ["modality:ct", "anatomy:brain", "author:me"]
}
```

Response:
```json
{ "sample_id": "..." }
```

Validation behavior:
- Each referenced file ID must exist in `data-store`, otherwise the API returns 404.

Related endpoints:
- `GET /api/v0/sample/{sample_id}` returns the stored sample record
- `GET /api/v0/sample/all?positive_tags=...&negative_tags=...` filters by tags
- `DELETE /api/v0/sample/{sample_id}` deletes a sample

Notes:
- Files referenced by samples are not copied; the sample record just points to existing files.
- Files referenced by sample do not expire as long as the sample exists.

---

### Start training

Endpoint:
- `POST /api/v0/train-algorithm`

Payload model: `IncomingTrainingRequest`
- `algorithm_id`: string
- `training_data`: list of sample IDs
- `checkpoint_id`: optional checkpoint to start from
- `algorithm_minor_version`: optional minor version string
- `tags`: list of strings
- `additional_parameters`: dict (free-form)

Example:
```json
{
  "algorithm_id": "<algorithm_id>",
  "training_data": ["<sample_id>"],
  "checkpoint_id": null,
  "algorithm_minor_version": null,
  "tags": ["experiment:42", "author:me"],
  "additional_parameters": {
    "learning_rate": 0.001,
    "batch_size": 4,
    "num_epochs": 10
  }
}
```

Response:
```json
{ "training_id": "..." }
```

Validation behavior (from `training_controller.py`):
- All referenced sample IDs must exist in `sample-store`.
- All files referenced by those samples must exist in `data-store`.
- The algorithm ID must exist in `algorithm-store` (via `find_algorithm_by_id`).

Execution mode:
- If `inference.backend_settings.executor` is `fastapi_background_tasks`, training runs via `training_task_fastapi`.
- If executor is `celery`, the task is queued under `training_task`.

Progress/status details (from `TrainingHandler` and `TaskHandler`):
- Status transitions are written to `training-store` via `TrainingHandler.status` (inherited from `TaskHandler`).
- Valid statuses are `PENDING`, `STARTED`, `RUNNING`, `COMPLETED`, `FAILED`, `STOPPED`.
- `training_controller.py` creates the record with `status="PENDING"` and `progress=0.0`.
- `TrainingHandler.mark_as_completed()` sets `progress=1.0`, `time_completed`, updates the log, and sets `status="COMPLETED"`.
- If a stop request is posted, `TaskHandler._check_for_stop_request()` acknowledges it and calls `mark_as_stopped()`, which sets `status="STOPPED"` and raises `TaskStoppedException`.
- `TaskHandler.mark_as_failed()` sets `status="FAILED"`, `progress=1.0`, clears `output_dataset_ids`, and stores the exception in the log.

---

### Check training status

Endpoint:
- `GET /api/v0/training/{training_id}`

Response model: `TrainingRecord`
- Includes `status`, `progress`, `log`, `output_checkpoint_ids`, etc.

Example response (shape):
```json
{
  "training_id": "...",
  "algorithm_id": "...",
  "status": "RUNNING",
  "progress": 0.3,
  "time_started": "...",
  "time_completed": null,
  "log": "",
  "training_data": ["<sample_id>"],
  "state": {},
  "tags": ["experiment:42"],
  "checkpoint_id": null,
  "algorithm_minor_version": null,
  "output_checkpoint_ids": []
}
```

Notes:
- `output_checkpoint_ids` is the key field for downstream retrieval of training results.

---

### Benchmark algorithm (optional pre-check)

Endpoint:
- `POST /api/v0/benchmark-algorithm`

Payload model: `IncomingBenchmarkRequest`
- `algorithm_id`: string
- `additional_parameters`: dict

Example:
```json
{
  "algorithm_id": "<algorithm_id>",
  "additional_parameters": {
    "sigma": 5,
    "winsize": 9
  }
}
```

Response:
```json
{ "benchmark_id": "..." }
```

Status endpoint:
- `GET /api/v0/benchmarks/{benchmark_id}`

Response model: `BenchmarkRecord`
- Includes `status`, `time_started`, `time_completed`, `log`, and `data`.

Example status response (shape):
```json
{
  "benchmark_id": "...",
  "algorithm_id": "...",
  "additional_parameters": {
    "sigma": 5,
    "winsize": 9
  },
  "status": "COMPLETED",
  "time_started": "...",
  "time_completed": "...",
  "log": "...",
  "data": {}
}
```

Notes:
- This step is useful as a pre-check before starting long-running training jobs.
- Valid statuses are `PENDING`, `STARTED`, `RUNNING`, `COMPLETED`, `FAILED`, `STOPPED`.
- Benchmark records are initialized with `status="PENDING"` and `time_started`.
- On terminal states (`COMPLETED`, `FAILED`, `STOPPED`), `time_completed` is filled.

---

### Stop training (optional)

Endpoint:
- `POST /api/v0/training/{training_id}/stop`

Behavior:
- Only `PENDING`, `RUNNING` or `STARTED` statuses are stoppable.
- A stop request is posted to `stop-requests`, which the training task checks.

---

### Retrieve training results (checkpoints)

Endpoint:
- `GET /api/v0/checkpoint/{checkpoint_id}`
- `GET /api/v0/checkpoint/all` (filtering supported by query params; see `checkpoint_controller.py`)

Results:
- Checkpoint metadata is returned (not the model bytes directly).

Checkpoint behavior (from `TrainingHandler.save_checkpoint()`):
- `save_checkpoint()` validates that every asset path exists in the algorithm’s assets.
- New assets are stored in `asset-store`, and a new `checkpoint_id` is created.
- A checkpoint manifest is saved in `algorithm-checkpoint-store`.
- The new checkpoint ID is appended to `output_checkpoint_ids` in the training handler, so it appears in the training record.

Checkpoint metadata shape (from `AlgorithmCheckpointRecord`):
```json
{
  "checkpoint_id": "...",
  "training_id": "...",
  "parent_algorithm_id": "...",
  "created_at": "...",
  "properties": {},
  "tags": [],
  "parent_checkpoint_id": null
}
```

Details on `properties` and `tags`:
- `properties` is a **free-form dictionary** provided by the algorithm when it calls
  `save_checkpoint(assets, properties)`. This is the place to store metrics,
  hyperparameters, dataset IDs, evaluation scores, or any other metadata you want to
  query later.
- `tags` are inherited from the **training run tags** (`IncomingTrainingRequest.tags`).
  When the checkpoint is created, `TrainingHandler.save_checkpoint()` copies the
  training record’s `tags` into the checkpoint manifest.
- `parent_checkpoint_id` is copied from the training record’s `checkpoint_id`, so you
  can track lineage if you trained from an existing checkpoint.

Filtering by tags:
- `GET /api/v0/checkpoint/all?positive_tags=tag1&positive_tags=tag2`
- `GET /api/v0/checkpoint/all?negative_tags=tag_to_exclude`
- You can combine both `positive_tags` and `negative_tags` in the same request.

---

### Export trained algorithm (optional)

Endpoint:
- `GET /api/v0/algorithm/{algorithm_name}/{algorithm_major_version}/export`

Query params:
- `algorithm_minor_version` (optional)
- `checkpoint_id` (optional; overrides assets with the checkpoint)

Response:
- Streaming zip download (`application/zip`) of the algorithm package.

What `algorithm_minor_version` means:
- The **minor version** is the build number stored for a given algorithm name + major version.
- Supplying it lets you export a **specific build**; if you omit it, the **latest build** is exported.

What `checkpoint_id` means:
- A checkpoint is a **snapshot of trained assets** (typically weights).
- Supplying `checkpoint_id` tells the exporter to **swap the algorithm’s assets** with the checkpoint’s assets before packaging.
- This is how you get a **trained** package out of a training run.

Export protection:
- If the algorithm is marked with `"exportable": false` in its `AlgorithmConfigSchema`, the export endpoint returns 403 Forbidden.

What the zip file is:
- A complete **deployable algorithm package**: `Runner.py`, `pyproject.toml`, and the assets under `files/`.
- If `checkpoint_id` is provided, those asset files come from the checkpoint instead of the original algorithm assets.

---

### End-to-end summary

1. Upload HDF5 files → get `file_id`s
2. Create training sample(s) referencing file IDs → get `sample_id`s
3. Optionally benchmark algorithm (`/benchmark-algorithm`) and poll benchmark record (`/benchmarks/{id}`)
4. Start training with algorithm ID + sample IDs → get `training_id`
5. Poll training record → read status + `output_checkpoint_ids`
6. Optionally stop training or fetch checkpoint metadata
7. Optionally export an algorithm package using checkpoint ID
