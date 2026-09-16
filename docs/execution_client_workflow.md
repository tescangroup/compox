## Execution workflow

This doc covers the client-facing flow for execution in Compox: upload data, start an execution, poll status, and retrieve results. All endpoints and payloads are derived from the current server code under `compox/src/compox/routers`.

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

### Execute an algorithm

Endpoint:
- `POST /api/v0/execute-algorithm`

Payload model: `IncomingExecutionRequest`
- `algorithm_id`: string
- `input_dataset_ids`: list of file IDs
- `checkpoint_id`: optional checkpoint to load assets from
- `algorithm_minor_version`: optional minor version to execute
- `execution_device_override`: optional device override (e.g. `"cpu"`, `"cuda:0"`)
- `additional_parameters`: dict (free-form)
- `session_token`: optional session identifier

Example:
```json
{
  "algorithm_id": "<algorithm_id>",
  "input_dataset_ids": ["<file_id_1>", "<file_id_2>"],
  "checkpoint_id": null,
  "algorithm_minor_version": null,
  "execution_device_override": null,
  "additional_parameters": {
    "threshold": 0.5,
    "tile_size": 512
  },
  "session_token": null
}
```

Response:
```json
{ "execution_id": "..." }
```

Validation behavior (from `execution_controller.py`):
- All referenced input file IDs must exist in `data-store`.
- The algorithm ID must exist in `algorithm-store` (via `find_algorithm_by_id`).

Execution mode:
- If `inference.backend_settings.executor` is `fastapi_background_tasks`, execution runs via `execution_task_fastapi`.
- If executor is `celery`, the task is queued under `task` (Celery).

Progress/status details (from `TaskHandler`):
- Valid statuses are `PENDING`, `STARTED`, `RUNNING`, `COMPLETED`, `FAILED`, `STOPPED`.
- `execution_controller.py` creates the record with `status="PENDING"` and `progress=0.0`.
- `TaskHandler.mark_as_completed()` sets `progress=1.0`, `time_completed`, `output_dataset_ids`, and `status="COMPLETED"`.
- `TaskHandler.mark_as_failed()` sets `status="FAILED"`, `progress=1.0`, clears `output_dataset_ids`, and stores the exception in the log.
- If a stop request is posted, `TaskHandler._check_for_stop_request()` acknowledges it and calls `mark_as_stopped()`, which sets `status="STOPPED"` and raises `TaskStoppedException`.

---

### Sessions (optional)

The execution API supports an optional `session_token` that can be used to share an in‑memory cache across multiple executions.

Behavior (from `TaskSession` and `TaskHandler`):
- If you **omit** `session_token` and the backend uses **FastAPI background tasks**, a new session token is generated and stored in the execution record.
- You can retrieve it via `GET /api/v0/executions/{execution_id}` and pass it in subsequent executions to reuse the cache.
- Sessions are **in‑memory only** (single process), expire after ~24 hours, and are capped in size/number of caches.
- **Celery does not support sessions**: if you attempt to use session features in Celery mode, a `NotImplementedError` is raised internally and `session_token` will be `None` in the execution record.

Practical guidance:
- Use sessions only when running with `fastapi_background_tasks`.
- Treat sessions as a performance optimization (e.g., caching model intermediates), not a persistent store.

---

### Check execution status

Endpoint:
- `GET /api/v0/executions/{execution_id}`

Response model: `ExecutionRecord`
- Includes `status`, `progress`, `log`, and `output_dataset_ids`.

Example response (shape):
```json
{
  "execution_id": "...",
  "algorithm_id": "...",
  "status": "RUNNING",
  "progress": 0.3,
  "time_started": "...",
  "time_completed": "",
  "log": "",
  "input_dataset_ids": ["<file_id_1>", "<file_id_2>"],
  "output_dataset_ids": [],
  "execution_device_override": null,
  "additional_parameters": {},
  "session_token": null,
  "checkpoint_id": null,
  "algorithm_minor_version": null
}
```

Notes:
- `output_dataset_ids` is the key field for downstream retrieval of results.

---

### Benchmark algorithm (optional)

Endpoint:
- `POST /api/v0/benchmark-algorithm`

Payload model: `IncomingBenchmarkRequest`
- `algorithm_id`: string
- `additional_parameters`: dict (free-form benchmark arguments)

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
  "data": {
    "latency_s": 0.021,
    "throughput": 47.6
  }
}
```

Notes:
- Valid statuses are `PENDING`, `STARTED`, `RUNNING`, `COMPLETED`, `FAILED`, `STOPPED`.
- Benchmark records are initialized with `status="PENDING"` and `time_started`.
- On storage fallback failure paths, records are finalized as `FAILED` with `time_completed` and log details.

---

### Stop execution (optional)

Endpoint:
- `POST /api/v0/executions/{execution_id}/stop`

Behavior:
- Only `PENDING`, `RUNNING` or `STARTED` statuses are stoppable.
- A stop request is posted to `stop-requests`, which the task checks.

---

### Retrieve output datasets

Endpoint:
- `GET /api/v0/files/{file_id}`

Behavior:
- Returns the raw HDF5 bytes for each dataset ID returned in `output_dataset_ids`.

---

### Delete files (optional)

Endpoint:
- `DELETE /api/v0/files/{file_id}`

Behavior:
- Deletes a file from `data-store` immediately.

Notes:
- Files already expire automatically (default 1 day), but you may want to delete
  them earlier to free up storage.

---

### End-to-end summary

1. Upload HDF5 files -> get `file_id`s
2. Execute algorithm with `algorithm_id` + `input_dataset_ids` -> get `execution_id`
3. Optionally reuse a `session_token` across executions (FastAPI background tasks only)
4. Optionally benchmark algorithm (`/benchmark-algorithm`) and poll benchmark record (`/benchmarks/{id}`)
5. Poll execution record -> read status + `output_dataset_ids` (and `session_token` if used)
6. Optionally stop execution
7. Download each output dataset by ID
8. Optionally delete files to free up storage early
