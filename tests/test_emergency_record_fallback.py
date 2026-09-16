"""
Copyright 2026 Tescan group, a.s.
All rights reserved
"""

from __future__ import annotations

import errno
import json
from pathlib import Path
from types import SimpleNamespace

from fastapi import FastAPI
from fastapi.testclient import TestClient

from compox.components.api_builder import ApiBuilder
from compox.database_connection.InMemoryConnection import InMemoryConnection
from compox.internal.EmergencyRecordStore import EmergencyRecordStore
from compox.routers import (
    deployment_controller,
    execution_controller,
    training_controller,
)
from compox.tasks.deploy_task_fastapi import deploy_task_fastapi
from compox.tasks.TaskHandler import TaskHandler


class CaptureExecutor:
    def __init__(self):
        self.calls: list[tuple[object, tuple, dict]] = []

    def submit(self, fn, *args, **kwargs):
        self.calls.append((fn, args, kwargs))
        return object()


class FailAfterFirstPutConnection(InMemoryConnection):
    def __init__(self, fail_collections: set[str], fail_from_count: int = 1):
        super().__init__()
        self.fail_collections = fail_collections
        self.put_counts: dict[str, int] = {}
        self.fail_from_count = fail_from_count

    def put_objects(self, collection_name, object_names, object):
        count = self.put_counts.get(collection_name, 0)
        if (
            collection_name in self.fail_collections
            and count >= self.fail_from_count
        ):
            raise OSError(errno.ENOSPC, "No space left on device")
        self.put_counts[collection_name] = count + 1
        return super().put_objects(collection_name, object_names, object)


def test_emergency_record_store_default_root_dir_uses_compox_subdir(tmp_path):
    """
    Verify that log-path-derived emergency roots use the Compox-specific subdirectory name.
    """
    log_path = tmp_path / "logs" / "compox.log"
    resolved = EmergencyRecordStore.default_root_dir(str(log_path))

    assert resolved == str(
        log_path.resolve().parent / "compox_emergency_records"
    )


def _make_execution_record(execution_id: str = "exec-1") -> dict:
    return {
        "execution_id": execution_id,
        "algorithm_id": "algo-1",
        "checkpoint_id": None,
        "algorithm_minor_version": None,
        "input_dataset_ids": ["file-1"],
        "execution_device_override": None,
        "additional_parameters": {},
        "session_token": None,
        "output_dataset_ids": [],
        "status": "RUNNING",
        "progress": 0.5,
        "time_started": "2026-04-02 10:00:00",
        "time_completed": "",
        "log": "",
    }


def _make_deploy_record(deploy_id: str = "dep-1") -> dict:
    return {
        "deploy_id": deploy_id,
        "status": "PENDING",
        "path": "C:/algorithms/foo",
        "algorithm_id": None,
        "algorithm_name": None,
        "algorithm_major_version": None,
        "time_started": None,
        "time_completed": None,
        "log": None,
    }


def _make_training_record(training_id: str = "train-1") -> dict:
    return {
        "training_id": training_id,
        "algorithm_id": "algo-1",
        "status": "RUNNING",
        "progress": 0.5,
        "time_started": "2026-04-02 10:00:00",
        "time_completed": None,
        "log": "",
        "training_data": ["sample-1"],
        "additional_parameters": {},
        "state": {},
        "tags": [],
        "checkpoint_id": None,
        "algorithm_minor_version": None,
        "output_checkpoint_ids": [],
    }


def test_task_handler_writes_failed_state_to_emergency_store(tmp_path: Path):
    """TaskHandler should persist a failed fallback record when record storage fails."""
    emergency_store = EmergencyRecordStore(root_dir=str(tmp_path))
    database_connection = FailAfterFirstPutConnection({"execution-store"})
    database_connection.create_collections(["execution-store"])
    database_connection.put_objects(
        "execution-store",
        ["exec-1"],
        [json.dumps(_make_execution_record()).encode()],
    )

    task_handler = TaskHandler(
        "exec-1",
        database_connection=database_connection,
        database_update=False,
        emergency_record_store=emergency_store,
    )
    task_handler.mark_as_failed(RuntimeError("disk space exhausted"))

    fallback_record = emergency_store.read_record("execution-store", "exec-1")
    assert fallback_record is not None
    assert fallback_record["status"] == "FAILED"
    assert fallback_record["progress"] == 1.0
    assert fallback_record["output_dataset_ids"] == []
    assert "_emergency_storage_error" in fallback_record
    assert "_emergency_storage_error_code" in fallback_record
    assert fallback_record["_emergency_storage_retryable"] is False


def test_emergency_record_store_uses_reserve_on_enospc(tmp_path: Path):
    """Emergency writes should succeed by releasing the reserve file after ENOSPC."""

    class FlakyEmergencyRecordStore(EmergencyRecordStore):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self.write_attempts = 0

        def _write_bytes(self, path: Path, data: bytes) -> None:
            self.write_attempts += 1
            if self.write_attempts == 1:
                raise OSError(errno.ENOSPC, "No space left on device")
            return super()._write_bytes(path, data)

    emergency_store = FlakyEmergencyRecordStore(
        root_dir=str(tmp_path), reserve_bytes=4096
    )

    reserve_path = emergency_store._reserve_path()
    assert reserve_path.exists()
    assert reserve_path.stat().st_size == 4096

    emergency_store.write_record(
        "execution-store",
        "exec-1",
        _make_execution_record() | {"status": "FAILED"},
    )

    fallback_record = emergency_store.read_record("execution-store", "exec-1")
    assert fallback_record is not None
    assert fallback_record["status"] == "FAILED"
    assert emergency_store.write_attempts == 2
    assert reserve_path.exists()
    assert reserve_path.stat().st_size == 4096


def test_emergency_record_store_returns_none_for_corrupted_json(tmp_path: Path):
    """Corrupted emergency JSON should be ignored instead of breaking the read path."""
    emergency_store = EmergencyRecordStore(root_dir=str(tmp_path))
    record_path = emergency_store._record_path("execution-store", "exec-1")
    record_path.parent.mkdir(parents=True, exist_ok=True)
    record_path.write_text("{not valid json", encoding="utf-8")

    assert emergency_store.read_record("execution-store", "exec-1") is None


def test_emergency_record_store_purge_all_records_clears_json_files(
    tmp_path: Path,
):
    """Startup purge should clear stale fallback JSON files while preserving the reserve file."""
    emergency_store = EmergencyRecordStore(root_dir=str(tmp_path))
    emergency_store.write_record(
        "execution-store",
        "exec-1",
        _make_execution_record() | {"status": "FAILED"},
    )
    emergency_store.write_record(
        "training-store",
        "train-1",
        _make_training_record("train-1") | {"status": "FAILED"},
    )

    emergency_store.purge_all_records()

    assert emergency_store.read_record("execution-store", "exec-1") is None
    assert emergency_store.read_record("training-store", "train-1") is None
    assert emergency_store._reserve_path().exists()


def test_emergency_record_store_purge_all_records_keeps_unrelated_json(
    tmp_path: Path,
):
    """Startup purge should only remove Compox-owned fallback files in known collections."""
    emergency_store = EmergencyRecordStore(root_dir=str(tmp_path))
    emergency_store.write_record(
        "execution-store",
        "exec-1",
        _make_execution_record() | {"status": "FAILED"},
    )

    unrelated_same_dir = (
        tmp_path / "execution-store" / "some_other_component_state.json"
    )
    unrelated_same_dir.write_text('{"ok": true}', encoding="utf-8")

    unrelated_other_dir = tmp_path / "custom-dir" / "random.json"
    unrelated_other_dir.parent.mkdir(parents=True, exist_ok=True)
    unrelated_other_dir.write_text('{"ok": true}', encoding="utf-8")

    emergency_store.purge_all_records()

    assert not emergency_store.has_record("execution-store", "exec-1")
    assert unrelated_same_dir.exists()
    assert unrelated_other_dir.exists()


def test_api_builder_purges_emergency_records_at_startup_build(tmp_path: Path):
    """Building the shared API store should purge stale emergency fallback records."""
    log_path = tmp_path / "logs" / "compox.log"
    emergency_root = Path(EmergencyRecordStore.default_root_dir(str(log_path)))
    stale_store = EmergencyRecordStore(root_dir=str(emergency_root))
    stale_store.write_record(
        "execution-store",
        "exec-1",
        _make_execution_record() | {"status": "FAILED"},
    )

    app = (
        ApiBuilder()
        .with_settings(SimpleNamespace(log_path=str(log_path)))
        .with_database_connection(InMemoryConnection())
        .with_algorithm_exporter(object())
        .build()
    )

    assert (
        app.state.emergency_record_store.read_record(
            "execution-store", "exec-1"
        )
        is None
    )


def test_emergency_record_store_ignores_reserve_release_race(tmp_path: Path):
    """Reserve-file release races should not prevent the fallback record from being written."""

    class RaceyEmergencyRecordStore(EmergencyRecordStore):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self.write_attempts = 0

        def _write_bytes(self, path: Path, data: bytes) -> None:
            self.write_attempts += 1
            if self.write_attempts == 1:
                raise OSError(errno.ENOSPC, "No space left on device")
            return super()._write_bytes(path, data)

        def _release_reserve_file(self) -> None:
            reserve_path = self._reserve_path()
            if reserve_path.exists():
                reserve_path.unlink()
            super()._release_reserve_file()

    emergency_store = RaceyEmergencyRecordStore(
        root_dir=str(tmp_path), reserve_bytes=4096
    )

    emergency_store.write_record(
        "execution-store",
        "exec-1",
        _make_execution_record() | {"status": "FAILED"},
    )

    fallback_record = emergency_store.read_record("execution-store", "exec-1")
    assert fallback_record is not None
    assert fallback_record["status"] == "FAILED"


def test_get_execution_record_uses_emergency_fallback(tmp_path: Path):
    """Execution record reads should fall back to the emergency store when needed."""
    emergency_store = EmergencyRecordStore(root_dir=str(tmp_path))
    database_connection = FailAfterFirstPutConnection({"execution-store"})
    database_connection.create_collections(["execution-store"])
    database_connection.put_objects(
        "execution-store",
        ["exec-1"],
        [json.dumps(_make_execution_record()).encode()],
    )

    task_handler = TaskHandler(
        "exec-1",
        database_connection=database_connection,
        database_update=False,
        emergency_record_store=emergency_store,
    )
    task_handler.mark_as_failed(RuntimeError("disk space exhausted"))

    app = FastAPI()
    app.state.database_connection = database_connection
    app.state.emergency_record_store = emergency_store
    app.include_router(execution_controller.router)

    with TestClient(app) as client:
        response = client.get("/api/v0/executions/exec-1")

    assert response.status_code == 200
    assert response.json()["status"] == "FAILED"


def test_get_execution_record_prefers_primary_terminal_record_over_failed_fallback(
    tmp_path: Path,
):
    """A terminal primary execution record should win over a stale failed fallback."""
    emergency_store = EmergencyRecordStore(root_dir=str(tmp_path))
    database_connection = InMemoryConnection()
    database_connection.create_collections(["execution-store"])
    database_connection.put_objects(
        "execution-store",
        ["exec-1"],
        [
            json.dumps(
                _make_execution_record("exec-1")
                | {
                    "status": "COMPLETED",
                    "progress": 1.0,
                    "time_completed": "2026-04-02 10:05:00",
                }
            ).encode()
        ],
    )
    emergency_store.write_record(
        "execution-store",
        "exec-1",
        _make_execution_record("exec-1")
        | {"status": "FAILED", "progress": 1.0},
    )

    app = FastAPI()
    app.state.database_connection = database_connection
    app.state.emergency_record_store = emergency_store
    app.include_router(execution_controller.router)

    with TestClient(app) as client:
        response = client.get("/api/v0/executions/exec-1")

    assert response.status_code == 200
    assert response.json()["status"] == "COMPLETED"


def test_get_execution_record_ignores_corrupted_fallback_when_primary_exists(
    tmp_path: Path,
):
    """A corrupted fallback file should not block returning a valid primary execution record."""
    emergency_store = EmergencyRecordStore(root_dir=str(tmp_path))
    database_connection = InMemoryConnection()
    database_connection.create_collections(["execution-store"])
    database_connection.put_objects(
        "execution-store",
        ["exec-1"],
        [json.dumps(_make_execution_record("exec-1")).encode()],
    )
    corrupt_path = emergency_store._record_path("execution-store", "exec-1")
    corrupt_path.parent.mkdir(parents=True, exist_ok=True)
    corrupt_path.write_text("{broken", encoding="utf-8")

    app = FastAPI()
    app.state.database_connection = database_connection
    app.state.emergency_record_store = emergency_store
    app.include_router(execution_controller.router)

    with TestClient(app) as client:
        response = client.get("/api/v0/executions/exec-1")

    assert response.status_code == 200
    assert response.json()["status"] == "RUNNING"


def test_execute_algorithm_writes_initial_failed_fallback(tmp_path: Path):
    """Execution submission should emit an immediate failed fallback if the initial record save fails."""
    emergency_store = EmergencyRecordStore(root_dir=str(tmp_path))
    database_connection = FailAfterFirstPutConnection(
        {"execution-store"}, fail_from_count=0
    )
    database_connection.create_collections(["algorithm-store", "data-store"])
    database_connection.put_objects(
        "algorithm-store",
        ["algo-1~demo~1"],
        [
            json.dumps(
                {
                    "algorithm_id": "algo-1",
                    "algorithm_name": "demo",
                    "algorithm_major_version": "1",
                }
            ).encode()
        ],
    )
    database_connection.put_objects("data-store", ["file-1"], [b"test"])

    app = FastAPI()
    app.state.database_connection = database_connection
    app.state.emergency_record_store = emergency_store
    app.state.settings = SimpleNamespace(
        inference=SimpleNamespace(
            backend_settings=SimpleNamespace(
                executor="fastapi_background_tasks"
            )
        )
    )
    app.state.executor = object()
    app.include_router(execution_controller.router)

    with TestClient(app) as client:
        response = client.post(
            "/api/v0/execute-algorithm",
            json={
                "algorithm_id": "algo-1",
                "input_dataset_ids": ["file-1"],
                "additional_parameters": {},
            },
        )

    assert response.status_code == 200
    execution_id = response.json()["execution_id"]
    fallback_record = emergency_store.read_record(
        "execution-store", execution_id
    )
    assert fallback_record is not None
    assert fallback_record["status"] == "FAILED"
    assert "Failed to save execution record" in fallback_record["log"]
    assert "_emergency_storage_error" in fallback_record
    assert "_emergency_storage_error_code" in fallback_record
    assert fallback_record["_emergency_storage_retryable"] is False


def test_execute_algorithm_passes_shared_emergency_store_to_background_task(
    tmp_path: Path,
):
    """Execution background submission should reuse the app-level emergency record store."""
    emergency_store = EmergencyRecordStore(root_dir=str(tmp_path / "emergency"))
    database_connection = InMemoryConnection()
    database_connection.create_collections(["algorithm-store", "data-store"])
    database_connection.put_objects(
        "algorithm-store",
        ["algo-1~demo~1"],
        [
            json.dumps(
                {
                    "algorithm_id": "algo-1",
                    "algorithm_name": "demo",
                    "algorithm_major_version": "1",
                }
            ).encode()
        ],
    )
    database_connection.put_objects("data-store", ["file-1"], [b"test"])

    executor = CaptureExecutor()
    app = FastAPI()
    app.state.database_connection = database_connection
    app.state.emergency_record_store = emergency_store
    app.state.settings = SimpleNamespace(
        inference=SimpleNamespace(
            backend_settings=SimpleNamespace(
                executor="fastapi_background_tasks"
            )
        )
    )
    app.state.executor = executor
    app.include_router(execution_controller.router)

    with TestClient(app) as client:
        response = client.post(
            "/api/v0/execute-algorithm",
            json={
                "algorithm_id": "algo-1",
                "input_dataset_ids": ["file-1"],
                "additional_parameters": {},
            },
        )

    assert response.status_code == 200
    assert len(executor.calls) == 1
    _, _, kwargs = executor.calls[0]
    assert kwargs["emergency_record_store"] is emergency_store


def test_get_training_record_uses_emergency_fallback(tmp_path: Path):
    """Training record reads should fall back to the emergency store when needed."""
    emergency_store = EmergencyRecordStore(root_dir=str(tmp_path))
    database_connection = InMemoryConnection()
    app = FastAPI()
    app.state.database_connection = database_connection
    app.state.emergency_record_store = emergency_store
    app.include_router(training_controller.router)

    emergency_store.write_record(
        "training-store",
        "train-1",
        _make_training_record("train-1")
        | {"status": "FAILED", "progress": 1.0, "log": "disk full"},
    )

    with TestClient(app) as client:
        response = client.get("/api/v0/training/train-1")

    assert response.status_code == 200
    assert response.json()["status"] == "FAILED"
    assert response.json()["log"] == "disk full"


def test_get_training_record_prefers_primary_terminal_record_over_failed_fallback(
    tmp_path: Path,
):
    """A terminal primary training record should win over a stale failed fallback."""
    emergency_store = EmergencyRecordStore(root_dir=str(tmp_path))
    database_connection = InMemoryConnection()
    database_connection.create_collections(["training-store"])
    database_connection.put_objects(
        "training-store",
        ["train-1"],
        [
            json.dumps(
                _make_training_record("train-1")
                | {
                    "status": "COMPLETED",
                    "progress": 1.0,
                    "time_completed": "2026-04-02 10:05:00",
                }
            )
        ],
    )
    emergency_store.write_record(
        "training-store",
        "train-1",
        _make_training_record("train-1")
        | {"status": "FAILED", "progress": 1.0},
    )

    app = FastAPI()
    app.state.database_connection = database_connection
    app.state.emergency_record_store = emergency_store
    app.include_router(training_controller.router)

    with TestClient(app) as client:
        response = client.get("/api/v0/training/train-1")

    assert response.status_code == 200
    assert response.json()["status"] == "COMPLETED"


def test_get_deploy_record_uses_emergency_fallback(tmp_path: Path):
    """Deploy record reads should fall back to the emergency store when needed."""
    emergency_store = EmergencyRecordStore(root_dir=str(tmp_path))
    database_connection = InMemoryConnection()
    app = FastAPI()
    app.state.database_connection = database_connection
    app.state.emergency_record_store = emergency_store
    app.include_router(deployment_controller.router)

    emergency_store.write_record(
        "deploy-store",
        "dep-1",
        _make_deploy_record("dep-1") | {"status": "FAILED", "log": "disk full"},
    )

    with TestClient(app) as client:
        response = client.get("/api/v0/deploy/dep-1")

    assert response.status_code == 200
    assert response.json()["status"] == "FAILED"
    assert response.json()["log"] == "disk full"


def test_get_deploy_record_prefers_primary_terminal_record_over_failed_fallback(
    tmp_path: Path,
):
    """A terminal primary deploy record should win over a stale failed fallback."""
    emergency_store = EmergencyRecordStore(root_dir=str(tmp_path))
    database_connection = InMemoryConnection()
    database_connection.create_collections(["deploy-store"])
    database_connection.put_objects(
        "deploy-store",
        ["dep-1"],
        [
            json.dumps(
                _make_deploy_record("dep-1")
                | {
                    "status": "COMPLETED",
                    "time_completed": "2026-04-02 10:05:00",
                }
            )
        ],
    )
    emergency_store.write_record(
        "deploy-store",
        "dep-1",
        _make_deploy_record("dep-1") | {"status": "FAILED", "log": "disk full"},
    )

    app = FastAPI()
    app.state.database_connection = database_connection
    app.state.emergency_record_store = emergency_store
    app.include_router(deployment_controller.router)

    with TestClient(app) as client:
        response = client.get("/api/v0/deploy/dep-1")

    assert response.status_code == 200
    assert response.json()["status"] == "COMPLETED"


def test_deploy_task_writes_fallback_to_shared_emergency_store(tmp_path: Path):
    """Deploy background failures should write their fallback record into the shared emergency store."""
    emergency_store = EmergencyRecordStore(root_dir=str(tmp_path / "emergency"))
    database_connection = FailAfterFirstPutConnection({"deploy-store"})
    database_connection.create_collections(["deploy-store"])
    database_connection.put_objects(
        "deploy-store",
        ["dep-1"],
        [json.dumps(_make_deploy_record("dep-1"))],
    )

    missing_path = str(tmp_path / "missing-algorithm")

    try:
        deploy_task_fastapi(
            database_connection=database_connection,
            deploy_id="dep-1",
            path=missing_path,
            emergency_record_store=emergency_store,
        )
    except FileNotFoundError:
        pass

    fallback_record = emergency_store.read_record("deploy-store", "dep-1")
    assert fallback_record is not None
    assert fallback_record["status"] == "FAILED"
    assert fallback_record["path"] == missing_path
    assert "_emergency_storage_error" in fallback_record
    assert "_emergency_storage_error_code" in fallback_record
    assert fallback_record["_emergency_storage_retryable"] is False


def test_async_deploy_passes_shared_emergency_store_to_background_task(
    tmp_path: Path,
):
    """Async deploy submission should reuse the app-level emergency record store."""
    emergency_store = EmergencyRecordStore(root_dir=str(tmp_path / "emergency"))
    database_connection = InMemoryConnection()
    database_connection.create_collections(["deploy-store"])
    algorithm_dir = tmp_path / "algo"
    algorithm_dir.mkdir()

    executor = CaptureExecutor()
    app = FastAPI()
    app.state.database_connection = database_connection
    app.state.emergency_record_store = emergency_store
    app.state.settings = SimpleNamespace(
        inference=SimpleNamespace(
            backend_settings=SimpleNamespace(
                executor="fastapi_background_tasks"
            )
        )
    )
    app.state.executor = executor
    app.include_router(deployment_controller.router)

    with TestClient(app) as client:
        response = client.post(
            "/api/v0/deploy/local-async",
            params={"path": str(algorithm_dir)},
        )

    assert response.status_code == 200
    assert len(executor.calls) == 1
    _, _, kwargs = executor.calls[0]
    assert kwargs["emergency_record_store"] is emergency_store
