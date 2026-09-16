"""
Copyright 2024 TESCAN 3DIM, s.r.o.
All rights reserved
"""

import pytest
import json
import io
import zipfile
from collections import deque
from unittest.mock import patch, MagicMock
from pydantic import BaseModel, ConfigDict, ValidationError
import numpy as np
import h5py
from datetime import datetime

from compox.server_utils import algorithm_cache
from compox.tasks.TaskHandler import TaskHandler
from compox.exceptions import CompoxExecutionError, CompoxTaskError
from compox.tasks.context_handler import current_handler


class DummySchema(BaseModel):
    """
    Simple Pydantic model used for testing HDF5 data fetching.

    Attributes
    ----------
    array1 : np.ndarray
        Required NumPy array field.
    array2 : np.ndarray | None
        Optional NumPy array field, defaults to None.
    """

    array1: np.ndarray
    array2: np.ndarray | None = None

    model_config = ConfigDict(arbitrary_types_allowed=True)


class StringListSchema(BaseModel):
    region_names: list[str]

    model_config = ConfigDict(arbitrary_types_allowed=True)


class DummySession:
    """
    In-memory session storage for testing session CRUD operations.
    """

    def __init__(self):
        self.store = {}

    def add_item(self, obj, key):
        """
        Store an object under `key`.
        """
        self.store[key] = obj

    def __getitem__(self, key):
        """
        Return the object stored under `key` (raises KeyError if missing).
        """
        return self.store[key]

    def remove_item(self, key):
        """
        Remove the object stored under `key` (raises KeyError if missing).
        """
        del self.store[key]


def _create_runner_zip_with_version(version: str) -> bytes:
    """
    Build a minimal runner module zip with an identifiable class VERSION.
    """
    buffer = io.BytesIO()
    runner_code = f"""
class Runner:
    VERSION = "{version}"

    def initialize(self, device=None):
        self.device = device

    def _load_assets(self):
        pass

    def register_task_handler(self, handler, algorithm_json):
        pass
"""
    with zipfile.ZipFile(buffer, "w") as z:
        z.writestr("Runner.py", runner_code.strip() + "\n")
    buffer.seek(0)
    return buffer.read()


def _clear_task_handler_algorithm_cache() -> None:
    """
    Clear the shared TaskHandler algorithm cache between tests.
    """
    cache_func = TaskHandler._TaskHandler__cached_fetch_algorithm
    cache_dict, access_order = _get_algorithm_cache_state(cache_func)
    cache_dict.clear()
    access_order.clear()


def _get_algorithm_cache_state(cache_func) -> tuple[dict, deque]:
    """
    Return the cache dictionary and access-order deque from a decorated method.
    """
    cache_dict = None
    access_order = None

    for cell in cache_func.__closure__ or ():
        value = cell.cell_contents
        if isinstance(value, dict):
            cache_dict = value
        elif isinstance(value, deque):
            access_order = value

    if cache_dict is None or access_order is None:
        raise AssertionError(
            "Failed to locate algorithm cache state in closure."
        )

    return cache_dict, access_order


def _get_cached_fetch_algorithm_impl():
    """
    Return the undecorated TaskHandler cached-fetch implementation from the wrapper closure.
    """
    cache_func = TaskHandler._TaskHandler__cached_fetch_algorithm
    for cell in cache_func.__closure__ or ():
        value = cell.cell_contents
        if (
            callable(value)
            and getattr(value, "__name__", "") == "__cached_fetch_algorithm"
        ):
            return value
    raise AssertionError(
        "Failed to locate original __cached_fetch_algorithm implementation."
    )


def _set_task_handler_algorithm_cache_maxsize(
    monkeypatch, maxsize: int
) -> None:
    """
    Rebind TaskHandler cached algorithm fetch with a test-specific cache size.
    """
    monkeypatch.setattr(
        TaskHandler,
        "_TaskHandler__cached_fetch_algorithm",
        algorithm_cache(maxsize=maxsize)(_get_cached_fetch_algorithm_impl()),
    )


def _configure_algorithm_cache_test_store(
    mock_connection, algorithms: dict
) -> None:
    """
    Configure mocked algorithm/module storage for multiple algorithm cache tests.
    """
    original_get_objects = mock_connection.get_objects.side_effect

    algorithm_records = {}
    module_archives = {}

    for algorithm_id, version in algorithms.items():
        algorithm_key = f"{algorithm_id}~{algorithm_id}_name~1"
        module_id = f"module_{algorithm_id}"
        algorithm_records[algorithm_key] = {
            "algorithm_id": algorithm_id,
            "algorithm_name": f"{algorithm_id}_name",
            "algorithm_major_version": "1",
            "supported_devices": ["cpu"],
            "default_device": "cpu",
            "latest_algorithm_minor_version": "0",
            "algorithm_minor_version": {
                "0": {
                    "module_id": module_id,
                    "assets": {
                        f"asset-{algorithm_id}": f"asset-{algorithm_id}"
                    },
                }
            },
        }
        module_archives[module_id] = _create_runner_zip_with_version(version)

    def list_objects(bucket):
        if bucket == "algorithm-store":
            return [{"Key": key} for key in algorithm_records]
        if bucket == "module-store":
            return [{"Key": key} for key in module_archives]
        return []

    def get_objects(bucket, keys):
        if bucket == "algorithm-store":
            return [json.dumps(algorithm_records[keys[0]])]
        if bucket == "module-store":
            return [module_archives[keys[0]]]
        return original_get_objects(bucket, keys)

    mock_connection.list_objects.side_effect = list_objects
    mock_connection.get_objects.side_effect = get_objects


def _count_bucket_reads(mock_connection, bucket_name: str) -> int:
    """
    Count how many times a specific storage bucket was read during a test.
    """
    return sum(
        1
        for call in mock_connection.get_objects.call_args_list
        if call.args[0] == bucket_name
    )


def verify_storage_and_get_saved_json(mock_connection):
    """
    Retrieve the last JSON payload passed to `put_objects` and returns the resulting dict.

    Parameters
    ----------
    mock_connection : MagicMock
        An S3-style mock that has been called with `put_objects()`.

    Returns
    -------
    dict
        The Python object obtained by `json.loads` of the saved payload.
    """
    args, _ = mock_connection.put_objects.call_args
    payload_json = args[2][0]
    payload = json.loads(payload_json)
    return payload


@pytest.fixture
def handler_with_session(task_handler):
    """
    Attach a DummySession instance to a TaskHandler and return both.
    """
    session = DummySession()
    task_handler.task_session = session
    return task_handler, session


@pytest.fixture(autouse=True)
def reset_task_handler_algorithm_cache_state():
    """
    Reset the shared TaskHandler algorithm cache configuration between tests.
    """
    TaskHandler._ALGORITHM_CACHE_MAXSIZE = 1
    _clear_task_handler_algorithm_cache()
    yield
    TaskHandler._ALGORITHM_CACHE_MAXSIZE = 1
    _clear_task_handler_algorithm_cache()


# test 1 - progress, status, dataset_ids, session_token
def test_updates_db(task_handler, mock_connection):
    """
    Verify that task_handler correctly store:
        - progress
        - status
        - output_dataset_ids
        - time_completed
        - session_token
    """
    task_handler.progress = 0.75
    task_handler.status = "RUNNING"
    task_handler.output_dataset_ids = [
        "dataset-id1",
        "dataset-id2",
        "dataset-id3",
    ]
    task_handler.time_completed = "1.23"
    task_handler.session_token = "uuid1"
    payload = verify_storage_and_get_saved_json(mock_connection)

    assert (
        payload["progress"] == 0.75
    ), f"Expected 'progress' to be '0.75', got {payload['progress']!r}"
    assert (
        payload["status"] == "RUNNING"
    ), f"Expected 'status' to be 'RUNNING', got {payload['status']!r}"
    assert payload["output_dataset_ids"] == [
        "dataset-id1",
        "dataset-id2",
        "dataset-id3",
    ], (
        f"Expected 'output_dataset_ids' to be '['dataset-id1', 'dataset-id2', 'dataset-id3']', "
        f"got {payload['output_dataset_ids']!r}"
    )
    assert (
        payload["time_completed"] == "1.23"
    ), f"Expected 'time_completed' to be '1.23', got {payload['time_completed']!r}"
    assert (
        payload["session_token"] == "uuid1"
    ), f"Expected 'session_token' to be 'uuid1', got {payload['session_token']!r}"


# Test 2 - Mark as Completed
def test_mark_as_completed(task_handler, mock_connection):
    """
    Veify that 'mark_as_completed':
        - set 'progress' to 1.0
        - set 'status to' 'COMPLETED'
        - keep 'output_dataset_ids' same
    """
    task_handler.output_dataset_ids = ["test"]
    task_handler.mark_as_completed(task_handler.output_dataset_ids)
    assert (
        task_handler._progress == 1.0
    ), f"Expected 'progress' to be '1.0', got {task_handler._progress!r}"
    assert (
        task_handler._status == "COMPLETED"
    ), f"Expected 'status' to be 'COMPLETED', got {task_handler._status!r}"
    assert task_handler._output_dataset_ids == [
        "test"
    ], f"Expected 'output_dataset_ids' to be '['test']', got {task_handler._output_dataset_ids!r}"
    payload = verify_storage_and_get_saved_json(mock_connection)
    assert (
        payload["progress"] == 1.0
    ), f"Expected 'progress' to be '1.0', got {payload['progress']!r}"
    assert (
        payload["status"] == "COMPLETED"
    ), f"Expected 'status' to be 'COMPLETED', got {payload['status']!r}"
    assert payload["output_dataset_ids"] == [
        "test"
    ], f"Expected 'output_dataset_ids' to be '['test']', got {payload['output_dataset_ids']!r}"
    try:
        datetime.fromisoformat(payload["time_completed"])
    except:
        pytest.fail(f"'time_completed' is not valid ISO-formatted date/time")


# Test 3 - Mark as Failed
def test_mark_as_failed(task_handler, mock_connection):
    """
    Veify that 'mark_as_failed':
        - set 'progress' to 1.0
        - set 'status to' 'FAILED'
        - keep 'output_dataset_ids' same
    """
    task_handler.mark_as_failed()
    assert (
        task_handler._progress == 1.0
    ), f"Expected 'progress' to be '1.0', got {task_handler._progress!r}"
    assert (
        task_handler._status == "FAILED"
    ), f"Expected 'status' to be 'FAILED', got {task_handler._status!r}"
    assert (
        task_handler._output_dataset_ids == []
    ), f"Expected 'output_dataset_ids' to be '[]', got {task_handler._output_dataset_ids!r}"
    payload = verify_storage_and_get_saved_json(mock_connection)
    assert (
        payload["progress"] == 1.0
    ), f"Expected 'progress' to be '1.0', got {payload['progress']!r}"
    assert (
        payload["status"] == "FAILED"
    ), f"Expected 'status' to be 'FAILED', got {payload['status']!r}"
    assert (
        payload["output_dataset_ids"] == []
    ), f"Expected 'output_dataset_ids' to be '[]', got {payload['output_dataset_ids']!r}"
    try:
        datetime.fromisoformat(payload["time_completed"])
    except:
        pytest.fail(f"'time_completed' is not valid ISO-formatted date/time")


def test_mark_as_failed_persists_structured_error_payload(
    task_handler, mock_connection
):
    cause = RuntimeError("db write failed")
    error = CompoxTaskError(
        "Failed to update task record.",
        code="task_record_update_failed",
        details={"field": "progress"},
        cause=cause,
    )

    task_handler.mark_as_failed(error)

    payload = verify_storage_and_get_saved_json(mock_connection)
    assert payload["error"]["type"] == "CompoxTaskError"
    assert payload["error"]["code"] == "task_record_update_failed"
    assert payload["error"]["details"] == {"field": "progress"}
    assert payload["error"]["cause"] == {
        "type": "RuntimeError",
        "message": "db write failed",
    }


# Test 4 - Test Invalid Progress and Status
def test_invalid_progress_raises(task_handler):
    """
    Verify that invalid status or progress raises ValueError
    """
    with pytest.raises(ValueError):
        task_handler.progress = -0.1
    with pytest.raises(ValueError):
        task_handler.progress = 1.1
    with pytest.raises(ValueError):
        task_handler.status = " "


def test_task_record_update_failure_is_typed(task_handler):
    """
    Verify task-record update failures use one typed exception path.
    """
    with patch.object(
        task_handler,
        "_save_task_record",
        side_effect=RuntimeError("database write failed"),
    ):
        with pytest.raises(CompoxTaskError) as exc_info:
            task_handler.progress = 0.5

    assert exc_info.value.code == "task_record_update_failed"
    assert exc_info.value.details == {"field": "progress"}
    assert isinstance(exc_info.value.cause, RuntimeError)


# Test 5 - Test Fetch Algorithm
def test_fetch_algorithm(task_handler):
    """
    Verify that fetch_algorithm calls the private cached method, stores the
    resolved runtime device and registers the runner.
    """
    dummy_runner = MagicMock(name="RunnerInstance")
    dummy_assets = ["asset-1", "asset-2"]
    resolved_runtime_device = "cuda"

    with patch.object(
        TaskHandler,
        "_TaskHandler__cached_fetch_algorithm",
        return_value=(
            dummy_runner,
            dummy_assets,
            resolved_runtime_device,
        ),
    ) as mock_cached:

        returned = task_handler.fetch_algorithm("1")

    assert (
        returned == dummy_runner
    ), f"Expected returned runner to be the same dummy_runner, got {returned!r}"
    assert task_handler.algorithm_assets == dummy_assets, (
        f"Expected algorithm_assets to be {dummy_assets!r}, "
        f"got {task_handler.algorithm_assets!r}"
    )
    stored_record = task_handler._get_task_record()
    assert (
        stored_record["resolved_execution_device"] == resolved_runtime_device
    ), (
        "Expected fetch_algorithm to persist the resolved runtime device, "
        f"got {stored_record.get('resolved_execution_device')!r}"
    )
    task_handler.set_as_current_handler()

    context_task_handler = current_handler.get()
    assert (
        context_task_handler == task_handler
    ), f"Expected current_task_handler to be the same task_handler, got {context_task_handler!r}"


# Test 6 - Test __cached_fetch_algorithm
def test_cached_fetch_algorithm_uses_cache(task_handler, mock_connection):
    """
    Verify that the private cached fetch algorithm method caches after first call.
    """
    _clear_task_handler_algorithm_cache()

    class DummyRunner:
        def __new__(cls):
            instance = super().__new__(cls)
            instance.initialize = MagicMock()
            instance._load_assets = MagicMock()
            return instance

    with patch("compox.tasks.TaskHandler.ZipImporter") as mock_import:
        dummy_mod = MagicMock()
        dummy_mod.Runner = DummyRunner
        mock_import.return_value.__enter__.return_value = dummy_mod
        runner1 = task_handler.fetch_algorithm("1")
        calls_first = mock_connection.get_objects.call_count

        assert (
            calls_first > 0
        ), f"Expected storage calls on first fetch, got {calls_first}"

        runner2 = task_handler.fetch_algorithm("1")
        calls_second = mock_connection.get_objects.call_count

    assert (
        runner1 == runner2
    ), "Expected same runner instance from cache on second fetch"
    # fetch_algorithm now always refreshes algorithm metadata and also reloads the
    # task record to persist the resolved runtime device, so only module loading
    # should stay cached between calls.
    assert (
        mock_import.call_count == 1
    ), f"Expected ZipImporter to be invoked once due to cache hit, got {mock_import.call_count}"
    assert calls_second - calls_first == 2, (
        "Expected one extra algorithm metadata fetch and one task-record fetch "
        f"on second call, got {calls_second - calls_first}"
    )


def test_cached_fetch_algorithm_reuses_multiple_cached_algorithms(
    task_handler, mock_connection, monkeypatch
):
    """
    Verify TaskHandler can reuse more than one cached runner when cache capacity allows it.
    """
    _set_task_handler_algorithm_cache_maxsize(monkeypatch, maxsize=3)
    _clear_task_handler_algorithm_cache()
    _configure_algorithm_cache_test_store(
        mock_connection,
        {
            "alg-a": "version-a",
            "alg-b": "version-b",
        },
    )

    runner_a_first = task_handler.fetch_algorithm("alg-a")
    runner_b_first = task_handler.fetch_algorithm("alg-b")
    runner_a_second = task_handler.fetch_algorithm("alg-a")

    assert getattr(runner_a_first, "VERSION", None) == "version-a"
    assert getattr(runner_b_first, "VERSION", None) == "version-b"
    assert (
        runner_a_first is runner_a_second
    ), "Expected alg-a runner to be reused from cache after switching to alg-b"
    assert (
        runner_a_first is not runner_b_first
    ), "Expected different algorithms to keep distinct runner instances in cache"
    assert (
        _count_bucket_reads(mock_connection, "module-store") == 2
    ), "Expected module-store to be read only for the first fetch of each unique algorithm"


def test_cached_fetch_algorithm_evicts_least_recently_used_runner(
    task_handler, mock_connection, monkeypatch
):
    """
    Verify TaskHandler evicts the least recently used runner when cache capacity is exceeded.
    """
    _set_task_handler_algorithm_cache_maxsize(monkeypatch, maxsize=3)
    _clear_task_handler_algorithm_cache()
    _configure_algorithm_cache_test_store(
        mock_connection,
        {
            "alg-a": "version-a",
            "alg-b": "version-b",
            "alg-c": "version-c",
            "alg-d": "version-d",
        },
    )

    runner_a_first = task_handler.fetch_algorithm("alg-a")
    runner_b_first = task_handler.fetch_algorithm("alg-b")
    task_handler.fetch_algorithm("alg-c")
    runner_b_second = task_handler.fetch_algorithm("alg-b")
    task_handler.fetch_algorithm("alg-d")
    runner_a_second = task_handler.fetch_algorithm("alg-a")

    assert (
        runner_b_first is runner_b_second
    ), "Expected alg-b to remain cached after being accessed again before eviction"
    assert (
        runner_a_first is not runner_a_second
    ), "Expected alg-a runner to be evicted and re-imported after cache overflow"
    assert (
        _count_bucket_reads(mock_connection, "module-store") == 5
    ), "Expected five module-store reads including one re-read after alg-a eviction"


def test_fetch_algorithm_resolves_new_latest_minor_when_minor_is_none(
    task_handler, mock_connection
):
    """
    If latest minor changes in storage and caller passes algorithm_minor_version=None,
    fetch_algorithm should load the new latest module (not stale cached one).
    """
    _clear_task_handler_algorithm_cache()

    algorithm_id = "alg-latest-cache-test"
    algorithm_key = f"{algorithm_id}~cache_test_algo~1"
    latest_minor_state = {"value": "0"}
    module_v0 = _create_runner_zip_with_version("v0")
    module_v1 = _create_runner_zip_with_version("v1")

    def list_objects(bucket):
        if bucket == "algorithm-store":
            return [{"Key": algorithm_key}]
        if bucket == "module-store":
            return [{"Key": "module_v0"}, {"Key": "module_v1"}]
        return []

    def get_objects(bucket, keys):
        if bucket == "algorithm-store":
            record = {
                "algorithm_id": algorithm_id,
                "algorithm_name": "cache_test_algo",
                "algorithm_major_version": "1",
                "supported_devices": ["cpu"],
                "default_device": "cpu",
                "latest_algorithm_minor_version": latest_minor_state["value"],
                "algorithm_minor_version": {
                    "0": {"module_id": "module_v0", "assets": {}},
                    "1": {"module_id": "module_v1", "assets": {}},
                },
            }
            return [json.dumps(record)]
        if bucket == "module-store":
            module_id = keys[0]
            return [module_v0 if module_id == "module_v0" else module_v1]
        return [json.dumps({})]

    mock_connection.list_objects.side_effect = list_objects
    mock_connection.get_objects.side_effect = get_objects

    runner_first = task_handler.fetch_algorithm(
        algorithm_id, algorithm_minor_version=None
    )
    assert getattr(runner_first, "VERSION", None) == "v0"

    # Simulate redeploy: latest minor is now 1.
    latest_minor_state["value"] = "1"

    runner_second = task_handler.fetch_algorithm(
        algorithm_id, algorithm_minor_version=None
    )
    assert getattr(runner_second, "VERSION", None) == "v1"


def test_cached_fetch_algorithm_uses_distinct_entries_for_device_override(
    task_handler, mock_connection
):
    """
    Verify different execution_device_override values do not reuse the same cache entry.
    """
    _clear_task_handler_algorithm_cache()

    algorithm_key = "alg-device-cache-test~cache_test_algo~1"
    module_archive = _create_runner_zip_with_version("device-test")

    def list_objects(bucket):
        if bucket == "algorithm-store":
            return [{"Key": algorithm_key}]
        if bucket == "module-store":
            return [{"Key": "module_device"}]
        return []

    def get_objects(bucket, keys):
        if bucket == "algorithm-store":
            return [
                json.dumps(
                    {
                        "algorithm_id": "alg-device-cache-test",
                        "algorithm_name": "cache_test_algo",
                        "algorithm_major_version": "1",
                        "supported_devices": ["cpu", "gpu"],
                        "default_device": "cpu",
                        "latest_algorithm_minor_version": "0",
                        "algorithm_minor_version": {
                            "0": {
                                "module_id": "module_device",
                                "assets": {},
                            }
                        },
                    }
                )
            ]
        if bucket == "module-store":
            return [module_archive]
        return [json.dumps({})]

    mock_connection.list_objects.side_effect = list_objects
    mock_connection.get_objects.side_effect = get_objects

    with patch(
        "compox.tasks.TaskHandler.check_system_gpu_availability",
        return_value=(True, 1),
    ):
        runner_cpu = task_handler.fetch_algorithm(
            "alg-device-cache-test", execution_device_override="cpu"
        )
        runner_gpu = task_handler.fetch_algorithm(
            "alg-device-cache-test", execution_device_override="gpu"
        )

    assert (
        runner_cpu is not runner_gpu
    ), "Expected CPU and GPU overrides to use distinct cached runner entries"
    assert (
        _count_bucket_reads(mock_connection, "module-store") == 2
    ), "Expected module-store to be read once per device override variant"


def test_cached_fetch_algorithm_uses_distinct_entries_for_checkpoint_id(
    task_handler, mock_connection
):
    """
    Verify different checkpoint_id values do not reuse the same cache entry.
    """
    _clear_task_handler_algorithm_cache()

    algorithm_key = "alg-checkpoint-cache-test~cache_test_algo~1"
    module_archive = _create_runner_zip_with_version("checkpoint-test")

    def list_objects(bucket):
        if bucket == "algorithm-store":
            return [{"Key": algorithm_key}]
        if bucket == "module-store":
            return [{"Key": "module_checkpoint"}]
        return []

    def get_objects(bucket, keys):
        if bucket == "algorithm-store":
            return [
                json.dumps(
                    {
                        "algorithm_id": "alg-checkpoint-cache-test",
                        "algorithm_name": "cache_test_algo",
                        "algorithm_major_version": "1",
                        "supported_devices": ["cpu"],
                        "default_device": "cpu",
                        "latest_algorithm_minor_version": "0",
                        "algorithm_minor_version": {
                            "0": {
                                "module_id": "module_checkpoint",
                                "assets": {"base-asset": "asset-base"},
                            }
                        },
                    }
                )
            ]
        if bucket == "module-store":
            return [module_archive]
        return [json.dumps({})]

    mock_connection.list_objects.side_effect = list_objects
    mock_connection.get_objects.side_effect = get_objects

    def checkpoint_factory(checkpoint_id, database_connection):
        checkpoint = MagicMock()
        checkpoint.checkpoint_manifest.assets = {
            "base-asset": f"asset-{checkpoint_id}"
        }
        return checkpoint

    with patch(
        "compox.tasks.TaskHandler.AlgorithmCheckpoint",
        side_effect=checkpoint_factory,
    ):
        runner_a = task_handler.fetch_algorithm(
            "alg-checkpoint-cache-test", checkpoint_id="checkpoint-a"
        )
        runner_b = task_handler.fetch_algorithm(
            "alg-checkpoint-cache-test", checkpoint_id="checkpoint-b"
        )

    assert (
        runner_a is not runner_b
    ), "Expected different checkpoint ids to use distinct cached runner entries"
    assert (
        _count_bucket_reads(mock_connection, "module-store") == 2
    ), "Expected module-store to be read once per checkpoint variant"


def test_cached_fetch_algorithm_default_maxsize_one_evicts_previous_runner(
    task_handler, mock_connection
):
    """
    Verify the shipped TaskHandler cache size of one evicts the previous runner.
    """
    _clear_task_handler_algorithm_cache()
    _configure_algorithm_cache_test_store(
        mock_connection,
        {
            "alg-a": "version-a",
            "alg-b": "version-b",
        },
    )

    runner_a_first = task_handler.fetch_algorithm("alg-a")
    runner_b = task_handler.fetch_algorithm("alg-b")
    runner_a_second = task_handler.fetch_algorithm("alg-a")

    assert getattr(runner_a_first, "VERSION", None) == "version-a"
    assert getattr(runner_b, "VERSION", None) == "version-b"
    assert (
        runner_a_first is not runner_a_second
    ), "Expected alg-a to be evicted after caching alg-b with maxsize=1"
    assert (
        _count_bucket_reads(mock_connection, "module-store") == 3
    ), "Expected module-store re-read after the default single-entry cache evicts alg-a"


def test_task_handler_class_cache_size_controls_algorithm_cache(
    mock_connection,
):
    """
    Verify TaskHandler cache capacity follows the configured class attribute.
    """
    TaskHandler._ALGORITHM_CACHE_MAXSIZE = 1
    _clear_task_handler_algorithm_cache()
    TaskHandler._ALGORITHM_CACHE_MAXSIZE = 2
    handler = TaskHandler(
        task_id="test-task-id",
        database_connection=mock_connection,
        database_update=True,
    )
    _clear_task_handler_algorithm_cache()
    _configure_algorithm_cache_test_store(
        mock_connection,
        {
            "alg-a": "version-a",
            "alg-b": "version-b",
        },
    )

    runner_a_first = handler.fetch_algorithm("alg-a")
    handler.fetch_algorithm("alg-b")
    runner_a_second = handler.fetch_algorithm("alg-a")

    assert (
        runner_a_first is runner_a_second
    ), "Expected TaskHandler class cache size to allow two cached runners"

    TaskHandler._ALGORITHM_CACHE_MAXSIZE = 1
    _clear_task_handler_algorithm_cache()


# Test 7 – Fetch Asset
def test_fetch_asset(task_handler, mock_connection):
    """
    Verify fetch_asset retrieves the correct bytes and calls the proper bucket.
    """

    class DummyRunner:
        def __new__(cls):
            instance = super().__new__(cls)
            return instance

        def initialize(self, device=None):
            self.device = device

        def _load_assets(self):
            pass

    with patch("compox.tasks.TaskHandler.ZipImporter") as mock_import:
        dummy_mod = MagicMock()
        dummy_mod.Runner = DummyRunner
        mock_import.return_value.__enter__.return_value = dummy_mod
        task_handler.fetch_algorithm("1")
        result = task_handler.fetch_asset("asset-1")

        assert (
            result.read() == b"dummy binary content"
        ), f"Expected asset bytes to be 'b'dummy binary content'', got {result.read()!r}"
        mock_connection.get_objects.assert_any_call("asset-store", ["asset-1"])


# Test 8 – Fetch Data (All keys)
def test_fetch_data_all_keys(task_handler):
    """
    Verify fetch_data returns all arrays when no keys specified.
    """
    result = task_handler.fetch_data(["file-id-1"], DummySchema)

    assert isinstance(result, list), f"Expected list, got {type(result)!r}"
    assert len(result) == 1, f"Expected list of length 1, got {result!r}"

    data = result[0]
    try:
        np.testing.assert_array_equal(data["array1"], np.array([1, 2]))
    except:
        pytest.fail(f"'array1' should be array([1,2]), got {data['array1']!r}")

    try:
        np.testing.assert_array_equal(data["array2"], np.array([3, 4, 5]))
    except:
        pytest.fail(
            f"'array2' should be array([3,4,5]), got {data['array2']!r}"
        )


# Test 9 – Fetch Data (One key)
def test_fetch_data_specific_key(task_handler):
    """
    Verify fetch_data returns only the specified key and sets missing to None.
    """
    result = task_handler.fetch_data(["file-id-2"], DummySchema, "array1")

    assert isinstance(result, list), f"Expected list, got {type(result)!r}"
    assert len(result) == 1, f"Expected list of length 1, got {result!r}"

    data = result[0]
    try:
        np.testing.assert_array_equal(data["array1"], np.array([1, 2]))
    except:
        pytest.fail(f"'array1' should be array([1,2]), got {data['array1']!r}")

    assert (
        data["array2"] == None
    ), f"Expected missing key 'array2' to be None, got {data.get('array2')!r}"


# Test 10 – Fetch Data (invalid HDF 5)
def test_fetch_data_invalid_hdf5_raises(task_handler, mock_connection):
    """
    Verify fetch_data raises on invalid HDF5 bytes.
    """
    mock_connection.get_objects.side_effect = lambda bucket, keys: [
        b"not a valid hdf5" if bucket == "data-store" else json.dumps({})
    ]
    with pytest.raises(Exception):
        task_handler.fetch_data(["bad-file"], DummySchema)


# Test 11 – Fetch Data (parallel)
def test_fetch_data_parallel(task_handler):
    """
    Verify parallel fetch_data increments stats and returns correct list.
    """
    ids = ["id1", "id2", "id3"]
    result = task_handler.fetch_data(ids, DummySchema, parallel=True)

    assert isinstance(result, list), f"Expected list, got {type(result)!r}"
    assert (
        len(result) == 3
    ), f"Expected list of length 3, got list of length {len(result)!r}"
    assert (
        task_handler.file_fetching_stats["count"] == 3
    ), f"Expected fetch count 3, got {task_handler.file_fetching_stats['count']}"

    data = result[0]
    try:
        np.testing.assert_array_equal(data["array1"], np.array([1, 2]))
    except:
        pytest.fail(f"'array1' should be array([1,2]), got {data['array1']!r}")

    try:
        np.testing.assert_array_equal(data["array2"], np.array([3, 4, 5]))
    except:
        pytest.fail(
            f"'array2' should be array([3,4,5]), got {data['array2']!r}"
        )


# Test 12 - Post Data
def test_post_data(task_handler, mock_connection):
    """
    Verify post_data uploads HDF5 with correct datasets and returns IDs.
    """
    # Create 2 test data dictionaries
    data1 = {"array1": np.array([1, 2, 3]), "array2": np.array([10, 20, 30])}

    data2 = {"array1": np.array([-1, -2]), "array2": None}

    # Patch generate_uuid
    with patch("compox.tasks.TaskHandler.generate_uuid") as mock_uuid:
        mock_uuid.side_effect = ["id1", "id2"]
        out_ids = task_handler.post_data([data1, data2], DummySchema)

        assert out_ids == [
            "id1",
            "id2",
        ], f"Expected output IDs '['id1','id2']', got {out_ids!r}"
        assert (
            mock_connection.put_objects.call_count == 3
        ), f"Expected 2 put_objects calls, got {mock_connection.put_objects.call_count}"

        call_args_list = mock_connection.put_objects.call_args_list
        uploaded_0 = call_args_list[0]
        uploaded_1 = call_args_list[1]
        uploaded_2 = call_args_list[2]
        bucket0, keys0, _ = uploaded_0[0]
        bucket1, keys1, vals1 = uploaded_1[0]
        bucket2, keys2, vals2 = uploaded_2[0]

        assert (
            bucket0 == "execution-store"
        ), f"Expected first put_object call to be 'execution-store', got {bucket0!r}"
        assert keys0 == [
            "test-task-id"
        ], f"Expected execution-store key to be test-task-id, got {keys0!r}"
        assert (
            bucket1 == "data-store"
        ), f"Expected first put_object call to be 'data-store', got {bucket1!r}"
        assert keys1 == [
            "id1"
        ], f"Expected data-store key to be '['id1']', got {keys1!r}"
        assert (
            bucket2 == "data-store"
        ), f"Expected second put_object call to be 'data-store', got {bucket2!r}"
        assert keys2 == [
            "id2"
        ], f"Expected data-store key to be '['id2']', got {keys2!r}"

        # Dictionary 1
        h5_bytes_1 = vals1[0]
        fh1 = io.BytesIO(h5_bytes_1)

        with h5py.File(fh1, "r") as f1:
            try:
                np.testing.assert_array_equal(
                    data1["array1"], np.array([1, 2, 3])
                )
            except:
                pytest.fail(
                    f"'array1' should be array([1,2,3]), got {data1['array1']!r}"
                )

            try:
                np.testing.assert_array_equal(
                    data1["array2"], np.array([10, 20, 30])
                )
            except:
                pytest.fail(
                    f"'array2' should be array([10,20,30]), got {data1['array2']!r}"
                )

        # Dictionary 2
        h5_bytes_2 = vals2[0]
        fh2 = io.BytesIO(h5_bytes_2)

        with h5py.File(fh2, "r") as f2:
            try:
                np.testing.assert_array_equal(
                    data2["array1"], np.array([-1, -2])
                )
            except:
                pytest.fail(
                    f"'array1' should be array([-1,-2]), got {data2['array1']!r}"
                )
            assert (
                "array2" not in f2.keys()
            ), f"Expected 'array2' missing, keys: {list(f2.keys())!r}"


def test_post_and_fetch_data_roundtrip_string_list(task_handler, mock_connection):
    stored_data = {}
    original_get_objects = mock_connection.get_objects.side_effect

    def put_objects(bucket, keys, values):
        if bucket == "execution-store":
            return True
        if bucket == "data-store":
            for key, value in zip(keys, values):
                stored_data[key] = value
            return True
        raise ValueError(f"Unexpected bucket {bucket!r}")

    def get_objects(bucket, keys):
        if bucket == "data-store":
            return [stored_data[key] for key in keys]
        return original_get_objects(bucket, keys)

    mock_connection.put_objects.side_effect = put_objects
    mock_connection.get_objects.side_effect = get_objects

    with patch("compox.tasks.TaskHandler.generate_uuid") as mock_uuid:
        mock_uuid.return_value = "string-list-id"
        out_ids = task_handler.post_data(
            [{"region_names": ["mid_intensity", "high_intensity"]}],
            StringListSchema,
        )

    assert out_ids == ["string-list-id"]

    result = task_handler.fetch_data(out_ids, StringListSchema)
    assert result == [
        {"region_names": ["mid_intensity", "high_intensity"]}
    ]


# Test 13 - Post invalid data
def test_post_data_validation_error(task_handler):
    """
    Verify post_data raises ValidationError on schema mismatch.
    """
    data1 = {"array1": np.array([1, 2, 3]), "array2": "not an array"}

    # Patch generate_uuid
    with patch("compox.tasks.TaskHandler.generate_uuid") as mock_uuid:
        mock_uuid.side_effect = ["id1", "id2"]
        with pytest.raises(ValidationError):
            task_handler.post_data(data1, DummySchema)


# test 14 - Post data + Exception
def test_post_data_storage_exception(task_handler, mock_connection):
    """
    Verify post_data propagates storage exceptions.
    """
    mock_connection.put_objects.side_effect = RuntimeError("S3 down")
    with pytest.raises(CompoxExecutionError):
        task_handler.post_data(
            [{"array1": np.array([0]), "array2": np.array([1])}], DummySchema
        )


# Test 15 - Save Item to Session
def test_save_item_to_session(handler_with_session):
    """
    Verify save_item_to_session stores obj under given key.
    """
    handler, session = handler_with_session
    assert (
        "my_key" not in session.store
    ), f"Did not expect 'my_key' in 'session.store'"
    handler.save_item_to_session(obj={"foo": 123}, key="my_key")
    assert (
        "my_key" in session.store
    ), f"Expect 'my_key' to be in 'session.store'"
    assert session.store["my_key"] == {
        "foo": 123
    }, f"Expected session['my_key'] to be ('foo' : 123), got {session.store.get('my_key')!r}"


# Test 16 - Load Item from Session
def test_load_item_from_session(handler_with_session):
    """
    Verify load_item_from_session returns stored object.
    """
    handler, session = handler_with_session
    session.store["my_key2"] = [1, 2, 3]
    result = handler.load_item_from_session("my_key2")
    assert result == [1, 2, 3], f"Expected loaded '[1,2,3]', got {result!r}"
    assert (
        "my_key2" in session.store
    ), f"Expect 'my_key2' to be in 'session.store'"


# Test 17 - Remove Item from Session
def test_remove_item_from_session(handler_with_session):
    """
    Verify remove_item_from_session deletes the key.
    """
    handler, session = handler_with_session
    session.store["my_key3"] = "value"
    handler.remove_item_from_session("my_key3")
    assert (
        "my_key3" not in session.store
    ), f"Expected 'my_key3' removed from session.store"


# Test 18 - Load Nonexistent key
def test_load_nonexistent_key_raises(handler_with_session):
    """
    Verify loading missing key raises KeyError.
    """
    handler, session = handler_with_session
    session.store.clear()

    with pytest.raises(KeyError):
        handler.load_item_from_session("my_key4")


# Test 19 - Remove Nonexistent Key
def test_remove_nonexistent_key_raises(handler_with_session):
    """
    Verify removing missing key raises KeyError.
    """
    handler, session = handler_with_session
    session.store.clear()

    with pytest.raises(KeyError):
        handler.remove_item_from_session("my_key5")


# Test 20 - Load/Remove/Save Key=None
def test_use_none_key(handler_with_session):
    """
    Verify session operations raise when session is None.
    """
    handler, _ = handler_with_session
    handler.task_session = None

    with pytest.raises(Exception):
        handler.load_item_from_session("unknown_key")
    with pytest.raises(Exception):
        handler.save_item_to_session("unknown_key")
    with pytest.raises(Exception):
        handler.remove_item_from_session("unknown_key")


# Test 21 - Test log
def test_update_log_writes_to_db(task_handler, mock_connection):
    """
    Verify update_log stores stream contents in database.
    """
    task_handler.logger.info("some message")
    task_handler.update_log()
    payload = verify_storage_and_get_saved_json(mock_connection)
    assert (
        "some message" in payload["log"]
    ), f"Expected 'some message' in log, got {payload.get('log')!r}"


# Test 22 - Test get Device (no cuda --> get cpu)
def test_get_device_no_cuda(task_handler):
    """
    Verify _get_device returns 'cpu' when CUDA unavailable.
    """
    with patch(
        "compox.tasks.TaskHandler.check_system_gpu_availability",
        return_value=(False, None),
    ):
        algo = {"default_device": "gpu", "supported_devices": ["cpu", "gpu"]}

        dev = task_handler._TaskHandler__get_device(
            algo, execution_device_override=None
        )
        assert dev == "cpu", f" Expected device to be 'cpu', got {dev!r}"


# Test 23 - Test get Device (cuda available)
def test_get_device_cuda_available(task_handler):
    """
    Verify _get_device returns 'cpu' when CUDA available.
    """
    with patch(
        "compox.tasks.TaskHandler.check_system_gpu_availability",
        return_value=(True, None),
    ):
        algo = {"default_device": "gpu", "supported_devices": ["cpu", "gpu"]}

        dev = task_handler._TaskHandler__get_device(
            algo, execution_device_override=None
        )
        assert dev == "cuda", f" Expected device to be 'cuda', got {dev!r}"


# Test 24 - Test get Device (cuda available, but override --> get cpu)
def test_get_device_respects_override(task_handler):
    """
    Verify _get_device returns 'cpu' when CUDA available with override.
    """
    with patch(
        "compox.tasks.TaskHandler.check_system_gpu_availability",
        return_value=(True, None),
    ):
        algo = {"default_device": "gpu", "supported_devices": ["cpu", "gpu"]}

        dev = task_handler._TaskHandler__get_device(
            algo, execution_device_override="cpu"
        )
        assert dev == "cpu", f" Expected device to be 'cpu', got {dev!r}"
