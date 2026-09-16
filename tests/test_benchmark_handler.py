"""
Copyright 2026 Tescan group, a.s.
All rights reserved
"""

import json
from unittest.mock import MagicMock

import pytest

from compox.tasks.BenchmarkHandler import BenchmarkHandler
from compox.tasks.TaskHandler import TaskStoppedException


@pytest.fixture
def benchmark_connection():
    connection = MagicMock()
    record = {
        "benchmark_id": "benchmark-id",
        "algorithm_id": "algorithm-id",
        "additional_parameters": {},
        "status": "PENDING",
        "time_started": "2026-01-01 10:00:00",
        "time_completed": "",
        "log": "",
        "data": {},
    }

    def get_objects(bucket, keys):
        if bucket == "benchmark-store":
            return [json.dumps(record)]
        return [json.dumps({})]

    def put_objects(bucket, keys, values):
        if bucket == "benchmark-store":
            record.update(json.loads(values[0]))

    def check_objects_exist(bucket, keys):
        return [False for _ in keys]

    connection.get_objects.side_effect = get_objects
    connection.put_objects.side_effect = put_objects
    connection.check_objects_exist.side_effect = check_objects_exist
    return connection, record


def test_benchmark_handler_mark_as_completed(benchmark_connection):
    connection, record = benchmark_connection
    handler = BenchmarkHandler("benchmark-id", connection)

    handler.mark_as_completed({"latency_s": 0.5})

    assert record["data"] == {"latency_s": 0.5}
    assert record["status"] == "COMPLETED"
    assert record["time_completed"] != ""
    assert "File fetching stats" in record["log"]
    assert set(record) == {
        "benchmark_id",
        "algorithm_id",
        "additional_parameters",
        "status",
        "time_started",
        "time_completed",
        "log",
        "data",
    }


def test_benchmark_handler_mark_as_failed(benchmark_connection):
    connection, record = benchmark_connection
    handler = BenchmarkHandler("benchmark-id", connection)

    handler.mark_as_failed(RuntimeError("boom"))

    assert record["data"] == {}
    assert record["status"] == "FAILED"
    assert record["time_completed"] != ""
    assert "Benchmark failed" in record["log"]
    assert set(record) == {
        "benchmark_id",
        "algorithm_id",
        "additional_parameters",
        "status",
        "time_started",
        "time_completed",
        "log",
        "data",
    }
