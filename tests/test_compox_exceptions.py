"""
Copyright 2026 Tescan group, a.s.
All rights reserved
"""

from types import SimpleNamespace

from fastapi import APIRouter
from fastapi.testclient import TestClient

from compox.components.api_builder import ApiBuilder
from compox.database_connection.InMemoryConnection import InMemoryConnection
from compox.exceptions import CompoxTaskError, CompoxNotFoundError
from compox.routers import execution_controller, sample_controller


def _settings(tmp_path):
    return SimpleNamespace(log_path=str(tmp_path / "compox.log"))


def test_compox_error_handler_returns_structured_error(tmp_path):
    router = APIRouter()

    @router.get("/typed-error")
    def typed_error():
        raise CompoxNotFoundError(
            "Missing thing",
            code="thing_not_found",
            details={"thing_id": "thing-1"},
        )

    app = (
        ApiBuilder()
        .with_settings(_settings(tmp_path))
        .with_database_connection(InMemoryConnection())
        .with_algorithm_exporter(object())
        .with_route(router)
        .build()
    )

    with TestClient(app) as client:
        response = client.get("/typed-error")

    assert response.status_code == 404
    assert response.json() == {
        "detail": "Missing thing",
        "code": "thing_not_found",
        "retryable": False,
        "details": {"thing_id": "thing-1"},
    }


def test_compox_error_handler_sanitizes_internal_errors(tmp_path):
    router = APIRouter()

    @router.get("/internal-typed-error")
    def internal_typed_error():
        raise CompoxTaskError(
            "Task record write failed for internal collection task-store",
            code="task_record_update_failed",
            details={"collection": "task-store", "key": "task-1"},
        )

    app = (
        ApiBuilder()
        .with_settings(_settings(tmp_path))
        .with_database_connection(InMemoryConnection())
        .with_algorithm_exporter(object())
        .with_route(router)
        .build()
    )

    with TestClient(app) as client:
        response = client.get("/internal-typed-error")

    assert response.status_code == 500
    assert response.json() == {
        "detail": "Failed due to an internal server error.",
        "code": "task_record_update_failed",
        "retryable": False,
    }


def test_request_validation_handler_keeps_detail_and_adds_code(tmp_path):
    router = APIRouter()

    @router.get("/items/{item_id}")
    def get_item(item_id: int):
        return {"item_id": item_id}

    app = (
        ApiBuilder()
        .with_settings(_settings(tmp_path))
        .with_database_connection(InMemoryConnection())
        .with_algorithm_exporter(object())
        .with_route(router)
        .build()
    )

    with TestClient(app) as client:
        response = client.get("/items/not-an-int")

    payload = response.json()
    assert response.status_code == 422
    assert "detail" in payload
    assert payload["code"] == "request_validation_error"
    assert payload["retryable"] is False


def test_execute_algorithm_missing_file_returns_typed_error(tmp_path):
    db = InMemoryConnection()
    db.create_collections(["algorithm-store", "data-store"])
    app = (
        ApiBuilder()
        .with_settings(_settings(tmp_path))
        .with_database_connection(db)
        .with_algorithm_exporter(object())
        .with_route(execution_controller.router)
        .build()
    )
    app.state.settings = SimpleNamespace(
        inference=SimpleNamespace(
            backend_settings=SimpleNamespace(
                executor="fastapi_background_tasks"
            )
        )
    )

    with TestClient(app) as client:
        response = client.post(
            "/api/v0/execute-algorithm",
            json={
                "algorithm_id": "algo-1",
                "input_dataset_ids": ["missing-file"],
                "additional_parameters": {},
            },
        )

    payload = response.json()
    assert response.status_code == 404
    assert payload["detail"]
    assert payload["code"] == "input_datasets_not_found"
    assert payload["details"] == {"missing_dataset_ids": ["missing-file"]}


def test_sample_invalid_raw_body_returns_validation_error(tmp_path):
    app = (
        ApiBuilder()
        .with_settings(_settings(tmp_path))
        .with_database_connection(InMemoryConnection())
        .with_algorithm_exporter(object())
        .with_route(sample_controller.router)
        .build()
    )

    with TestClient(app) as client:
        response = client.post("/api/v0/sample", data="invalid")

    payload = response.json()
    assert response.status_code == 422
    assert "detail" in payload
    assert payload["code"] == "request_validation_error"
    assert payload["retryable"] is False
