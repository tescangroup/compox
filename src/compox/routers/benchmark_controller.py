"""
Copyright 2026 Tescan group, a.s.
All rights reserved
"""

import json
from datetime import datetime

from fastapi import APIRouter, Request
from fastapi.responses import JSONResponse

from compox.pydantic_models import (
    BenchmarkRecord,
    BenchmarkResponse,
    IncomingBenchmarkRequest,
    ResponseMessage,
)
from compox.server_utils import generate_uuid, find_algorithm_by_id
from compox.tasks.StopRequest import StopRequest


router = APIRouter(prefix="/api", tags=["benchmark-controller"])


@router.post(
    "/v0/benchmark-algorithm",
    summary="Benchmarks an algorithm using dummy data",
    response_model=BenchmarkResponse,
    responses={
        500: {"model": ResponseMessage},
        404: {"model": ResponseMessage},
    },
)
def benchmark_algorithm(
    request: Request,
    incoming_benchmark_request: IncomingBenchmarkRequest,
) -> BenchmarkResponse:
    """
    Triggers a benchmark run for an algorithm.  Instead of operating on
    uploaded datasets the runner generates its own dummy input, making it
    possible to measure performance without preparing real data.

    Parameters
    ----------
    request : Request
        The request.
    incoming_benchmark_request : IncomingBenchmarkRequest
        The incoming benchmark request.

    Returns
    -------
    BenchmarkResponse
        The benchmark response containing the benchmark id.

    Raises
    ------
    Exception
        If the server backend is not supported or saving the benchmark record fails.
    """
    benchmark_id = generate_uuid()
    database_connection = request.app.state.database_connection
    settings = request.app.state.settings

    # check if algorithm exists
    _, algorithm_id, _, _, _ = find_algorithm_by_id(
        incoming_benchmark_request.algorithm_id,
        database_connection.list_objects("algorithm-store"),
    )
    if algorithm_id is None:
        return JSONResponse(
            status_code=404,
            content={"detail": "Algorithm not found"},
        )

    benchmark_record = BenchmarkRecord(
        benchmark_id=benchmark_id,
        algorithm_id=incoming_benchmark_request.algorithm_id,
        additional_parameters=incoming_benchmark_request.additional_parameters,
        status="PENDING",
        time_started=str(datetime.now()),
        time_completed="",
        log="",
        data={},
    )
    try:
        database_connection.put_objects(
            "benchmark-store",
            [benchmark_id],
            [json.dumps(benchmark_record.model_dump())],
        )
    except Exception as e:
        fallback_record = benchmark_record.model_dump()
        fallback_record["status"] = "FAILED"
        fallback_record["time_completed"] = str(datetime.now())
        fallback_record["log"] = f"Failed to save benchmark record: {e}"
        request.app.state.emergency_record_store.write_record(
            "benchmark-store",
            benchmark_id,
            fallback_record,
            storage_error=e,
        )
        return BenchmarkResponse(benchmark_id=benchmark_id)

    if settings.inference.backend_settings.executor == "celery":
        request.app.state.executor.send_task(
            "benchmark_task",
            args=[
                json.dumps(benchmark_record.model_dump()),
            ],
            task_id=benchmark_id,
            retries=2,
        )
    elif (
        settings.inference.backend_settings.executor
        == "fastapi_background_tasks"
    ):
        from compox.tasks.benchmark_task_fastapi import (
            benchmark_task_fastapi,
        )

        request.app.state.executor.submit(
            benchmark_task_fastapi,
            database_connection=database_connection,
            benchmark_record=benchmark_record,
            emergency_record_store=request.app.state.emergency_record_store,
        )
    else:
        raise Exception(
            "Server backend {} not supported:".format(
                settings.inference.backend_settings.executor
            )
        )

    return BenchmarkResponse(benchmark_id=benchmark_id)


@router.get(
    "/v0/benchmarks/{benchmark_id}",
    summary="Get benchmark record by id",
    response_model=BenchmarkRecord,
    responses={500: {"model": ResponseMessage}},
)
async def get_benchmark_record(
    benchmark_id: str, request: Request
) -> BenchmarkRecord:
    """
    Get benchmark record by id.

    Parameters
    ----------
    benchmark_id : str
        The id of the benchmark record.
    request : Request
        The request.

    Returns
    -------
    BenchmarkRecord
        The benchmark record.
    """
    database_connection = request.app.state.database_connection
    emergency_record_store = request.app.state.emergency_record_store
    try:
        fallback_record = emergency_record_store.read_record(
            "benchmark-store", benchmark_id
        )
        object_exists = database_connection.check_objects_exist(
            "benchmark-store", [benchmark_id]
        )[0]
        if not object_exists:
            if fallback_record is not None:
                return BenchmarkRecord(**fallback_record)
            return JSONResponse(
                status_code=404,
                content={"detail": "Benchmark record not found"},
            )
        return BenchmarkRecord(
            **json.loads(
                database_connection.get_objects(
                    "benchmark-store", [benchmark_id]
                )[0]
            )
        )
    except Exception as e:
        fallback_record = emergency_record_store.read_record(
            "benchmark-store", benchmark_id
        )
        if fallback_record is not None:
            return BenchmarkRecord(**fallback_record)
        return JSONResponse(
            status_code=500,
            content={"detail": f"Failed to get benchmark record: {e}"},
        )
