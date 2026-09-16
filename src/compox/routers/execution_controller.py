"""
Copyright 2024 TESCAN 3DIM, s.r.o.
All rights reserved
"""

from fastapi import APIRouter, Request
from compox.pydantic_models import (
    ExecutionRecord,
    ExecutionResponse,
    IncomingExecutionRequest,
    ResponseMessage,
)
import json
from datetime import datetime
from compox.server_utils import generate_uuid, find_algorithm_by_id
from compox.tasks.StopRequest import StopRequest
from compox.exceptions import (
    CompoxConfigurationError,
    CompoxError,
    CompoxNotFoundError,
    CompoxTaskError,
    CompoxValidationError,
)


STOPPABLE_STATES = {"PENDING", "RUNNING", "STARTED"}
TERMINAL_STATES = {"STOPPED", "FAILED", "COMPLETED"}


router = APIRouter(prefix="/api", tags=["execution-controller"])


# post inference request to torchserve
@router.post(
    "/v0/execute-algorithm",
    summary="Executes an algorithm on a dataset",
    response_model=ExecutionResponse,
    responses={
        500: {"model": ResponseMessage},
        404: {"model": ResponseMessage},
    },
)
def execute_algorithm(
    request: Request,
    incoming_execution_request: IncomingExecutionRequest,
) -> ExecutionResponse:
    """
    Executes an algorithm on a dataset.

    Parameters
    ----------
    request : Request
        The request.
    incoming_execution_request : IncomingExecutionRequest
        The incoming execution request.

    Returns
    -------
    ExecutionResponse
        The execution response.

    Raises
    ------
    Exception
        If the server backend is not supported or saving the execution record fails.
    """
    execution_id = generate_uuid()
    database_connection = request.app.state.database_connection
    settings = request.app.state.settings
    # check if input datasets exist
    files_exist = database_connection.check_objects_exist(
        "data-store", incoming_execution_request.input_dataset_ids
    )
    if False in files_exist:
        not_found_files = [
            incoming_execution_request.input_dataset_ids[i]
            for i in range(len(files_exist))
            if not files_exist[i]
        ]
        raise CompoxNotFoundError(
            "Input datasets with the following identifiers not found: {}".format(
                "\n".join(not_found_files)
            ),
            code="input_datasets_not_found",
            details={"missing_dataset_ids": not_found_files},
        )

    # check if algorithm exists
    _, algorithm_id, _, _, _ = find_algorithm_by_id(
        incoming_execution_request.algorithm_id,
        database_connection.list_objects("algorithm-store"),
    )
    if algorithm_id is None:
        raise CompoxNotFoundError(
            "Algorithm not found",
            code="algorithm_not_found",
            details={
                "algorithm_id": incoming_execution_request.algorithm_id
            },
        )
    # save execution record to db
    execution_record = ExecutionRecord(
        execution_id=execution_id,
        algorithm_id=incoming_execution_request.algorithm_id,
        input_dataset_ids=incoming_execution_request.input_dataset_ids,
        execution_device_override=incoming_execution_request.execution_device_override,
        resolved_execution_device=None,
        additional_parameters=incoming_execution_request.additional_parameters,
        session_token=incoming_execution_request.session_token,
        checkpoint_id=incoming_execution_request.checkpoint_id,
        algorithm_minor_version=incoming_execution_request.algorithm_minor_version,
        output_dataset_ids=[],
        status="PENDING",
        progress=0,
        time_started=str(datetime.now()),
        time_completed="",
        log="",
    )
    try:
        database_connection.put_objects(
            "execution-store",
            [execution_id],
            [json.dumps(execution_record.model_dump())],
        )
    except Exception as e:
        fallback_record = execution_record.model_dump()
        fallback_record["status"] = "FAILED"
        fallback_record["progress"] = 1.0
        fallback_record["time_completed"] = str(datetime.now())
        fallback_record["log"] = f"Failed to save execution record: {e}"
        request.app.state.emergency_record_store.write_record(
            "execution-store",
            execution_id,
            fallback_record,
            storage_error=e,
        )
        return ExecutionResponse(execution_id=execution_id)

    if settings.inference.backend_settings.executor == "celery":
        request.app.state.executor.send_task(
            "task",
            args=[
                json.dumps(execution_record.model_dump()),
            ],
            task_id=execution_id,
            retries=2,
        )
    elif (
        settings.inference.backend_settings.executor
        == "fastapi_background_tasks"
    ):
        from compox.tasks.fastapi_background_task import (
            execution_task_fastapi,
        )

        request.app.state.executor.submit(
            execution_task_fastapi,
            database_connection=database_connection,
            execution_record=execution_record,
            emergency_record_store=request.app.state.emergency_record_store,
        )
    else:
        raise CompoxConfigurationError(
            "Server backend {} not supported".format(
                settings.inference.backend_settings.executor
            ),
            code="invalid_server_backend",
        )

    return ExecutionResponse(execution_id=execution_id)


@router.post(
    "/v0/executions/{id}/stop",
    summary="Stops an execution by id",
    response_model=ResponseMessage,
)
async def stop_execution(id: str, request: Request) -> ResponseMessage:
    """
    Stops an execution by id.

    Parameters
    ----------
    id : str
        The id of the execution to stop.
    request : Request
        The request.

    Returns
    -------
    ResponseMessage
        The response message.
    """

    database_connection = request.app.state.database_connection

    # get execution record
    try:
        object_exists = database_connection.check_objects_exist(
            "execution-store", [id]
        )[0]
        if not object_exists:
            raise CompoxNotFoundError(
                "Execution record not found",
                code="execution_record_not_found",
                details={"execution_id": id},
            )
        execution_record = ExecutionRecord(
            **json.loads(
                database_connection.get_objects("execution-store", [id])[0]
            )
        )
    except CompoxError:
        raise
    except Exception as e:
        raise CompoxTaskError(
            "Failed to get execution record",
            code="execution_record_read_failed",
            cause=e,
        )

    status = execution_record.status.upper()
    if status not in STOPPABLE_STATES:
        raise CompoxValidationError(
            f"Execution in state {status} cannot be stopped",
            code="execution_not_stoppable",
            details={"execution_id": id, "status": status},
        )

    try:
        stop_request = StopRequest(id, database_connection)
        stop_request.submit()
        return ResponseMessage(detail="Stop request posted successfully")
    except Exception as e:
        raise CompoxTaskError(
            "Failed to post stop request",
            code="stop_request_failed",
            cause=e,
        )


@router.get(
    "/v0/executions/{id}",
    summary="Get execution record by id",
    response_model=ExecutionRecord,
    responses={500: {"model": ResponseMessage}},
)
async def get_execution_record(id: str, request: Request) -> ExecutionRecord:
    """
    Get execution record by id.

    Parameters
    ----------
    id : str
        The id of the execution record.

    request : Request
        The request.

    Returns
    -------
    ExecutionRecord
        The execution record.
    """
    database_connection = request.app.state.database_connection
    emergency_record_store = request.app.state.emergency_record_store
    try:
        fallback_record = emergency_record_store.read_record(
            "execution-store", id
        )
        object_exists = database_connection.check_objects_exist(
            "execution-store", [id]
        )[0]
        if not object_exists:
            if fallback_record is not None:
                return ExecutionRecord(**fallback_record)
            raise CompoxNotFoundError(
                "Execution record not found",
                code="execution_record_not_found",
                details={"execution_id": id},
            )
        primary_record = ExecutionRecord(
            **json.loads(
                database_connection.get_objects("execution-store", [id])[0]
            )
        )
        if fallback_record is not None and fallback_record.get("status") == "FAILED":
            if primary_record.status.upper() not in TERMINAL_STATES:
                return ExecutionRecord(**fallback_record)
        return primary_record
    except CompoxError:
        raise
    except Exception as e:
        fallback_record = emergency_record_store.read_record(
            "execution-store", id
        )
        if fallback_record is not None:
            return ExecutionRecord(**fallback_record)
        raise CompoxTaskError(
            "Failed to get execution record",
            code="execution_record_read_failed",
            cause=e,
        )
