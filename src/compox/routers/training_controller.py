"""
Copyright 2025 TESCAN 3DIM, s.r.o.
All rights reserved
"""

from fastapi import APIRouter, Request
import json
from datetime import datetime
from compox.pydantic_models import (
    TrainingRecord,
    TrainingResponse,
    IncomingTrainingRequest,
    ResponseMessage,
)
from compox.server_utils import generate_uuid, find_algorithm_by_id
from compox.training.TrainingSample import TrainingSample
from compox.tasks.StopRequest import StopRequest
from compox.exceptions import (
    CompoxError,
    CompoxNotFoundError,
    CompoxTaskError,
    CompoxValidationError,
)

STOPPABLE_STATES = {"PENDING", "RUNNING", "STARTED"}
TERMINAL_STATES = {"STOPPED", "FAILED", "COMPLETED"}

router = APIRouter(prefix="/api", tags=["training-controller"])


# post training request to torchserve
@router.post(
    "/v0/train-algorithm",
    summary="Trains an algorithm on sample(s)",
    response_model=TrainingResponse,
    responses={
        500: {"model": ResponseMessage},
        404: {"model": ResponseMessage},
    },
)
def train_algorithm(
    request: Request,
    incoming_training_request: IncomingTrainingRequest,
):
    """
    Executes a training of an algorithm on sample(s).

    Parameters
    ----------
    request : Request
        The request.
    incoming_training_request : IncomingTrainingRequest
        The incoming training request.

    Returns
    -------
    TrainingResponse
        The training response.
    """
    algorithm_collection_name = "algorithm-store"
    sample_collection_name = "sample-store"
    training_collection_name = "training-store"

    training_id = generate_uuid()
    database_connection = request.app.state.database_connection
    settings = request.app.state.settings

    # check if input samples are valid
    samples_exist = database_connection.check_objects_exist(
        sample_collection_name, incoming_training_request.training_data
    )
    if False in samples_exist:
        not_found_samples = [
            incoming_training_request.training_data[i]
            for i in range(len(samples_exist))
            if not samples_exist[i]
        ]
        raise CompoxNotFoundError(
            "Input samples with the following identifiers not found: {}".format(
                "\n".join(not_found_samples)
            ),
            code="input_samples_not_found",
            details={"missing_sample_ids": not_found_samples},
        )

    # check whether files associated with samples are present
    not_found_files = []
    for sample_id in incoming_training_request.training_data:
        sample = TrainingSample(
            database_connection,
            sample_id=sample_id,
        )
        _, missing_files = sample._validate_files_exist()
        not_found_files.extend(missing_files)

    if len(not_found_files) > 0:
        raise CompoxNotFoundError(
            "Files (associated with input sample) with the following identifiers not found: {}".format(
                "\n".join(not_found_files)
            ),
            code="sample_files_not_found",
            details={"missing_file_ids": not_found_files},
        )

    # check if algorithm exists and get its JSON
    algorithm_key, _, _, _, _ = find_algorithm_by_id(
        incoming_training_request.algorithm_id,
        database_connection.list_objects(algorithm_collection_name),
    )

    if algorithm_key is None:
        raise CompoxNotFoundError(
            "Algorithm not found",
            code="algorithm_not_found",
            details={"algorithm_id": incoming_training_request.algorithm_id},
        )
    # create a new algorithm id for the trained algorithm

    training_record = TrainingRecord(
        training_id=training_id,
        algorithm_id=incoming_training_request.algorithm_id,
        status="PENDING",
        checkpoint_id=incoming_training_request.checkpoint_id,
        algorithm_minor_version=incoming_training_request.algorithm_minor_version,
        tags=incoming_training_request.tags,
        progress=0.0,
        time_started=str(datetime.now()),
        time_completed=None,
        log="",
        training_data=incoming_training_request.training_data,
        additional_parameters=incoming_training_request.additional_parameters,
        state={},
        output_checkpoint_ids=[],
    )
    try:
        database_connection.put_objects(
            training_collection_name,
            [training_id],
            [json.dumps(training_record.model_dump())],
        )
    except Exception as e:
        fallback_record = training_record.model_dump()
        fallback_record["status"] = "FAILED"
        fallback_record["progress"] = 1.0
        fallback_record["time_completed"] = str(datetime.now())
        fallback_record["log"] = f"Failed to save training record: {e}"
        request.app.state.emergency_record_store.write_record(
            training_collection_name,
            training_id,
            fallback_record,
            storage_error=e,
        )
        return TrainingResponse(training_id=training_id)

    if settings.inference.backend_settings.executor == "celery":
        request.app.state.executor.send_task(
            "training_task",
            args=[
                json.dumps(training_record.model_dump()),
            ],
            task_id=training_id,
            retries=2,
        )

    elif (
        settings.inference.backend_settings.executor
        == "fastapi_background_tasks"
    ):
        from compox.training.training_task_fastapi import (
            training_task_fastapi,
        )

        request.app.state.executor.submit(
            training_task_fastapi,
            database_connection=database_connection,
            training_record=training_record,
            emergency_record_store=request.app.state.emergency_record_store,
        )

    return TrainingResponse(training_id=training_id)


@router.get(
    "/v0/training/{training_id}",
    summary="Get training record by id",
    response_model=TrainingRecord,
    responses={
        500: {"model": ResponseMessage},
        404: {"model": ResponseMessage},
    },
)
async def get_training_record(training_id: str, request: Request):
    """
    Get training record by id.

    Parameters
    ----------
    training_id : str
        The id of the training record.

    Returns
    -------
    TrainingRecord
        The training record.
    """
    training_collection_name = "training-store"
    database_connection = request.app.state.database_connection
    emergency_record_store = request.app.state.emergency_record_store
    try:
        fallback_record = emergency_record_store.read_record(
            training_collection_name, training_id
        )
        primary_record = None

        object_exists = database_connection.check_objects_exist(
            training_collection_name, [training_id]
        )[0]
        if object_exists:
            primary_record = TrainingRecord(
                **json.loads(
                    database_connection.get_objects(
                        training_collection_name, [training_id]
                    )[0]
                )
            )
        else:
            try:
                primary_record = TrainingRecord(
                    **json.loads(
                        database_connection.get_objects(
                            training_collection_name, [training_id]
                        )[0]
                    )
                )
            except Exception:
                primary_record = None

        if primary_record is None:
            if fallback_record is not None:
                return TrainingRecord(**fallback_record)
            raise CompoxNotFoundError(
                "Training record not found",
                code="training_record_not_found",
                details={"training_id": training_id},
            )

        if fallback_record is not None and fallback_record.get("status") == "FAILED":
            if primary_record.status.upper() not in TERMINAL_STATES:
                return TrainingRecord(**fallback_record)
        return primary_record

    except CompoxError:
        raise
    except Exception as e:
        fallback_record = emergency_record_store.read_record(
            training_collection_name, training_id
        )
        if fallback_record is not None:
            return TrainingRecord(**fallback_record)
        raise CompoxTaskError(
            "Failed to get training record",
            code="training_record_read_failed",
            cause=e,
        )


@router.post(
    "/v0/training/{training_id}/stop",
    summary="Stop training by id",
    response_model=ResponseMessage,
    responses={
        500: {"model": ResponseMessage},
        404: {"model": ResponseMessage},
    },
)
async def stop_training(training_id: str, request: Request):
    """
    Stop training by id.

    Parameters
    ----------
    training_id : str
        The id of the training record.

    Returns
    -------
    ResponseMessage
        The response message.
    """
    database_connection = request.app.state.database_connection

    # get execution record
    try:
        object_exists = database_connection.check_objects_exist(
            "training-store", [training_id]
        )[0]
        if not object_exists:
            raise CompoxNotFoundError(
                "Training record not found",
                code="training_record_not_found",
                details={"training_id": training_id},
            )
        training_record = TrainingRecord(
            **json.loads(
                database_connection.get_objects(
                    "training-store", [training_id]
                )[0]
            )
        )
    except CompoxError:
        raise
    except Exception as e:
        raise CompoxTaskError(
            "Failed to get training record",
            code="training_record_read_failed",
            cause=e,
        )

    status = training_record.status.upper()
    if status not in STOPPABLE_STATES:
        raise CompoxValidationError(
            f"Training in state {status} cannot be stopped",
            code="training_not_stoppable",
            details={"training_id": training_id, "status": status},
        )

    try:
        stop_request = StopRequest(training_id, database_connection)
        stop_request.submit()
        return ResponseMessage(detail="Stop request posted successfully")
    except Exception as e:
        raise CompoxTaskError(
            "Failed to post stop request",
            code="stop_request_failed",
            cause=e,
        )
