"""
Copyright 2024 TESCAN 3DIM, s.r.o.
All rights reserved
"""

import json
import os
from typing import List, Optional, Union
from fastapi import APIRouter, Request, Query
from fastapi.responses import StreamingResponse
from pydantic import ValidationError
from starlette.background import BackgroundTask

from compox.pydantic_models import (
    AlgorithmRegisteredResponse,
    ResponseMessage,
    S3ModelFileRecord,
    FailedAlgorithmRegisteredResponse,
)
from compox.exceptions import (
    CompoxAlgorithmError,
    CompoxError,
    CompoxNotFoundError,
)
from compox.algorithm_utils.AlgorithmStorageMetrics import (
    AlgorithmStorageMetrics,
)

router = APIRouter(prefix="/api", tags=["algorithms-controller"])


@router.get(
    "/v0/algorithm/{algorithm_name}/{algorithm_major_version}",
    summary="Returns algorithm by its name and version.",
    response_model=Union[
        AlgorithmRegisteredResponse, FailedAlgorithmRegisteredResponse
    ],
    responses={
        500: {"model": ResponseMessage},
        404: {"model": ResponseMessage},
    },
)
def get_algorithm(
    algorithm_name: str, algorithm_major_version: str, request: Request
) -> Union[AlgorithmRegisteredResponse, FailedAlgorithmRegisteredResponse]:
    """
    Returns algorithm by its name and version.

    Parameters
    ----------
    algorithm_name : str
        Algorithm name.

    algorithm_major_version : str
        Algorithm version.

    request: Request
        The request.

    Returns
    -------
    Union[AlgorithmRegisteredResponse, FailedAlgorithmRegisteredResponse]
        The algorithm.
    """
    database_connection = request.app.state.database_connection
    algorithm_collection = "algorithm-store"
    storage_metrics = AlgorithmStorageMetrics(database_connection)
    try:
        # get all algoerithms
        all_algorithms = database_connection.list_objects(algorithm_collection)

        if len(all_algorithms) == 0:
            raise CompoxNotFoundError(
                "No algorithms found in the algorithm store",
                code="no_algorithms_found",
            )

        # find the algorithm with the requested name and major version
        found_algorithm = None

        for key in all_algorithms:
            algorithm_json = storage_metrics.load_record(key["Key"])

            if (
                algorithm_json["algorithm_name"].lower()
                == algorithm_name.lower()
                and algorithm_json["algorithm_major_version"].lower()
                == algorithm_major_version.lower()
            ):
                found_algorithm = algorithm_json
                break

        if found_algorithm is not None:
            try:
                return AlgorithmRegisteredResponse(
                    algorithm_id=found_algorithm["algorithm_id"],
                    algorithm_minor_versions=list(
                        found_algorithm["algorithm_minor_version"].keys()
                    ),
                    latest_algorithm_minor_version=found_algorithm[
                        "latest_algorithm_minor_version"
                    ],
                    algorithm_minor_version=found_algorithm[
                        "latest_algorithm_minor_version"
                    ],
                    algorithm_name=found_algorithm["algorithm_name"],
                    algorithm_version=found_algorithm[
                        "algorithm_major_version"
                    ],
                    algorithm_type=found_algorithm["algorithm_type"],
                    algorithm_tags=found_algorithm["algorithm_tags"],
                    algorithm_description=found_algorithm[
                        "algorithm_description"
                    ],
                    supported_devices=found_algorithm["supported_devices"],
                    default_device=found_algorithm["default_device"],
                    additional_parameters=found_algorithm[
                        "additional_parameters"
                    ],
                    training_parameters=found_algorithm.get(
                        "training_parameters", {}
                    ),
                    benchmark_outputs=found_algorithm.get(
                        "benchmark_outputs", []
                    ),
                    removable=found_algorithm.get("removable", False),
                    exportable=found_algorithm.get("exportable", True),
                    storage_metrics=found_algorithm.get("storage_metrics"),
                )
            except ValidationError as e:
                return FailedAlgorithmRegisteredResponse(
                    algorithm_name=found_algorithm["algorithm_name"],
                    algorithm_version=found_algorithm[
                        "algorithm_major_version"
                    ],
                    message=f"The algorithm has not been configured correctly.\n{e}",
                )
        else:
            raise CompoxNotFoundError(
                "Model with the requested name and major version not found in the model store",
                code="algorithm_not_found",
                details={
                    "algorithm_name": algorithm_name,
                    "algorithm_major_version": algorithm_major_version,
                },
            )
    except CompoxError:
        raise
    except Exception as e:
        raise CompoxAlgorithmError(
            "Failed to get algorithm due to an internal server error.",
            code="algorithm_read_failed",
            cause=e,
        ) from e


@router.get(
    "/v0/algorithm/all",
    summary="Lists all available algorithms",
    response_model=List[
        Union[AlgorithmRegisteredResponse, FailedAlgorithmRegisteredResponse]
    ],
    responses={500: {"model": ResponseMessage}},
)
async def list_model_files(
    request: Request,
    positive_tag: Optional[List[str]] | None = Query([]),
    negative_tag: Optional[List[str]] | None = Query([]),
    algorithm_type: Optional[str] | None = Query(None),
    supported_devices: Optional[List[str]] | None = Query([]),
) -> List[S3ModelFileRecord]:
    """
    Lists all available algorithms.

    Parameters
    ----------
    request : Request
        The request.
    positive_tag : Optional[List[str]] | None
        A list of tags the algorithm must have.
    negative_tag : Optional[List[str]] | None
        A list of tags the algorithm must not have.
    algorithm_type : Optional[str] | None
        The type of the algorithm.
    supported_devices : Optional[List[str]] | None
        The devices the algorithm is compatible with.

    Returns
    -------
    List[S3ModelFileRecord]
        The list of algorithms.
    """
    positive_tags = positive_tag
    negative_tags = negative_tag

    database_connection = request.app.state.database_connection
    algorithm_collection = "algorithm-store"
    storage_metrics = AlgorithmStorageMetrics(database_connection)

    try:
        # get all algoerithms
        all_algorithms = database_connection.list_objects(algorithm_collection)

        if len(all_algorithms) == 0:
            raise CompoxNotFoundError(
                "No algorithms found in the algorithm store",
                code="no_algorithms_found",
            )

        algorithms = []
        for key in all_algorithms:
            algorithm_json = storage_metrics.load_record(key["Key"])

            # check if the algorithm has all the positive tags
            if positive_tags:
                if not all(
                    tag in algorithm_json["algorithm_tags"]
                    for tag in positive_tags
                ):
                    continue

            # check if the algorithm has any of the negative tags
            if negative_tags:
                if any(
                    tag in algorithm_json["algorithm_tags"]
                    for tag in negative_tags
                ):
                    continue

            # check if the algorithm has the requested type
            if algorithm_type is not None:
                if (
                    algorithm_json["algorithm_type"].lower()
                    != algorithm_type.lower()
                ):
                    continue

            # check if the algorithm has the requested device
            if supported_devices:
                if not any(
                    d.lower()
                    in [
                        algorithm_device.lower()
                        for algorithm_device in algorithm_json[
                            "supported_devices"
                        ]
                    ]
                    for d in supported_devices
                ):
                    continue
            try:
                algorithms.append(
                    AlgorithmRegisteredResponse(
                        algorithm_id=algorithm_json["algorithm_id"],
                        algorithm_minor_versions=list(
                            algorithm_json["algorithm_minor_version"].keys()
                        ),
                        latest_algorithm_minor_version=algorithm_json[
                            "latest_algorithm_minor_version"
                        ],
                        algorithm_minor_version=algorithm_json[
                            "latest_algorithm_minor_version"
                        ],
                        algorithm_name=algorithm_json["algorithm_name"],
                        algorithm_version=algorithm_json[
                            "algorithm_major_version"
                        ],
                        algorithm_type=algorithm_json["algorithm_type"],
                        algorithm_tags=algorithm_json["algorithm_tags"],
                        algorithm_description=algorithm_json[
                            "algorithm_description"
                        ],
                        default_device=algorithm_json["default_device"],
                        supported_devices=algorithm_json["supported_devices"],
                        additional_parameters=algorithm_json[
                            "additional_parameters"
                        ],
                        training_parameters=algorithm_json.get(
                            "training_parameters", {}
                        ),
                        benchmark_outputs=algorithm_json.get(
                            "benchmark_outputs", []
                        ),
                        removable=algorithm_json.get("removable", False),
                        exportable=algorithm_json.get("exportable", True),
                        storage_metrics=algorithm_json.get("storage_metrics"),
                    )
                )
            except ValidationError as _:
                continue

        return algorithms
    except CompoxError:
        raise
    except Exception as e:
        raise CompoxAlgorithmError(
            "Failed to list algorithms due to an internal server error.",
            code="algorithm_list_failed",
            cause=e,
        ) from e


@router.get(
    "/v0/algorithm/{algorithm_name}/{algorithm_major_version}/export",
    summary="Export an algorithm",
    description=(
        "Exports an algorithm by name and major version. "
        "Optionally allows selecting a minor version and a specific checkpoint."
    ),
)
async def export_algorithm(
    request: Request,
    algorithm_name: str,
    algorithm_major_version: int,
    algorithm_minor_version: Optional[int] = Query(
        None, description="Optional minor version of the algorithm"
    ),
    checkpoint_id: Optional[str] = Query(
        None, description="Optional checkpoint identifier"
    ),
) -> StreamingResponse:
    """
    Export an algorithm by its name and version.

    Parameters
    ----------
    request : Request
        The request.
    algorithm_name : str
        Algorithm name.
    algorithm_major_version : int
        Algorithm major version.
    algorithm_minor_version : Optional[int]
        Optional minor version of the algorithm.
    checkpoint_id : Optional[str]
        Optional checkpoint identifier.

    Returns
    -------
    StreamingResponse
        The exported algorithm as a zip file.
    """

    exporter = request.app.state.algorithm_exporter
    database_connection = request.app.state.database_connection
    algorithm_collection = "algorithm-store"
    for item in database_connection.list_objects(algorithm_collection):
        key = item["Key"]
        algorithm_json = json.loads(
            database_connection.get_objects(
                algorithm_collection,
                [key],
            )[0]
        )
        if (
            algorithm_json["algorithm_name"].lower() == algorithm_name.lower()
            and algorithm_json["algorithm_major_version"].lower()
            == str(algorithm_major_version).lower()
        ):
            if not algorithm_json.get("exportable", True):
                raise CompoxAlgorithmError(
                    "Algorithm is not exportable.",
                    code="algorithm_not_exportable",
                    http_status=403,
                    details={
                        "algorithm_name": algorithm_name,
                        "algorithm_major_version": str(algorithm_major_version),
                    },
                )
            break

    try:
        fname, tmp_zip_path, export_stream = (
            exporter.export_algorithm_zip_stream(
                algorithm_name=algorithm_name,
                algorithm_major_version=str(algorithm_major_version),
                algorithm_minor_version=(
                    str(algorithm_minor_version)
                    if algorithm_minor_version is not None
                    else None
                ),
                algorithm_checkpoint_id=checkpoint_id,
            )
        )
    except CompoxError:
        raise
    except Exception as e:
        raise CompoxAlgorithmError(
            "Failed to export algorithm due to an internal server error.",
            code="algorithm_export_failed",
            cause=e,
        ) from e

    return StreamingResponse(
        export_stream,
        media_type="application/zip",
        headers={"Content-Disposition": f'attachment; filename="{fname}"'},
        background=BackgroundTask(os.remove, tmp_zip_path),
    )
