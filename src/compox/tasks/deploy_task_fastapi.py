"""
Copyright 2026 TESCAN 3DIM, s.r.o.
All rights reserved
"""

import json
import os
from datetime import datetime
from typing import Optional

from loguru import logger

from compox.algorithm_utils.AlgorithmDeployer import AlgorithmDeployer
from compox.database_connection.BaseConnection import BaseConnection
from compox.internal.EmergencyRecordStore import EmergencyRecordStore
from compox.exceptions import CompoxValidationError


DEPLOY_COLLECTION = "deploy-store"
ALGORITHM_COLLECTION = "algorithm-store"


def _update_deploy_record(
    database_connection: BaseConnection,
    deploy_id: str,
    updates: dict,
) -> None:
    """
    Update a deploy record with new fields.

    Parameters
    ----------
    database_connection : BaseConnection
        Database connection used to read/write deploy records.
    deploy_id : str
        Identifier of the deploy job record in the deploy store.
    updates : dict
        Key-value pairs to merge into the deploy record.

    Returns
    -------
    None
    """
    record = json.loads(
        database_connection.get_objects(DEPLOY_COLLECTION, [deploy_id])[0]
    )
    record.update(updates)
    database_connection.put_objects(
        DEPLOY_COLLECTION, [deploy_id], [json.dumps(record)]
    )


def _build_algorithm_meta(
    database_connection: BaseConnection,
    algorithm_id: str,
) -> dict:
    """
    Build algorithm metadata for the deploy record.

    Parameters
    ----------
    database_connection : BaseConnection
        Database connection used to read algorithm records.
    algorithm_id : str
        Identifier of the deployed algorithm.

    Returns
    -------
    dict
        Algorithm metadata with id, name, major version, and latest minor version.
    """
    for item in database_connection.list_objects(ALGORITHM_COLLECTION):
        key = item["Key"]
        if key.startswith(f"{algorithm_id}~"):
            algorithm_json = json.loads(
                database_connection.get_objects(
                    ALGORITHM_COLLECTION, [key]
                )[0]
            )
            return {
                "algorithm_id": algorithm_id,
                "algorithm_name": algorithm_json["algorithm_name"],
                "algorithm_major_version": algorithm_json[
                    "algorithm_major_version"
                ],
                "algorithm_minor_version": algorithm_json[
                    "latest_algorithm_minor_version"
                ],
            }
    return {
        "algorithm_id": algorithm_id,
        "algorithm_name": "",
        "algorithm_major_version": "",
        "algorithm_minor_version": "",
    }


def _set_algorithm_removable(
    database_connection: BaseConnection,
    algorithm_id: str,
    removable: bool = True,
) -> None:
    """
    Mark an algorithm as removable in the algorithm store.

    Parameters
    ----------
    database_connection : BaseConnection
        Database connection used to read/write algorithm records.
    algorithm_id : str
        Identifier of the deployed algorithm.
    removable : bool, optional
        Whether the algorithm can be removed via the deploy delete endpoint.

    Returns
    -------
    None
    """
    for item in database_connection.list_objects(ALGORITHM_COLLECTION):
        key = item["Key"]
        if key.startswith(f"{algorithm_id}~"):
            algorithm_json = json.loads(
                database_connection.get_objects(
                    ALGORITHM_COLLECTION, [key]
                )[0]
            )
            algorithm_json["removable"] = removable
            database_connection.put_objects(
                ALGORITHM_COLLECTION, [key], [json.dumps(algorithm_json)]
            )
            return


def _set_algorithm_exportable(
    database_connection: BaseConnection,
    algorithm_id: str,
    exportable: bool = True,
) -> None:
    """
    Mark an algorithm as exportable in the algorithm store.

    Parameters
    ----------
    database_connection : BaseConnection
        Database connection used to read/write algorithm records.
    algorithm_id : str
        Identifier of the deployed algorithm.
    exportable : bool, optional
        Whether the algorithm can be exported via the export endpoint.

    Returns
    -------
    None
    """
    for item in database_connection.list_objects(ALGORITHM_COLLECTION):
        key = item["Key"]
        if key.startswith(f"{algorithm_id}~"):
            algorithm_json = json.loads(
                database_connection.get_objects(
                    ALGORITHM_COLLECTION, [key]
                )[0]
            )
            algorithm_json["exportable"] = exportable
            database_connection.put_objects(
                ALGORITHM_COLLECTION, [key], [json.dumps(algorithm_json)]
            )
            return


@logger.catch
def deploy_task_fastapi(
    database_connection: BaseConnection,
    deploy_id: str,
    path: str,
    emergency_record_store: EmergencyRecordStore | None = None,
    algorithm_name_override: Optional[str] = None,
    algorithm_major_version_override: Optional[str] = None,
    removable_override: Optional[bool] = None,
    exportable_override: Optional[bool] = None,
) -> None:
    """
    FastAPI background task for deploying an algorithm from a local path.

    Parameters
    ----------
    database_connection : BaseConnection
        Database connection used to read/write deploy and algorithm records.
    deploy_id : str
        Identifier of the deploy job record in the deploy store.
    path : str
        Local filesystem path to an algorithm directory or a .zip archive.
    algorithm_name_override : Optional[str], optional
        Optional override for the algorithm name stored in the database.
    algorithm_major_version_override : Optional[str], optional
        Optional override for the algorithm major version stored in the database.
    removable_override : Optional[bool], optional
        Optional override for whether the algorithm is removable.
    exportable_override : Optional[bool], optional
        Optional override for whether the algorithm is exportable.

    Returns
    -------
    None

    Raises
    ------
    FileNotFoundError
        If the provided path does not exist.
    ValueError
        If the path is neither a directory nor a .zip file.
    Exception
        Any exception during deployment is re-raised after recording failure.
    """
    try:
        emergency_record_store = emergency_record_store or EmergencyRecordStore()
        _update_deploy_record(
            database_connection,
            deploy_id,
            {"status": "RUNNING", "time_started": str(datetime.now())},
        )

        if not os.path.exists(path):
            raise CompoxValidationError(
                f"Path not found: {path}",
                code="deploy_path_not_found",
                details={"path": path},
            )

        if os.path.isdir(path):
            deployer = AlgorithmDeployer(path)
            algorithm_id = deployer.store_algorithm(
                database_connection=database_connection,
                algorithm_name_override=algorithm_name_override,
                algorithm_major_version_override=algorithm_major_version_override,
            )
            _set_algorithm_removable(
                database_connection,
                algorithm_id,
                deployer.removable
                if removable_override is None
                else removable_override,
            )
            _set_algorithm_exportable(
                database_connection,
                algorithm_id,
                deployer.exportable
                if exportable_override is None
                else exportable_override,
            )
            algorithm_meta = {
                "algorithm_id": algorithm_id,
                "algorithm_name": deployer.algorithm_name,
                "algorithm_major_version": deployer.algorithm_major_version,
                "algorithm_minor_version": deployer.algorithm_minor_version,
            }
        elif os.path.isfile(path) and path.lower().endswith(".zip"):
            algorithm_id = AlgorithmDeployer.deploy_from_zip(
                path,
                database_connection=database_connection,
                algorithm_name_override=algorithm_name_override,
                algorithm_major_version_override=algorithm_major_version_override,
            )
            algorithm_record = None
            for item in database_connection.list_objects(ALGORITHM_COLLECTION):
                key = item["Key"]
                if key.startswith(f"{algorithm_id}~"):
                    algorithm_record = json.loads(
                        database_connection.get_objects(
                            ALGORITHM_COLLECTION, [key]
                        )[0]
                    )
                    break
            _set_algorithm_removable(
                database_connection,
                algorithm_id,
                (
                    bool(algorithm_record and algorithm_record.get("removable"))
                    if removable_override is None
                    else removable_override
                ),
            )
            _set_algorithm_exportable(
                database_connection,
                algorithm_id,
                (
                    bool(
                        algorithm_record
                        and algorithm_record.get("exportable", True)
                    )
                    if exportable_override is None
                    else exportable_override
                ),
            )
            algorithm_meta = _build_algorithm_meta(
                database_connection, algorithm_id
            )
        else:
            raise CompoxValidationError(
                "Path must be a directory or a .zip file.",
                code="invalid_deploy_path_type",
                details={"path": path},
            )

        _update_deploy_record(
            database_connection,
            deploy_id,
            {
                "status": "COMPLETED",
                "time_completed": str(datetime.now()),
                **algorithm_meta,
            },
        )
    except Exception as e:
        failed_record = {
            "deploy_id": deploy_id,
            "status": "FAILED",
            "path": path,
            "time_completed": str(datetime.now()),
            "log": str(e),
        }
        try:
            record = json.loads(
                database_connection.get_objects(DEPLOY_COLLECTION, [deploy_id])[0]
            )
            failed_record = {**record, **failed_record}
        except Exception:
            pass
        emergency_record_store.write_record(
            DEPLOY_COLLECTION,
            deploy_id,
            failed_record,
            storage_error=e,
        )
        raise
