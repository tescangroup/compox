"""
Copyright 2024 TESCAN 3DIM, s.r.o.
All rights reserved
"""

import io
import h5py
import json
from typing import Type, Any
import time
from datetime import datetime
from loguru import logger
from concurrent.futures import ThreadPoolExecutor
from pydantic import ValidationError
from compox.server_utils import (
    find_algorithm_by_id,
    generate_uuid,
    algorithm_cache,
    check_system_gpu_availability,
    check_mps_availability,
)
from compox.algorithm_utils.zip_importer import ZipImporter
from compox.algorithm_utils.io_schemas import DataSchema
from compox.internal.hdf5_io import HDF5IO
from compox.session.TaskSession import TaskSession
from compox.database_connection.S3Connection import S3Connection
from compox.training.AlgorithmCheckpoint import AlgorithmCheckpoint
from compox.tasks.StopRequest import StopRequest
from compox.internal.EmergencyRecordStore import EmergencyRecordStore
from compox.exceptions import (
    CompoxAlgorithmError,
    CompoxAssetError,
    CompoxError,
    CompoxExecutionError,
    CompoxImportError,
    CompoxNotFoundError,
    CompoxStateError,
    CompoxTaskError,
    error_failure_payload,
)


class TaskStoppedException(CompoxTaskError):
    """
    Raised when a task has been stopped by request.

    The class name is kept for backward compatibility with existing callers.
    """

    def __init__(self, message: str = "Task has been stopped.") -> None:
        super().__init__(
            message,
            code="task_stopped",
            http_status=409,
            retryable=False,
        )


class TaskHandler:
    """
    Task handler class for the execution task. This class is used to update
    the progress, status and log of the execution task. Also contains methods
    to fetch the algorithm, assets and data from the database server of choice.

    Parameters
    ----------
    task_id : str
        The identifier of the task. Typically a UUID.
    database_connection : S3Connection
        The database connection object instance. Must inherit from the
        BaseConnection class and implement the required methods.
    database_update : bool, optional
        Whether to the execution record in the database, by default True.
        Can be set to False for example when debugging locally.
    task_session : TaskSession | None, optional
        The task session object instance. Must inherit from the TaskSession
        class, by default None.
    """

    _RECORD_STORAGE_NAME = "execution-store"
    _ALGORITHM_CACHE_MAXSIZE = 1

    def __init__(
        self,
        task_id: str,
        database_connection: S3Connection,
        database_update: bool = True,
        task_session: TaskSession | None = None,
        emergency_record_store: EmergencyRecordStore | None = None,
    ):
        self.stop_request = StopRequest(task_id, database_connection)
        self._task_id = task_id
        self._progress = 0.0
        self.database_update = database_update
        self.database_connection = database_connection
        self.emergency_record_store = (
            emergency_record_store or EmergencyRecordStore()
        )
        self.algorithm_assets = None
        self.stream = io.StringIO()
        self.logger = logger.bind(log_type="TASK", task_id=task_id)
        self.logger_sink_id = self.logger.add(
            self.stream,
            format="{time:YYYY-MM-DD HH:mm:ss} {level} {message}",
            level="INFO",
            filter=lambda record: record["extra"].get("task_id")
            == self.task_id,
        )

        self.file_fetching_stats = {
            "count": 0.0,
            "time": 0.0,
        }
        self.file_posting_stats = {
            "count": 0.0,
            "time": 0.0,
        }

        self.status = "STARTED"
        self.task_session = task_session
        if task_session is not None:
            self.session_token = task_session.session_token

    def _get_task_record(self) -> dict:
        """
        Get the task record from the database.

        Returns
        -------
        dict
            The task record as a dictionary.

        Raises
        ------
        Exception
            If getting the task record fails.
        """
        self._check_for_stop_request()
        return json.loads(
            self.database_connection.get_objects(
                self._RECORD_STORAGE_NAME,
                [self._task_id],
            )[0]
        )

    def _check_for_stop_request(self) -> None:
        """
        Check if a stop request exists for the task. If a stop request exists,
        mark the task as stopped.

        Returns
        -------
        None

        Raises
        ------
        Exception
            If checking for stop request fails.
        """
        try:
            if (
                self.stop_request.exists()
                and not self.stop_request.is_acknowledged()
            ):
                self.logger.info("Received stop request. Stopping task.")
                self.stop_request.acknowledge()
                self.mark_as_stopped()
                return
        except TaskStoppedException:
            raise
        except Exception as e:
            error = CompoxTaskError(
                "Failed to check task stop request.",
                code="stop_request_check_failed",
                cause=e,
            )
            self.mark_as_failed(error)
            raise error from e

    def _save_task_record(self, task_record: dict) -> None:
        """
        Save the task record to the database.

        Parameters
        ----------
        task_record : dict
            The task record as a dictionary.

        Returns
        -------
        None

        Raises
        ------
        Exception
            If saving the task record fails.
        """
        self.database_connection.put_objects(
            self._RECORD_STORAGE_NAME,
            [self._task_id],
            [json.dumps(task_record).encode()],
        )

    def _update_task_record_field(self, field: str, value: Any) -> None:
        """
        Persist one field change to the task record.

        This keeps task-record update failures on one typed path instead of
        duplicating catch/fail/re-raise blocks in each property setter.
        """
        if not self.database_update:
            return

        try:
            task_record = self._get_task_record()
            task_record[field] = value
            self._save_task_record(task_record)
        except TaskStoppedException:
            raise
        except CompoxError as e:
            self.mark_as_failed(e)
            raise
        except Exception as e:
            error = CompoxTaskError(
                "Failed to update task record.",
                code="task_record_update_failed",
                details={"field": field},
                cause=e,
            )
            self.mark_as_failed(error)
            raise error from e

    @property
    def task_id(self):
        """
        The identifier of the task. Typically a UUID.

        :getter: Returns the task id.
        :setter: Sets the task id.
        :type: str
        """
        return self._task_id

    @property
    def progress(self):
        """
        The progress of the task in the range [0., 1.].

        :getter: Returns the progress of the task.
        :setter: Sets the progress of the task.
        :type: float
        """
        return self._progress

    @progress.setter
    def progress(self, progress: float) -> None:
        """
        Update the progress of the task.

        Parameters
        ----------
        progress : float
            The progress of the task in the range [0., 1.].

        Returns
        -------
        None

        Raises
        ------
        Exception
        """
        if not (0.0 <= progress <= 1.0):
            raise ValueError(
                f"Progress must be between 0 and 1. Got: {progress}"
            )

        self._progress = progress
        self._update_task_record_field("progress", progress)

    @property
    def status(self):
        """
        The status of the task. e.g. "RUNNING", "COMPLETED", "FAILED"

        :getter: Returns the status of the task.
        :setter: Sets the status of the task.
        :type: str
        """

        return self._status

    @status.setter
    def status(self, status: str) -> None:
        """
        Update the status of the task.

        Parameters
        ----------
        status : str
            The status of the task. e.g. "RUNNING", "COMPLETED", "FAILED"

        Returns
        -------
        None

        Raises
        ------
        Exception
        """
        if status not in [
            "RUNNING",
            "COMPLETED",
            "FAILED",
            "PENDING",
            "STARTED",
            "STOPPED",
        ]:
            raise ValueError(f"Invalid status. Got: {status}")

        self._status = status
        self._update_task_record_field("status", status)

    @property
    def output_dataset_ids(self):
        """
        The output dataset identifiers of the task.

        :getter: Returns the output dataset identifiers of the task.
        :setter: Sets the output dataset identifiers of the task.
        :type: list[str]
        """
        return self._output_dataset_ids

    @output_dataset_ids.setter
    def output_dataset_ids(self, output_dataset_ids: list[str]) -> None:
        """
        Update the output dataset identifiers of the task.

        Parameters
        ----------
        output_dataset_ids : list[str]
            The output dataset identifiers of the task.

        Returns
        -------
        None

        Raises
        ------
        Exception
        """
        self._output_dataset_ids = output_dataset_ids
        self._update_task_record_field("output_dataset_ids", output_dataset_ids)

    @property
    def time_completed(self):
        """
        The time the task was completed.

        :getter: Returns the time the task was completed.
        :setter: Sets the time the task was completed.
        :type: str
        """
        return self._time_completed

    @time_completed.setter
    def time_completed(self, time_completed: str) -> None:
        """
        Update the time the task was completed.

        Parameters
        ----------
        time_completed : str
            The time the task was completed.

        Returns
        -------
        None

        Raises
        ------
        Exception
        """
        self._time_completed = time_completed
        self._update_task_record_field("time_completed", time_completed)

    @property
    def session_token(self):
        """
        The identifier of the session. Typically a UUID.

        :getter: Returns the session id.
        :setter: Sets the session id.
        :type: str
        """
        return self._session_token

    @session_token.setter
    def session_token(self, session_token: str) -> None:
        """
        Update the session token of the task.

        Parameters
        ----------
        session_token : str
            The session token of the task.

        Returns
        -------
        None

        Raises
        ------
        Exception
        """
        self._session_token = session_token
        self._update_task_record_field("session_token", session_token)

    def set_as_current_handler(self) -> None:
        """
        Set this task handler as the current task handler in the
        current_task_handler context variable. This is used to access the
        current task handler from anywhere in the code.

        Returns
        -------
        None
        """
        from compox.tasks.context_handler import current_handler

        current_handler.set(self)

    def mark_as_completed(self, output_dataset_ids: list[str]) -> None:
        """
        Mark the task as completed and update its record in the database. This
        will set the progress to 1.0, the status to "COMPLETED" and the time
        completed to the current time.

        Parameters
        ----------
        output_dataset_ids : list[str]
            The output dataset identifiers of the task.

        Returns
        -------
        None
        """
        self.progress = 1.0
        self.output_dataset_ids = output_dataset_ids
        self.time_completed = str(datetime.now())

        # log the posting and fetching stats
        self._log_file_stats()
        self.update_log()
        self.status = "COMPLETED"

        logger.remove(self.logger_sink_id)

    def _log_file_stats(self) -> None:
        """
        Log the file fetching and posting stats.
        """
        self.logger.info(
            f"File fetching stats: {self.file_fetching_stats['count']} files "
            f"fetched in {self.file_fetching_stats['time']:.4f} seconds."
        )
        self.logger.info(
            f"File posting stats: {self.file_posting_stats['count']} files "
            f"posted in {self.file_posting_stats['time']:.4f} seconds."
        )

    def mark_as_failed(self, e: Exception | None = None) -> None:
        """
        Mark the task as failed and update its record in the database. This
        will set the progress to 1.0, the status to "FAILED" and the time
        completed to the current time. The exception that caused the task to
        fail will be logged in the task log.

        Parameters
        ----------
        e : Exception | None, optional
            The exception that caused the task to fail, by default None. It will
            be logged in the task log.

        Returns
        -------
        None
        """

        if isinstance(e, TaskStoppedException):
            return

        try:
            if e is not None:
                self.logger.opt(exception=e).error("Task failed")
            self._log_file_stats()
            self.log = str(self.stream.getvalue())

            failed_record = self._build_failed_record(
                time_completed=str(datetime.now()),
                error=e,
            )
            try:
                self.database_connection.put_objects(
                    self._RECORD_STORAGE_NAME,
                    [self._task_id],
                    [json.dumps(failed_record).encode()],
                )
                self.emergency_record_store.delete_record(
                    self._RECORD_STORAGE_NAME, self._task_id
                )
            except Exception as storage_error:
                self.emergency_record_store.write_record(
                    self._RECORD_STORAGE_NAME,
                    self._task_id,
                    failed_record,
                    storage_error=storage_error,
                )

            self._time_completed = failed_record["time_completed"]
            self._status = failed_record["status"]
            self._progress = failed_record["progress"]
            self._output_dataset_ids = failed_record["output_dataset_ids"]

        except TaskStoppedException:
            raise

        finally:
            try:
                logger.remove(self.logger_sink_id)
            except Exception:
                pass

    def mark_as_stopped(self) -> None:
        """
        Mark the task as stopped and update its record in the database. This
        will set the status to "STOPPED" and the time completed to the current
        time.

        Returns
        -------
        None
        """

        try:
            self.time_completed = str(datetime.now())
            self._log_file_stats()
            self.update_log()

        except CompoxError as e:
            self.mark_as_failed(e)
        except Exception as e:
            error = CompoxTaskError(
                "Failed while marking task as stopped.",
                code="task_stop_update_failed",
                cause=e,
            )
            self.mark_as_failed(error)
        finally:
            try:
                logger.remove(self.logger_sink_id)
            except Exception:
                pass
            try:
                self.stop_request.delete()
            except Exception:
                pass
            self.status = "STOPPED"
        raise TaskStoppedException("Task has been stopped.")

    def update_log(self) -> None:
        """
        Update the log of the task in the database. This method is called
        automatically when the task is completed or failed. It can also be
        called manually to update the log during the execution of the task.

        Returns
        -------
        None

        Raises
        ------
        Exception
        """
        self.log = str(self.stream.getvalue())
        self._update_task_record_field("log", self.log)

    def _build_failed_record(
        self,
        time_completed: str,
        error: Exception | str | None = None,
    ) -> dict:
        """
        Build a terminal failed task record from the current stored record.
        """
        try:
            task_record = json.loads(
                self.database_connection.get_objects(
                    self._RECORD_STORAGE_NAME,
                    [self._task_id],
                )[0]
            )
        except Exception:
            task_record = {self._record_id_field_name(): self._task_id}

        task_record["status"] = "FAILED"
        task_record["progress"] = 1.0
        task_record["time_completed"] = time_completed
        task_record["output_dataset_ids"] = []
        task_record["log"] = self.log
        if error is not None:
            task_record["error"] = error_failure_payload(error)
        return task_record

    def _record_id_field_name(self) -> str:
        if self._RECORD_STORAGE_NAME == "execution-store":
            return "execution_id"
        if self._RECORD_STORAGE_NAME == "training-store":
            return "training_id"
        return "record_id"

    def fetch_algorithm(
        self,
        algorithm_id: str,
        execution_device_override: str | None = None,
        checkpoint_id: str | None = None,
        algorithm_minor_version: str | None = None,
    ) -> object:
        """
        Fetches the algorithm from the database and imports its corresponding
        Python module and runner class.

        Parameters
        ----------
        algorithm_id : str
            The id of the algorithm.

        execution_device_override : str | None, optional
            The requested abstract execution device class, by default None.
            This uses the algorithm metadata vocabulary (for example ``cpu``,
            ``gpu`` or ``mps``). Compox resolves this request to a concrete
            runtime device string passed into the runner, such as ``cpu``,
            ``cuda`` or ``mps``.

        checkpoint_id : str | None, optional
            The id of the checkpoint, by default None. If provided, the
            checkpoint will be used to load the model assets.

        algorithm_minor_version : str | None, optional
            The minor version of the algorithm, by default None. If provided, the
            minor version will be used to load the model assets.

        Returns
        -------
        object
            The algorithm Runner object.

        Raises
        ------
        ValueError
            If fetch algorithm failed.
        """

        # get the algorithm json file from algorithm-store bucket

        try:
            self.logger.info(
                f"Fetching algorithm {algorithm_id} from the database."
            )
            self.logger.info("Loading the algorithm.")
            start = time.time()
            found_algorithm_key, _, _, _, _ = find_algorithm_by_id(
                algorithm_id,
                self.database_connection.list_objects("algorithm-store"),
            )
            if found_algorithm_key is None:
                raise CompoxNotFoundError(
                    f"Algorithm with id {algorithm_id} not found.",
                    code="algorithm_not_found",
                    details={"algorithm_id": algorithm_id},
                )

            # Always resolve latest algorithm metadata before entering cache.
            algorithm_json = json.loads(
                self.database_connection.get_objects(
                    "algorithm-store",
                    [found_algorithm_key],
                )[0]
            )
            algorithm_minor_version = (
                algorithm_json["latest_algorithm_minor_version"]
                if algorithm_minor_version is None
                else algorithm_minor_version
            )
            runner, algorithm_assets, resolved_execution_device = (
                self.__cached_fetch_algorithm(
                    algorithm_json,
                    execution_device_override,
                    checkpoint_id,
                    algorithm_minor_version,
                )
            )
            self._set_resolved_execution_device(resolved_execution_device)
            # call the runners initialize method to reset the state of the runner
            runner.initialize()
            self.algorithm_assets = algorithm_assets

            self.logger.info(
                "Algorithm runner successfully loaded in {} seconds.".format(
                    round(time.time() - start, 8)
                )
            )
            algorithm_minor_version = (
                algorithm_json["latest_algorithm_minor_version"]
                if algorithm_minor_version is None
                else algorithm_minor_version
            )
            self.logger = logger.bind(
                algorithm=f"{algorithm_json['algorithm_name']} {algorithm_json['algorithm_major_version']}.{algorithm_minor_version}",
                log_type="TASK",
                task_id=self.task_id,
            )
            return runner
        except TaskStoppedException:
            raise
        except CompoxError as e:
            self.mark_as_failed(e)
            raise
        except Exception as e:
            error = CompoxAlgorithmError(
                "Failed to fetch algorithm",
                code="algorithm_fetch_failed",
                details={"algorithm_id": algorithm_id},
                cause=e,
            )
            self.mark_as_failed(error)
            raise error from e

    @algorithm_cache(maxsize=1, maxsize_attr="_ALGORITHM_CACHE_MAXSIZE")
    def __cached_fetch_algorithm(
        self,
        algorithm_json: dict,
        execution_device_override: str | None = None,
        checkpoint_id: str | None = None,
        algorithm_minor_version: str | None = None,
    ) -> tuple[object, dict, str]:
        """
        Fetches the algorithm from the database and imports its corresponding
        Python module and runner class. This method is cached to avoid
        unnecessary fetches from the database when repeated calls are made
        to the same algorithm.

        Parameters
        ----------
        algorithm_json : dict
            The algorithm json file as a dictionary. The algorithm json file
            contains the metadata of the algorithm (e.g. algorithm type, tags).

        execution_device_override : str | None, optional
            The requested abstract execution device class, by default None.
            This uses the algorithm metadata vocabulary (for example ``cpu``,
            ``gpu`` or ``mps``). Compox resolves this request to a concrete
            runtime device string passed into the runner, such as ``cpu``,
            ``cuda`` or ``mps``.

        checkpoint_id : str | None, optional
            The id of the checkpoint, by default None. If provided, the
            checkpoint will be used to load the model assets.

        algorithm_minor_version : str | None, optional
            The minor version of the algorithm, by default None. If provided, the
            minor version will be used to load the model assets.

        Returns
        -------
        object
            The algorithm Runner object.
        dict
            The assets of the algorithm, such as model weights, configuration files, etc.
        str
            The concrete runtime device resolved for the runner.

        Raises
        ------
        ValueError
            If fetch algorithm failed.
        """

        # get the algorithm module from the module-store bucket
        if not algorithm_minor_version:
            algorithm_minor_version = algorithm_json[
                "latest_algorithm_minor_version"
            ]

        # load the checkpoint if provided
        if checkpoint_id is not None:
            try:
                algorithm_checkpoint = AlgorithmCheckpoint(
                    checkpoint_id=checkpoint_id,
                    database_connection=self.database_connection,
                )
            except CompoxError:
                raise
            except Exception as e:
                error = CompoxAlgorithmError(
                    "Failed to load checkpoint",
                    code="checkpoint_load_failed",
                    details={"checkpoint_id": checkpoint_id},
                    cause=e,
                )
                self.mark_as_failed(error)
                raise error from e
            # override the algorithm assets with the ones from the checkpoint
            for (
                key,
                value,
            ) in algorithm_checkpoint.checkpoint_manifest.assets.items():
                try:
                    algorithm_json["algorithm_minor_version"][
                        algorithm_minor_version
                    ]["assets"][key] = value
                except KeyError as ke:
                    error = CompoxAlgorithmError(
                        f"Asset {key} from checkpoint not found in algorithm assets.",
                        code="checkpoint_asset_incompatible",
                        details={
                            "checkpoint_id": checkpoint_id,
                            "asset_path": key,
                        },
                        cause=ke,
                    )
                    self.logger.error(
                        f"{error.message} Make sure the checkpoint is compatible with the algorithm."
                    )
                    self.mark_as_failed(error)
                    raise error from ke
        module_id = algorithm_json["algorithm_minor_version"][
            algorithm_minor_version
        ]["module_id"]
        algorithm_assets = algorithm_json["algorithm_minor_version"][
            algorithm_minor_version
        ]["assets"]
        self.algorithm_assets = algorithm_assets
        device = self.__get_device(algorithm_json, execution_device_override)

        try:
            module_archive_bytes = io.BytesIO(
                self.database_connection.get_objects(
                    "module-store",
                    [module_id],
                )[0]
            )
            with ZipImporter(module_archive_bytes.getvalue(), module_id) as m:
                runner = m.Runner.__new__(m.Runner)
                runner.initialize(device=device)
                runner._load_assets()
        except CompoxError as e:
            self.mark_as_failed(e)
            raise
        except Exception as e:
            error = CompoxImportError(
                "Failed to import algorithm module",
                code="algorithm_module_import_failed",
                details={"module_id": module_id},
                cause=e,
            )
            self.mark_as_failed(error)
            raise error from e
        return runner, algorithm_assets, device

    def _set_resolved_execution_device(
        self, resolved_execution_device: str | None
    ) -> None:
        """
        Persist the concrete runtime device resolved for this execution.

        Parameters
        ----------
        resolved_execution_device : str | None
            The resolved runtime device string that is actually passed into the
            runner, for example ``cpu``, ``cuda`` or ``mps``.
        """
        task_record = self._get_task_record()
        task_record["resolved_execution_device"] = resolved_execution_device
        self._save_task_record(task_record)

    def __get_device(
        self, algorithm_json: dict, execution_device_override: str | None = None
    ) -> str:
        """
        Resolve the concrete runtime device to run the model and inference on.
        The algorithm metadata and per-run request use abstract values such as
        ``cpu``, ``gpu`` and ``mps``. This resolver maps those values to the
        concrete runtime string passed into the runner, such as ``cpu``,
        ``cuda`` or ``mps``.

        Parameters
        ----------
        algorithm_json : dict
            The algorithm json file as a dictionary. The algorithm json file
            contains the metadata of the algorithm (e.g. algorithm type, tags).

        execution_device_override : str | None, optional
            The requested abstract execution device class, by default None.
            If provided, Compox tries to honor that request and falls back only
            when the requested class is unsupported or unavailable.

        Returns
        -------
        device : str
            The concrete runtime device string used to initialize the runner.

        Raises
        ------
        ValueError
            If device is not supported.
        """
        assert algorithm_json["default_device"].lower() in [
            d.lower() for d in algorithm_json["supported_devices"]
        ], (
            f"Default device {algorithm_json['default_device']} is not supported. "
            f"Supported devices are {algorithm_json['supported_devices']}. Check the "
            "algorithm pyproject.toml file."
        )

        # set the device to run the model and inference on
        gpu_available, _ = check_system_gpu_availability()
        if not execution_device_override:
            if algorithm_json["default_device"].lower() == "cpu":
                self.logger.info(
                    "No execution device override requested. "
                    "Default device class 'cpu' resolved to runtime device 'cpu'."
                )
                device = "cpu"
            elif (
                algorithm_json["default_device"].lower() == "gpu"
                and gpu_available
            ):
                self.logger.info(
                    "No execution device override requested. "
                    "Default device class 'gpu' resolved to runtime device 'cuda'."
                )
                device = "cuda"
            elif (
                algorithm_json["default_device"].lower() == "gpu"
                and not gpu_available
            ):
                self.logger.warning(
                    "No execution device override requested. "
                    "Default device class 'gpu' could not be resolved because CUDA is not available. "
                    "Falling back to runtime device 'cpu'."
                )
                device = "cpu"
            elif (
                (algorithm_json["default_device"].lower() == "")
                and ("gpu" in algorithm_json["supported_devices"])
                and (gpu_available)
            ):
                self.logger.info(
                    "No execution device override requested. "
                    "No default device class specified, GPU is supported and CUDA is available. "
                    "Resolved runtime device 'cuda'."
                )
                device = "cuda"
            elif (
                (algorithm_json["default_device"].lower() == "")
                and ("gpu" in algorithm_json["supported_devices"])
                and (not gpu_available)
            ):
                self.logger.warning(
                    "No execution device override requested. "
                    "No default device class specified, GPU is supported but CUDA is not available. "
                    "Falling back to runtime device 'cpu'."
                )
                device = "cpu"
            elif algorithm_json["default_device"].lower() == "mps":
                mps_available = check_mps_availability()
                if not mps_available:
                    self.logger.warning(
                        "No execution device override requested. "
                        "Default device class 'mps' could not be resolved because MPS is not available. "
                        "Falling back to runtime device 'cpu'."
                    )
                    device = "cpu"
                else:
                    self.logger.info(
                        "No execution device override requested. "
                        "Default device class 'mps' resolved to runtime device 'mps'."
                    )
                    device = "mps"
            else:
                raise ValueError(
                    f"Default device {algorithm_json['default_device']} is not supported."
                )
        else:
            if (
                execution_device_override.lower()
                in algorithm_json["supported_devices"]
                and execution_device_override.lower() == "cpu"
            ):
                self.logger.info(
                    f"Execution device override requested '{execution_device_override}'. "
                    "Resolved runtime device 'cpu'."
                )
                device = "cpu"
            elif (
                execution_device_override.lower()
                in algorithm_json["supported_devices"]
                and execution_device_override.lower() == "gpu"
                and gpu_available
            ):
                self.logger.info(
                    f"Execution device override requested '{execution_device_override}'. "
                    "Resolved runtime device 'cuda'."
                )
                device = "cuda"
            elif (
                execution_device_override.lower()
                in algorithm_json["supported_devices"]
                and execution_device_override.lower() == "gpu"
                and not gpu_available
            ):
                self.logger.warning(
                    f"Execution device override requested '{execution_device_override}', "
                    "but CUDA is not available. Falling back to runtime device 'cpu'."
                )
                device = "cpu"
            else:
                self.logger.warning(
                    f"Execution device override '{execution_device_override}' is not supported. "
                    f"Falling back to the algorithm default device class '{algorithm_json['default_device']}'."
                )
                device = self.__get_device(algorithm_json)

        return device

    def fetch_asset(self, asset_path: str) -> io.BytesIO:
        """
        Fetches an asset as bytes from the database by its path relative to the
        algorithm Runner class.

        Parameters
        ----------
        asset_path : str
            The path to the asset relative to the algorithm Runner class. e.g.
            "files/weights.pth"

        Returns
        -------
        io.BytesIO
            The asset as bytes.

        Raises
        ------
        ValueError
            If fetch asset failed.
        """
        if self.algorithm_assets is None:
            raise CompoxStateError(
                "Algorithm assets are not initialized.",
                code="algorithm_assets_not_initialized",
            )

        try:
            asset_id = self.algorithm_assets[asset_path]
        except KeyError as e:
            raise CompoxAssetError(
                f"Asset path '{asset_path}' is not registered for the algorithm.",
                code="asset_path_not_found",
                details={"asset_path": asset_path},
                cause=e,
            ) from e
        self.logger.info(f"Fetching asset {asset_id} from the database.")
        try:
            start = time.time()
            asset = io.BytesIO(
                self.database_connection.get_objects(
                    "asset-store",
                    [asset_id],
                )[0]
            )
            end = time.time()
            # log the fetching time with 4 decimal places
            self.logger.info(
                f"Asset {asset_id} fetched in {round(end - start, 4)} seconds."
            )
            return asset
        except CompoxError as e:
            self.mark_as_failed(e)
            raise
        except Exception as e:
            error = CompoxAssetError(
                "Failed to fetch asset",
                code="asset_fetch_failed",
                details={"asset_path": asset_path, "asset_id": asset_id},
                cause=e,
            )
            self.mark_as_failed(error)
            raise error from e

    def fetch_data(
        self,
        file_ids: list[str],
        pydantic_data_schema: Type[DataSchema],
        *keys: str,
        parallel: bool = False,
    ) -> list[dict]:
        """
        Fetches the data from the database. A pydantic schema must be provided
        to validate the data. The data is fetches as a list of dictionaries, where
        each dictionary represents a dataset. Specific keys can be provided to
        fetch from the HDF5 file, if not provided, all keys will be fetched.

        Parameters
        ----------
        file_ids : list[str]
            The identifiers of the data files in the database.
        pydantic_data_schema : Type[DataSchema]
            The pydantic schema of the data. Must inherit from the DataSchema class.
        *keys : str
            Optional keys to fetch from the HDF5 file, if not provided, all keys
            will be fetched.
        parallel : bool, optional
            If True, the data will be fetched in parallel. Default is False.

        Returns
        -------
        list[dict]
            List of the datasets fetched from the database as dictionaries.

        Raises
        ------
        Exception
        """

        # self.logger.info("Fetching data from the database.")
        # get data object
        datasets = []

        def fetch_file(file_id):
            # convert to file-like object
            file_like_obj = io.BytesIO(
                self.database_connection.get_objects(
                    "data-store",
                    [file_id],
                )[0]
            )
            # read from file-like object
            data_dict = {}
            if len(keys) == 0:
                with h5py.File(file_like_obj, "r") as f:
                    for key in f.keys():
                        data_dict[key] = HDF5IO.read_dataset(f[key])
            else:
                with h5py.File(file_like_obj, "r") as f:
                    for key in keys:
                        try:
                            data_dict[key] = HDF5IO.read_dataset(f[key])
                        except KeyError:
                            data_dict[key] = None
            # validate and dump
            data_dict = pydantic_data_schema.model_validate(data_dict)
            data_dict = data_dict.model_dump()
            return data_dict

        try:
            start = time.time()

            if parallel:
                with ThreadPoolExecutor() as executor:
                    datasets = list(executor.map(fetch_file, file_ids))
            else:
                datasets = [fetch_file(file_id) for file_id in file_ids]
            end = time.time()
            self.file_fetching_stats["count"] += len(file_ids)
            self.file_fetching_stats["time"] += end - start
            return datasets

        except ValidationError as e:
            self.mark_as_failed(e)
            raise
        except KeyError as e:
            self.mark_as_failed(e)
            raise
        except CompoxError as e:
            self.mark_as_failed(e)
            raise
        except Exception as e:
            error = CompoxExecutionError(
                "Failed to fetch data",
                code="data_fetch_failed",
                details={"file_ids": file_ids},
                cause=e,
            )
            self.mark_as_failed(error)
            raise error from e

    def post_data(
        self,
        result: list[dict],
        pydantic_data_schema: Type[DataSchema],
        parallel: bool = False,
    ) -> list[str]:
        """
        Uploads a list of datasets to the database. The dataset is a dictionary
        where the keys are the names of the datasets and the values are the
        datasets themselves (e.g. numpy arrays). A pydantic schema must be provided
        to validate the data before uploading. The data is uploaded as HDF5 files.

        Parameters
        ----------
        result : list[dict]
            The result to upload to the database.
        pydantic_data_schema : Type[DataSchema]
            The pydantic schema of the data. Must inherit from the DataSchema class.
        parallel : bool, optional
            If True, the data will be uploaded in parallel. Default is False.

        Returns
        -------
        list[str]
            The dataset identifiers of the uploaded datasets.

        Raises
        ------
        Exception
        """
        # TODO: this is not working for all algorithms currently, must be fixed
        self.logger.info(
            f"Uploading {str(len(result))} results to the database."
        )

        def post_file(r):
            r = pydantic_data_schema.model_validate(r)
            r = r.model_dump()
            bio = io.BytesIO()
            with h5py.File(bio, "w") as f:
                for key in r.keys():
                    if r[key] is not None:
                        HDF5IO.write_dataset(f, key, r[key])
            # upload response to minio
            output_dataset_id = generate_uuid()
            self.database_connection.put_objects(
                "data-store",
                [output_dataset_id],
                [bio.getvalue()],
            )
            return output_dataset_id

        try:
            start = time.time()
            if parallel:
                with ThreadPoolExecutor() as executor:
                    output_dataset_ids = list(executor.map(post_file, result))
            else:
                output_dataset_ids = [post_file(file_id) for file_id in result]
            end = time.time()
            self.file_posting_stats["count"] += len(result)
            self.file_posting_stats["time"] += end - start

        except ValidationError as e:
            self.mark_as_failed(e)
            raise
        except CompoxError as e:
            self.mark_as_failed(e)
            raise
        except Exception as e:
            error = CompoxExecutionError(
                "Failed to post data",
                code="data_post_failed",
                cause=e,
            )
            self.mark_as_failed(error)
            raise error from e
        return output_dataset_ids

    def save_item_to_session(self, obj: Any, key: str) -> None:
        """
        Save an object to the task session.

        Parameters
        ----------
        obj : Any
            The object to save.
        key : str
            The key to save the object under.

        Returns
        -------
        None

        Raises
        ------
        Exception
        ValueError
            If task session is not initialized.
        """

        if self.task_session is None:
            raise CompoxStateError(
                "Task session is not initialized.",
                code="task_session_not_initialized",
            )

        try:
            self.task_session.add_item(obj, key)
            self.logger.info(
                f"Saved object with key {key} to the task session."
            )
        except CompoxError as e:
            self.mark_as_failed(e)
            raise
        except Exception as e:
            error = CompoxTaskError(
                "Failed to save item to task session",
                code="task_session_save_failed",
                details={"key": key},
                cause=e,
            )
            self.mark_as_failed(error)
            raise error from e

    def load_item_from_session(self, key: str) -> Any:
        """
        Load an object from the task session.

        Parameters
        ----------
        key : str
            The key of the object to load.

        Returns
        -------
        Any
            The object loaded from the task session.

        Raises
        ------
        Exception
            If the task session is not initialized.
        ValueError
            If task session is not initialized.
        """

        if self.task_session is None:
            raise CompoxStateError(
                "Task session is not initialized.",
                code="task_session_not_initialized",
            )

        try:
            obj = self.task_session[key]
            self.logger.info(
                f"Loaded object with key {key} from the task session."
            )
            return obj
        except KeyError as e:
            self.mark_as_failed(e)
            raise
        except CompoxError as e:
            self.mark_as_failed(e)
            raise
        except Exception as e:
            if self.task_session is None:
                self.logger.error(
                    "The algorithm is attempting to load an object from the task",
                    "session, but the task session is not initialized.",
                    "Please make sure you are providing the session token in the",
                    "execution request.",
                )
            error = CompoxTaskError(
                "Failed to load item from task session",
                code="task_session_load_failed",
                details={"key": key},
                cause=e,
            )
            self.mark_as_failed(error)
            raise error from e

    def remove_item_from_session(self, key: str) -> None:
        """
        Remove an object from the task session.

        Parameters
        ----------
        key : str
            The key of the object to remove.

        Returns
        -------
        None

        Raises
        ------
        Exception
        ValueError
            If task session is not initialized.
        """

        if self.task_session is None:
            raise CompoxStateError(
                "Task session is not initialized.",
                code="task_session_not_initialized",
            )

        try:
            self.task_session.remove_item(key)
            self.logger.info(
                f"Removed object with key {key} from the task session."
            )
        except KeyError as e:
            self.mark_as_failed(e)
            raise
        except CompoxError as e:
            self.mark_as_failed(e)
            raise
        except Exception as e:
            error = CompoxTaskError(
                "Failed to remove item from task session",
                code="task_session_remove_failed",
                details={"key": key},
                cause=e,
            )
            self.mark_as_failed(error)
            raise error from e
