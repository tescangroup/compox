"""
Copyright 2024 TESCAN 3DIM, s.r.o.
All rights reserved
"""

import time
import io
from typing import Type, Any, List
from abc import ABC, abstractmethod
from pathlib import Path

from compox.algorithm_utils.io_schemas import DataSchema
from compox.algorithm_utils.runner_context import current_runner_context
from compox.tasks import TaskHandler
from compox.tasks.context_handler import current_handler
from compox.training.TrainingHandler import TrainingHandler
from compox.training.TrainingDataset import TrainingDataset
from compox.exceptions import (
    CompoxCheckpointError,
    CompoxError,
    CompoxStateError,
)


class BaseRunner(ABC):
    """
    Base class for all runners. Specifies the architecture of a runner and the
    required methods.

    When implementing a new runner, the following methods need to be implemented:
    - preprocess: Preprocess the input data.
    - inference: Run the inference on the output of the preprocessing.
    - postprocess: Postprocess the output of the inference.
    """

    algorithm_type = "Generic"

    def __init__(self): ...

    def initialize(self, device: str | None = None) -> None:
        """
        Initialize the runner with the given device. This method is called by
        the TaskHandler when the algorithm is fetched. It is used to set the
        device on which the model and inference will be run.

        Parameters
        ----------
        device : str | None
            The device on which the model and inference will be run. e.g. "cpu", "cuda:0"
            or "cuda:1". This is set during the initialization of the runner.

        Returns
        -------
        None
        """

        if device:
            self._device = device
        current_runner_context.set({})
        self._initializing = True
        self.__init__()
        self._initializing = False

    @property
    def task_handler(self) -> TaskHandler.TaskHandler:
        """
        Get the current task handler. This is used to access the task handler
        methods and attributes.

        Returns
        -------
        TaskHandler.TaskHandler
            Current task handler.

        Raises
        ------
        ValueError
            If task handler is not set.
        """
        task_handler = current_handler.get(None)
        if task_handler is None:
            raise CompoxStateError(
                "Task handler is not set.",
                code="task_handler_not_set",
            )
        return task_handler

    @property
    def runner_context(self) -> dict:
        """
        Get the current runner context. This is used to access the runner context
        methods and attributes.

        Returns
        -------
        dict
            current runner context
        """
        runner_context = current_runner_context.get({})
        return runner_context

    @property
    def device(self) -> str:
        """
        Get the device on which the model and inference will be run.
        This is set during the initialization of the runner.

        Returns
        -------
        str
            The device that will be used to run the model and inference
        """
        return self._device

    def __setattr__(self, name: str, value: Any) -> None:
        """
        Set an attribute in the runner context.
        Parameters
        ----------
        name : str
            The name of the attribute to set.
        value : Any
            The value of the attribute to set.
        Returns
        -------
        None

        Raises
        ------
        AttributeError
            Attribute cannot be modified.
        """

        if name in {
            "_locked_attributes",
            "_locking_assets",
            "_device",
            "task_handler",
            "runner_context",
            "device",
            "_initializing",
        }:
            # Allow setting selected internal attributes directly
            super().__setattr__(name, value)
            return

        if getattr(self, "_locking_assets", False):
            # if the assets are being loaded, add their names to the locked attributes
            self._locked_attributes.add(name)
            super().__setattr__(name, value)
            return

        if hasattr(self, "_locked_attributes"):
            if name in self._locked_attributes and getattr(
                self, "_initializing", False
            ):
                # if the runner is being reinitialized, do not overwrite locked attributes
                # but also do not raise an error
                return
            elif name in self._locked_attributes and not getattr(
                self, "_initializing", False
            ):
                # if the attribute is locked, and the runner is not being reinitialized,
                # raise an error
                raise AttributeError(
                    f"Attribute '{name}' is locked and cannot be modified."
                )
        # set the attribute in the runner context
        self.runner_context[name] = value

    def __getattribute__(self, name: str) -> Any:
        """
        Get an attribute from the runner context.

        Parameters
        ----------
        name : str
            The name of the attribute to get.

        Returns
        -------
        Any
            The value of the attribute.
        """
        runner_context = super().__getattribute__("runner_context")

        if name in runner_context:
            return runner_context[name]
        else:
            return super().__getattribute__(name)

    def __delattr__(self, name):
        if name in self.runner_context:
            del self.runner_context[name]
        else:
            super().__delattr__(name)

    def load_assets(self):
        """
        This method should be overridden to load all necessary assets for the algorithm,
        such as trained models, precomputed data, or other resources.

        Assets must be loaded using `self.fetch_asset()` instead of accessing the file
        system directly. All assets should be stored as attributes on the runner instance.

        WARNING: The attributes set in this method will be protected against reassignment
        in other parts of the code, so they should not be modified after this method is called.
        However, this protection does not hold for mutating mutable types with in-place operations
        (e.g., appending to a list or modifying a dictionary). If you need to modify such attributes,
        consider using a different approach.
        """
        pass

    def _load_assets(self):
        """
        Internal wrapper for the load_assets method. This is supposed to be called
        from the TaskHandler class when fetching the algorithm assets. It is done
        like this because we need to store the attribute names as strings in order
        to prevent modifying the attributes which are shared between threads.

        E.g. the user loads some heave ML model as self.model and the Runner with model gets
        cached in the TaskHandler runner cache. If the developer attempts to modify
        self.model in the algorithm code, we want to raise an error, because
        the model is shared between threads and modifying it would lead to
        unpredictable behavior.
        """
        self._locked_attributes = set()
        self._locking_assets = True
        self.load_assets()
        self._locking_assets = False

    def run(self, input_data: dict, args: dict = None) -> None:
        """
        Run the algorithm.

        Parameters
        ----------
        input_data : dict
            The input data.

        args : dict
            Additional arguments.

        Returns
        -------
        None

        Raises
        ------
        Exception
            If an error occurs during the execution.

        """
        self.task_handler.logger.info("Starting execution.")
        start = time.time()
        if not args:
            args = {}
        try:
            out = self.postprocess_base(
                self.inference_base(
                    self.preprocess_base(input_data, args), args
                ),
                args,
            )
            self.task_handler.logger.info(
                "Execution completed in {} seconds.".format(
                    round(time.time() - start, 2)
                )
            )
            self.task_handler.mark_as_completed(out)
            return None

        except TaskHandler.TaskStoppedException as _:
            raise

        except Exception as e:
            self.task_handler.mark_as_failed(e)
            raise

    def preprocess_base(self, input_data: dict, args: dict = None) -> Any:
        """
        Preprocess the input data.

        Parameters
        ----------
        input_data : dict
            The input data.

        args : dict
            The additional arguments

        Returns
        -------
        Any
            The preprocessed input data.
        """
        if not args:
            args = {}
        start = time.time()
        # update status of the execution to running
        self.task_handler.status = "RUNNING"
        out = self.preprocess(input_data, args)
        end = time.time()
        self.task_handler.logger.info(
            "Data preprocessing finished in {} seconds".format(
                round(end - start, 2)
            )
        )
        self.task_handler.update_log()
        return out

    @abstractmethod
    def preprocess(self, input_data: dict, args: dict = None) -> Any:
        """
        Preprocess the input data.

        Parameters
        ----------
        input_data : dict
            The input data.

        args : dict
            Additional arguments.

        Returns
        -------
        Any
            The preprocessed input data.

        Raises
        ------
        NotImplementedError
        """
        raise NotImplementedError

    def inference_base(self, data: Any, args: dict = None) -> Any:
        """
        Run the inference.

        Parameters
        ----------
        data : Any
            The input data.
        args : dict
            Additional arguments.

        Returns
        -------
        Any
            The output data.
        """
        if not args:
            args = {}
        start = time.time()
        self.task_handler.logger.info("Running inference.")
        out = self.inference(data, args)
        end = time.time()
        self.task_handler.logger.info(
            "Inference finished in {} seconds".format(round(end - start, 2))
        )
        self.task_handler.update_log()

        return out

    @abstractmethod
    def inference(self, data: Any, args: dict = None) -> Any:
        """
        Run the inference.

        Parameters
        ----------
        data : Any
            The input data.
        args : dict
            Additional arguments.

        Returns
        -------
        Any
            The output data.

        Raises
        ------
        NotImplementedError
        """
        raise NotImplementedError

    def postprocess_base(self, data: Any, args: dict = None) -> list[str]:
        """
        Postprocess the output data.

        Parameters
        ----------
        data : Any
            The input data.
        args : dict
            Additional arguments.

        Returns
        -------
        list[str]
            The ids of the output datasets.
        """
        if not args:
            args = {}
        start = time.time()
        self.task_handler.logger.info("Postprocessing output data.")
        output_dataset_ids = self.postprocess(data, args)
        end = time.time()
        self.task_handler.logger.info(
            "Postprocessing finished in {} seconds".format(
                round(end - start, 2)
            )
        )
        self.task_handler.update_log()
        return output_dataset_ids

    @abstractmethod
    def postprocess(self, data: Any, args: dict = None) -> list[str]:
        """
        Postprocess the output data.

        Parameters
        ----------
        data : Any
            The input data.
        args : dict
            Additional arguments.

        Returns
        -------
        list[str]
            The ids of the output datasets.

        Raises
        ------
        NotImplementedError
        """
        raise NotImplementedError

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
        fetch from the HDF5 file, if not provided, all keys will be fetched. This
        method is wrapper around the fetch_data method of the TaskHandler class.

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
        """
        return self.task_handler.fetch_data(
            file_ids, pydantic_data_schema, *keys, parallel=parallel
        )

    def save_item_to_session(self, obj: Any, key: str) -> None:
        """
        Save an item to the session cache.

        Parameters
        ----------
        obj : Any
            The item to save.
        key : str
            The key to save the item.

        Returns
        -------
        None
        """
        self.task_handler.save_item_to_session(obj, key)
        return None

    def load_item_from_session(self, key: str) -> Any:
        """
        Fetch an item from the session cache.

        Parameters
        ----------
        key : str
            The key to fetch the item.

        Returns
        -------
        Any
            The item fetched from the session cache.
        """
        return self.task_handler.load_item_from_session(key)

    def remove_item_from_session(self, key: str) -> None:
        """
        Remove an item from the session cache.

        Parameters
        ----------
        key : str
            The key to remove the item.

        Returns
        -------
        None
        """
        self.task_handler.remove_item_from_session(key)
        return None

    def post_data(
        self,
        data: list[dict],
        pydantic_data_schema: Type[DataSchema],
        parallel: bool = False,
    ) -> list[str]:
        """
        Uploads a list of datasets to the database. The dataset is a dictionary
        where the keys are the names of the datasets and the values are the
        datasets themselves (e.g. numpy arrays). A pydantic schema must be provided
        to validate the data before uploading. The data is uploaded as HDF5 files.
        This method is wrapper around the post_data method of the TaskHandler class.

        Parameters
        ----------
        data : list[dict]
            List of the datasets to upload. Each dataset is a defined as a dictionary.
        pydantic_data_schema : Type[DataSchema]
            The pydantic schema of the data. Must inherit from the DataSchema class.
        parallel : bool, optional
            If True, the data will be uploaded in parallel. Default is False.

        Returns
        -------
        list[str]
            List of the identifiers of the uploaded datasets.
        """
        return self.task_handler.post_data(data, pydantic_data_schema, parallel)

    def fetch_asset(self, asset_path: str) -> io.BytesIO:
        """
        Fetches an asset as bytes from the database by its path relative to the
        algorithm Runner class.

        Parameters
        ----------
        asset_path : str
            TThe path to the asset relative to the algorithm Runner class. e.g.
            "files/weights.pth"

        Returns
        -------
        io.BytesIO
            The asset as bytes.
        """
        return self.task_handler.fetch_asset(asset_path)

    def set_progress(self, progress: float) -> None:
        """
        Set the progress of the execution. The progress must be a float between
        0 and 1.

        Parameters
        ----------
        progress : float
            The progress of the execution.

        Raises
        ------
        ValueError
            If progress is not between 0 a 1 or float
        """
        # check if the progress is a float between 0 and 1
        if not isinstance(progress, float):
            raise ValueError("Progress must be a float.")
        if progress < 0 or progress > 1:
            raise ValueError("Progress must be between 0 and 1.")
        self.task_handler.progress = progress
        return None

    def log_message(self, message: str, logging_level: str = "INFO") -> None:
        """
        Log a message.

        Parameters
        ----------
        message : str
            The message to log.
        logging_level : str
            The logging level as defined in the logging module. Default is "INFO".

        Returns
        -------
        None

        Raises
        ------
        ValueError
            If an invalid logging level is provided.
        """
        if logging_level == "INFO":
            self.task_handler.logger.info(message)
        elif logging_level == "WARNING":
            self.task_handler.logger.warning(message)
        elif logging_level == "ERROR":
            self.task_handler.logger.error(message)
        elif logging_level == "DEBUG":
            self.task_handler.logger.debug(message)
        else:
            raise ValueError("Invalid logging level provided.")
        return None

    def run_training(
        self, training_data: list[str], args: dict = None
    ) -> tuple[str, str, str]:
        """
        Train the algorithm.

        Parameters
        ----------
        training_data : list[str]
            The training samples ids.
        args : dict
            Additional arguments for training.
        Returns
        -------
        tuple[str, str, str]
            The trained algorithm id, name and major version.
        """
        self._require_training_handler(
            "Training can only be run with a TrainingHandler."
        )
        if not args:
            args = {}
        self.task_handler.logger.info("Starting training.")
        self.task_handler.status = "RUNNING"
        start = time.time()
        try:
            self.train(training_data, args)
            self.task_handler.logger.info(
                "Training completed in {} seconds.".format(
                    round(time.time() - start, 2)
                )
            )
            if len(self.task_handler.output_checkpoint_ids) == 0:
                self.task_handler.mark_as_failed(
                    "Training failed: At least one output checkpoint must be created before marking the training as completed."
                )
            self.task_handler.mark_as_completed()
            return None

        except TaskHandler.TaskStoppedException:
            raise

        except Exception as e:
            self.task_handler.mark_as_failed(e)
            raise

    def train(self, training_data: list[str], args: dict = None) -> None:
        """
        Train the algorithm.

        Parameters
        ----------
        training_data : list[str]
            The training samples ids.
        args : dict
            Additional arguments for training.

        Returns
        -------
        None

        Raises
        ------
        NotImplementedError
        """
        self.task_handler.logger.error(
            "Training is not implemented for this algorithm. A train() method"
            " must be implemented in the Runner subclass."
        )
        raise NotImplementedError

    def save_checkpoint(
        self, checkpoint: dict[str, bytes], properties: dict = {}
    ) -> str:
        """
        Save a training checkpoint to the database.

        Parameters
        ----------
        checkpoint : dict[str, bytes]
            The dictionary containing the checkpoint files. The keys are the file names
            which must correspond to the asset keys used to load the assets in the load_assets()
            method. Values are the file contents as bytes e.g. Pytorch model weight converted
            to bytes using io.BytesIO().

        properties : dict, optional
            Additional properties to associate with the checkpoint. Default is an empty dictionary.

        Returns
        -------
        str
            The identifier of the saved checkpoint.

        Raises
        ------
        ValueError
            If saving the checkpoint fails.
        """
        self._require_training_handler(
            "Checkpoints can only be saved with a TrainingHandler."
        )
        try:
            checkpoint_id = self.task_handler.save_checkpoint(
                checkpoint, properties
            )
            return checkpoint_id

        except TaskHandler.TaskStoppedException as _:
            raise
        except CompoxError:
            raise
        except Exception as e:
            error = CompoxCheckpointError(
                "Failed to save checkpoint",
                code="runner_checkpoint_save_failed",
                cause=e,
            )
            self.task_handler.mark_as_failed(error)
            raise error from e

    def set_state(self, state: dict) -> None:
        """
        Set the state of the runner. The state is a dictionary that can be used
        to store any information that might be useful for the client, such as
        intermediate metrics, loss values, etc.

        WARNING: The state is always overwritten, not merged, so it is up to the
        developer to either fetch the current state using get_state() or keep
        track of the state in the algorithm code.

        Parameters
        ----------
        state : dict
            The state of the runner.

        Returns
        -------
        None
        """
        self.task_handler.state = state
        return None

    def get_state(self) -> dict:
        """
        Get the current state of the runner. The state is a dictionary that can be used
        to store any information that might be useful for the client, such as
        intermediate metrics, loss values, etc.

        Returns
        -------
        dict
            The current state of the runner.
        """
        return self.task_handler.state

    def get_training_dataset(
        self, training_sample_ids: list[str]
    ) -> TrainingDataset:
        """
        Retrieves the training dataset record from the database.

        Parameters
        ----------
        training_sample_ids : list[str]
            The training sample ids.

        Returns
        -------
        TrainingDataset
            The training dataset record.

        Raises
        ------
        ValueError
            If training dataset could not be fetched.
        """

        self._require_training_handler(
            "Training datasets can only be fetched with a TrainingHandler."
        )
        dataset = self.task_handler.get_training_dataset(training_sample_ids)
        self.task_handler.logger.info(
            f"Fetched training dataset with {len(dataset)} samples."
        )
        return dataset

    def save_training_files_to_temp_store(
        self,
        folder_path: str | Path,
        files: List[dict],
        pydantic_data_schema: Type[DataSchema],
        parallel: bool = True,
    ) -> list[Path]:
        """
        Saves training files represented by a list of dictionaries to a specific folder
        in a temporary storage created specifically for training purposes. You must provide
        a pydantic schema to validate the data before saving. This method should be used
        when some data (mainly numpy arrays) are loaded in the memory after some preprocessing
        and need to be saved to the temporary storage so that they can be accessed during
        training.

        Parameters
        ----------
        folder_path : str | Path
            The path to the folder in the temporary storage where the files will be saved.
        files : List[dict]
            The list of files to save. Each file is represented as a dictionary.
        pydantic_data_schema : Type[DataSchema]
            The pydantic schema of the data. Must inherit from the DataSchema class.
        parallel : bool, optional
            If True, the files will be saved in parallel. Default is True.
        Returns
        -------
        list[Path]
            List of the paths to the saved files.
        """
        self._require_training_handler(
            "Training files can only be saved with a TrainingHandler."
        )
        return self.task_handler.save_training_files_to_temp_store(
            folder_path, files, pydantic_data_schema, parallel
        )

    def download_files_to_temp_store(
        self,
        folder_path: str | Path,
        file_ids: List[str],
        pydantic_data_schema: Type[DataSchema],
        batch_size: int = 8,
        *keys: str,
    ):
        """
        Downloads files from the database to a specific folder in a temporary storage
        created specifically for training purposes. This method works directly with the
        file identifiers in the database, which means that the files do not need to be loaded
        to the memory, but are downloaded directly to the temporary storage.
        You must provide a pydantic schema to validate the data before saving.

        Parameters
        ----------
        folder_path : str | Path
            The path to the folder in the temporary storage where the files will be saved.
        file_ids : List[str]
            The list of file identifiers in the database.
        pydantic_data_schema : Type[DataSchema]
            The pydantic schema of the data. Must inherit from the DataSchema class.
        batch_size : int, optional
            The number of files to download in a single batch. Default is 8.
        *keys : str
            Optional keys to filter the files to download.
        """
        self._require_training_handler(
            "Files can only be downloaded with a TrainingHandler."
        )
        return self.task_handler.download_files_to_temp_store(
            folder_path, file_ids, pydantic_data_schema, batch_size, *keys
        )

    def download_dataset_to_temp_store(
        self,
        dataset: TrainingDataset,
        pydantic_data_schemas: dict[str, Type[DataSchema]],
    ) -> list[list[dict]]:
        """
        Downloads the entire training dataset to the temporary store while
        preserving the directory structure logically represented in the
        sample manifests.

        A subdirectory named after each sample's ID will be created
        within the specified folder path in the temporary store. Then subdirectories
        for each sample key will be created within the sample ID directory.
        Finally, the files associated with each sample key will be saved in
        their respective subdirectories.

        Parameters
        ----------
        dataset : TrainingDataset
            The training dataset to be downloaded.
        pydantic_data_schemas : dict[str, Type[DataSchema]]
            A dictionary mapping sample keys to their corresponding Pydantic
            data schema classes for validating the data.

            e.g. {"input": InputDataSchema, "label": LabelDataSchema}
        -------
        list[list[dict]]
            A list of lists of dictionaries with the individual samples represented
            as dictionaries following the structure of the sample manifests, but
            with local paths in the temporary store instead of file IDs.
        """
        self._require_training_handler(
            "Datasets can only be downloaded with a TrainingHandler."
        )
        return self.task_handler.download_dataset_to_temp_store(
            dataset, pydantic_data_schemas
        )

    def load_dataset_from_temp_store(
        self,
        local_samples: list[list[dict]],
    ) -> list[list[dict]]:
        """
        Loads the entire training dataset from the temporary store while
        preserving the directory structure logically represented in the
        sample manifests.

        Parameters
        ----------
        local_samples : list[list[dict]]
            A list of lists of dictionaries with the individual samples represented
            as dictionaries following the structure of the sample manifests, but
            with local paths in the temporary store instead of file IDs.
        -------
        list[list[dict]]
            A list of lists of dictionaries with the individual samples represented
            as dictionaries following the structure of the sample manifests, but
            with loaded data dictionaries instead of file IDs.
        """
        self._require_training_handler(
            "Datasets can only be loaded with a TrainingHandler."
        )
        return self.task_handler.load_dataset_from_temp_store(local_samples)

    def load_files_from_temp_store(
        self,
        paths: List[str | Path],
        parallel: bool = True,
        *keys: str,
    ) -> List[dict]:
        """
        Loads files from the temporary store.

        Parameters
        ----------
        paths : List[str | Path]
            The list of file paths in the temporary store to be loaded.
        parallel : bool, optional
            Whether to load the files in parallel, by default True.
        *keys : str
            The keys to extract from the loaded data dictionaries. If no keys are
            provided, the entire data dictionary will be returned.

        Returns
        -------
        list[dict]
            The list of loaded data dictionaries.
        """
        self._require_training_handler(
            "Files can only be loaded with a TrainingHandler."
        )
        return self.task_handler.load_files_from_temp_store(
            paths, parallel, *keys
        )

    def _require_training_handler(self, message: str) -> None:
        """
        Ensure the current task handler supports training-only operations.
        """
        if not isinstance(self.task_handler, TrainingHandler):
            raise CompoxStateError(
                f"The task handler is not a TrainingHandler. {message}",
                code="training_handler_required",
            )
        

    def run_benchmark(self, args: dict = None) -> None:
        """
        Run the benchmark pipeline for the algorithm.

        This method drives the full benchmark lifecycle: it calls
        ``benchmark()`` (which subclasses override to generate dummy data and
        run inference), records timing, persists the returned result dict via
        the task handler's ``mark_as_completed`` path, and handles failures
        consistently with ``run()``.

        Parameters
        ----------
        args : dict, optional
            Additional parameters forwarded to ``benchmark()``.

        Returns
        -------
        None
        """
        self.task_handler.logger.info("Starting benchmark.")
        start = time.time()
        args = args or {}
        try:
            results = self.benchmark(args) or {}
            self.task_handler.logger.info(
                "Benchmark completed in {} seconds.".format(
                    round(time.time() - start, 2)
                )
            )
            self.task_handler.mark_as_completed(results)
            return None

        except TaskHandler.TaskStoppedException:
            raise

        except Exception as e:
            self.task_handler.mark_as_failed(e)
            raise

    def benchmark(self, args: dict = None) -> dict:
        """
        Run the benchmark for this algorithm.

        Override this method in a subclass to implement algorithm-specific
        benchmarking.  The implementation should:
        1. Generate dummy / synthetic input data without relying on uploaded
           datasets.
        2. Run the full inference pipeline (preprocess -> inference ->
           postprocess or an equivalent subset).
        3. Collect and return timing and/or quality metrics as a plain dict.

        The returned dict is stored in the ``data`` field of the benchmark
        record and is available to callers via
        ``GET /api/v0/benchmarks/{benchmark_id}``.

        Parameters
        ----------
        args : dict, optional
            Additional parameters that were forwarded from the benchmark
            request.

        Returns
        -------
        dict
            A plain dictionary of benchmark results (e.g. timing in seconds,
            throughput, accuracy metrics).  Return an empty dict if there are
            no results to report.

        Raises
        ------
        NotImplementedError
            If the subclass has not implemented this method.
        """
        self.task_handler.logger.error(
            "Benchmarking is not implemented for this algorithm. A benchmark()"
            " method must be implemented in the Runner subclass."
        )
        raise NotImplementedError
