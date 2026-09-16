"""
Copyright 2025 TESCAN 3DIM, s.r.o.
All rights reserved
"""

import json
import os
from datetime import datetime, timezone
from loguru import logger
from pathlib import Path
from typing import Type, List

from compox.server_utils import generate_uuid, find_algorithm_by_id
from compox.tasks.TaskHandler import TaskHandler
from compox.session.TaskSession import TaskSession
from compox.training.TrainingDataset import TrainingDataset
from compox.training.TrainingSample import TrainingSample
from compox.training.TempStore import TempStore
from compox.training.AlgorithmCheckpoint import AlgorithmCheckpoint
from compox.algorithm_utils.io_schemas import DataSchema
from compox.internal.EmergencyRecordStore import EmergencyRecordStore
from compox.exceptions import (
    CompoxAssetError,
    CompoxCheckpointError,
    CompoxError,
    CompoxNotFoundError,
    CompoxStateError,
    CompoxTrainingError,
)


class TrainingHandler(TaskHandler):
    """
    Training handler class for the training task. This class is used to update
    the progress, status and log of the training task.

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

    _RECORD_STORAGE_NAME = "training-store"

    def __init__(
        self,
        training_id: str,
        database_connection,
        database_update: bool = True,
        task_session: TaskSession | None = None,
        temp_store: TempStore | None = None,
        emergency_record_store: EmergencyRecordStore | None = None,
    ):
        super().__init__(
            training_id,
            database_connection,
            database_update,
            task_session,
            emergency_record_store=emergency_record_store,
        )
        self.temp_store = temp_store
        self.output_checkpoint_ids = []

    @property
    def state(self):
        """
        The state of the training task.

        :getter: Returns the state.
        :setter: Sets the state.
        :type: dict
        """
        return self._state

    @state.setter
    def state(self, state: dict) -> None:
        """
        Update the state of the training task.

        Parameters
        ----------
        state : dict
            The state of the training task.

        Returns
        -------
        None

        Raises
        ------
        Exception
        """
        self._state = state
        self._update_task_record_field("state", state)

    def _post_assets(
        self, assets: dict[str, bytes], asset_ids: list[str]
    ) -> None:
        """
        Uploads multiple assets as bytes to the database.

        Parameters
        ----------
        assets : dict[str, bytes]
            The dictionary of assets as bytes, where the key is the asset path
            and the value is the asset content.
        asset_ids : list[str]
            The list of asset identifiers.

        Raises
        ------
        ValueError
            If upload asset failed.
        """

        assets = list(assets.values())

        if len(assets) != len(asset_ids):
            raise ValueError(
                "The number of assets must be the same as the number of asset ids."
            )

        try:
            self.database_connection.put_objects(
                "asset-store",
                asset_ids,
                assets,
            )
            # log the posting time with 4 decimal places
            self.logger.info(f"Uploaded {len(assets)} assets to the database.")
        except CompoxError as e:
            self.mark_as_failed(e)
            raise
        except Exception as e:
            error = CompoxAssetError(
                "Failed to upload assets",
                code="asset_upload_failed",
                cause=e,
            )
            self.mark_as_failed(error)
            raise error from e

    def _remove_assets(self, asset_ids: list[str]) -> None:
        """
        Removes multiple assets from the database.

        Parameters
        ----------
        asset_ids : list[str]
            The list of asset identifiers.

        Raises
        ------
        ValueError
            If removal of asset failed.
        """

        try:
            self.logger.info("Removing assets from the database.")
            self.database_connection.delete_objects(
                "asset-store",
                asset_ids,
            )
            # log the removal time with 4 decimal places
            self.logger.info(
                f"Removed {len(asset_ids)} assets from the database."
            )
        except CompoxError as e:
            self.mark_as_failed(e)
            raise
        except Exception as e:
            error = CompoxAssetError(
                "Failed to remove assets",
                code="asset_remove_failed",
                details={"asset_ids": asset_ids},
                cause=e,
            )
            self.mark_as_failed(error)
            raise error from e

    def get_training_dataset(
        self, training_sample_ids: list[str]
    ) -> TrainingDataset:
        """
        Retrieves the training dataset record from the database.

        Parameters
        ----------
        training_sample_ids : list[str]
            The list of training sample identifiers.

        Returns
        -------
        TrainingDataset
            The training dataset record.

        Raises
        ------
        ValueError
            If retrieval of training dataset failed.
        """
        try:
            if not training_sample_ids:
                raise CompoxNotFoundError(
                    "Training sample ID not found in training record.",
                    code="training_samples_not_found",
                )
            training_samples = [
                TrainingSample(self.database_connection, sample_id=sample_id)
                for sample_id in training_sample_ids
            ]

            combined_dataset = TrainingDataset(samples=training_samples)
            return combined_dataset
        except CompoxError as e:
            self.mark_as_failed(e)
            raise
        except Exception as e:
            error = CompoxTrainingError(
                "Failed to retrieve training dataset",
                code="training_dataset_retrieval_failed",
                details={"training_sample_ids": training_sample_ids},
                cause=e,
            )
            self.mark_as_failed(error)
            raise error from e

    def save_training_files_to_temp_store(
        self,
        folder_path: str | Path,
        files: List[dict],
        pydantic_data_schema: Type[DataSchema],
        parallel: bool = True,
    ) -> list[Path]:
        """
        Saves files to the temporary store. This method is useful if you
        need to pre-load the files into memory so that they can be processed
        before being saved to the temporary store.

        Parameters
        ----------
        folder_path : str | Path
            The folder path in the temporary store where the files will be saved.
        files : list[bytes]
            The list of files as bytes.
        pydantic_data_schema : Type[DataSchema]
            The Pydantic data schema class for validating the data.
        parallel : bool, optional
            Whether to save the files in parallel, by default True.

        Returns
        -------
        list[Path]
            The list of paths to the saved files.

        Raises
        ------
        ValueError
            If saving files to temporary store failed.
        """
        if self.temp_store is None:
            raise CompoxStateError(
                "TempStore is not initialized.",
                code="temp_store_not_initialized",
            )

        try:
            self.logger.info("Saving files to temporary store.")
            self.temp_store.mkdir(folder_path)
            paths = self.temp_store.save(
                folder_path,
                files,
                pydantic_data_schema,
                parallel=parallel,
            )
            self.logger.info("Saved files to temporary store.")
            return paths
        except CompoxError as e:
            self.mark_as_failed(e)
            raise
        except Exception as e:
            error = CompoxTrainingError(
                "Failed to save files to temporary store",
                code="temp_store_save_failed",
                cause=e,
            )
            self.mark_as_failed(error)
            raise error from e

    def download_files_to_temp_store(
        self,
        folder_path: str | Path,
        file_ids: List[str],
        pydantic_data_schema: Type[DataSchema],
        parallel: bool = True,
        batch_size: int = 8,
        *keys: str,
    ) -> list[Path]:
        """
        Downloads files from the database and saves them to the temporary store.

        Parameters
        ----------
        folder_path : str | Path
            The folder path in the temporary store where the files will be saved.
        file_ids : List[str]
            The list of file identifiers to be downloaded.
        pydantic_data_schema : Type[DataSchema]
            The Pydantic data schema class for validating the data.
        parallel : bool, optional
            Whether to download and save the files in parallel, by default True.
        batch_size : int, optional
            The number of files to download in each batch, by default 8.
        *keys : str
            The keys to extract from the loaded data dictionaries. If no keys are
            provided, the entire data dictionary will be returned.

        Returns
        -------
        list[Path]
            The list of paths to the saved files.

        Raises
        ------
        ValueError
            If downloading files to temporary store failed.
        """
        if self.temp_store is None:
            raise CompoxStateError(
                "TempStore is not initialized.",
                code="temp_store_not_initialized",
            )

        try:
            self.temp_store.mkdir(folder_path)
            paths = []
            for i in range(0, len(file_ids), batch_size):
                batch_file_ids = file_ids[i : i + batch_size]
                files = self.fetch_data(
                    batch_file_ids, pydantic_data_schema, *keys, parallel=True
                )
                batch_paths = self.temp_store.save(
                    folder_path,
                    files,
                    pydantic_data_schema,
                    parallel=parallel,
                )
                paths.extend(batch_paths)
            return paths
        except CompoxError as e:
            self.mark_as_failed(e)
            raise
        except Exception as e:
            error = CompoxTrainingError(
                "Failed to download files to temporary store",
                code="temp_store_download_failed",
                details={"file_ids": file_ids},
                cause=e,
            )
            self.mark_as_failed(error)
            raise error from e

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
        if self.temp_store is None:
            raise CompoxStateError(
                "TempStore is not initialized.",
                code="temp_store_not_initialized",
            )

        local_samples = []
        try:
            for sample in dataset:
                sample_id = sample.sample_id
                self.temp_store.mkdir(sample_id)
                local_samples.append([])
                for i, file in enumerate(sample.sample_manifest.files):
                    file_name = f"file_{i:06d}"
                    self.temp_store.mkdir(os.path.join(sample_id, file_name))
                    local_samples[-1].append({})
                    for key, file_ids in file.items():
                        if key not in pydantic_data_schemas:
                            raise CompoxTrainingError(
                                f"Pydantic data schema for key '{key}' not provided.",
                                code="training_data_schema_missing",
                                details={"sample_key": key},
                            )
                        self.temp_store.mkdir(
                            os.path.join(sample_id, file_name, key)
                        )
                        if not file_ids:
                            local_samples[-1][-1][key] = []
                            continue
                        paths = self.download_files_to_temp_store(
                            os.path.join(sample_id, file_name, key),
                            file_ids,
                            pydantic_data_schemas[key],
                        )
                        local_samples[-1][-1][key] = paths

            return local_samples
        except CompoxError as e:
            self.mark_as_failed(e)
            raise
        except Exception as e:
            error = CompoxTrainingError(
                "Failed to download training dataset to temporary store",
                code="training_dataset_download_failed",
                cause=e,
            )
            self.mark_as_failed(error)
            raise error from e

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
        local_samples : list[dict]
            List of dictionaries with the individual samples represented as dictionaries
            following the structure of the sample manifests, but with local paths
            in the temporary store instead of file IDs.
        -------
        list[dict]
            List of dictionaries with the individual samples represented as dictionaries
            following the structure of the sample manifests, but with loaded data
            dictionaries instead of file IDs.
        """
        if self.temp_store is None:
            raise CompoxStateError(
                "TempStore is not initialized.",
                code="temp_store_not_initialized",
            )

        data_list = []
        try:
            for sample in local_samples:
                data_list.append([])
                for file in sample:
                    data_list[-1].append({})
                    for key, paths in file.items():
                        if not paths:
                            data_list[-1][-1][key] = []
                            continue
                        data_dicts = self.load_files_from_temp_store(paths)
                        data_list[-1][-1][key] = data_dicts
            return data_list
        except CompoxError as e:
            self.mark_as_failed(e)
            raise
        except Exception as e:
            error = CompoxTrainingError(
                "Failed to load training dataset from temporary store",
                code="training_dataset_load_failed",
                cause=e,
            )
            self.mark_as_failed(error)
            raise error from e

    def load_files_from_temp_store(
        self,
        paths: List[str | Path],
        parallel: bool = True,
        *keys: str,
    ) -> list[dict]:
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

        Raises
        ------
        ValueError
            If loading files from temporary store failed.
        """
        if self.temp_store is None:
            raise CompoxStateError(
                "TempStore is not initialized.",
                code="temp_store_not_initialized",
            )

        try:
            data_dicts = self.temp_store.load(
                paths,
                parallel=parallel,
                *keys,
            )
            return data_dicts
        except CompoxError as e:
            self.mark_as_failed(e)
            raise
        except Exception as e:
            error = CompoxTrainingError(
                "Failed to load files from temporary store",
                code="temp_store_load_failed",
                cause=e,
            )
            self.mark_as_failed(error)
            raise error from e

    def save_checkpoint(
        self, assets: dict[str, bytes], properties: dict
    ) -> str:
        """
        Saves a new algorithm checkpoint to the database.

        Parameters
        ----------
        assets : dict[str, bytes]
            The dictionary of assets as bytes. The keys are the asset paths
            defined in the algorithm and the values are the asset bytes.
        properties : dict
            A dictionary of arbitrary properties associated with the checkpoint.
        Returns
        -------
        str
            The checkpoint id.
        """
        try:
            training_record = self._get_task_record()
            algorithm_id = training_record["algorithm_id"]
            training_run_tags = training_record.get("tags", [])
            parent_checkpoint_id = training_record.get("checkpoint_id", None)
            found_algorithm_key, _, _, _, _ = find_algorithm_by_id(
                algorithm_id,
                self.database_connection.list_objects("algorithm-store"),
            )
            if found_algorithm_key is None:
                raise ValueError(
                    f"Parent algorithm {algorithm_id} not found in algorithm-store."
                )

            algorithm_json = json.loads(
                self.database_connection.get_objects(
                    "algorithm-store",
                    [found_algorithm_key],
                )[0]
            )
            latest_minor_version = algorithm_json[
                "latest_algorithm_minor_version"
            ]
            algorithm_assets = algorithm_json["algorithm_minor_version"][
                latest_minor_version
            ].get("assets", {})

            for asset_path in assets.keys():
                if asset_path not in algorithm_assets.keys():
                    raise CompoxCheckpointError(
                        f"Asset path {asset_path} does not exist in the algorithm assets.",
                        code="checkpoint_asset_path_invalid",
                        details={
                            "asset_path": asset_path,
                            "existing_asset_paths": list(
                                algorithm_assets.keys()
                            ),
                        },
                    )
            asset_ids = [generate_uuid() for _ in assets]
            self._post_assets(assets, asset_ids)

            new_algorithm_assets = {
                asset_path: asset_id
                for asset_path, asset_id in zip(assets.keys(), asset_ids)
            }
            checkpoint_id = generate_uuid()
            algorithm_checkpoint = AlgorithmCheckpoint(
                self.database_connection,
                checkpoint_manifest={
                    "checkpoint_id": checkpoint_id,
                    "parent_algorithm_id": algorithm_id,
                    "parent_algorithm_key": found_algorithm_key,
                    "training_id": self.task_id,
                    "assets": new_algorithm_assets,
                    "created_at": datetime.now(timezone.utc).isoformat(
                        timespec="seconds"
                    ),
                    "properties": properties,
                    "tags": training_run_tags,
                    "parent_checkpoint_id": parent_checkpoint_id,
                },
            )
            algorithm_checkpoint.register_checkpoint()
            self.output_checkpoint_ids.append(checkpoint_id)
            return checkpoint_id
        except CompoxError as e:
            self.mark_as_failed(e)
            raise
        except Exception as e:
            error = CompoxCheckpointError(
                "Failed to save checkpoint",
                code="checkpoint_save_failed",
                cause=e,
            )
            self.mark_as_failed(error)
            raise error from e

    def delete_checkpoint(self, checkpoint_id: str) -> None:
        """
        Deletes an existing algorithm checkpoint from the database.

        Parameters
        ----------
        checkpoint_id : str
            The identifier of the checkpoint to be deleted.

        Returns
        -------
        None
        """
        try:
            algorithm_checkpoint = AlgorithmCheckpoint(
                self.database_connection, checkpoint_id=checkpoint_id
            )
            algorithm_checkpoint.delete_checkpoint()
            if checkpoint_id in self.output_checkpoint_ids:
                self.output_checkpoint_ids.remove(checkpoint_id)
            else:
                self.logger.warning(
                    f"Attempted to remove checkpoint id {checkpoint_id} from output_checkpoint_ids, but it was not found in the list."
                )
        except CompoxError as e:
            self.mark_as_failed(e)
            raise
        except Exception as e:
            error = CompoxCheckpointError(
                "Failed to delete checkpoint",
                code="checkpoint_delete_failed",
                details={"checkpoint_id": checkpoint_id},
                cause=e,
            )
            self.mark_as_failed(error)
            raise error from e

    def mark_as_completed(
        self,
    ) -> None:
        """
        Mark the training task as completed and update its record in the database. This
        will set the progress to 1.0, the status to "COMPLETED" and the time
        completed to the current time.

        Parameters
        ----------
            output_checkpoint_id : str

        Returns
        -------
            None
        """
        self.progress = 1.0
        self.time_completed = datetime.now(timezone.utc).isoformat(
            timespec="seconds"
        )
        self.update_log()
        self.status = "COMPLETED"
        logger.remove(self.logger_sink_id)
