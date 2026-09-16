"""
Copyright 2025 TESCAN 3DIM, s.r.o.
All rights reserved
"""

import json
from typing import Optional
from loguru import logger
from compox.database_connection.BaseConnection import BaseConnection
from compox.algorithm_utils.AlgorithmStorageMetrics import (
    AlgorithmStorageMetrics,
)
from compox.training.CheckpointManifest import CheckpointManifest
from compox.server_utils import find_algorithm_by_id
from compox.exceptions import (
    CompoxCheckpointError,
    CompoxError,
    CompoxNotFoundError,
    CompoxValidationError,
)


class AlgorithmCheckpoint:
    """
    Adapter for managing algorithm checkpoints in the database.

    Parameters
    ----------
    database_connection : Optional[BaseConnection]
        An instance of a database connection. If None, the instance will be
        created when needed.
    checkpoint_id : Optional[str]
        The unique identifier of the checkpoint to load. If provided, the
        checkpoint will be loaded from the database during initialization.
    checkpoint_manifest : dict
        A dictionary representing the checkpoint manifest. If provided, it will
        be used to initialize the instance instead of loading from the database.
    """

    def __init__(
        self,
        database_connection: Optional[BaseConnection] = None,
        checkpoint_id: Optional[str] = None,
        checkpoint_manifest: Optional[dict] = None,
    ):
        if database_connection is not None:
            self.database_connection = database_connection

        if (checkpoint_id is None) == (checkpoint_manifest is None):
            raise CompoxValidationError(
                "Either 'checkpoint_id' or 'checkpoint_manifest' must be provided, "
                "but not both.",
                code="invalid_checkpoint_initialization",
            )

        if checkpoint_manifest:
            self.checkpoint_manifest = CheckpointManifest.model_validate(
                checkpoint_manifest
            )
            self._checkpoint_id = self.checkpoint_manifest.checkpoint_id
        else:
            self._checkpoint_id = checkpoint_id
            self.checkpoint_manifest = self.load_checkpoint_manifest()

    @property
    def checkpoint_id(self) -> str:
        """
        Returns the unique identifier of the checkpoint.

        Returns
        -------
        str
            The checkpoint identifier (UUID).
        """
        return self._checkpoint_id

    def load_checkpoint_manifest(self) -> dict | None:
        """
        Load the checkpoint manifest from the database.

        Returns
        -------
        dict | None
            The checkpoint manifest as a dictionary, or None if not found or failed to load.
        """
        try:
            checkpoint_manifest = json.loads(
                self.database_connection.get_objects(
                    "algorithm-checkpoint-store", [self.checkpoint_id]
                )[0]
            )
        except CompoxError:
            raise
        except Exception as e:
            raise CompoxNotFoundError(
                f"Failed to load checkpoint manifest with ID {self.checkpoint_id}",
                code="checkpoint_not_found",
                details={"checkpoint_id": self.checkpoint_id},
                cause=e,
            ) from e
        try:
            checkpoint_manifest = CheckpointManifest.model_validate(
                checkpoint_manifest
            )
            return checkpoint_manifest
        except CompoxError:
            raise
        except Exception as e:
            logger.error(
                f"Checkpoint manifest with ID {self.checkpoint_id} is invalid: {e}"
            )
            raise CompoxCheckpointError(
                f"Checkpoint manifest with ID {self.checkpoint_id} is invalid.",
                code="checkpoint_manifest_invalid",
                details={"checkpoint_id": self.checkpoint_id},
                cause=e,
            ) from e

    def register_checkpoint(self) -> bool:
        """
        Saves the checkpoint to the checkpoint storafe and add the reference to
        the checkpoint to its parent algorithm and training record.

        Returns
        -------
        bool
            True if the save operation was successful, False otherwise.
        """
        try:
            self.database_connection.put_objects(
                "algorithm-checkpoint-store",
                [str(self.checkpoint_id)],
                [self.checkpoint_manifest.model_dump_json()],
            )
            self._add_checkpoint_id_to_algorithm_record()
            self._add_checkpoint_id_to_training_record()
            return True
        except CompoxError:
            raise
        except Exception as e:
            logger.error(
                f"Failed to save checkpoint manifest with ID {self.checkpoint_id}: {e}"
            )
            raise CompoxCheckpointError(
                "Failed to save checkpoint manifest",
                code="checkpoint_manifest_save_failed",
                details={"checkpoint_id": self.checkpoint_id},
                cause=e,
            ) from e

    def _remove_assets_associated_with_checkpoint(self) -> None:
        """
        Delete all assets associated with the checkpoint from the database.
        """
        try:
            asset_ids = list(self.checkpoint_manifest.assets.values())
            if asset_ids:
                self.database_connection.delete_objects(
                    "asset-store", asset_ids
                )
        except CompoxError:
            raise
        except Exception as e:
            logger.error(
                f"Failed to delete assets associated with checkpoint ID {self.checkpoint_id}: {e}"
            )
            raise CompoxCheckpointError(
                "Failed to delete assets associated with checkpoint",
                code="checkpoint_asset_delete_failed",
                details={"checkpoint_id": self.checkpoint_id},
                cause=e,
            ) from e

    def _add_checkpoint_id_to_algorithm_record(self) -> None:
        """
        Add the checkpoint ID to the parent algorithm's list of checkpoints.
        """
        try:
            found_algorithm_key = self._resolve_parent_algorithm_key()
            algorithm_json = json.loads(
                self.database_connection.get_objects(
                    "algorithm-store",
                    [found_algorithm_key],
                )[0]
            )
            if "checkpoints" not in algorithm_json:
                algorithm_json["checkpoints"] = []
            if self.checkpoint_id not in algorithm_json["checkpoints"]:
                algorithm_json["checkpoints"].append(self.checkpoint_id)
                algorithm_json = AlgorithmStorageMetrics(
                    self.database_connection
                ).mark_stale(algorithm_json)
                self.database_connection.put_objects(
                    "algorithm-store",
                    [found_algorithm_key],
                    [json.dumps(algorithm_json)],
                )
        except CompoxError:
            raise
        except Exception as e:
            logger.error(
                f"Failed to add checkpoint ID {self.checkpoint_id} to algorithm ID {self.checkpoint_manifest.parent_algorithm_id}: {e}"
            )
            raise CompoxCheckpointError(
                "Failed to add checkpoint ID to algorithm record",
                code="checkpoint_algorithm_record_update_failed",
                details={
                    "checkpoint_id": self.checkpoint_id,
                    "algorithm_id": self.checkpoint_manifest.parent_algorithm_id,
                },
                cause=e,
            ) from e

    def _remove_checkpoint_id_from_algorithm_record(self) -> None:
        """
        Remove the checkpoint ID from the parent algorithm's list of checkpoints.
        """
        try:
            found_algorithm_key = self._resolve_parent_algorithm_key()
            algorithm_json = json.loads(
                self.database_connection.get_objects(
                    "algorithm-store",
                    [found_algorithm_key],
                )[0]
            )
            if (
                "checkpoints" in algorithm_json
                and self.checkpoint_id in algorithm_json["checkpoints"]
            ):
                algorithm_json["checkpoints"].remove(self.checkpoint_id)
                algorithm_json = AlgorithmStorageMetrics(
                    self.database_connection
                ).mark_stale(algorithm_json)
                self.database_connection.put_objects(
                    "algorithm-store",
                    [found_algorithm_key],
                    [json.dumps(algorithm_json)],
                )
        except CompoxError:
            raise
        except Exception as e:
            logger.error(
                f"Failed to remove checkpoint ID {self.checkpoint_id} from algorithm ID {self.checkpoint_manifest.parent_algorithm_id}: {e}"
            )
            raise CompoxCheckpointError(
                "Failed to remove checkpoint ID from algorithm record",
                code="checkpoint_algorithm_record_update_failed",
                details={
                    "checkpoint_id": self.checkpoint_id,
                    "algorithm_id": self.checkpoint_manifest.parent_algorithm_id,
                },
                cause=e,
            ) from e

    def _resolve_parent_algorithm_key(self) -> str:
        """
        Resolve the parent algorithm-store key for this checkpoint.

        Prefer the key carried in the checkpoint manifest. Fall back to the
        legacy algorithm-id bucket scan for older manifests.
        """
        parent_algorithm_key = self.checkpoint_manifest.parent_algorithm_key
        if parent_algorithm_key:
            return parent_algorithm_key

        found_algorithm_key, _, _, _, _ = find_algorithm_by_id(
            self.checkpoint_manifest.parent_algorithm_id,
            self.database_connection.list_objects("algorithm-store"),
        )
        if found_algorithm_key is None:
            raise ValueError(
                "Parent algorithm "
                f"{self.checkpoint_manifest.parent_algorithm_id} not found in algorithm-store."
            )
        return found_algorithm_key

    def _add_checkpoint_id_to_training_record(self) -> None:
        """
        Add the checkpoint ID to the parent training record's list of output checkpoints.
        """
        try:
            training_json = json.loads(
                self.database_connection.get_objects(
                    "training-store",
                    [self.checkpoint_manifest.training_id],
                )[0]
            )
            if "output_checkpoint_ids" not in training_json:
                training_json["output_checkpoint_ids"] = []
            if self.checkpoint_id not in training_json["output_checkpoint_ids"]:
                training_json["output_checkpoint_ids"].append(
                    self.checkpoint_id
                )
                self.database_connection.put_objects(
                    "training-store",
                    [self.checkpoint_manifest.training_id],
                    [json.dumps(training_json)],
                )
        except CompoxError:
            raise
        except Exception as e:
            logger.error(
                f"Failed to add checkpoint ID {self.checkpoint_id} to training ID {self.checkpoint_manifest.training_id}: {e}"
            )
            raise CompoxCheckpointError(
                "Failed to add checkpoint ID to training record",
                code="checkpoint_training_record_update_failed",
                details={
                    "checkpoint_id": self.checkpoint_id,
                    "training_id": self.checkpoint_manifest.training_id,
                },
                cause=e,
            ) from e

    def _remove_checkpoint_id_from_training_record(self) -> None:
        """
        Remove the checkpoint ID from the parent training record's list of output checkpoints.
        """
        try:
            training_json = json.loads(
                self.database_connection.get_objects(
                    "training-store",
                    [self.checkpoint_manifest.training_id],
                )[0]
            )
            if (
                "output_checkpoint_ids" in training_json
                and self.checkpoint_id in training_json["output_checkpoint_ids"]
            ):
                training_json["output_checkpoint_ids"].remove(
                    self.checkpoint_id
                )
                self.database_connection.put_objects(
                    "training-store",
                    [self.checkpoint_manifest.training_id],
                    [json.dumps(training_json)],
                )
        except CompoxError:
            raise
        except Exception as e:
            logger.error(
                f"Failed to remove checkpoint ID {self.checkpoint_id} from training ID {self.checkpoint_manifest.training_id}: {e}"
            )
            raise CompoxCheckpointError(
                "Failed to remove checkpoint ID from training record",
                code="checkpoint_training_record_update_failed",
                details={
                    "checkpoint_id": self.checkpoint_id,
                    "training_id": self.checkpoint_manifest.training_id,
                },
                cause=e,
            ) from e

    def delete_checkpoint(self) -> None:
        """
        Delete the checkpoint from the database.

        Returns
        -------
        None

        Raises
        -------
        Exception
            Returns an Exception if the deletion fails.
        """
        try:
            self._remove_assets_associated_with_checkpoint()
            self._remove_checkpoint_id_from_algorithm_record()
            self._remove_checkpoint_id_from_training_record()
            self.database_connection.delete_objects(
                "algorithm-checkpoint-store", [self.checkpoint_id]
            )
        except CompoxError:
            raise
        except Exception as e:
            logger.error(
                f"Could not delete checkpoint {self.checkpoint_id}: {e}"
            )
            raise CompoxCheckpointError(
                "Could not delete checkpoint",
                code="checkpoint_delete_failed",
                details={"checkpoint_id": self.checkpoint_id},
                cause=e,
            ) from e

    def add_tags(self, new_tags: list[str]) -> None:
        """
        Add new tags to the checkpoint manifest.

        Parameters
        ----------
        new_tags : list[str]
            A list of new tags to add.
        """
        if not isinstance(new_tags, list):
            raise TypeError("new_tags must be a list of strings")
        for tag in new_tags:
            if not isinstance(tag, str):
                raise TypeError("new_tags must be a list of strings")
            self.checkpoint_manifest.tags.append(tag)

    def check_tags(
        self,
        query_positive_tags: Optional[list[str]] = None,
        query_negative_tags: Optional[list[str]] = None,
    ) -> bool:
        """
        Check if the checkpoint manifest contains all specified tags and does not
        contain any of the specified negative tags.

        Parameters
        ----------
        query_positive_tags : Optional[list[str]]
            A list of tags to check for presence in the manifest.
        query_negative_tags : Optional[list[str]]
            A list of tags to check for absence in the manifest.

        Returns
        -------
        bool
            True if all query_tags are present and none of the query_negative_tags
            are present, False otherwise.
        """

        if not query_positive_tags:
            query_positive_tags = []
        if not query_negative_tags:
            query_negative_tags = []

        checkpoint_tags = set(self.checkpoint_manifest.tags)
        if not checkpoint_tags.issuperset(query_positive_tags):
            return False
        if not checkpoint_tags.isdisjoint(query_negative_tags):
            return False
        return True
