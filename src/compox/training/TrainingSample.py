"""
Copyright 2025 TESCAN 3DIM, s.r.o.
All rights reserved
"""

import json
from typing import Optional, Union, Any
from loguru import logger
from compox.database_connection.BaseConnection import BaseConnection
from compox.training.SampleManifest import SampleManifest
from compox.exceptions import (
    CompoxError,
    CompoxNotFoundError,
    CompoxTrainingError,
    CompoxValidationError,
)


class TrainingSample:
    """
    Adapter class for loading and validating training samples.
    """

    def __init__(
        self,
        database_connection: BaseConnection,
        sample_id: Optional[str] = None,
        sample_manifest: Optional[dict] = None,
    ):
        """
        Class serves as an adapter for loading and validating training sample files
        for model training.

        Parameters
        ----------
        database_connection : BaseConnection
            The database connection object instance.
        sample_id : Optional[str]
            The identifier of the sample to load.
        sample_manifest : Optional[dict]
            The sample manifest dictionary. If provided, this is used directly
            instead of loading from the database.
        """

        self.database_connection = database_connection

        if (sample_id is None) == (sample_manifest is None):
            raise CompoxValidationError(
                "Provide exactly one of 'sample_id' or 'sample_manifest'.",
                code="sample_source_invalid",
            )

        if sample_manifest is not None:
            self.sample_manifest = SampleManifest.model_validate(
                sample_manifest
            )
            self._sample_id = self.sample_manifest.sample_id
        else:
            self._sample_id = sample_id
            self.sample_manifest = self.load_sample_manifest()

    @property
    def sample_id(self) -> str:
        """
        The identifier of the sample.

        Returns
        -------
        str
            The sample identifier.
        """
        return self._sample_id

    def __getitem__(
        self, key: Union[int, slice, tuple]
    ) -> Union[dict[str, Any], list[dict[str, Any]], str, list[str]]:
        """
        Retrieve training sample entries or specific fields using Python indexing syntax.

        Supported forms
        ---------------
        adapter[i] : dict[str, Any]
            Return the full dict for the i-th training sample entry.
            Example → {'input': 'fid1', 'target': 'fid1_y'}

        adapter[i, "field"] : list[str]
            Return a single field value from the i-th entry.
            Example → 'fid1'

        adapter[start:stop] : list[dict[str, Any]]
            Return a list of dicts for the given slice.
            Example → [{'input': 'fid2', 'target': 'fid2_y'}, {'input': 'fid3', 'target': 'fid3_y'}]

        adapter[start:stop, "field"] : list[str]
            Return a list of values for the given field across a slice.
            Example → ['fid2', 'fid3']

        adapter[:, "field"] : list[str]
            Return all values for a given field across the entire training sample.
            Example → ['fid1', 'fid2', 'fid3']

        adapter[:, :] : list[str]
            Return all values across all fields for all entries, flattened into a single list.
            Example → ['fid1', 'fid1_y', 'fid2', 'fid2_y', 'fid3', 'fid3_y']

        Parameters
        ----------
        key : int | slice | tuple
            Index or key describing what to fetch.

        Returns
        -------
        dict[str, Any] | list[dict[str, Any]] | str | list[str]
            Depending on the indexing mode:
            - dict for a single entry
            - str for a single field
            - list for slices or multiple entries
        """
        if isinstance(key, int):
            return self.sample_manifest.files[key]

        elif isinstance(key, slice):
            return self.sample_manifest.files[key]

        elif isinstance(key, tuple) and len(key) == 2:
            idx, field = key

            # Special case: [:, :]
            if (
                isinstance(idx, slice)
                and idx == slice(None)
                and isinstance(field, slice)
                and field == slice(None)
            ):
                # merge lists from all entries and all fields
                return [
                    fid
                    for entry in self.sample_manifest.files
                    for field_list in entry.values()
                    for fid in field_list
                ]

            # [:, "field"]
            if isinstance(idx, slice) and isinstance(field, str):
                # merge lists from all entries
                return [
                    fid
                    for entry in self.sample_manifest.files[idx]
                    for fid in entry.get(field, [])
                ]

            # [i, "field"]
            if isinstance(idx, int) and isinstance(field, str):
                return self.sample_manifest.files[idx][field]

        raise TypeError(f"Unsupported key type: {type(key)}")

    def __len__(self) -> int:
        """
        Get the number of training sample entries.

        Returns
        -------
        int
            The number of files in the training sample.
        """
        return len(self.sample_manifest.files)

    def __repr__(self):
        return f"TrainingSample({self.sample_manifest})"

    def __str__(self):
        return f"TrainingSample({self.sample_manifest})"

    def get_file_list(self) -> list[dict[str, Any]]:
        """
        Returns a raw list of all files in the sample.

        Returns
        -------
        list[dict[str, Any]]
        """
        return self.sample_manifest.files

    def load_sample_manifest(self) -> SampleManifest:
        """
        Load the sample manifest from the database.

        Returns
        -------
        SampleManifest
            The loaded sample manifest.
        """
        try:
            sample_manifest = json.loads(
                self.database_connection.get_objects(
                    "sample-store", [self.sample_id]
                )[0]
            )
        except CompoxError:
            raise
        except Exception as e:
            raise CompoxNotFoundError(
                f"Failed to load sample manifest for sample {self.sample_id}.",
                code="sample_not_found",
                details={"sample_id": self.sample_id},
                cause=e,
            ) from e

        try:
            sample_manifest = SampleManifest.model_validate(sample_manifest)
            return sample_manifest
        except CompoxError:
            raise
        except Exception as e:
            raise CompoxValidationError(
                f"Invalid sample manifest for sample {self.sample_id}.",
                code="invalid_sample_manifest",
                details={"sample_id": self.sample_id},
                cause=e,
            ) from e

    def save_sample_manifest(self) -> bool:
        """
        Save the sample manifest to the database.

        Returns
        -------
        bool
            True if the save was successful, False otherwise.
        """

        try:
            self._promote_files_to_training_files()
            self.database_connection.put_objects(
                "sample-store",
                [str(self.sample_id)],
                [json.dumps(self.sample_manifest.model_dump())],
            )
            return True
        except CompoxError:
            raise
        except Exception as e:
            logger.error(f"Failed to save sample {self.sample_id}: {e}")
            raise CompoxTrainingError(
                f"Failed to save sample {self.sample_id}.",
                code="sample_save_failed",
                details={"sample_id": self.sample_id},
                cause=e,
            ) from e

    def add_tags(self, new_tags: list[str]) -> None:
        """
        Add new tags to the sample manifest.

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
            self.sample_manifest.tags.append(tag)

    def check_tags(
        self,
        query_positive_tags: list[str] = None,
        query_negative_tags: list[str] = None,
    ) -> bool:
        """
        Check if the sample manifest contains all specified tags.

        Parameters
        ----------
        query_positive_tags : list[str]
            List of positive tags to check for presence in the sample manifest.
        query_negative_tags : list[str]
            List of negative tags to check for absence in the sample manifest.

        Returns
        -------
        bool
            True if all specified tags are present, False otherwise.
        """
        if not query_positive_tags:
            query_positive_tags = []
        if not query_negative_tags:
            query_negative_tags = []
        sample_tags = set(self.sample_manifest.tags)
        if not sample_tags.issuperset(query_positive_tags):
            return False
        if not sample_tags.isdisjoint(query_negative_tags):
            return False
        return True

    def delete_sample_manifest(self) -> bool:
        """
        Delete the sample manifest from the database.

        Returns
        -------
        bool
            True if the deletion was successful, raises an exception otherwise.
        """
        try:
            self._demote_files_from_training_files()
            self.database_connection.delete_objects(
                "sample-store", [self.sample_id]
            )
            return True
        except CompoxError:
            raise
        except Exception as e:
            logger.error(f"Failed to delete sample {self.sample_id}: {e}")
            raise CompoxTrainingError(
                f"Failed to delete sample {self.sample_id}.",
                code="sample_delete_failed",
                details={"sample_id": self.sample_id},
                cause=e,
            ) from e

    def _validate_files_exist(self) -> tuple[bool, list[str]]:
        """
        Validate that all files in the sample manifest exist.

        Returns
        -------
        tuple[bool, list[str]]
            A tuple where the first element is True if all files exist, False otherwise.
            The second element is a list of missing file IDs if any are missing.
        """
        # get all file ids present in the "files" field
        file_ids = self[:, :]

        exist = self.database_connection.check_objects_exist(
            "data-store", file_ids
        )

        if not all(exist):
            missing_files = [
                file_id
                for file_id, exists in zip(file_ids, exist)
                if not exists
            ]
            return False, missing_files
        else:
            return True, []

    def get_key_list(self) -> list[str]:
        """
        Get a list of all unique keys present in the sample manifest files.

        Returns
        -------
        list[str]
            A list of unique keys.
        """
        return list(self.sample_manifest.files[0].keys())

    def _promote_files_to_training_files(self):
        """
        This method promotes files from the temporary storage to the training file store.
        """
        files_exist, missing_files = self._validate_files_exist()
        if not files_exist:
            raise CompoxNotFoundError(
                "Cannot promote files to training store. Some files are missing.",
                code="sample_files_not_found",
                details={"missing_files": missing_files},
            )
        file_ids = set(self[:, :])
        for file_id in file_ids:
            tags = self.database_connection.get_object_tags(
                "data-store", file_id
            )
            try:
                ref = int(tags.get("training_ref", "0"))
            except ValueError:
                ref = 0
            tags["training_ref"] = str(ref + 1)
            self.database_connection.put_object_tags(
                "data-store", file_id, tags
            )

    def _demote_files_from_training_files(self):
        """
        This method demotes files from the training file store back to the temporary storage.
        """
        file_ids = set(self[:, :])
        for file_id in file_ids:
            tags = self.database_connection.get_object_tags(
                "data-store", file_id
            )
            try:
                ref = int(tags.get("training_ref", "0"))
            except ValueError:
                ref = 0
            ref = max(ref - 1, 0)
            tags["training_ref"] = str(ref)
            self.database_connection.put_object_tags(
                "data-store", file_id, tags
            )


if __name__ == "__main__":

    dummydb = "dummy_db"
    adapter = TrainingSample(
        dummydb,
        sample_manifest={
            "sample_id": "3f8b2b65-1c3d-4a55-8a1d-7c2d4c0b9a90",
            "files": [
                {"input": ["fid1", "fid1b"], "target": ["fid1_y"]},
                {"input": ["fid2"], "target": ["fid2_y"]},
                {"input": ["fid3"], "target": ["fid3_y"]},
            ],
            "tags": ["segmentation:skull", "author:jan"],
            "time_created": "2025-09-02T11:42:00+02:00",
        },
    )

    adapter2 = TrainingSample(
        dummydb,
        sample_manifest={
            "sample_id": "4f8b2b65-1c3d-4a55-8a1d-7c2d4c0b9a91",
            "files": [
                {"input": ["fid4"], "target": ["fid4_y"]},
                {"input": ["fid5", "fid5b"], "target": ["fid5_y"]},
                {"input": ["fid6"], "target": ["fid6_y"]},
            ],
            "tags": ["modality:CT", "author:jana"],
            "time_created": "2025-09-02T11:42:00+02:00",
        },
    )

    assert adapter.check_tags(["segmentation:skull", "author:jan"])  # True
    assert not adapter2.check_tags(query_negative_tags=["author:jana"])  # False
