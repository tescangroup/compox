"""
Copyright 2024 TESCAN 3DIM, s.r.o.
All rights reserved
"""

from compox.database_connection.BaseConnection import BaseConnection
from compox.database_connection.exceptions import reraised_storage_error
import tempfile
import os
import json


class TempfileConnection(BaseConnection):
    """
    A connection class for a local file system "database". This class inherits from
    the BaseConnection class and implements the methods for interacting with a local
    tempfile file structure mimicking an object storage database. Can be used for
    testing and debugging purposes, or when a real database is not available for
    local deployment of the application.

    Parameters
    ----------
    temp_folder_name : str
        The name of the temporary folder.
    """

    def __init__(self, temp_folder_name: str = "pcb_temp"):
        super().__init__()
        self.temp_folder = tempfile.TemporaryDirectory(prefix=temp_folder_name)

    def list_collections(self) -> list:
        """
        List all subdirectories in the temporary folder.

        Returns
        -------
        list
            The list of subdirectories.
        """
        return os.listdir(self.temp_folder.name)

    def check_collections_exists(self, collection_names: list[str]) -> list[bool]:
        """
        Check if the subdirectories exist in the temporary folder.

        Parameters
        ----------
        collection_names : list[str]
            The subdirectory names.

        Returns
        -------
        list[bool]
            The list of booleans indicating if the subdirectories exist.
        """
        return [
            os.path.isdir(os.path.join(self.temp_folder.name, name))
            for name in collection_names
        ]

    def delete_collections(self, collection_names: list[str]) -> None:
        """
        Delete the subdirectories in the temporary folder including all files.

        Parameters
        ----------
        collection_names : list[str]
            The subdirectory names.
        """
        for name in collection_names:
            os.rmdir(os.path.join(self.temp_folder.name, name))

    def create_collections(self, collection_names: list[str]) -> None:
        """
        Create subdirectories in the temporary folder.

        Parameters
        ----------
        collection_names : list[str]
            The subdirectory names.
        """
        for name in collection_names:
            os.mkdir(os.path.join(self.temp_folder.name, name))

    def list_objects(self, collection_name: str) -> list[str]:
        """
        List all files in a subdirectory.

        Parameters
        ----------
        collection_name : str
            The subdirectory name.

        Returns
        -------
        list[str]
            The list of files.
        """
        return os.listdir(os.path.join(self.temp_folder.name, collection_name))

    def check_objects_exist(
        self, collection_name: str, object_names: list[str]
    ) -> list[bool]:
        """
        Check if files exist in a subdirectory.

        Parameters
        ----------
        collection_name : str
            The subdirectory name.
        object_names : list[str]
            The file names.

        Returns
        -------
        list[bool]
            The list of booleans indicating if the files exist.
        """
        return [
            os.path.isfile(os.path.join(self.temp_folder.name, collection_name, name))
            for name in object_names
        ]

    def delete_objects(self, collection_name: str, object_names: list[str]) -> None:
        """
        Delete files in a subdirectory.

        Parameters
        ----------
        collection_name : str
            The subdirectory name.
        object_names : list[str]
            The file names.
        """
        for name in object_names:
            os.remove(os.path.join(self.temp_folder.name, collection_name, name))

    def get_objects(self, collection_name: str, object_names: list[str]) -> list[bytes]:
        """
        Get files from a subdirectory.

        Parameters
        ----------
        collection_name : str
            The subdirectory name.
        object_names : list[str]
            The file names.

        Returns
        -------
        list[bytes]
            The list of file bytes.
        """
        return [
            open(
                os.path.join(self.temp_folder.name, collection_name, name), "rb"
            ).read()
            for name in object_names
        ]

    def get_object_sizes(
        self, collection_name: str, object_names: list[str]
    ) -> list[int]:
        """
        Get file sizes in bytes for objects in a subdirectory.
        """
        return [
            os.path.getsize(
                os.path.join(self.temp_folder.name, collection_name, name)
            )
            for name in object_names
        ]

    def put_objects(
        self, collection_name: str, object_names: list[str], object: list[bytes]  | list[str]
    ) -> None:
        """
        Put files in a subdirectory.

        Parameters
        ----------
        collection_name : str
            The subdirectory name.
        object_names : list[str]
            The file names.
        object : list[bytes]  | list[str]
            The file bytes.
        """
        try:
            for name, obj in zip(object_names, object):
                if isinstance(obj, bytes):
                    with open(
                        os.path.join(self.temp_folder.name, collection_name, name), "wb"
                    ) as f:
                        f.write(obj)
            if collection_name == "data-store":
                for name in object_names:
                    tags_path = os.path.join(
                        self.temp_folder.name, collection_name, f"{name}.tags"
                    )
                    if not os.path.isfile(tags_path):
                        with open(tags_path, "w", encoding="utf-8") as f:
                            f.write(json.dumps({"training_ref": "0"}))
        except Exception as exc:
            raise reraised_storage_error(
                exc,
                operation=f"put_objects to {collection_name}",
            ) from exc

    def put_objects_with_duplicity_check(
        self, collection_name: str, object_names: list[str], object: list[bytes]
    ) -> list[bool]:
        """
        Put files in a subdirectory with a check for existing files.

        Parameters
        ----------
        collection_name : str
            The subdirectory name.
        object_names : list[str]
            The file names.
        object : list[bytes]
            The file bytes.

        Returns
        -------
        list[bool]
            The list of booleans indicating if the files were put.
        """
        try:
            object_exists = self.check_objects_exist(collection_name, object_names)
            for i, (name, obj, exists) in enumerate(
                zip(object_names, object, object_exists)
            ):
                if not exists:
                    with open(
                        os.path.join(self.temp_folder.name, collection_name, name), "wb"
                    ) as f:
                        f.write(obj)
            return object_exists
        except Exception as exc:
            raise reraised_storage_error(
                exc,
                operation=f"put_objects_with_duplicity_check to {collection_name}",
            ) from exc

    def get_object_tags(
        self, collection_name: str, object_name: str
    ) -> dict[str, str]:
        """
        Get object tags for a file. Tags are stored in a sidecar .tags JSON file.
        """
        tags_path = os.path.join(
            self.temp_folder.name, collection_name, f"{object_name}.tags"
        )
        if not os.path.isfile(tags_path):
            return {}
        with open(tags_path, "r", encoding="utf-8") as f:
            return json.loads(f.read())

    def put_object_tags(
        self, collection_name: str, object_name: str, tags: dict[str, str]
    ) -> None:
        """
        Put object tags for a file. Tags are stored in a sidecar .tags JSON file.
        """
        tags_path = os.path.join(
            self.temp_folder.name, collection_name, f"{object_name}.tags"
        )
        with open(tags_path, "w", encoding="utf-8") as f:
            f.write(json.dumps({k: str(v) for k, v in tags.items()}))
