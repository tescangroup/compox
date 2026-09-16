"""
Copyright 2026 Tescan group, a.s.
All rights reserved
"""

from __future__ import annotations

import copy
import json
from datetime import datetime, timezone
from typing import Any

from compox.database_connection import BaseConnection


class AlgorithmRecordRegistrar:
    """
    Shared semantic logic for algorithm record lookup, versioning and storage.

    This class operates on already prepared module ids and asset mappings. It
    does not build modules from source or collect assets from the filesystem.
    """

    @staticmethod
    def find_existing_algorithm_by_name_and_major(
        database_connection: BaseConnection.BaseConnection,
        algorithm_name: str,
        algorithm_major_version: str,
        algorithm_collection_name: str = "algorithm-store",
    ) -> tuple[str, dict[str, Any]] | None:
        """
        Find an existing algorithm record by `(name, major_version)`.

        Parameters
        ----------
        database_connection : BaseConnection.BaseConnection
            Database connection used to query the algorithm store.
        algorithm_name : str
            Name of the algorithm to find.
        algorithm_major_version : str
            Major version of the algorithm to find.
        algorithm_collection_name : str, optional
            Name of the collection containing algorithm records.

        Returns
        -------
        tuple[str, dict[str, Any]] | None
            A tuple of `(algorithm_store_key, algorithm_record)` if found,
            otherwise `None`.
        """
        if (
            algorithm_collection_name
            not in database_connection.list_collections()
        ):
            return None

        algorithm_keys = database_connection.list_objects(
            algorithm_collection_name
        )
        for algorithm in algorithm_keys:
            key = (
                str(algorithm["Key"])
                if isinstance(algorithm, dict) and "Key" in algorithm
                else str(algorithm)
            )
            parts = key.split("~")
            if len(parts) < 3:
                continue
            name = parts[1]
            major_version = parts[2]
            if (
                name == algorithm_name
                and major_version == str(algorithm_major_version)
            ):
                return key, json.loads(
                    database_connection.get_objects(
                        algorithm_collection_name, [key]
                    )[0]
                )
        return None

    @staticmethod
    def compose_new_algorithm_record(
        algorithm_id: str,
        algorithm_name: str,
        algorithm_major_version: str,
        algorithm_minor_version: str,
        module_id: str | None,
        assets_dict: dict[str, Any],
        metadata: dict[str, Any],
        timestamp: str | None = None,
    ) -> dict[str, Any]:
        """
        Compose a new algorithm record from prepared metadata and dependencies.

        Parameters
        ----------
        algorithm_id : str
            Target algorithm identifier.
        algorithm_name : str
            Name of the algorithm.
        algorithm_major_version : str
            Major version of the algorithm.
        algorithm_minor_version : str
            Initial minor version to store as the latest deployed version.
        module_id : str | None
            Identifier of the prepared module artifact.
        assets_dict : dict[str, Any]
            Mapping of logical asset names to stored asset ids.
        metadata : dict[str, Any]
            Top-level algorithm metadata fields to be copied into the record.
        timestamp : str | None, optional
            Timestamp to use for the record and initial minor version. If not
            provided, the current time is used.

        Returns
        -------
        dict[str, Any]
            Newly composed algorithm record ready to be stored.
        """
        if timestamp is None:
            timestamp = datetime.now(timezone.utc).isoformat(
                timespec="seconds"
            )

        return {
            "algorithm_id": algorithm_id,
            "algorithm_name": algorithm_name,
            "algorithm_major_version": algorithm_major_version,
            "latest_algorithm_minor_version": algorithm_minor_version,
            "algorithm_minor_version": {
                str(algorithm_minor_version): {
                    "timestamp": timestamp,
                    "module_id": module_id,
                    "assets": assets_dict,
                }
            },
            **copy.deepcopy(metadata),
            "timestamp": timestamp,
        }

    @staticmethod
    def insert_new_minor_version(
        existing_algorithm_record: dict[str, Any],
        module_id: str | None,
        assets_dict: dict[str, Any],
        timestamp: str | None = None,
    ) -> tuple[dict[str, Any], bool]:
        """
        Insert a new minor version if module id or asset mapping changed.

        Parameters
        ----------
        existing_algorithm_record : dict[str, Any]
            Existing algorithm record to extend.
        module_id : str | None
            Module id of the prepared algorithm version.
        assets_dict : dict[str, Any]
            Asset mapping of the prepared algorithm version.
        timestamp : str | None, optional
            Timestamp to store on the new minor version. If not provided, the
            current time is used.

        Returns
        -------
        tuple[dict[str, Any], bool]
            A tuple containing the updated record and a boolean indicating
            whether a new minor version was inserted.
        """
        modified_algorithm_json = copy.deepcopy(existing_algorithm_record)
        modified_algorithm_json.setdefault("algorithm_minor_version", {})
        if timestamp is None:
            timestamp = datetime.now(timezone.utc).isoformat(
                timespec="seconds"
            )

        if "latest_algorithm_minor_version" not in modified_algorithm_json:
            modified_algorithm_json["latest_algorithm_minor_version"] = -1
            latest_minor_version_record = {}
        else:
            latest_minor_version_record = modified_algorithm_json[
                "algorithm_minor_version"
            ][str(modified_algorithm_json["latest_algorithm_minor_version"])]

        latest_minor_version = modified_algorithm_json[
            "latest_algorithm_minor_version"
        ]

        if (
            latest_minor_version_record.get("module_id") == module_id
            and latest_minor_version_record.get("assets") == assets_dict
        ):
            return modified_algorithm_json, False

        new_minor_version = str(int(latest_minor_version) + 1)
        modified_algorithm_json["latest_algorithm_minor_version"] = (
            new_minor_version
        )
        modified_algorithm_json["algorithm_minor_version"][new_minor_version] = {
            "timestamp": timestamp,
            "module_id": module_id,
            "assets": assets_dict,
        }
        return modified_algorithm_json, True

    @staticmethod
    def upsert_algorithm_record(
        database_connection: BaseConnection.BaseConnection,
        algorithm_record: dict[str, Any],
        algorithm_collection_name: str = "algorithm-store",
        existing_algorithm: tuple[str, dict[str, Any]] | None = None,
    ) -> str:
        """
        Store an algorithm record and return the final algorithm-store key.

        Parameters
        ----------
        database_connection : BaseConnection.BaseConnection
            Database connection used to write the algorithm record.
        algorithm_record : dict[str, Any]
            Algorithm record to be stored.
        algorithm_collection_name : str, optional
            Name of the collection containing algorithm records.
        existing_algorithm : tuple[str, dict[str, Any]] | None, optional
            Existing algorithm-store key and record, if one has already been
            found for the same `(name, major_version)`.

        Returns
        -------
        str
            Final algorithm-store key used for the stored record.
        """
        if (
            algorithm_collection_name
            not in database_connection.list_collections()
        ):
            database_connection.create_collections([algorithm_collection_name])

        algorithm_key = (
            f"{algorithm_record['algorithm_id']}~"
            f"{algorithm_record['algorithm_name']}~"
            f"{algorithm_record['algorithm_major_version']}"
        )

        if (
            existing_algorithm is not None
            and existing_algorithm[0] != algorithm_key
        ):
            database_connection.delete_objects(
                algorithm_collection_name, [existing_algorithm[0]]
            )

        database_connection.put_objects(
            algorithm_collection_name,
            [algorithm_key],
            [json.dumps(algorithm_record, indent=4)],
        )
        return algorithm_key
