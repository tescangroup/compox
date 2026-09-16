"""
Copyright 2026 Tescan group, a.s.
All rights reserved
"""

from __future__ import annotations

import json
import threading
from datetime import datetime, timezone
from typing import Any

from compox.database_connection.BaseConnection import BaseConnection


class AlgorithmStorageMetrics:
    """
    Compute and persist logical storage metrics for algorithm records.

    The metrics represent the logical footprint of one algorithm record:
    unique module objects referenced by all minor versions, unique asset
    objects referenced by all minor versions and associated checkpoints, and
    the checkpoint manifest objects themselves.

    The calculated size is a logical per-algorithm footprint. Shared module
    or asset objects are de-duplicated within one algorithm graph, but the
    same shared object can still contribute to the size of multiple
    algorithms.

    Parameters
    ----------
    database_connection : BaseConnection
        Storage backend used to inspect algorithm, module, checkpoint,
        and asset objects and to persist refreshed metrics back into the
        `algorithm-store`.
    """

    _STALE_FLAG = "storage_metrics_stale"
    _LOCKS_GUARD = threading.Lock()
    _LOCKS: dict[str, threading.RLock] = {}

    def __init__(self, database_connection: BaseConnection):
        self.database_connection = database_connection

    def attach_metrics(
        self, algorithm_record: dict[str, Any]
    ) -> dict[str, Any]:
        """
        Return a copy of an algorithm record with refreshed storage metrics.

        Parameters
        ----------
        algorithm_record : dict[str, Any]
            Algorithm record in the same shape as stored in
            `algorithm-store`.

        Returns
        -------
        dict[str, Any]
            Shallow copy of the input record with the `storage_metrics`
            field updated to the current logical footprint.
        """
        updated_record = dict(algorithm_record)
        updated_record.pop(self._STALE_FLAG, None)
        updated_record["storage_metrics"] = self.calculate_metrics(
            updated_record
        )
        return updated_record

    def mark_stale(
        self, algorithm_record: dict[str, Any]
    ) -> dict[str, Any]:
        """
        Return a copy of an algorithm record marked for lazy metric refresh.
        """
        updated_record = dict(algorithm_record)
        updated_record[self._STALE_FLAG] = True
        return updated_record

    def update_record(
        self, algorithm_key: str, algorithm_record: dict[str, Any]
    ) -> dict[str, Any]:
        """
        Persist refreshed storage metrics for one stored algorithm record.

        Parameters
        ----------
        algorithm_key : str
            Object key of the algorithm inside `algorithm-store`.
        algorithm_record : dict[str, Any]
            Algorithm record to recompute and write back.

        Returns
        -------
        dict[str, Any]
            Updated algorithm record containing the recomputed
            `storage_metrics` field.
        """
        with self._record_lock(algorithm_key):
            updated_record = self.attach_metrics(algorithm_record)
            self.database_connection.put_objects(
                "algorithm-store",
                [algorithm_key],
                [json.dumps(updated_record, indent=4)],
            )
            return updated_record

    def load_record(self, algorithm_key: str) -> dict[str, Any]:
        """
        Load one algorithm record and lazily refresh its storage metrics.

        Parameters
        ----------
        algorithm_key : str
            Object key of the algorithm inside `algorithm-store`.

        Returns
        -------
        dict[str, Any]
            Stored algorithm record with fresh `storage_metrics`.
        """
        algorithm_record = json.loads(
            self.database_connection.get_objects(
                "algorithm-store", [algorithm_key]
            )[0]
        )
        if not self.is_stale(algorithm_record):
            return algorithm_record

        with self._record_lock(algorithm_key):
            algorithm_record = json.loads(
                self.database_connection.get_objects(
                    "algorithm-store", [algorithm_key]
                )[0]
            )
            if self.is_stale(algorithm_record):
                return self.update_record(algorithm_key, algorithm_record)
            return algorithm_record

    def recompute_all(self) -> int:
        """
        Recompute storage metrics for every record in `algorithm-store`.

        Returns
        -------
        int
            Number of algorithm records updated. Returns `0` when the
            backend does not contain `algorithm-store`.
        """
        if "algorithm-store" not in self.database_connection.list_collections():
            return 0

        updated_count = 0
        for item in self.database_connection.list_objects("algorithm-store"):
            algorithm_key = self._normalize_key(item)
            algorithm_record = json.loads(
                self.database_connection.get_objects(
                    "algorithm-store", [algorithm_key]
                )[0]
            )
            self.update_record(algorithm_key, algorithm_record)
            updated_count += 1
        return updated_count

    @classmethod
    def is_stale(cls, algorithm_record: dict[str, Any]) -> bool:
        """
        Return True when storage metrics are missing or explicitly stale.
        """
        return bool(
            algorithm_record.get(cls._STALE_FLAG)
            or algorithm_record.get("storage_metrics") is None
        )

    def calculate_metrics(
        self, algorithm_record: dict[str, Any]
    ) -> dict[str, Any]:
        """
        Calculate logical storage metrics for one algorithm record.

        Parameters
        ----------
        algorithm_record : dict[str, Any]
            Algorithm record whose referenced modules, assets, and
            checkpoints should be analysed.

        Returns
        -------
        dict[str, Any]
            Dictionary with the current logical size and its breakdown:
            `logical_size_bytes`, `module_bytes`, `asset_bytes`,
            `checkpoint_manifest_bytes`, `module_count`, `asset_count`,
            `checkpoint_count`, and `last_recomputed_at`.
        """
        module_ids: set[str] = set()
        asset_ids: set[str] = set()
        checkpoint_ids: set[str] = set()

        for version_record in algorithm_record.get(
            "algorithm_minor_version", {}
        ).values():
            module_id = version_record.get("module_id")
            if module_id:
                module_ids.add(str(module_id))
            for asset_id in version_record.get("assets", {}).values():
                if asset_id:
                    asset_ids.add(str(asset_id))

        checkpoint_manifest_bytes = 0
        existing_checkpoint_ids: set[str] = set()
        for checkpoint_id in algorithm_record.get("checkpoints", []):
            if checkpoint_id is None:
                continue
            checkpoint_id = str(checkpoint_id)
            if checkpoint_id in checkpoint_ids:
                continue
            checkpoint_ids.add(checkpoint_id)
            if not self._object_exists(
                "algorithm-checkpoint-store", checkpoint_id
            ):
                continue
            existing_checkpoint_ids.add(checkpoint_id)
            checkpoint_manifest_bytes += (
                self.database_connection.get_object_sizes(
                    "algorithm-checkpoint-store", [checkpoint_id]
                )[0]
            )
            checkpoint_manifest = json.loads(
                self.database_connection.get_objects(
                    "algorithm-checkpoint-store", [checkpoint_id]
                )[0]
            )
            for asset_id in checkpoint_manifest.get("assets", {}).values():
                if asset_id:
                    asset_ids.add(str(asset_id))

        existing_module_ids = self._existing_object_ids(
            "module-store", module_ids
        )
        existing_asset_ids = self._existing_object_ids(
            "asset-store", asset_ids
        )
        module_bytes = self._sum_object_sizes(
            "module-store", existing_module_ids
        )
        asset_bytes = self._sum_object_sizes("asset-store", existing_asset_ids)

        return {
            "logical_size_bytes": (
                module_bytes + asset_bytes + checkpoint_manifest_bytes
            ),
            "module_bytes": module_bytes,
            "asset_bytes": asset_bytes,
            "checkpoint_manifest_bytes": checkpoint_manifest_bytes,
            "module_count": len(existing_module_ids),
            "asset_count": len(existing_asset_ids),
            "checkpoint_count": len(existing_checkpoint_ids),
            "last_recomputed_at": datetime.now(timezone.utc).isoformat(
                timespec="seconds"
            ),
        }

    def _sum_object_sizes(
        self, collection_name: str, object_ids: set[str]
    ) -> int:
        existing_ids = self._existing_object_ids(collection_name, object_ids)
        if not existing_ids:
            return 0
        return sum(
            self.database_connection.get_object_sizes(
                collection_name, existing_ids
            )
        )

    def _existing_object_ids(
        self, collection_name: str, object_ids: set[str]
    ) -> list[str]:
        return [
            object_id
            for object_id in sorted(object_ids)
            if self._object_exists(collection_name, object_id)
        ]

    def _object_exists(self, collection_name: str, object_id: str) -> bool:
        if collection_name not in self.database_connection.list_collections():
            return False
        return self.database_connection.check_objects_exist(
            collection_name, [object_id]
        )[0]

    @classmethod
    def _record_lock(cls, algorithm_key: str) -> threading.RLock:
        with cls._LOCKS_GUARD:
            lock = cls._LOCKS.get(algorithm_key)
            if lock is None:
                lock = threading.RLock()
                cls._LOCKS[algorithm_key] = lock
            return lock

    @staticmethod
    def _normalize_key(item: dict[str, Any] | str) -> str:
        if isinstance(item, dict) and "Key" in item:
            return str(item["Key"])
        return str(item)
