"""
Copyright 2026 Tescan group, a.s.
All rights reserved
"""

from __future__ import annotations

import importlib.metadata
import copy
import hashlib
import json
import os
import threading
from datetime import datetime
from pathlib import Path
from typing import Any

from loguru import logger

from compox.algorithm_utils.AlgorithmRecordRegistrar import (
    AlgorithmRecordRegistrar,
)
from compox.database_connection.BaseConnection import BaseConnection
from compox.database_connection.CompoxAlgorithmBundleConnection import (
    CompoxAlgorithmBundleConnection,
)
from compox.exceptions import CompoxBundleError, CompoxError


class BuiltinAlgorithmImporter:
    """
    Import a bundled snapshot of builtin algorithms into live storage.

    The bundle is still a four-store snapshot, but import is performed
    semantically per algorithm rather than as a blind store restore.

    For each bundled algorithm, the importer:
    - checks for an existing target record by `(algorithm_name, major_version)`
    - copies only the referenced module, asset and checkpoint objects
    - rewrites imported checkpoint manifests to the target algorithm id
    - merges the latest bundled minor version using the same minor-version
      insertion rule as `AlgorithmDeployer`

    User data stores such as datasets, samples, executions or trainings are
    never touched by this importer.

    Parameters
    ----------
    database_connection : BaseConnection
        Live storage connection where objects will be imported.
    settings : Any
        Server settings object containing storage bundle configuration.
    """

    _lock = threading.Lock()
    _STATE_COLLECTION = "system-store"
    _STATE_KEY = "migration-state"
    _SOURCE_COLLECTIONS = (
        "algorithm-store",
        "module-store",
        "asset-store",
        "algorithm-checkpoint-store",
    )

    def __init__(self, database_connection: BaseConnection, settings: Any):
        """
        Initialize the builtin algorithm importer.

        Parameters
        ----------
        database_connection : BaseConnection
            Live storage connection where bundled algorithms will be imported.
        settings : Any
            Server settings object providing
            `storage.builtin_storage_bundle_path` and
            `storage.builtin_storage_bundle_key`.
        """
        self._db = database_connection
        self._settings = settings
        self._logger = logger.bind(log_type="MIGRATION")
        self._backend_version = self._get_backend_version()
        self._record_registrar = AlgorithmRecordRegistrar()

    def run_startup_migration(self) -> None:
        """
        Import the configured builtin algorithm bundle during application startup.

        The import is gated by the SHA256 of the configured bundle file and
        records its state in `system-store/migration-state`.
        """
        self._logger.info("Starting startup migration")
        with self._lock:
            self._ensure_state_collection()
            state = self._read_state()

            resolved_bundle_path = self._resolve_bundle_path()
            bundle_hash = self._compute_bundle_sha256(resolved_bundle_path)
            if (
                bundle_hash is not None
                and state.get("last_imported_bundle_sha256") == bundle_hash
                and state.get("last_import_status") == "COMPLETED"
            ):
                self._logger.info(
                    "Migration already completed for this algorithm bundle. Skipping."
                )
                return

            self._write_state(
                {
                    **state,
                    "last_import_status": "RUNNING",
                    "last_import_started_at": str(datetime.now()),
                    "target_backend_version": self._backend_version,
                    "target_bundle_sha256": bundle_hash,
                }
            )

            imported = 0
            failed = 0
            rollback_journal: list[dict[str, Any]] = []
            try:
                source_db = self._build_bundle_storage_connection(
                    resolved_bundle_path
                )
                if source_db is None:
                    self._write_state(
                        {
                            **state,
                            "last_imported_backend_version": state.get(
                                "last_imported_backend_version"
                            ),
                            "last_import_status": "SKIPPED",
                            "last_import_completed_at": str(datetime.now()),
                            "imported_algorithms": 0,
                            "failed_algorithms": 0,
                            "last_imported_bundle_sha256": state.get(
                                "last_imported_bundle_sha256"
                            ),
                            "reason": "builtin_storage_bundle_path/builtin_storage_bundle_key not configured or bundle file not found",
                        }
                    )
                    return

                if not self._source_collections_exist(source_db):
                    raise CompoxBundleError(
                        "Bundle storage is missing required collections.",
                        code="bundle_missing_required_collections",
                    )

                for algorithm_key in self._normalize_keys(
                    source_db.list_objects("algorithm-store")
                ):
                    try:
                        algorithm_raw = source_db.get_objects(
                            "algorithm-store", [algorithm_key]
                        )[0]
                        if isinstance(algorithm_raw, bytes):
                            algorithm_raw = algorithm_raw.decode("utf-8")
                        algorithm_record = json.loads(algorithm_raw)

                        existing = self._record_registrar.find_existing_algorithm_by_name_and_major(
                            self._db,
                            str(algorithm_record.get("algorithm_name", "")),
                            str(
                                algorithm_record.get(
                                    "algorithm_major_version", ""
                                )
                            ),
                        )
                        self._import_algorithm_semantically(
                            source_db,
                            algorithm_record,
                            existing,
                            rollback_journal,
                        )
                        imported += 1
                    except Exception as e:
                        failed += 1
                        raise CompoxBundleError(
                            f"Failed to import builtin algorithm '{algorithm_key}' from storage bundle.",
                            code="builtin_algorithm_import_failed",
                            details={"algorithm_key": algorithm_key},
                            cause=e,
                        ) from e

                self._write_state(
                    {
                        "last_imported_bundle_sha256": bundle_hash,
                        "last_imported_backend_version": self._backend_version,
                        "last_import_status": "COMPLETED",
                        "last_import_completed_at": str(datetime.now()),
                        "imported_algorithms": imported,
                        "failed_algorithms": failed,
                    }
                )
            except Exception as e:
                failure = (
                    e
                    if isinstance(e, CompoxError)
                    else CompoxBundleError(
                        "Builtin algorithm bundle import failed.",
                        code="builtin_algorithm_bundle_import_failed",
                        cause=e,
                    )
                )
                rollback_error = self._rollback_journal(rollback_journal)
                self._write_state(
                    {
                        "last_imported_bundle_sha256": state.get(
                            "last_imported_bundle_sha256"
                        ),
                        "last_imported_backend_version": state.get(
                            "last_imported_backend_version"
                        ),
                        "last_import_status": "FAILED",
                        "last_import_completed_at": str(datetime.now()),
                        "imported_algorithms": imported,
                        "failed_algorithms": failed,
                        "error": str(failure),
                        "error_code": failure.code,
                        "retryable": failure.retryable,
                        "rollback_status": (
                            "FAILED"
                            if rollback_error is not None
                            else "COMPLETED"
                        ),
                        "rollback_error": (
                            str(rollback_error)
                            if rollback_error is not None
                            else None
                        ),
                    }
                )
                if failure is e:
                    raise
                raise failure from e

    def _build_bundle_storage_connection(
        self, resolved_bundle_path: str | None = None
    ) -> BaseConnection | None:
        """
        Build a read-only connection to the configured algorithm bundle.
        """
        bundle_path = self._settings.storage.builtin_storage_bundle_path
        bundle_key = self._settings.storage.builtin_storage_bundle_key
        if resolved_bundle_path is None:
            resolved_bundle_path = self._resolve_bundle_path()
        self._logger.info(
            f"Algorithm bundle config: path='{bundle_path}', resolved_path='{resolved_bundle_path}', cwd='{os.getcwd()}', key_configured={bundle_key is not None}"
        )
        if bundle_path is None or not bundle_key:
            self._logger.warning(
                "Algorithm bundle path/key is not fully configured."
            )
            return None
        if resolved_bundle_path is None or not os.path.isfile(
            resolved_bundle_path
        ):
            self._logger.warning(
                f"Algorithm bundle file not found at resolved path: {resolved_bundle_path}"
            )
            return None
        self._logger.info(
            f"Using algorithm bundle file: {resolved_bundle_path}"
        )
        return CompoxAlgorithmBundleConnection(resolved_bundle_path, bundle_key)

    def _resolve_bundle_path(self) -> str | None:
        """
        Resolve the configured builtin algorithm bundle path to an absolute path.
        """
        bundle_path = self._settings.storage.builtin_storage_bundle_path
        if bundle_path is None:
            return None
        return str(Path(bundle_path).expanduser().resolve(strict=False))

    @staticmethod
    def _compute_bundle_sha256(bundle_path: str | None) -> str | None:
        """
        Compute the SHA256 digest of the configured algorithm bundle file.
        """
        if bundle_path is None or not os.path.isfile(bundle_path):
            return None
        hasher = hashlib.sha256()
        with open(bundle_path, "rb") as fh:
            for chunk in iter(lambda: fh.read(1024 * 1024), b""):
                hasher.update(chunk)
        return hasher.hexdigest()

    def _source_collections_exist(self, source_db: BaseConnection) -> bool:
        """
        Validate that the bundle declares all required collections.
        """
        existing = set(source_db.list_collections())
        return all(name in existing for name in self._SOURCE_COLLECTIONS)

    def _ensure_state_collection(self) -> None:
        """
        Ensure the system collection used for importer state exists.
        """
        if self._STATE_COLLECTION not in self._db.list_collections():
            self._db.create_collections([self._STATE_COLLECTION])

    def _read_state(self) -> dict[str, Any]:
        """
        Load the persisted importer state from live storage.
        """
        exists = self._db.check_objects_exist(
            self._STATE_COLLECTION, [self._STATE_KEY]
        )[0]
        if not exists:
            return {}
        raw = self._db.get_objects(self._STATE_COLLECTION, [self._STATE_KEY])[0]
        if isinstance(raw, bytes):
            raw = raw.decode("utf-8")
        try:
            return json.loads(raw)
        except (json.JSONDecodeError, TypeError, UnicodeDecodeError):
            return {}

    def _write_state(self, state: dict[str, Any]) -> None:
        """
        Persist importer state into live storage.
        """
        self._db.put_objects(
            self._STATE_COLLECTION,
            [self._STATE_KEY],
            [json.dumps(state)],
        )

    def _import_algorithm_semantically(
        self,
        source_db: BaseConnection,
        source_algorithm_record: dict[str, Any],
        existing_algorithm: tuple[str, dict[str, Any]] | None,
        rollback_journal: list[dict[str, Any]],
    ) -> None:
        """
        Import one bundled algorithm while preserving standard minor-version semantics.
        """
        target_algorithm_id = (
            existing_algorithm[1]["algorithm_id"]
            if existing_algorithm is not None
            else source_algorithm_record["algorithm_id"]
        )

        if existing_algorithm is None:
            self._copy_all_minor_version_dependencies(
                source_db, source_algorithm_record, rollback_journal
            )
        else:
            latest_minor_record = self._get_latest_minor_version_record(
                source_algorithm_record
            )
            if latest_minor_record is not None:
                self._copy_minor_version_dependencies(
                    source_db, latest_minor_record, rollback_journal
                )

        for checkpoint_id in source_algorithm_record.get("checkpoints", []):
            self._copy_checkpoint_if_missing(
                source_db,
                str(checkpoint_id),
                target_algorithm_id,
                rollback_journal,
            )

        merged_record = self._merge_algorithm_record(
            source_algorithm_record,
            existing_algorithm[1] if existing_algorithm is not None else None,
            target_algorithm_id,
        )
        self._upsert_algorithm_record(
            merged_record, existing_algorithm, rollback_journal
        )

    def _copy_all_minor_version_dependencies(
        self,
        source_db: BaseConnection,
        algorithm_record: dict[str, Any],
        rollback_journal: list[dict[str, Any]],
    ) -> None:
        """
        Copy modules and assets referenced by all minor versions of an algorithm record.
        """
        for version_record in algorithm_record.get(
            "algorithm_minor_version", {}
        ).values():
            self._copy_minor_version_dependencies(
                source_db, version_record, rollback_journal
            )

    def _copy_minor_version_dependencies(
        self,
        source_db: BaseConnection,
        version_record: dict[str, Any],
        rollback_journal: list[dict[str, Any]],
    ) -> None:
        """
        Copy one minor version's referenced module and assets.
        """
        module_id = version_record.get("module_id")
        if module_id:
            self._copy_object_if_missing(
                source_db, "module-store", module_id, rollback_journal
            )

        for asset_id in version_record.get("assets", {}).values():
            self._copy_object_if_missing(
                source_db, "asset-store", asset_id, rollback_journal
            )

    def _copy_checkpoint_if_missing(
        self,
        source_db: BaseConnection,
        checkpoint_id: str,
        target_algorithm_id: str,
        rollback_journal: list[dict[str, Any]],
        visited: set[str] | None = None,
    ) -> None:
        """
        Copy a checkpoint manifest, rewrite its parent algorithm id and copy its assets.
        """
        if visited is None:
            visited = set()
        if checkpoint_id in visited:
            return
        visited.add(checkpoint_id)

        checkpoint_payload = source_db.get_objects(
            "algorithm-checkpoint-store", [checkpoint_id]
        )[0]
        if isinstance(checkpoint_payload, bytes):
            checkpoint_payload = checkpoint_payload.decode("utf-8")
        checkpoint_manifest = json.loads(checkpoint_payload)
        checkpoint_manifest["parent_algorithm_id"] = target_algorithm_id

        for asset_id in checkpoint_manifest.get("assets", {}).values():
            self._copy_object_if_missing(
                source_db, "asset-store", asset_id, rollback_journal
            )

        self._put_object_with_journal(
            "algorithm-checkpoint-store",
            checkpoint_id,
            json.dumps(checkpoint_manifest),
            rollback_journal,
        )

        parent_checkpoint_id = checkpoint_manifest.get("parent_checkpoint_id")
        if parent_checkpoint_id:
            self._copy_checkpoint_if_missing(
                source_db,
                str(parent_checkpoint_id),
                target_algorithm_id,
                rollback_journal,
                visited,
            )

    def _merge_algorithm_record(
        self,
        source_algorithm_record: dict[str, Any],
        existing_algorithm_record: dict[str, Any] | None,
        target_algorithm_id: str,
    ) -> dict[str, Any]:
        """
        Merge one bundled algorithm record into the target backend record.

        If the algorithm already exists, only the latest bundled minor version
        is considered for semantic insertion, matching normal deployment flow.
        """
        if existing_algorithm_record is None:
            merged_record = copy.deepcopy(source_algorithm_record)
            merged_record["algorithm_id"] = target_algorithm_id
        else:
            merged_record = copy.deepcopy(existing_algorithm_record)
            latest_minor_record = self._get_latest_minor_version_record(
                source_algorithm_record
            )
            if latest_minor_record is not None:
                merged_record, _ = (
                    self._record_registrar.insert_new_minor_version(
                        merged_record,
                        latest_minor_record.get("module_id"),
                        latest_minor_record.get("assets", {}),
                    )
                )

        self._apply_source_metadata(merged_record, source_algorithm_record)
        merged_record["algorithm_id"] = target_algorithm_id
        merged_record.setdefault("checkpoints", [])
        for checkpoint_id in source_algorithm_record.get("checkpoints", []):
            if checkpoint_id not in merged_record["checkpoints"]:
                merged_record["checkpoints"].append(checkpoint_id)

        return merged_record

    @staticmethod
    def _get_latest_minor_version_record(
        algorithm_record: dict[str, Any],
    ) -> dict[str, Any] | None:
        """
        Return the minor-version record pointed to by latest_algorithm_minor_version.
        """
        latest_minor = algorithm_record.get("latest_algorithm_minor_version")
        minor_versions = algorithm_record.get("algorithm_minor_version", {})
        if latest_minor is None:
            return None
        return minor_versions.get(str(latest_minor))

    @staticmethod
    def _apply_source_metadata(
        target_record: dict[str, Any],
        source_record: dict[str, Any],
    ) -> None:
        """
        Copy top-level algorithm metadata fields from the source record.
        """
        for field in (
            "algorithm_type",
            "algorithm_tags",
            "algorithm_description",
            "supported_devices",
            "default_device",
            "additional_parameters",
            "training_parameters",
            "removable",
            "exportable",
        ):
            if field in source_record:
                target_record[field] = copy.deepcopy(source_record[field])

    def _copy_object_if_missing(
        self,
        source_db: BaseConnection,
        collection: str,
        object_key: str,
        rollback_journal: list[dict[str, Any]],
    ) -> None:
        """
        Copy one object from the bundle into the target backend if it is missing.
        """
        if self._db.check_objects_exist(collection, [object_key])[0]:
            return
        payload = source_db.get_objects(collection, [object_key])[0]
        self._put_object_with_journal(
            collection, object_key, payload, rollback_journal
        )

    def _upsert_algorithm_record(
        self,
        source_algorithm_record: dict[str, Any],
        existing_algorithm: tuple[str, dict[str, Any]] | None,
        rollback_journal: list[dict[str, Any]],
    ) -> None:
        """
        Upsert one builtin algorithm record into live storage.
        """
        record = dict(source_algorithm_record)
        record["source_backend_version"] = self._backend_version

        algorithm_key = (
            f"{record['algorithm_id']}~{record['algorithm_name']}~"
            f"{record['algorithm_major_version']}"
        )
        if (
            existing_algorithm is not None
            and existing_algorithm[0] != algorithm_key
        ):
            self._delete_object_with_journal(
                "algorithm-store", existing_algorithm[0], rollback_journal
            )
        self._put_object_with_journal(
            "algorithm-store",
            algorithm_key,
            json.dumps(record, indent=4),
            rollback_journal,
        )

    def _put_object_with_journal(
        self,
        collection: str,
        object_key: str,
        payload: bytes | str,
        rollback_journal: list[dict[str, Any]],
    ) -> None:
        """
        Write one object and remember how to undo the write.
        """
        if collection not in self._db.list_collections():
            self._db.create_collections([collection])

        exists = self._db.check_objects_exist(collection, [object_key])[0]
        if exists:
            previous_payload = self._db.get_objects(collection, [object_key])[0]
            rollback_journal.append(
                {
                    "op": "restore",
                    "collection": collection,
                    "key": object_key,
                    "payload": previous_payload,
                }
            )
        else:
            rollback_journal.append(
                {"op": "delete", "collection": collection, "key": object_key}
            )
        self._db.put_objects(collection, [object_key], [payload])

    def _delete_object_with_journal(
        self,
        collection: str,
        object_key: str,
        rollback_journal: list[dict[str, Any]],
    ) -> None:
        """
        Delete one object and remember how to restore it.
        """
        exists = self._db.check_objects_exist(collection, [object_key])[0]
        if not exists:
            return
        previous_payload = self._db.get_objects(collection, [object_key])[0]
        rollback_journal.append(
            {
                "op": "restore",
                "collection": collection,
                "key": object_key,
                "payload": previous_payload,
            }
        )
        self._db.delete_objects(collection, [object_key])

    def _rollback_journal(
        self, rollback_journal: list[dict[str, Any]]
    ) -> Exception | None:
        """
        Undo best-effort writes performed during the current import run.
        """
        try:
            for entry in reversed(rollback_journal):
                if entry["op"] == "delete":
                    if self._db.check_objects_exist(
                        entry["collection"], [entry["key"]]
                    )[0]:
                        self._db.delete_objects(
                            entry["collection"], [entry["key"]]
                        )
                elif entry["op"] == "restore":
                    self._db.put_objects(
                        entry["collection"],
                        [entry["key"]],
                        [entry["payload"]],
                    )
            return None
        except Exception as e:
            self._logger.error(f"Rollback failed: {e}")
            return e

    @staticmethod
    def _normalize_keys(items: list[dict] | list[str]) -> list[str]:
        """
        Normalize list_objects output to plain string keys.
        """
        keys = []
        for item in items:
            if isinstance(item, dict) and "Key" in item:
                keys.append(str(item["Key"]))
            else:
                keys.append(str(item))
        return keys

    @staticmethod
    def _get_backend_version() -> str:
        """
        Read installed backend package version.
        """
        try:
            return importlib.metadata.version("compox")
        except importlib.metadata.PackageNotFoundError:
            return "unknown"
