"""
Copyright 2026 Tescan group, a.s.
All rights reserved
"""

import errno
import json
import os
import tempfile
import threading
from pathlib import Path
from typing import Any

from compox.database_connection.exceptions import normalize_storage_error
from compox.exceptions import CompoxError
from compox.exceptions import error_metadata


class EmergencyRecordStore:
    """
    Minimal sidecar filesystem store for terminal fallback records.

    It is intentionally narrow: Compox uses it only to persist a failed
    execution, training, or deploy record when the primary storage backend
    refuses the final state write.

    Parameters
    ----------
    root_dir : str | None
        The root directory for the emergency record store. If None, it will be resolved
        using the following precedence:
        1. If log_path is provided, a subdirectory named "compox_emergency_records"
           in the same directory as log_path.
        2. A directory named "compox_emergency_records" in the system's temporary directory.
    reserve_bytes : int
        Size of the local reserve file used to guarantee enough room for one
        emergency JSON write if the local disk reports ENOSPC. The reserve is
        released only on that failure path and recreated on the next successful
        write when space is available again.
    """

    DEFAULT_RESERVE_BYTES = 1024 * 1024
    RECORD_FILE_PREFIX = "compox_emergency_"
    KNOWN_COLLECTIONS = ("execution-store", "training-store", "deploy-store")

    def __init__(
        self, root_dir: str | None = None, reserve_bytes: int = DEFAULT_RESERVE_BYTES
    ) -> None:
        self.root_dir = Path(root_dir or self.default_root_dir())
        self.reserve_bytes = max(int(reserve_bytes), 0)
        self._reserve_lock = threading.Lock()
        self.root_dir.mkdir(parents=True, exist_ok=True)
        self._ensure_reserve_file()

    @staticmethod
    def default_root_dir(log_path: str | None = None) -> str:
        """
        Resolve the default emergency record store path.

        Parameters
        ----------
        log_path : str | None
            Optional log path to derive the emergency record directory from.

        Returns
        -------
        str
            The resolved default root directory for the emergency record store.
        """
        if log_path:
            return str(
                Path(log_path).resolve().parent / "compox_emergency_records"
            )
        return os.path.join(tempfile.gettempdir(), "compox_emergency_records")

    def read_record(
        self, collection_name: str, record_id: str
    ) -> dict[str, Any] | None:
        """
        Read a record from the emergency store.

        Parameters
        ----------
        collection_name : str
            The name of the collection the record belongs to.
            record_id : str
        The unique identifier of the record.

        Returns
        -------
        dict[str, Any] | None
            The record data if it exists, otherwise None.
        """
        path = self._record_path(collection_name, record_id)
        if not path.exists():
            return None
        try:
            return json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return None

    def write_record(
        self,
        collection_name: str,
        record_id: str,
        record: dict[str, Any],
        storage_error: Exception | str | None = None,
    ) -> None:
        """
        Write a record to the emergency store.

        Parameters
        ----------
        collection_name : str
            The name of the collection the record belongs to.
        record_id : str
            The unique identifier of the record.
        record : dict[str, Any]
            The record data to be written.
        storage_error : Exception | str | None, optional
            An optional storage error to be recorded, if any.
        """
        payload = dict(record)
        if storage_error is not None:
            if isinstance(storage_error, Exception) and not isinstance(
                storage_error, CompoxError
            ):
                storage_error = normalize_storage_error(
                    storage_error, operation="fallback record source write"
                )
            payload.update(error_metadata(storage_error))

        path = self._record_path(collection_name, record_id)
        path.parent.mkdir(parents=True, exist_ok=True)
        data = json.dumps(payload, indent=4).encode("utf-8")
        self._atomic_write(path, data)

    def delete_record(self, collection_name: str, record_id: str) -> None:
        """
        Delete a record from the emergency store.

        Parameters
        ----------
        collection_name : str
            The name of the collection the record belongs to.
        record_id : str
            The unique identifier of the record.
        """
        path = self._record_path(collection_name, record_id)
        if path.exists():
            path.unlink()

    def has_record(self, collection_name: str, record_id: str) -> bool:
        """
        Check if a record exists in the emergency store.

        Parameters
        ----------
        collection_name : str
            The name of the collection the record belongs to.
        record_id : str
            The unique identifier of the record.

        Returns
        -------
        bool
            True if the record exists, False otherwise.
        """
        return self._record_path(collection_name, record_id).exists()

    def purge_all_records(self) -> None:
        """
        Remove all Compox emergency fallback record files under the store root.

        The reserve file is preserved and recreated if needed. This method is
        intended for explicit startup hygiene on the shared application store,
        not for ad hoc runtime store instances.
        """
        for collection_name in self.KNOWN_COLLECTIONS:
            collection_dir = self.root_dir / collection_name
            if not collection_dir.is_dir():
                continue
            for path in collection_dir.glob(f"{self.RECORD_FILE_PREFIX}*.json"):
                try:
                    path.unlink()
                except FileNotFoundError:
                    continue
                except OSError:
                    continue

        for directory in sorted(self.root_dir.rglob("*"), reverse=True):
            if not directory.is_dir():
                continue
            if directory == self.root_dir:
                continue
            try:
                directory.rmdir()
            except OSError:
                continue

        self._ensure_reserve_file()

    def _record_path(self, collection_name: str, record_id: str) -> Path:
        return (
            self.root_dir
            / collection_name
            / f"{self.RECORD_FILE_PREFIX}{record_id}.json"
        )

    def _atomic_write(self, path: Path, data: bytes) -> None:
        temp_path = path.with_suffix(path.suffix + ".tmp")
        try:
            self._write_bytes(temp_path, data)
        except OSError as exc:
            if not self._is_no_space_error(exc):
                raise
            self._release_reserve_file()
            self._write_bytes(temp_path, data)
        os.replace(temp_path, path)
        self._ensure_reserve_file()

    def _write_bytes(self, path: Path, data: bytes) -> None:
        with open(path, "wb") as handle:
            handle.write(data)

    def _reserve_path(self) -> Path:
        return self.root_dir / ".emergency_reserve.bin"

    def _ensure_reserve_file(self) -> None:
        if self.reserve_bytes <= 0:
            return
        reserve_path = self._reserve_path()
        with self._reserve_lock:
            try:
                if (
                    reserve_path.exists()
                    and reserve_path.stat().st_size >= self.reserve_bytes
                ):
                    return
            except FileNotFoundError:
                pass
            try:
                with open(reserve_path, "wb") as handle:
                    handle.truncate(self.reserve_bytes)
            except OSError:
                pass

    def _release_reserve_file(self) -> None:
        reserve_path = self._reserve_path()
        with self._reserve_lock:
            try:
                if reserve_path.exists():
                    reserve_path.unlink()
            except FileNotFoundError:
                pass
            except OSError:
                return

    @staticmethod
    def _is_no_space_error(exc: OSError) -> bool:
        return exc.errno == errno.ENOSPC
