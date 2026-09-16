"""
Copyright 2026 Tescan group, a.s.
All rights reserved
"""

from __future__ import annotations

import errno
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest
from botocore.exceptions import ClientError

from compox.database_connection.InMemoryConnection import InMemoryConnection
from compox.database_connection.S3Connection import S3Connection
from compox.database_connection.TempfileConnection import TempfileConnection
from compox.database_connection.exceptions import (
    CompoxDiskFullError,
    CompoxStorageAccessError,
    CompoxStorageUnavailableError,
    CompoxStorageWriteError,
    normalize_storage_error,
)
from compox.exceptions import CompoxError


def test_normalize_storage_error_maps_enospc_to_disk_full() -> None:
    """ENOSPC should be normalized into a disk-full storage exception."""
    error = normalize_storage_error(
        OSError(errno.ENOSPC, "No space left on device"),
        operation="record write",
    )

    assert isinstance(error, CompoxDiskFullError)
    assert isinstance(error, CompoxError)
    assert error.code == "disk_full"


def test_normalize_storage_error_maps_permission_denied_to_access_error() -> None:
    """Filesystem access errors should become storage access failures."""
    error = normalize_storage_error(
        PermissionError(errno.EACCES, "Permission denied"),
        operation="record write",
    )

    assert isinstance(error, CompoxStorageAccessError)
    assert isinstance(error, CompoxError)
    assert error.code == "storage_access_denied"


def test_normalize_storage_error_maps_client_error_access_denied() -> None:
    """S3 AccessDenied responses should become access-denied storage failures."""
    error = normalize_storage_error(
        ClientError(
            {
                "Error": {
                    "Code": "AccessDenied",
                    "Message": "Access Denied",
                }
            },
            "PutObject",
        ),
        operation="put_objects to execution-store",
    )

    assert isinstance(error, CompoxStorageAccessError)
    assert error.code == "storage_access_denied"


def test_normalize_storage_error_maps_client_error_unavailable() -> None:
    """Transient S3 service failures should become unavailable storage failures."""
    error = normalize_storage_error(
        ClientError(
            {
                "Error": {
                    "Code": "ServiceUnavailable",
                    "Message": "Service unavailable",
                }
            },
            "PutObject",
        ),
        operation="put_objects to execution-store",
    )

    assert isinstance(error, CompoxStorageUnavailableError)
    assert error.code == "storage_unavailable"
    assert error.retryable is True

def test_normalize_storage_error_falls_back_to_generic_write_error() -> None:
    """Unknown exceptions should still become a stable generic storage write error."""
    error = normalize_storage_error(RuntimeError("boom"), operation="record write")

    assert isinstance(error, CompoxStorageWriteError)
    assert error.code == "storage_write_failed"


def test_tempfile_connection_put_objects_reraises_normalized_storage_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Tempfile backend should normalize write failures like the other file-backed stores."""
    connection = TempfileConnection()
    connection.create_collections(["execution-store"])

    def _raise_enospc(*args, **kwargs):
        raise OSError(errno.ENOSPC, "No space left on device")

    monkeypatch.setattr("builtins.open", _raise_enospc)

    with pytest.raises(CompoxDiskFullError):
        connection.put_objects("execution-store", ["record-1"], [b"payload"])


def test_inmemory_connection_reraises_normalized_storage_error_from_injected_failure(
) -> None:
    """In-memory backend should still present the normalized hierarchy if a test injects a write failure."""
    connection = InMemoryConnection()

    class BrokenCollection(dict):
        def __setitem__(self, key, value):
            raise RuntimeError("injected failure")

    connection.store = {"execution-store": BrokenCollection()}
    connection.tags = {"execution-store": {}}

    with pytest.raises(CompoxStorageWriteError):
        connection.put_objects("execution-store", ["record-1"], [b"payload"])


def test_s3_lifecycle_hook_adds_content_md5_header() -> None:
    """MinIO requires Content-MD5 on lifecycle configuration requests."""
    params = {"headers": {}, "body": b"<LifecycleConfiguration />"}

    S3Connection._add_content_md5_header(params)

    assert params["headers"]["Content-MD5"] == "7YDDAdXc+T/IZIKADDT00g=="


def test_s3_lifecycle_hook_keeps_existing_content_md5_header() -> None:
    """Do not replace a checksum that botocore or the caller already supplied."""
    params = {
        "headers": {"Content-MD5": "existing-checksum"},
        "body": b"<LifecycleConfiguration />",
    }

    S3Connection._add_content_md5_header(params)

    assert params["headers"]["Content-MD5"] == "existing-checksum"


def test_s3_put_objects_retries_with_exponential_backoff(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Transient S3 write failures should be retried with increasing delays."""
    access_denied = ClientError(
        {
            "Error": {
                "Code": "AccessDenied",
                "Message": "Access Denied",
            }
        },
        "PutObject",
    )
    connection = S3Connection.__new__(S3Connection)
    connection.s3_client = MagicMock()
    connection.s3_client.put_object.side_effect = [
        access_denied,
        access_denied,
        access_denied,
        None,
    ]
    connection.uploader = SimpleNamespace(chunk_size=1024)
    connection.collection_prefix = ""
    connection.post_data_retries = 5
    connection.post_data_retry_initial_delay = 0.1
    connection.post_data_retry_max_delay = 2.0
    connection.logger = MagicMock()
    sleep_calls = []
    monkeypatch.setattr(
        "compox.database_connection.S3Connection.time.sleep",
        sleep_calls.append,
    )

    connection.put_objects(
        "execution-store",
        ["record-1"],
        [b"payload"],
    )

    assert sleep_calls == [0.1, 0.2, 0.4]
    assert connection.s3_client.put_object.call_count == 4
