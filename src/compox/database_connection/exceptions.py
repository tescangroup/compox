"""
Copyright 2026 Tescan group, a.s.
All rights reserved
"""

from __future__ import annotations

import errno

from botocore.exceptions import ClientError

from compox.exceptions import CompoxStorageError


class CompoxDiskFullError(CompoxStorageError):
    """Storage failed because the local or backing filesystem is out of space."""

    def __init__(self, message: str, *, cause: Exception | None = None) -> None:
        super().__init__(
            code="disk_full", message=message, retryable=False, cause=cause
        )


class CompoxStorageAccessError(CompoxStorageError):
    """Storage failed because access was denied or permissions are insufficient."""

    def __init__(self, message: str, *, cause: Exception | None = None) -> None:
        super().__init__(
            code="storage_access_denied",
            message=message,
            retryable=False,
            cause=cause,
        )


class CompoxStorageUnavailableError(CompoxStorageError):
    """Storage backend is temporarily unavailable or unreachable."""

    def __init__(self, message: str, *, cause: Exception | None = None) -> None:
        super().__init__(
            code="storage_unavailable",
            message=message,
            retryable=True,
            cause=cause,
        )


class CompoxStorageWriteError(CompoxStorageError):
    """Generic storage write failure that does not match a more specific category."""

    def __init__(
        self,
        message: str,
        *,
        retryable: bool = False,
        cause: Exception | None = None,
    ) -> None:
        super().__init__(
            code="storage_write_failed",
            message=message,
            retryable=retryable,
            cause=cause,
        )


def normalize_storage_error(
    exc: Exception, *, operation: str = "storage write"
) -> CompoxStorageError:
    """
    Convert backend-specific storage exceptions into stable Compox storage errors.
    """

    if isinstance(exc, CompoxStorageError):
        return exc

    if isinstance(exc, OSError):
        if exc.errno == errno.ENOSPC:
            return CompoxDiskFullError(
                f"{operation} failed because the disk is full.", cause=exc
            )
        if exc.errno in {errno.EACCES, errno.EPERM}:
            return CompoxStorageAccessError(
                f"{operation} failed due to insufficient filesystem permissions.",
                cause=exc,
            )
        return CompoxStorageWriteError(
            f"{operation} failed: {exc}", retryable=False, cause=exc
        )

    if isinstance(exc, ClientError):
        error = exc.response.get("Error", {})
        code = str(error.get("Code", ""))
        message = str(error.get("Message", exc))

        if code in {"AccessDenied", "Forbidden"}:
            return CompoxStorageAccessError(
                f"{operation} failed due to storage access denial: {message}",
                cause=exc,
            )
        if code in {"InternalError", "ServiceUnavailable", "SlowDown"}:
            return CompoxStorageUnavailableError(
                f"{operation} failed because the storage backend is unavailable: {message}",
                cause=exc,
            )
        if code in {"InsufficientStorage", "NoSuchBucket"}:
            return CompoxStorageWriteError(
                f"{operation} failed: {message}",
                retryable=code != "NoSuchBucket",
                cause=exc,
            )
        return CompoxStorageWriteError(
            f"{operation} failed: {message}", retryable=False, cause=exc
        )

    return CompoxStorageWriteError(
        f"{operation} failed: {exc}", retryable=False, cause=exc
    )


def reraised_storage_error(
    exc: Exception, *, operation: str = "storage write"
) -> CompoxStorageError:
    """
    Convenience wrapper returning a normalized storage exception suitable for ``raise``.
    """

    return normalize_storage_error(exc, operation=operation)
