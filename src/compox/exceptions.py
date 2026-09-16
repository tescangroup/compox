"""
Copyright 2026 Tescan group, a.s.
All rights reserved
"""

from __future__ import annotations

from typing import Any


class CompoxError(Exception):
    """
    Base class for typed Compox failures.

    The fields are intentionally stable so API handlers, task records, CLI
    commands, and tests can all reason about failures without parsing messages.
    """

    def __init__(
        self,
        message: str,
        *,
        code: str = "compox_error",
        http_status: int = 500,
        retryable: bool = False,
        details: dict[str, Any] | None = None,
        cause: Exception | None = None,
    ) -> None:
        super().__init__(message)
        self.code = code
        self.message = message
        self.http_status = http_status
        self.retryable = retryable
        self.details = details
        self.cause = cause

    def to_response_body(self) -> dict[str, Any]:
        """
        Convert the error into the public API response shape.
        """
        body: dict[str, Any] = {
            "detail": self.message,
            "code": self.code,
            "retryable": self.retryable,
        }
        if self.details is not None:
            body["details"] = self.details
        return body

    def to_emergency_metadata(self) -> dict[str, Any]:
        """
        Convert the error into structured fallback-record metadata.
        """
        return {
            "_emergency_storage_error": self.message,
            "_emergency_storage_error_code": self.code,
            "_emergency_storage_retryable": self.retryable,
        }


class CompoxConfiguredError(CompoxError):
    """
    Error with class-level defaults for code/status/retryable.

    Subclasses can override class attributes to avoid repeating constructors
    when only defaults differ.
    """

    DEFAULT_CODE = "compox_error"
    DEFAULT_HTTP_STATUS = 500
    DEFAULT_RETRYABLE = False

    def __init__(
        self,
        message: str,
        *,
        code: str | None = None,
        http_status: int | None = None,
        retryable: bool | None = None,
        details: dict[str, Any] | None = None,
        cause: Exception | None = None,
    ) -> None:
        super().__init__(
            message,
            code=code if code is not None else self.DEFAULT_CODE,
            http_status=(
                http_status
                if http_status is not None
                else self.DEFAULT_HTTP_STATUS
            ),
            retryable=(
                retryable if retryable is not None else self.DEFAULT_RETRYABLE
            ),
            details=details,
            cause=cause,
        )


class CompoxClientError(CompoxConfiguredError):
    DEFAULT_HTTP_STATUS = 400


class CompoxServerError(CompoxConfiguredError):
    DEFAULT_HTTP_STATUS = 500


class CompoxValidationError(CompoxClientError):
    DEFAULT_CODE = "validation_error"


class CompoxNotFoundError(CompoxConfiguredError):
    DEFAULT_CODE = "not_found"
    DEFAULT_HTTP_STATUS = 404


class CompoxConflictError(CompoxConfiguredError):
    DEFAULT_CODE = "conflict"
    DEFAULT_HTTP_STATUS = 409


class CompoxConfigurationError(CompoxServerError):
    DEFAULT_CODE = "configuration_error"


class CompoxDomainError(CompoxServerError):
    """Base for domain-specific server errors (task, bundle, training, ...)."""


class CompoxTaskError(CompoxDomainError):
    DEFAULT_CODE = "task_error"


class CompoxAlgorithmError(CompoxDomainError):
    DEFAULT_CODE = "algorithm_error"


class CompoxDeploymentError(CompoxDomainError):
    DEFAULT_CODE = "deployment_error"


class CompoxAssetError(CompoxDomainError):
    DEFAULT_CODE = "asset_error"


class CompoxFileError(CompoxDomainError):
    DEFAULT_CODE = "file_error"


class CompoxCheckpointError(CompoxDomainError):
    DEFAULT_CODE = "checkpoint_error"


class CompoxImportError(CompoxDomainError):
    DEFAULT_CODE = "import_error"


class CompoxBundleError(CompoxDomainError):
    DEFAULT_CODE = "bundle_error"


class CompoxTrainingError(CompoxDomainError):
    DEFAULT_CODE = "training_error"


class CompoxExecutionError(CompoxDomainError):
    DEFAULT_CODE = "execution_error"


class CompoxStateError(CompoxDomainError):
    DEFAULT_CODE = "state_error"


class CompoxStorageError(CompoxServerError):
    def __init__(
        self,
        *,
        code: str,
        message: str,
        retryable: bool = False,
        cause: Exception | None = None,
        details: dict[str, Any] | None = None,
    ) -> None:
        super().__init__(
            message,
            code=code,
            http_status=500,
            retryable=retryable,
            details=details,
            cause=cause,
        )


def error_metadata(exc: Exception | str) -> dict[str, Any]:
    """
    Convert any exception into structured fallback-record metadata.
    """
    if isinstance(exc, CompoxError):
        return exc.to_emergency_metadata()
    return {
        "_emergency_storage_error": str(exc),
        "_emergency_storage_error_code": (
            exc.__class__.__name__
            if isinstance(exc, Exception)
            else "storage_error"
        ),
        "_emergency_storage_retryable": False,
    }


def _json_safe(value: Any) -> Any:
    """
    Convert arbitrary values into a JSON-safe representation.
    """
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_json_safe(item) for item in value]
    return repr(value)


def error_failure_payload(exc: Exception | str) -> dict[str, Any]:
    """
    Convert an exception into structured metadata for failed task records.
    """
    if isinstance(exc, CompoxError):
        payload: dict[str, Any] = {
            "type": exc.__class__.__name__,
            "message": exc.message,
            "code": exc.code,
            "details": _json_safe(exc.details),
            "retryable": exc.retryable,
            "http_status": exc.http_status,
        }
        if exc.cause is not None:
            payload["cause"] = {
                "type": exc.cause.__class__.__name__,
                "message": str(exc.cause),
            }
        return payload

    if isinstance(exc, Exception):
        payload = {
            "type": exc.__class__.__name__,
            "message": str(exc),
            "code": exc.__class__.__name__,
        }
        if exc.__cause__ is not None:
            payload["cause"] = {
                "type": exc.__cause__.__class__.__name__,
                "message": str(exc.__cause__),
            }
        return payload

    return {
        "type": "error",
        "message": str(exc),
        "code": "error",
    }
