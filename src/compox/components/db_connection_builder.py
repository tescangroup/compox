"""
Copyright 2024 TESCAN 3DIM, s.r.o.
All rights reserved
"""

from compox.config.server_settings import Settings
from compox.database_connection.S3Connection import S3Connection
from compox.database_connection import BaseConnection
from compox.exceptions import CompoxConfigurationError


_DATABASE_BACKEND_TYPE = "s3"


def build_database_connection(
    settings: Settings,
) -> BaseConnection.BaseConnection:
    """
    Build a database connection based on the settings provided.

    Parameters
    ----------
    settings : Settings
        The settings object containing the database configuration.

    Returns
    -------
    BaseConnection.BaseConnection
        The database connection instance.

    Raises
    ------
    CompoxConfigurationError
        If Database backend is not supported.
    """

    if _DATABASE_BACKEND_TYPE == "s3":

        database_connection = S3Connection(
            settings.storage.backend_settings.s3_endpoint_url,
            settings.storage.access_key_id,
            settings.storage.secret_access_key,
            settings.storage.backend_settings.aws_region,
            data_store_expire_days=settings.storage.data_store_expire_days,
            execution_store_expire_days=settings.storage.execution_store_expire_days,
            training_store_expire_days=settings.storage.training_store_expire_days,
            deploy_store_expire_days=settings.storage.deploy_store_expire_days,
            stop_requests_expire_days=settings.storage.stop_requests_expire_days,
            collection_prefix=settings.storage.collection_prefix,
        )
        return database_connection
    else:
        raise CompoxConfigurationError(
            "Database backend not supported",
            code="unsupported_database_backend",
            details={"backend": _DATABASE_BACKEND_TYPE},
        )
