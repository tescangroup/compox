"""
Copyright 2024 TESCAN 3DIM, s.r.o.
All rights reserved
"""

from fastapi import APIRouter, Request
from urllib.parse import urlparse, urlunparse
from compox.pydantic_models import ResponseMessage, UrlResponse
from compox.exceptions import CompoxFileError, CompoxNotFoundError

router = APIRouter(prefix="/api/v1/files", tags=["file-controller"])


@router.get(
    "/{object_name}/upload-url",
    summary="Returns presigned URL to upload a file to the database",
    response_model=UrlResponse,
    responses={500: {"model": ResponseMessage}},
)
async def get_upload_url(object_name: str, request: Request) -> UrlResponse:
    """
    Returns URL to upload a file to the database.

    Parameters
    ----------
    object_name : str
        The name of the object.
    request : Request
        The request.

    Returns
    -------
    UrlResponse
    """

    database_connection = request.app.state.database_connection
    settings = request.app.state.settings

    try:
        url = rewrite_s3_url(
            database_connection.get_presigned_upload_url(
                "data-store", object_name
            ),
            settings.storage.backend_settings.s3_domain_name,
        )
        return UrlResponse(url=url)
    except Exception as e:
        raise CompoxFileError(
            "Failed to get upload URL.",
            code="file_upload_url_failed",
            details={"object_name": object_name},
            cause=e,
        ) from e


@router.get(
    "/{object_name}/download-url",
    summary="Returns presigned URL to download a file from the database",
    response_model=UrlResponse,
    responses={
        500: {"model": ResponseMessage},
        404: {"model": ResponseMessage},
    },
)
async def get_download_url(object_name: str, request: Request) -> UrlResponse:
    """
    Returns URL to download a file from the database.

    Parameters
    ----------
    object_name : str
        The name of the object.
    request : Request
        The request.

    Returns
    -------
    UrlResponse
    """

    database_connection = request.app.state.database_connection
    settings = request.app.state.settings

    try:
        objects_exist = database_connection.check_objects_exist(
            "data-store", [object_name]
        )

        if not objects_exist[0]:
            raise CompoxNotFoundError(
                "File not found",
                code="file_not_found",
                details={"object_name": object_name},
            )

        url = rewrite_s3_url(
            database_connection.get_presigned_download_url(
                "data-store", object_name
            ),
            settings.storage.backend_settings.s3_domain_name,
        )
        return UrlResponse(url=url)
    except CompoxNotFoundError:
        raise
    except Exception as e:
        raise CompoxFileError(
            "Failed to get download URL.",
            code="file_download_url_failed",
            details={"object_name": object_name},
            cause=e,
        ) from e


@router.delete(
    "/{object_name}",
    summary="Deletes an object from the database",
    response_model=ResponseMessage,
    responses={
        500: {"model": ResponseMessage},
        404: {"model": ResponseMessage},
    },
)
async def delete_dataset(object_name: str, request: Request) -> ResponseMessage:
    """
    Deletes an object from the database.

    Parameters
    ----------
    object_name : str
        The name of the object.

    request : Request
        The request.

    Returns
    -------
    ResponseMessage
    """

    database_connection = request.app.state.database_connection

    try:
        objects_exist = database_connection.check_objects_exist(
            "data-store", [object_name]
        )

        if not objects_exist[0]:
            raise CompoxNotFoundError(
                "File not found",
                code="file_not_found",
                details={"object_name": object_name},
            )

        database_connection.delete_objects("data-store", [object_name])
        return ResponseMessage(detail="File deleted successfully")
    except CompoxNotFoundError:
        raise
    except Exception as e:
        raise CompoxFileError(
            "Failed to delete file.",
            code="file_delete_failed",
            details={"object_name": object_name},
            cause=e,
        ) from e


def rewrite_s3_url(url: str, domain: str | None) -> str:
    """
    Rewrite the domain URL with a custom domain.

    Parameters
    ----------
    url : str
        The original URL to rewrite.
    domain : str | None
        The domain to use in the rewritten URL.

    Returns
    -------
    str
        The rewritten URL with the new domain.
    """
    if not domain:
        return url

    original_url = urlparse(url)
    rewritten_url = urlunparse(
        (
            original_url.scheme,
            (
                domain + ":" + str(original_url.port)
                if original_url.port
                else domain
            ),
            original_url.path,
            original_url.params,
            original_url.query,
            original_url.fragment,
        )
    )
    return rewritten_url
