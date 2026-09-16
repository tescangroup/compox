"""
Copyright 2024 TESCAN 3DIM, s.r.o.
All rights reserved
"""

from fastapi import APIRouter, Request
from starlette.responses import StreamingResponse
import io
import h5py
from compox.server_utils import generate_uuid, calculate_s3_etag
from compox.pydantic_models import FileUploadResponse, ResponseMessage
from compox.exceptions import (
    CompoxFileError,
    CompoxNotFoundError,
    CompoxValidationError,
)

router = APIRouter(prefix="/api", tags=["file-controller"])


@router.post(
    "/v0/files",
    summary="Uploads an image stack as a hdf5 file to the database",
    response_model=FileUploadResponse,
    responses={
        500: {"model": ResponseMessage},
        422: {"model": ResponseMessage},
    },
)
async def upload_files(request: Request) -> FileUploadResponse:
    """
    Uploads an image stack as a hdf5 file to the database.

    Parameters
    ----------
    request : Request
        The request.

    Returns
    -------
    FileUploadResponse
        The file upload response.
    """
    # create database connection
    database_connection = request.app.state.database_connection
    bio_bytes: bytes = await request.body()
    bio = io.BytesIO(bio_bytes)

    # check if bio is valid hdf5 file
    try:
        with h5py.File(bio, "r") as f:
            assert f is not None
    except Exception as e:
        raise CompoxValidationError(
            "The provided file is not a valid hdf5 file.",
            code="invalid_hdf5_file",
            http_status=422,
            cause=e,
        ) from e

    try:

        # BUG: This is working but the upload times rise dramatically with the
        # number of files in the bucket. This is because the folowing code always
        # iterates over all files in the bucket. It should be possible to solve
        # this smarter. For now we just disable the check.
        if False:
            # calculate file etag hash
            etag = calculate_s3_etag(bio)
            s3_client = database_connection.s3_client

            # if file already exists, return the existing file id
            try:
                paginator = s3_client.get_paginator("list_objects")
                page_iterator = paginator.paginate(Bucket="data-store")
                for page in page_iterator:
                    for obj in page["Contents"]:
                        if obj["ETag"] == etag:
                            print(
                                "Existing file found with matching etag hash, returning..."
                            )
                            return FileUploadResponse(
                                file_id=obj["Key"],
                            )
            except Exception as _:
                print(
                    "No existing file found with matching etag hash found, uploading..."
                )

        file_id = generate_uuid()

        database_connection.put_objects(
            "data-store",
            [file_id],
            [bio.getvalue()],
        )
        return FileUploadResponse(
            file_id=file_id,
        )
    except CompoxValidationError:
        raise
    except Exception as e:
        raise CompoxFileError(
            "Failed to upload file.",
            code="file_upload_failed",
            cause=e,
        ) from e


@router.get(
    "/v0/files/{id}",
    summary="Downloads a file from the database",
    responses={
        500: {"model": ResponseMessage},
        404: {"model": ResponseMessage},
    },
)
async def download_file(id: str, request: Request):
    """
    Downloads a file from database.

    Parameters
    ----------
    id : str
        The id of the dataset.
    request : Request
        The request.
        The id of the file.

    Returns
    -------
    StreamingResponse
        The dataset.
    """
    database_connection = request.app.state.database_connection
    try:
        try:
            object = database_connection.get_objects("data-store", [id])[0]
        except Exception as e:
            raise CompoxNotFoundError(
                "File not found",
                code="file_not_found",
                details={"file_id": id},
                cause=e,
            ) from e

        file_like_obj = io.BytesIO(object)

        # stream response
        def chunk_generator():
            for chunk in iter(lambda: file_like_obj.read(8 * 1024 * 1024), b""):
                yield chunk

        return StreamingResponse(
            chunk_generator(), media_type="application/octet-stream"
        )
    except CompoxNotFoundError:
        raise
    except Exception as e:
        raise CompoxFileError(
            "Failed to download file.",
            code="file_download_failed",
            details={"file_id": id},
            cause=e,
        ) from e


@router.delete(
    "/v0/files/{id}",
    summary="Deletes a file from the database",
    response_model=ResponseMessage,
    responses={
        500: {"model": ResponseMessage},
        404: {"model": ResponseMessage},
    },
)
async def delete_file(id: str, request: Request) -> ResponseMessage:
    """
    Deletes a file from the database.

    Parameters
    ----------
    id : str
        The id of the file.
    request : Request
        The request.

        Returns
        -------
        ResponseMessage
    """

    database_connection = request.app.state.database_connection

    try:
        objects_exist = database_connection.check_objects_exist(
            "data-store", [id]
        )

        if not objects_exist[0]:
            raise CompoxNotFoundError(
                "File not found",
                code="file_not_found",
                details={"file_id": id},
            )

        database_connection.delete_objects("data-store", [id])
        return ResponseMessage(detail="File deleted successfully")
    except CompoxNotFoundError:
        raise
    except Exception as e:
        raise CompoxFileError(
            "Failed to delete file.",
            code="file_delete_failed",
            details={"file_id": id},
            cause=e,
        ) from e
