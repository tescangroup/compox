"""
Copyright 2024 TESCAN 3DIM, s.r.o.
All rights reserved
"""

import importlib.metadata
from fastapi import APIRouter, Request
from typing import Union

from compox.pydantic_models import RootMessage, ResponseMessage
from compox.server_utils import check_system_gpu_availability
from compox.exceptions import CompoxConfigurationError

router = APIRouter(prefix="", tags=["root"])


@router.get(
    "/",
    response_model=RootMessage,
    responses={200: {"model": RootMessage}, 500: {"model": ResponseMessage}},
)
def read_root(request: Request) -> Union[RootMessage, ResponseMessage]:
    """
    This is the root endpoint of the server. It returns the basic information
    about the server. This is used to check if the server is running and to
    get the server information, mainly about cuda availability and the number
    of cuda capable devices.

    Parameters
    ----------
    request : Request
        The request.

    Returns
    -------
    Union[RootMessage, ResponseMessage]

    """

    try:

        settings = request.app.state.settings

        name = settings.info.product_name
        tags = settings.info.server_tags
        group = settings.info.group_name
        organization = settings.info.organization_name
        domain = settings.info.organization_domain

        version = importlib.metadata.version("compox")
        cuda_available, cuda_capable_device_count = (
            check_system_gpu_availability()
        )

        return RootMessage(
            name=name,
            tags=tags,
            group=group,
            organization=organization,
            domain=domain,
            version=version,
            cuda_available=cuda_available,
            cuda_capable_devices_count=cuda_capable_device_count,
        )
    except Exception as e:
        raise CompoxConfigurationError(
            "Failed to build root server response.",
            code="root_response_failed",
            cause=e,
        ) from e
