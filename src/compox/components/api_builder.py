"""
Copyright 2024 TESCAN 3DIM, s.r.o.
All rights reserved
"""

import os
import tempfile
import atexit
from contextlib import asynccontextmanager
from concurrent.futures import _base, ThreadPoolExecutor

from fastapi import APIRouter, FastAPI, Request
from fastapi.encoders import jsonable_encoder
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.logger import logger as fastapi_logger
from fastapi.responses import JSONResponse
from celery import Celery
from starlette.exceptions import HTTPException as StarletteHTTPException

from compox.config.server_settings import Settings
from compox.components.minio_wrapper import MinIOWrapper
from compox.components.celery_builder import build_celery
from compox.components.db_connection_builder import build_database_connection
from compox.internal.EmergencyRecordStore import EmergencyRecordStore
from compox.components.builtin_algorithm_importer import (
    BuiltinAlgorithmImporter,
)
from compox.algorithm_utils.AlgorithmExporter import AlgorithmExporter
from compox.database_connection.BaseConnection import BaseConnection
from compox.tasks.TaskHandler import TaskHandler
from compox.exceptions import (
    CompoxConfigurationError,
    CompoxError,
)

from compox.server_utils import (
    check_and_create_database_collections,
    get_subprocess_fn,
)
from compox.algorithm_utils.zip_importer import ZipImporter

from compox.routers import (
    algorithms_controller,
    benchmark_controller,
    deployment_controller,
    execution_controller,
    execution_manager,
    file_controller,
    file_controller_v1,
    root,
    sample_controller,
    training_controller,
    checkpoint_controller,
)


class ApiBuilder:
    def __init__(self):
        self.lifespan = None
        self.settings = None
        self.database_connection = None
        self.celery = None
        self.executor = None
        self.middleware = None
        self.routes = []

    def with_lifespan(self, lifespan):
        self.lifespan = lifespan
        return self

    def with_settings(self, settings):
        self.settings = settings
        return self

    def with_database_connection(self, database_connection: BaseConnection):
        self.database_connection = database_connection
        return self

    def with_algorithm_exporter(self, algorithm_exporter: AlgorithmExporter):
        self.algorithm_exporter = algorithm_exporter
        return self

    def with_executor(self, executor: _base.Executor | Celery | None = None):
        self.executor = executor
        return self

    def with_route(self, route: APIRouter):
        self.routes.append(route)
        return self

    def with_middleware(self, middleware, middleware_settings):
        self.middleware = middleware
        self.middleware_settings = middleware_settings
        return self

    def build(self):
        app = FastAPI(lifespan=self.lifespan)
        app.state.database_connection = self.database_connection
        app.state.executor = self.executor
        app.state.settings = self.settings
        app.state.algorithm_exporter = self.algorithm_exporter
        app.state.emergency_record_store = EmergencyRecordStore(
            EmergencyRecordStore.default_root_dir(self.settings.log_path)
        )
        app.state.emergency_record_store.purge_all_records()
        self._register_exception_handlers(app)
        for route in self.routes:
            app.include_router(route)

        if self.middleware is not None:
            app.add_middleware(
                CORSMiddleware, **self.middleware_settings.model_dump()
            )

        return app

    @staticmethod
    def _register_exception_handlers(app: FastAPI) -> None:
        """
        Register Compox-wide API error translation.
        """

        @app.exception_handler(CompoxError)
        async def compox_error_handler(
            request: Request, exc: CompoxError
        ) -> JSONResponse:
            if exc.http_status >= 500:
                return JSONResponse(
                    status_code=exc.http_status,
                    content={
                        "detail": (
                            "Failed due to an internal server error."
                        ),
                        "code": exc.code,
                        "retryable": exc.retryable,
                    },
                )

            return JSONResponse(
                status_code=exc.http_status,
                content=exc.to_response_body(),
            )

        @app.exception_handler(RequestValidationError)
        async def validation_error_handler(
            request: Request, exc: RequestValidationError
        ) -> JSONResponse:
            return JSONResponse(
                status_code=422,
                content={
                    "detail": jsonable_encoder(exc.errors()),
                    "code": "request_validation_error",
                    "retryable": False,
                },
            )

        @app.exception_handler(StarletteHTTPException)
        async def http_error_handler(
            request: Request, exc: StarletteHTTPException
        ) -> JSONResponse:
            return JSONResponse(
                status_code=exc.status_code,
                content={
                    "detail": exc.detail,
                    "code": f"http_{exc.status_code}",
                    "retryable": False,
                },
            )

        @app.exception_handler(Exception)
        async def unexpected_error_handler(
            request: Request, exc: Exception
        ) -> JSONResponse:
            fastapi_logger.exception("Unhandled API exception")
            return JSONResponse(
                status_code=500,
                content={
                    "detail": (
                        "Failed due to an internal server error."
                    ),
                    "code": "internal_server_error",
                    "retryable": False,
                },
            )


# define app context manager
@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Lifespan context manager for the FastAPI app.
    The subprocess for minio is started here and killed when the app is closed.

    Parameters
    ----------
    app : FastAPI
        The FastAPI instance.

    Raises
    ------
    ValueError
        If minio storage path does not exist.
    """
    # setup subprocess mechanism that runs on both linux and win
    subprocess_fn = get_subprocess_fn()

    lifecycle_subprocesses = {}

    settings = app.state.settings

    # maybe run local minio subprocess
    if (
        settings.storage.backend_settings.provider == "minio"
        and settings.storage.backend_settings.start_instance
    ):

        if not os.path.exists(settings.storage.backend_settings.storage_path):
            raise CompoxConfigurationError(
                "Minio storage path does not exist!",
                code="minio_storage_path_missing",
            )

        minio_wrapper = MinIOWrapper(settings)
        lifecycle_subprocesses["minio"] = minio_wrapper.start(subprocess_fn)

    new_collections = check_and_create_database_collections(
        [
            "data-store",
            "execution-store",
            "algorithm-store",
            "module-store",
            "asset-store",
            "training-store",
            "sample-store",
            "algorithm-checkpoint-store",
            "stop-requests",
            "deploy-store",
            "system-store",
            "benchmark-store"
        ],
        database_connection=app.state.database_connection,
    )
    if len(new_collections) > 0:
        fastapi_logger.info(f"Created new collections: {new_collections}")

    builtin_algorithm_importer = BuiltinAlgorithmImporter(
        app.state.database_connection, settings
    )
    builtin_algorithm_importer.run_startup_migration()

    for lifecycle_subprocess in lifecycle_subprocesses.values():
        atexit.register(lifecycle_subprocess.kill)

    yield

    # if the app database connection has s3_client, close it
    if app.state.database_connection.s3_client:
        app.state.database_connection.s3_client.close()

    for lifecycle_subprocess in lifecycle_subprocesses.values():
        atexit.register(lifecycle_subprocess.kill)


def build_api(settings: Settings, with_lifespan: bool = True) -> FastAPI:
    """
    Build a FastAPI instance with the provided settings and lifecycle management.

    Parameters
    ----------
    settings : Settings
        The settings object containing the configuration for the API.
    with_lifespan : bool, optional
        Whether to include the lifespan context manager for the API. Default is True.

    Returns
    -------
    FastAPI
        The FastAPI instance.

    Raises
    ------
    ValueError
        If server backend is invalid.
    """

    # Configure persistent module import cache. Prefer storage-adjacent runtime path
    # when backend provides local storage_path, otherwise use system temp fallback.
    storage_backend = settings.storage.backend_settings
    storage_path = getattr(storage_backend, "storage_path", None)
    module_cache_dir = (
        os.path.join(storage_path, "runtime", "module_cache")
        if storage_path
        else os.path.join(tempfile.gettempdir(), "compox", "module_cache")
    )
    TaskHandler._ALGORITHM_CACHE_MAXSIZE = (
        settings.inference.algorithm_cache_maxsize
    )
    ZipImporter.configure_cache_dir(module_cache_dir)
    atexit.register(ZipImporter.cleanup_cache)

    # Build database connection
    database_connection = build_database_connection(settings)

    # build algorithm exporter
    algorithm_exporter = AlgorithmExporter(
        database_connection=database_connection
    )

    # Task execution
    match settings.inference.backend_settings.executor:
        case "fastapi_background_tasks":
            task_executor = ThreadPoolExecutor(
                max_workers=settings.inference.backend_settings.worker_number
            )
        case "celery":
            task_executor = build_celery(settings)
        case _:
            raise CompoxConfigurationError(
                "Invalid server backend",
                code="invalid_server_backend",
            )

    # build api with lifecycle management
    api_builder = (
        ApiBuilder()
        .with_settings(settings)
        .with_database_connection(database_connection)
        .with_algorithm_exporter(algorithm_exporter)
        .with_executor(task_executor)
        .with_route(root.router)
        .with_route(algorithms_controller.router)
        .with_route(deployment_controller.router)
        .with_route(execution_controller.router)
        .with_route(benchmark_controller.router)
        .with_route(file_controller.router)
        .with_route(file_controller_v1.router)
        .with_route(execution_manager.router)
        .with_route(sample_controller.router)
        .with_route(training_controller.router)
        .with_route(checkpoint_controller.router)
        .with_middleware(CORSMiddleware, settings.middleware)
    )
    if with_lifespan:
        api_builder.with_lifespan(lifespan)

    return api_builder.build()
