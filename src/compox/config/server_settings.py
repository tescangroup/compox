"""
Copyright 2024 TESCAN 3DIM, s.r.o.
All rights reserved
"""

import os
import sys
from typing import Literal, Union, Annotated
from importlib.resources import files
import json

import yaml
from pydantic import BaseModel, Field, ConfigDict, model_validator
from pydantic_settings import BaseSettings

from compox.server_utils import generate_uuid


# dont parse CLI arguments if running in pytest
# or if COMPOX_DISABLE_CLI is set to True
_PARSE_CLI = (
    not os.environ.get("COMPOX_DISABLE_CLI", False)
    and "pytest" not in sys.modules
)


# configuration sections
class CompoxInfo(BaseModel):
    """
    Pydantic model for Compox information.
    The model is configured to forbid extra fields that are not defined in the model.
    """

    model_config = ConfigDict(extra="forbid")
    product_name: str = "Tescan Compox Backend"
    server_tags: list[str] = []
    group_name: str = "TESCAN GROUP, a.s."
    organization_name: str = "TESCAN GROUP, a.s."
    organization_domain: str = "tescan.com"

    @model_validator(mode="after")
    def add_default_tag(self):
        if "compox" not in self.server_tags:
            self.server_tags.append("compox")
        return self


class GUISettings(BaseModel):
    """
    Pydantic model for GUI settings.
    The model is configured to forbid extra fields that are not defined in the model.
    """

    model_config = ConfigDict(extra="forbid")
    algorithm_add_remove_in_menus: bool = False
    use_systray: bool = False
    icon_path: str = os.path.join(
        files("compox"), "resources", "compoxbackend.ico"
    )


class MinioSettings(BaseModel):
    """
    Pydantic model for MinIO settings. s3_endpoint_url can be unset, in which case
    it will be set to http://localhost:{port} after the model is validated. Either
    this or the AWS scheme is used depending on the provider field in backend_settings.
    The model is configured to forbid extra fields that are not defined in the model.
    """

    model_config = ConfigDict(extra="forbid")
    provider: Literal["minio"] = "minio"
    start_instance: bool = True
    port: int = 5483
    console_port: int = 5482
    executable_path: str = (
        os.path.join("minio", "minio_bin")
        if os.name != "nt"
        else os.path.join("minio", "minio.exe")
    )
    storage_path: str = os.path.join("minio", "compox_store")
    # derived attributes
    aws_region: str | None = None
    s3_domain_name: str | None = None
    s3_endpoint_url: str | None = None

    @model_validator(mode="after")
    def set_endpoint_url_from_port(self):
        if self.s3_endpoint_url is None:
            self.s3_endpoint_url: str = f"http://localhost:{self.port}"
        return self


class AWSSettings(BaseModel):
    """
    Pydantic model for AWS settings. Either this or the MinIO scheme is used
    depending on the provider field in backend_settings. The model is configured to
    forbid extra fields that are not defined in the model.
    """

    model_config = ConfigDict(extra="forbid")
    provider: Literal["aws"] = "aws"
    s3_endpoint_url: str | None = None
    aws_region: str | None = None
    s3_domain_name: str | None = None


class StorageSettings(BaseModel):
    """
    Pydantic model for storage settings.
    The model is configured to forbid extra fields that are not defined in the model.
    """

    model_config = ConfigDict(extra="forbid")
    collection_prefix: str = ""
    data_store_expire_days: int = 1
    execution_store_expire_days: int = 30
    training_store_expire_days: int = 30
    deploy_store_expire_days: int = 30
    stop_requests_expire_days: int = 7
    builtin_storage_bundle_path: str | None = None
    builtin_storage_bundle_key: str | None = None
    access_key_id: str | None = generate_uuid(version=4)
    secret_access_key: str | None = generate_uuid(version=4)
    backend_settings: Annotated[
        Union[MinioSettings, AWSSettings], Field(discriminator="provider")
    ] = MinioSettings()


class FastAPITaskSettings(BaseModel):
    """
    Pydantic model for FastAPI task settings. Either this or the Celery scheme is used
    depending on the executor field in backend_settings. The model is configured to
    forbid extra fields that are not defined in the model.
    """

    model_config = ConfigDict(extra="forbid")
    executor: Literal["fastapi_background_tasks"] = "fastapi_background_tasks"
    worker_number: int = 1


class CelerySettings(BaseModel):
    """
    Pydantic model for Celery task settings. Either this or the FastAPI scheme is used
    depending on the executor field in backend_settings. The model is configured to
    forbid extra fields that are not defined in the model.
    """

    model_config = ConfigDict(extra="forbid")
    executor: Literal["celery"] = "celery"
    worker_name: str = "compox_worker"
    broker_url: str
    result_backend: str = "rpc://"
    run_flower: bool = False
    flower_port: int | None = None


class InferenceSettings(BaseModel):
    """
    Pydantic model for inference settings.
    The model is configured to forbid extra fields that are not defined in the model.
    """

    model_config = ConfigDict(extra="forbid")
    device: Literal["cuda", "cpu", "mps"] = "cuda"
    cuda_visible_devices: str = "0"
    algorithm_cache_maxsize: int = Field(default=1, ge=1)
    backend_settings: Annotated[
        Union[FastAPITaskSettings, CelerySettings],
        Field(discriminator="executor"),
    ] = FastAPITaskSettings()


class SSLSettings(BaseModel):
    """
    Pydantic model for SSL settings. If use_ssl is set to True,
    ssl_keyfile and ssl_certfile paths will be generated automatically unless provided.
    The model is configured to forbid extra fields that are not defined in the model.
    """

    model_config = ConfigDict(extra="forbid")
    use_ssl: bool = False
    ssl_keyfile: str | None = None
    ssl_certfile: str | None = None


class MiddlewareSettings(BaseModel):
    """
    Pydantic model for Middleware settings.
    The model is configured to forbid extra fields that are not defined in the model.
    """

    model_config = ConfigDict(extra="forbid")
    allow_origins: list = []
    allow_methods: list = ["GET"]
    allow_headers: list = []
    allow_credentials: bool = False
    expose_headers: list = []
    max_age: int = 3600


class LoggingSettings(BaseModel):
    """
    Pydantic model for logging settings.
    The model is configured to forbid extra fields that are not defined in the model.
    """

    model_config = ConfigDict(extra="forbid")
    console_level: Literal[
        "TRACE", "DEBUG", "INFO", "SUCCESS", "WARNING", "ERROR", "CRITICAL"
    ] = "INFO"
    file_level: Literal[
        "TRACE", "DEBUG", "INFO", "SUCCESS", "WARNING", "ERROR", "CRITICAL"
    ] = "INFO"


class Settings(BaseSettings, cli_parse_args=_PARSE_CLI):
    """
    Pydantic model for Compox settings.
    The model is configured to forbid extra fields that are not defined in the model.
    """

    model_config = ConfigDict(
        extra="forbid", env_nested_delimiter="__", env_prefix="COMPOX_"
    )

    port: int = 5481
    deploy_algorithms_from: str = (
        "./algorithms"  # points to default foo/bar algorithms
    )

    info: CompoxInfo = CompoxInfo()
    gui: GUISettings = GUISettings()
    storage: StorageSettings = StorageSettings()
    inference: InferenceSettings = InferenceSettings()
    ssl: SSLSettings = SSLSettings()
    middleware: MiddlewareSettings = MiddlewareSettings()
    logging: LoggingSettings = LoggingSettings()

    log_path: str = "LOG_DEFAULT:compox.log"

    # to allow parsing other arguments by runtime scripts
    config: str | None = None

    @model_validator(mode="after")
    def parse_paths(self):
        """
        Inject paths to the settings based on the OS and server info.
        """
        if os.name == "posix":  # linux standalone
            _LOG_DEFAULT_PREFIX: str = os.path.join(
                "var",
                "log",
                self.info.organization_name,
                self.info.product_name,
            )
            _PROGRAMDATA_DEFAULT_PREFIX: str = "."
            _RELATIVE_DEFAULT_PREFIX: str = (
                sys._MEIPASS if hasattr(sys, "_MEIPASS") else "."
            )
        elif os.name == "nt":  # windows
            _LOG_DEFAULT_PREFIX: str = os.path.join(
                os.getenv("TEMP"),
                self.info.organization_name,
                self.info.product_name,
            )
            _PROGRAMDATA_DEFAULT_PREFIX: str = os.path.join(
                os.getenv("PROGRAMDATA"),
                self.info.organization_name,
                self.info.product_name,
            )
            _RELATIVE_DEFAULT_PREFIX: str = (
                sys._MEIPASS if hasattr(sys, "_MEIPASS") else "."
            )
        else:
            raise ValueError(f"Unsupported OS: {os.name}")

        def prepend_path(path: str | None) -> str | None:
            if path is None:
                return path
            if path.startswith("LOG_DEFAULT:"):
                return os.path.join(
                    _LOG_DEFAULT_PREFIX,
                    os.path.normpath(path.split("LOG_DEFAULT:")[1]),
                )
            elif path.startswith("PROGRAMDATA_DEFAULT:"):
                return os.path.join(
                    _PROGRAMDATA_DEFAULT_PREFIX,
                    os.path.normpath(path.split("PROGRAMDATA_DEFAULT:")[1]),
                )
            elif path.startswith("RELATIVE_DEFAULT:"):
                return os.path.join(
                    _RELATIVE_DEFAULT_PREFIX,
                    os.path.normpath(path.split("RELATIVE_DEFAULT:")[1]),
                )
            return path

        self.log_path = prepend_path(self.log_path)
        self.ssl.ssl_keyfile = prepend_path(self.ssl.ssl_keyfile)
        self.ssl.ssl_certfile = prepend_path(self.ssl.ssl_certfile)
        self.gui.icon_path = prepend_path(self.gui.icon_path)
        self.storage.builtin_storage_bundle_path = prepend_path(
            self.storage.builtin_storage_bundle_path
        )
        if self.storage.backend_settings.provider == "minio":
            self.storage.backend_settings.executable_path = prepend_path(
                self.storage.backend_settings.executable_path
            )
            self.storage.backend_settings.storage_path = prepend_path(
                self.storage.backend_settings.storage_path
            )

        return self


def get_server_settings(
    config_path: str | None = None, verbose: bool = True
) -> Settings:
    """
    Get server settings from a yaml file or use default settings.
    """
    if config_path is not None:
        with open(config_path, "r") as file:
            conf = yaml.safe_load(file)
            settings = Settings(**(conf or {}))
    else:
        settings = Settings()
    if verbose:
        print(json.dumps(json.loads(settings.model_dump_json()), indent=4))
    return settings
