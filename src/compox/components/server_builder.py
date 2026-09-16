import os
import logging
import time
import threading
import importlib.metadata
from loguru import logger
from contextlib import contextmanager

import uvicorn
from fastapi import FastAPI

from compox.config.server_settings import Settings
from compox.exceptions import CompoxConfigurationError
from compox.internal.logging import configure_logging
import compox


class Compox(uvicorn.Server):
    _should_restart = False

    def __init__(
        self, config: uvicorn.Config, logger: logging.Logger | None = None
    ):
        super().__init__(config)
        self.config = config
        self.logger = logger
        self._version = importlib.metadata.version("compox")

    def install_signal_handlers(self):
        pass

    @contextmanager
    def run_in_thread(self, disable_logger: bool = False):
        """
        Run the server in a separate thread and yield control to the caller.
        This allows the server to run concurrently with other operations.
        If `disable_logger` is True, the logger will not be used.
        """
        if disable_logger:
            logger.remove()
        thread = threading.Thread(target=self.run)
        thread.start()
        try:
            while not self.started:
                time.sleep(1e-3)
            yield
        finally:
            self.should_exit = True
            thread.join()

    def restart(self):
        self._should_restart = True

    def should_restart(self):
        return self._should_restart


def build_server(
    app: FastAPI, settings: Settings, disable_logger: bool = False
) -> Compox:

    if disable_logger:
        backend_logger = logger
        backend_logger.remove()
    else:
        backend_logger = configure_logging(
            settings.log_path,
            console_level=settings.logging.console_level,
            file_level=settings.logging.file_level,
        )

    if settings.ssl.use_ssl:
        if not os.path.exists(settings.ssl.ssl_keyfile):
            backend_logger.error("Cannot find SSL keyfile!")
            raise CompoxConfigurationError(
                f"Cannot find SSL keyfile: {settings.ssl.ssl_keyfile}",
                code="ssl_keyfile_not_found",
                details={"path": settings.ssl.ssl_keyfile},
            )
        if not os.path.exists(settings.ssl.ssl_certfile):
            backend_logger.error("Cannot find SSL certfile!")
            raise CompoxConfigurationError(
                f"Cannot find SSL certfile: {settings.ssl.ssl_certfile}",
                code="ssl_certfile_not_found",
                details={"path": settings.ssl.ssl_certfile},
            )

    backend_logger.info("Configuring server")
    config = uvicorn.Config(
        app=app,
        host="0.0.0.0" if os.name == "posix" else "127.0.0.1",
        port=settings.port,
        reload=False,
        ws="none",
        log_config=None,
        log_level=logging.INFO,
        ssl_keyfile=settings.ssl.ssl_keyfile if settings.ssl.use_ssl else None,
        ssl_certfile=(
            settings.ssl.ssl_certfile if settings.ssl.use_ssl else None
        ),
    )

    backend_logger.info("Building server")
    server = Compox(config=config, logger=backend_logger)
    backend_logger.info(f"Server v{server._version} set up done")
    return server
