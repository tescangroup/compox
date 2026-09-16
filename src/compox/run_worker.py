"""
Copyright 2024 TESCAN 3DIM, s.r.o.
All rights reserved
"""

import argparse
from compox.server_utils import get_subprocess_fn
from compox.config.server_settings import get_server_settings
from compox.internal.logging import configure_logging
from compox.components.celery_builder import build_celery
from compox.tasks.celery_task import execution_task_celery  # noqa F401
from compox.tasks.benchmark_task_celery import benchmark_task_celery  # noqa F401
from compox.training.training_task_celery import (  # noqa F401
    training_task_celery,
)


def parse_args() -> object:
    """
    Argument parser for the server.

    Returns
    -------
    object
        The parsed arguments.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        help="Path to server configuration yaml file. If None, use defaults specified in server_config.py.",
        default=None,
        required=False,
    )
    args, _ = parser.parse_known_args()
    return args


def main():
    """
    Main function to run the Celery worker.
    """
    args = parse_args()
    settings = get_server_settings(args.config)
    _ = configure_logging(
        log_path=settings.log_path,
        console_level=settings.logging.console_level,
        file_level=settings.logging.file_level,
    )

    if settings.inference.backend_settings.executor == "celery":
        celery_app = build_celery(settings)

        if settings.inference.backend_settings.run_flower:
            subprocess_fn = get_subprocess_fn()
            _ = subprocess_fn(
                [
                    "celery",
                    f"--broker={settings.inference.backend_settings.broker_url}",
                    "flower",
                    f"--port={settings.inference.backend_settings.flower_port}",
                ]
            )
        celery_app.worker_main(
            [
                "worker",
                "--loglevel=info",
                "--pool=solo",
                f"-n {settings.inference.backend_settings.worker_name}",
            ]
        )
    else:
        print(
            f"Unsupported executor: {settings.inference.backend_settings.executor}"
        )


if __name__ == "__main__":
    main()
