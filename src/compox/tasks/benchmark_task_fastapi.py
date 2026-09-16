"""
Copyright 2026 Tescan group, a.s.
All rights reserved
"""

import json
from datetime import datetime
from typing import Any

from loguru import logger

from compox.database_connection.S3Connection import S3Connection
from compox.internal.CUDAMemoryManager import CUDAMemoryManager
from compox.pydantic_models import BenchmarkRecord
from compox.session.TaskSession import TaskSession
from compox.tasks.BenchmarkHandler import BenchmarkHandler
from compox.tasks.TaskHandler import TaskStoppedException


@logger.catch
def benchmark_task_fastapi(
    database_connection: S3Connection,
    benchmark_record: BenchmarkRecord,
    emergency_record_store=None,
) -> Any:
    """
    FastAPI background task for benchmarking an algorithm.

    Parameters
    ----------
    database_connection : S3Connection
        The database connection object instance.
    benchmark_record : BenchmarkRecord
        The benchmark record created when the benchmark request was accepted.
    emergency_record_store : optional
        Fallback record store used when the primary database is unavailable.

    Returns
    -------
    Any
        Current benchmark record dict retrieved from the database.
    """
    with CUDAMemoryManager(), TaskSession() as task_session:
        task_handler = BenchmarkHandler(
            benchmark_record.benchmark_id,
            database_connection=database_connection,
            database_update=True,
            task_session=task_session,
            emergency_record_store=emergency_record_store,
        )
        task_handler.initial_record = benchmark_record.model_dump()
        task_handler.set_as_current_handler()
        task_handler.logger.info("Fetching algorithm for benchmark...")
        start = datetime.now()
        runner = task_handler.fetch_algorithm(benchmark_record.algorithm_id)
        task_handler.logger.info(
            "Algorithm fetched in {} seconds.".format(
                (datetime.now() - start).total_seconds()
            )
        )
        try:
            runner.run_benchmark(
                args=benchmark_record.additional_parameters,
            )
        except TaskStoppedException:
            logger.info("Benchmark was interrupted by stop request.")

    return json.loads(
        database_connection.get_objects(
            "benchmark-store", [benchmark_record.benchmark_id]
        )[0]
    )
