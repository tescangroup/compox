"""
Copyright 2026 Tescan group, a.s.
All rights reserved
"""

import json
from datetime import datetime
from typing import Any

from celery import shared_task, Task
from loguru import logger

from compox.internal.CUDAMemoryManager import CUDAMemoryManager
from compox.pydantic_models import BenchmarkRecord
from compox.session.TaskSession import TaskSession
from compox.tasks.TaskHandler import TaskStoppedException
from compox.tasks.BenchmarkHandler import BenchmarkHandler


@logger.catch
@shared_task(
    bind=True,
    autoretry_for=(Exception,),
    max_retries=0,
    name="benchmark_task",
)
def benchmark_task_celery(
    self: Task,
    message: str,
) -> Any:
    """
    Celery task for benchmarking an algorithm.

    Parameters
    ----------
    self : Task
        The celery task object.
    message : str
        The serialized benchmark record.

    Returns
    -------
    Any
        Current benchmark record from database.
    """

    benchmark_record = BenchmarkRecord.model_validate_json(message)

    with CUDAMemoryManager(), TaskSession(not_implemented=True) as task_session:
        task_handler = BenchmarkHandler(
            benchmark_record.benchmark_id,
            self.app.database_connection,
            database_update=True,
            task_session=task_session,
            emergency_record_store=self.app.emergency_record_store,
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
        self.app.database_connection.get_objects(
            "benchmark-store", [benchmark_record.benchmark_id]
        )[0]
    )
