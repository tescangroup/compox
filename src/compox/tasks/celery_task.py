"""
Copyright 2024 TESCAN 3DIM, s.r.o.
All rights reserved
"""

import json
from datetime import datetime
from typing import Any

from celery import shared_task, Task
from loguru import logger

from compox.internal.CUDAMemoryManager import CUDAMemoryManager
from compox.pydantic_models import ExecutionRecord
from compox.session.TaskSession import TaskSession
from compox.tasks.TaskHandler import TaskHandler
from compox.tasks.TaskHandler import TaskStoppedException


@logger.catch
@shared_task(
    bind=True,
    autoretry_for=(Exception,),
    max_retries=0,
    name="task",
)
def execution_task_celery(
    self: Task,
    message: str,
) -> Any:
    """
    Celery task for the execution of an algorithm. This task is executed by a
    celery worker.

    Parameters
    ----------
    self : Task
        The celery task object.
    message : str
        The message.

    Returns
    -------
    Any
        Current execution record from database.
    """

    execution_record = ExecutionRecord.model_validate_json(message)

    with CUDAMemoryManager(), TaskSession(
        session_token=execution_record.session_token, not_implemented=True
    ) as task_session:
        task_handler = TaskHandler(
            execution_record.execution_id,
            self.app.database_connection,
            database_update=True,
            task_session=task_session,
            emergency_record_store=self.app.emergency_record_store,
        )
        task_handler.set_as_current_handler()
        start = datetime.now()
        runner = task_handler.fetch_algorithm(
            execution_record.algorithm_id,
            execution_device_override=execution_record.execution_device_override,
            checkpoint_id=execution_record.checkpoint_id,
            algorithm_minor_version=execution_record.algorithm_minor_version,
        )
        task_handler.logger.info(
            "Algorithm fetched in {} seconds.".format(
                (datetime.now() - start).total_seconds()
            )
        )
        try:
            runner.run(
                {
                    "input_dataset_ids": execution_record.input_dataset_ids,
                },
                args=execution_record.additional_parameters,
            )
        except TaskStoppedException:
            logger.info("Execution was interrupted by stop request.")

    # get current execution record from database
    execution_record = json.loads(
        self.app.database_connection.get_objects(
            "execution-store", [execution_record.execution_id]
        )[0]
    )

    return execution_record

