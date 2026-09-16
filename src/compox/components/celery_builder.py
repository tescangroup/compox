"""
Copyright 2024 TESCAN 3DIM, s.r.o.
All rights reserved
"""

from celery import Celery
from kombu import Queue

from compox.config.server_settings import Settings
from compox.components.db_connection_builder import build_database_connection
from compox.internal.EmergencyRecordStore import EmergencyRecordStore
from compox.tasks.TaskHandler import TaskHandler


def route_task(name, args, kwargs, options, task=None, **kw):
    if ":" in name:
        queue, _ = name.split(":")
        return {"queue": queue}
    return {"queue": "processing-queue"}


def build_celery(settings: Settings) -> Celery:
    """
    Build a Celery instance with the broker URL parsed from the settings object.

    Parameters
    ----------
    settings : Settings
        The settings object containing the broker URL.

    Returns
    -------
    Celery
        The Celery instance.
    """
    TaskHandler._ALGORITHM_CACHE_MAXSIZE = (
        settings.inference.algorithm_cache_maxsize
    )

    celery = Celery(
            broker=settings.inference.backend_settings.broker_url,
            task_create_missing_queues=True,
        )
    celery.conf.update(task_track_started=True)
    celery.conf.update(task_serializer="pickle")
    celery.conf.update(result_serializer="pickle")
    celery.conf.update(accept_content=["pickle", "json"])
    celery.conf.update(result_expires=200)
    celery.conf.update(result_persistent=True)
    celery.conf.update(worker_send_task_events=False)
    celery.conf.update(task_routes=(route_task,))
    celery.conf.update(broker_heartbeat=600)
    celery.conf.update(task_queues=(
        Queue('processing-queue', routing_key='task.#'),
    ))
    
    database_connection = build_database_connection(settings)
    celery.database_connection = database_connection
    celery.emergency_record_store = EmergencyRecordStore(
        EmergencyRecordStore.default_root_dir(settings.log_path)
    )
    celery.emergency_record_store.purge_all_records()
    
    return celery
