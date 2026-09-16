"""
Copyright 2026 Tescan group, a.s.
All rights reserved
"""

from datetime import datetime

from loguru import logger

from compox.tasks.TaskHandler import TaskHandler, TaskStoppedException


class BenchmarkHandler(TaskHandler):
    """
    Task handler for benchmark jobs. Stores records in ``benchmark-store``
    and uses ``benchmark_id`` as the record identifier. The
    ``mark_as_completed`` override accepts a plain ``dict`` (the benchmark
    result payload) and persists it to the ``data`` field of the record.
    """

    _RECORD_STORAGE_NAME = "benchmark-store"

    @property
    def progress(self):
        return self._progress

    @progress.setter
    def progress(self, progress: float) -> None:
        self._progress = progress

    @property
    def status(self):
        return self._status

    @status.setter
    def status(self, status: str) -> None:
        self._status = status
        if self.database_update:
            task_record = self._get_record_or_stub()
            task_record["status"] = status
            self._save_record_or_fallback(task_record)

    @property
    def session_token(self):
        return self._session_token

    @session_token.setter
    def session_token(self, session_token: str) -> None:
        self._session_token = session_token

    @property
    def time_completed(self):
        return self._time_completed

    @time_completed.setter
    def time_completed(self, time_completed: str) -> None:
        self._time_completed = time_completed
        if self.database_update:
            task_record = self._get_record_or_stub()
            task_record["time_completed"] = time_completed
            self._save_record_or_fallback(task_record)

    @property
    def output_dataset_ids(self):
        return self._output_dataset_ids

    @output_dataset_ids.setter
    def output_dataset_ids(self, output_dataset_ids: list[str]) -> None:
        self._output_dataset_ids = output_dataset_ids

    def _remove_logger_sink(self) -> None:
        try:
            logger.remove(self.logger_sink_id)
        except Exception:
            pass

    def _save_record_or_fallback(self, task_record: dict) -> None:
        try:
            self._save_task_record(task_record)
            self.emergency_record_store.delete_record(
                self._RECORD_STORAGE_NAME, self._task_id
            )
        except Exception as storage_error:
            self.emergency_record_store.write_record(
                self._RECORD_STORAGE_NAME,
                self._task_id,
                task_record,
                storage_error=storage_error,
            )

    def _get_record_or_stub(self) -> dict:
        try:
            return self._get_task_record()
        except Exception:
            return getattr(self, "initial_record", None) or {
                self._record_id_field_name(): self._task_id,
                "status": "PENDING",
                "time_started": "",
                "time_completed": "",
                "log": self.log,
                "data": {},
            }

    def _set_resolved_execution_device(
        self, resolved_execution_device: str | None
    ) -> None:
        self._resolved_execution_device = resolved_execution_device

    def update_log(self) -> None:
        """
        Persist only the benchmark log, keeping the benchmark record compact.
        """
        self.log = str(self.stream.getvalue())
        if self.database_update:
            task_record = self._get_record_or_stub()
            task_record["log"] = self.log
            self._save_record_or_fallback(task_record)

    def mark_as_completed(self, data: dict | None = None) -> None:
        """
        Mark the benchmark as completed and persist the result payload.

        Parameters
        ----------
        data : dict | None, optional
            The benchmark result payload, by default None.
        """
        try:
            self._log_file_stats()
            self.update_log()
            self.time_completed = str(datetime.now())
            self.status = "COMPLETED"
            if self.database_update:
                task_record = self._get_record_or_stub()
                task_record["data"] = data or {}
                self._save_record_or_fallback(task_record)
        except Exception as e:
            self.mark_as_failed(e)
            raise
        finally:
            self._remove_logger_sink()

    def mark_as_failed(self, e: Exception | None = None) -> None:
        """
        Mark the benchmark as failed by persisting the log to the record.
        """
        if isinstance(e, TaskStoppedException):
            return

        try:
            if e is not None:
                self.logger.opt(exception=e).error("Benchmark failed")
            self._log_file_stats()
            self.log = str(self.stream.getvalue())
            if self.database_update:
                task_record = self._get_record_or_stub()
                task_record["status"] = "FAILED"
                task_record["time_completed"] = str(datetime.now())
                task_record["data"] = {}
                task_record["log"] = self.log
                self._save_record_or_fallback(task_record)
        except TaskStoppedException:
            raise
        finally:
            self._remove_logger_sink()

    def mark_as_stopped(self) -> None:
        """
        Mark the benchmark as stopped and persist its latest log.
        """
        try:
            self._log_file_stats()
            self.update_log()
            self.time_completed = str(datetime.now())
            self.status = "STOPPED"
        except Exception as e:
            self.mark_as_failed(e)
        finally:
            self._remove_logger_sink()
            try:
                self.stop_request.delete()
            except Exception:
                pass
        raise TaskStoppedException("Benchmark has been stopped.")

    def _record_id_field_name(self) -> str:
        return "benchmark_id"
