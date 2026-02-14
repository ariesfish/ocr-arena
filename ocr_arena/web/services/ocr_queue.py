from __future__ import annotations

import tempfile
from dataclasses import dataclass
from pathlib import Path
from queue import Empty, Queue
from threading import Event, Thread

from ...utils.logging import get_logger
from ..stores.task_store import TaskStore
from .arena_runner import PipelineArenaRunner

logger = get_logger(__name__)


@dataclass
class OCRQueueItem:
    task_id: str
    request_id: str
    filename: str
    file_path: Path


class OCRTaskQueue:
    def __init__(self, task_store: TaskStore, arena_runner: PipelineArenaRunner):
        self._task_store = task_store
        self._arena_runner = arena_runner
        self._stop_event = Event()
        self._queue: Queue[OCRQueueItem] = Queue()
        self._thread = Thread(target=self._worker_loop, daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop_event.set()
        if self._thread.is_alive():
            self._thread.join(timeout=2.0)

    def submit(
        self,
        *,
        task_id: str,
        request_id: str,
        filename: str,
        file_bytes: bytes,
    ) -> None:
        suffix = Path(filename).suffix or ".png"
        with tempfile.NamedTemporaryFile(
            mode="wb",
            suffix=suffix,
            prefix="ocr_arena_upload_",
            delete=False,
        ) as temp_file:
            temp_file.write(file_bytes)
            temp_path = Path(temp_file.name).resolve()

        self._queue.put(
            OCRQueueItem(
                task_id=task_id,
                request_id=request_id,
                filename=filename,
                file_path=temp_path,
            )
        )

    def _worker_loop(self) -> None:
        while not self._stop_event.is_set():
            try:
                item = self._queue.get(timeout=0.2)
            except Empty:
                continue

            try:
                self._task_store.update_task(
                    item.task_id,
                    status="running",
                    progress=5,
                    stage="accepted",
                    message="task accepted by worker",
                )

                result = self._arena_runner.run_compare(
                    source_path=item.file_path,
                    request_id=item.request_id,
                    task_id=item.task_id,
                    persist_artifacts=True,
                    progress_callback=lambda progress, stage, message: self._task_store.update_task(
                        item.task_id,
                        status="running",
                        progress=progress,
                        stage=stage,
                        message=message,
                    ),
                )
                self._task_store.mark_succeeded(
                    item.task_id,
                    result=result,
                    message="ocr completed",
                )
            except Exception as exc:
                self._task_store.mark_failed(
                    item.task_id,
                    code="OCR_FAILED",
                    message=str(exc),
                    request_id=item.request_id,
                )
            finally:
                try:
                    item.file_path.unlink(missing_ok=True)
                except Exception:
                    logger.warning("Failed to delete temp input: %s", item.file_path)
                self._queue.task_done()
