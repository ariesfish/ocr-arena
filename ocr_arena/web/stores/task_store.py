from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from threading import Lock
from typing import Dict, List, Optional
from uuid import uuid4

from ..errors import TaskLookupError


def _utc_now() -> datetime:
    return datetime.now(timezone.utc)


@dataclass
class TaskRecord:
    task_id: str
    task_type: str
    status: str
    request_id: str
    created_at: datetime
    updated_at: datetime
    progress: int = 0
    stage: Optional[str] = None
    message: Optional[str] = None
    model_key: Optional[str] = None
    result: Optional[dict] = None
    error: Optional[dict] = None
    expires_at: Optional[datetime] = None

    def to_dict(self) -> dict:
        return {
            "task_id": self.task_id,
            "task_type": self.task_type,
            "status": self.status,
            "progress": int(self.progress),
            "stage": self.stage,
            "message": self.message,
            "model_key": self.model_key,
            "created_at": self.created_at.isoformat(),
            "updated_at": self.updated_at.isoformat(),
            "result": self.result,
            "error": self.error,
        }


class TaskStore:
    def __init__(self, ttl_seconds: int = 1800):
        self.ttl_seconds = int(ttl_seconds)
        self._lock = Lock()
        self._tasks: Dict[str, TaskRecord] = {}
        self._expired_ids: List[str] = []

    def _gen_task_id(self, prefix: str) -> str:
        now = _utc_now().strftime("%Y%m%d_%H%M%S")
        suffix = uuid4().hex[:6]
        return f"{prefix}_{now}_{suffix}"

    def _mark_expired_locked(self, task_id: str) -> None:
        self._expired_ids.append(task_id)
        if len(self._expired_ids) > 2048:
            self._expired_ids = self._expired_ids[-1024:]

    def cleanup_expired(self) -> None:
        now = _utc_now()
        with self._lock:
            expired_keys = [
                task_id
                for task_id, record in self._tasks.items()
                if record.expires_at is not None and now >= record.expires_at
            ]
            for task_id in expired_keys:
                self._tasks.pop(task_id, None)
                self._mark_expired_locked(task_id)

    def create_task(
        self,
        task_type: str,
        request_id: str,
        model_key: Optional[str] = None,
        task_id: Optional[str] = None,
    ) -> TaskRecord:
        now = _utc_now()
        resolved_task_id = task_id or self._gen_task_id(
            "ocr" if task_type == "ocr" else "dl"
        )
        record = TaskRecord(
            task_id=resolved_task_id,
            task_type=task_type,
            status="queued",
            request_id=request_id,
            created_at=now,
            updated_at=now,
            progress=0,
            stage="queued",
            model_key=model_key,
        )
        with self._lock:
            self._tasks[resolved_task_id] = record
        return record

    def get_task(self, task_id: str) -> TaskRecord:
        self.cleanup_expired()
        with self._lock:
            record = self._tasks.get(task_id)
            if record is None:
                raise TaskLookupError(
                    task_id=task_id, expired=task_id in self._expired_ids
                )
            return record

    def get_task_snapshot(self, task_id: str) -> dict:
        return self.get_task(task_id).to_dict()

    def update_task(
        self,
        task_id: str,
        *,
        status: Optional[str] = None,
        progress: Optional[int] = None,
        stage: Optional[str] = None,
        message: Optional[str] = None,
        result: Optional[dict] = None,
        error: Optional[dict] = None,
    ) -> TaskRecord:
        with self._lock:
            if task_id not in self._tasks:
                raise TaskLookupError(
                    task_id=task_id, expired=task_id in self._expired_ids
                )
            record = self._tasks[task_id]
            if status is not None:
                record.status = status
            if progress is not None:
                record.progress = max(0, min(100, int(progress)))
            if stage is not None:
                record.stage = stage
            if message is not None:
                record.message = message
            if result is not None:
                record.result = result
            if error is not None:
                record.error = error
            record.updated_at = _utc_now()
            return record

    def mark_succeeded(
        self,
        task_id: str,
        *,
        result: Optional[dict] = None,
        message: Optional[str] = None,
    ) -> TaskRecord:
        record = self.update_task(
            task_id,
            status="succeeded",
            progress=100,
            stage="finished",
            message=message or "completed",
            result=result,
            error=None,
        )
        with self._lock:
            record.expires_at = _utc_now() + timedelta(seconds=self.ttl_seconds)
            return record

    def mark_failed(
        self,
        task_id: str,
        *,
        code: str,
        message: str,
        request_id: str,
        progress: int = 100,
    ) -> TaskRecord:
        record = self.update_task(
            task_id,
            status="failed",
            progress=progress,
            stage="failed",
            message=message,
            error={
                "code": code,
                "message": message,
                "request_id": request_id,
            },
        )
        with self._lock:
            record.expires_at = _utc_now() + timedelta(seconds=self.ttl_seconds)
            return record

    def list_tasks(self) -> List[TaskRecord]:
        self.cleanup_expired()
        with self._lock:
            return list(self._tasks.values())

    def has_active_download(self, model_key: str) -> bool:
        active_status = {"queued", "running"}
        with self._lock:
            for record in self._tasks.values():
                if (
                    record.task_type == "download"
                    and record.model_key == model_key
                    and record.status in active_status
                ):
                    return True
        return False

    def latest_download_task(self, model_key: str) -> Optional[TaskRecord]:
        candidates = [
            item
            for item in self.list_tasks()
            if item.task_type == "download" and item.model_key == model_key
        ]
        if not candidates:
            return None
        return sorted(candidates, key=lambda item: item.updated_at)[-1]
