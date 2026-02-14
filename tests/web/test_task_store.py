from __future__ import annotations

from datetime import timedelta

import pytest

from ocr_arena.web.errors import TaskLookupError
from ocr_arena.web.stores.task_store import TaskStore, _utc_now


def test_task_store_lifecycle() -> None:
    store = TaskStore(ttl_seconds=1800)
    task = store.create_task(task_type="ocr", request_id="req_a")
    store.update_task(task.task_id, status="running", progress=50)
    snapshot = store.get_task_snapshot(task.task_id)
    assert snapshot["status"] == "running"
    assert snapshot["progress"] == 50

    store.mark_succeeded(task.task_id, result={"ok": True})
    done = store.get_task_snapshot(task.task_id)
    assert done["status"] == "succeeded"
    assert done["result"] == {"ok": True}


def test_task_store_expired_returns_410() -> None:
    store = TaskStore(ttl_seconds=1)
    task = store.create_task(task_type="ocr", request_id="req_a")
    store.mark_succeeded(task.task_id, result={"ok": True})

    record = store.get_task(task.task_id)
    record.expires_at = _utc_now() - timedelta(seconds=1)

    with pytest.raises(TaskLookupError):
        store.get_task(task.task_id)
