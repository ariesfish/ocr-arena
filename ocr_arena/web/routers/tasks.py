from __future__ import annotations

import json
from pathlib import Path

from fastapi import APIRouter, Request
from fastapi.responses import FileResponse

from ..errors import ArenaAPIError, TaskLookupError
from ..schemas import TaskSnapshot

router = APIRouter(prefix="/tasks", tags=["tasks"])


@router.get("/{task_id}", response_model=TaskSnapshot)
def get_task(task_id: str, request: Request) -> TaskSnapshot:
    services = request.app.state.services
    try:
        payload = services.task_store.get_task_snapshot(task_id)
    except TaskLookupError:
        raise
    return TaskSnapshot.model_validate(payload)


@router.get("/{task_id}/layout")
def get_task_layout_preview(task_id: str, request: Request, page: int = 0) -> FileResponse:
    if page < 0:
        raise ArenaAPIError(
            status_code=400,
            code="INVALID_PAGE_INDEX",
            message=f"page must be >= 0, got {page}",
        )

    services = request.app.state.services
    payload = services.task_store.get_task_snapshot(task_id)
    result = payload.get("result") if isinstance(payload, dict) else None

    if not isinstance(result, dict):
        raise ArenaAPIError(
            status_code=404,
            code="LAYOUT_PREVIEW_NOT_FOUND",
            message=f"Task has no result payload: {task_id}",
        )

    manifest_raw = result.get("manifest")
    if not isinstance(manifest_raw, str) or not manifest_raw.strip():
        raise ArenaAPIError(
            status_code=404,
            code="LAYOUT_PREVIEW_NOT_FOUND",
            message=f"Task result has no manifest: {task_id}",
        )

    manifest_path = Path(manifest_raw).expanduser().resolve()
    if not manifest_path.is_file():
        raise ArenaAPIError(
            status_code=404,
            code="MANIFEST_NOT_FOUND",
            message=f"Manifest file not found: {manifest_path}",
        )

    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ArenaAPIError(
            status_code=422,
            code="MANIFEST_INVALID",
            message=f"Manifest json parse failed: {exc}",
        ) from exc

    layout = manifest.get("layout") if isinstance(manifest, dict) else None
    files = layout.get("files") if isinstance(layout, dict) else None
    if not isinstance(files, list) or not files:
        raise ArenaAPIError(
            status_code=404,
            code="LAYOUT_PREVIEW_NOT_FOUND",
            message=f"No layout preview files found for task: {task_id}",
        )

    if page >= len(files):
        raise ArenaAPIError(
            status_code=404,
            code="LAYOUT_PAGE_NOT_FOUND",
            message=f"Layout page {page} not found, total={len(files)}",
        )

    layout_path = Path(str(files[page])).expanduser().resolve()
    if not layout_path.is_file():
        raise ArenaAPIError(
            status_code=404,
            code="LAYOUT_FILE_NOT_FOUND",
            message=f"Layout file missing: {layout_path}",
        )

    return FileResponse(path=layout_path)
