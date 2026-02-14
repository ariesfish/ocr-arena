from __future__ import annotations

import tempfile
from pathlib import Path

from fastapi import APIRouter, File, Request, UploadFile

from ..errors import ArenaAPIError
from ..schemas import ArenaResult, TaskCreatedResponse

router = APIRouter(prefix="/arena", tags=["arena"])


def _validate_upload(
    *,
    filename: str,
    file_bytes: bytes,
    max_size_bytes: int,
    allowed_suffixes: set[str],
) -> None:
    if not filename:
        raise ArenaAPIError(
            status_code=400, code="INVALID_FILE", message="Missing filename"
        )
    if not file_bytes:
        raise ArenaAPIError(
            status_code=400, code="EMPTY_FILE", message="Uploaded file is empty"
        )
    if len(file_bytes) > max_size_bytes:
        raise ArenaAPIError(
            status_code=400,
            code="FILE_TOO_LARGE",
            message=f"File too large, max bytes={max_size_bytes}",
        )
    suffix = Path(filename).suffix.lower().lstrip(".")
    if suffix not in allowed_suffixes:
        raise ArenaAPIError(
            status_code=400,
            code="UNSUPPORTED_FILE_TYPE",
            message=f"Unsupported file type: .{suffix}",
        )


@router.post("/jobs", response_model=TaskCreatedResponse)
async def create_ocr_job(
    request: Request, file: UploadFile = File(...)
) -> TaskCreatedResponse:
    services = request.app.state.services
    request_id = request.state.request_id

    file_bytes = await file.read()
    allowed_suffixes = {
        item.lower() for item in services.config.web.allowed_image_suffixes
    }
    max_size_bytes = int(services.config.web.upload_max_mb) * 1024 * 1024
    _validate_upload(
        filename=file.filename or "",
        file_bytes=file_bytes,
        max_size_bytes=max_size_bytes,
        allowed_suffixes=allowed_suffixes,
    )

    services.model_manager.assert_models_ready()

    task = services.task_store.create_task(task_type="ocr", request_id=request_id)
    services.ocr_queue.submit(
        task_id=task.task_id,
        request_id=request_id,
        filename=file.filename or "upload.png",
        file_bytes=file_bytes,
    )
    return TaskCreatedResponse(task_id=task.task_id, task_type="ocr", status="queued")


@router.post("/ocr", response_model=ArenaResult)
async def run_sync_ocr(request: Request, file: UploadFile = File(...)) -> ArenaResult:
    services = request.app.state.services
    request_id = request.state.request_id
    services.model_manager.assert_models_ready()

    file_bytes = await file.read()
    allowed_suffixes = {
        item.lower() for item in services.config.web.allowed_image_suffixes
    }
    max_size_bytes = int(services.config.web.upload_max_mb) * 1024 * 1024
    _validate_upload(
        filename=file.filename or "",
        file_bytes=file_bytes,
        max_size_bytes=max_size_bytes,
        allowed_suffixes=allowed_suffixes,
    )

    suffix = Path(file.filename or "upload.png").suffix or ".png"
    with tempfile.NamedTemporaryFile(
        mode="wb",
        suffix=suffix,
        prefix="ocr_arena_sync_",
        delete=False,
    ) as temp_file:
        temp_file.write(file_bytes)
        temp_path = Path(temp_file.name).resolve()

    try:
        result = services.arena_runner.run_compare(
            source_path=temp_path,
            request_id=request_id,
            persist_artifacts=False,
        )
    finally:
        temp_path.unlink(missing_ok=True)

    return ArenaResult.model_validate(result)
