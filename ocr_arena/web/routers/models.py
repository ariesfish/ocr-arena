from __future__ import annotations

from typing import Optional

from fastapi import APIRouter, Request

from ..schemas import DownloadRequest, ModelsStatusResponse, TaskCreatedResponse

router = APIRouter(prefix="/models", tags=["models"])


@router.get("/status", response_model=ModelsStatusResponse)
def models_status(request: Request) -> ModelsStatusResponse:
    services = request.app.state.services
    payload = services.model_manager.get_models_status()
    return ModelsStatusResponse.model_validate({"models": payload})


@router.post("/{model_key}/download", response_model=TaskCreatedResponse)
def create_download_task(
    model_key: str,
    request: Request,
    payload: Optional[DownloadRequest] = None,
) -> TaskCreatedResponse:
    services = request.app.state.services
    request_id = request.state.request_id
    task_id = services.model_manager.create_download_task(
        model_key=model_key,
        request_id=request_id,
        source=(payload.source if payload else None),
    )
    return TaskCreatedResponse(
        task_id=task_id,
        task_type="download",
        status="queued",
    )
