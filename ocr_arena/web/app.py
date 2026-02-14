from __future__ import annotations

from contextlib import asynccontextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Optional
from uuid import uuid4

from fastapi import FastAPI, HTTPException, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from starlette.exceptions import HTTPException as StarletteHTTPException

from ..config import Config, load_config
from ..utils.logging import ensure_logging_configured
from .errors import ArenaAPIError, TaskLookupError
from .routers import arena_router, health_router, models_router, tasks_router
from .services import ModelManager, OCRTaskQueue, PipelineArenaRunner
from .stores import TaskStore


@dataclass
class WebServices:
    config: Config
    task_store: TaskStore
    model_manager: ModelManager
    arena_runner: PipelineArenaRunner
    ocr_queue: OCRTaskQueue


def _resolve_frontend_dist() -> Optional[Path]:
    candidates = [
        Path("/app/frontend_dist"),
        Path(__file__).resolve().parents[2] / "web" / "dist",
    ]
    for candidate in candidates:
        index_file = candidate / "index.html"
        if candidate.is_dir() and index_file.is_file():
            return candidate
    return None


def _resolve_request_id(request: Request) -> str:
    incoming = request.headers.get("x-request-id")
    if incoming and incoming.strip():
        return incoming.strip()
    return f"arena_{uuid4().hex[:12]}"


def _error_payload(request: Request, code: str, message: str) -> dict:
    request_id = getattr(request.state, "request_id", None)
    if not request_id:
        request_id = _resolve_request_id(request)
    return {
        "code": code,
        "message": message,
        "request_id": request_id,
    }


def _http_error_code(status_code: int) -> str:
    mapping = {
        400: "BAD_REQUEST",
        404: "NOT_FOUND",
        409: "CONFLICT",
        410: "GONE",
        422: "INVALID_REQUEST",
        500: "INTERNAL_ERROR",
        504: "GATEWAY_TIMEOUT",
    }
    return mapping.get(int(status_code), f"HTTP_{int(status_code)}")


def _http_error_message(detail: object, status_code: int) -> str:
    if isinstance(detail, str) and detail.strip():
        return detail
    if detail is not None:
        return str(detail)
    return f"HTTP {int(status_code)}"


def build_services(config: Config) -> WebServices:
    task_store = TaskStore(ttl_seconds=config.web.task_ttl_seconds)
    arena_runner = PipelineArenaRunner(config=config)
    model_manager = ModelManager(config=config, task_store=task_store)
    ocr_queue = OCRTaskQueue(task_store=task_store, arena_runner=arena_runner)
    return WebServices(
        config=config,
        task_store=task_store,
        model_manager=model_manager,
        arena_runner=arena_runner,
        ocr_queue=ocr_queue,
    )


def create_app(config: Optional[Config] = None) -> FastAPI:
    resolved_config = config or load_config()
    ensure_logging_configured(
        level=resolved_config.logging.level,
        format_string=resolved_config.logging.format,
    )

    services = build_services(resolved_config)

    @asynccontextmanager
    async def lifespan(_app: FastAPI):
        try:
            yield
        finally:
            services.ocr_queue.stop()
            services.arena_runner.stop()

    app = FastAPI(title="OCR Arena API", version="0.1.0", lifespan=lifespan)
    app.state.services = services

    @app.middleware("http")
    async def request_context_middleware(request: Request, call_next):
        request_id = _resolve_request_id(request)
        request.state.request_id = request_id
        response = await call_next(request)
        response.headers["x-request-id"] = request_id
        return response

    @app.exception_handler(ArenaAPIError)
    async def arena_api_error_handler(request: Request, exc: ArenaAPIError):
        return JSONResponse(
            status_code=exc.status_code,
            content=_error_payload(request, exc.code, exc.message),
        )

    @app.exception_handler(TaskLookupError)
    async def task_lookup_error_handler(request: Request, exc: TaskLookupError):
        return JSONResponse(
            status_code=exc.status_code,
            content=_error_payload(request, exc.code, exc.message),
        )

    @app.exception_handler(RequestValidationError)
    async def request_validation_handler(request: Request, exc: RequestValidationError):
        return JSONResponse(
            status_code=422,
            content=_error_payload(
                request,
                _http_error_code(422),
                str(exc),
            ),
        )

    @app.exception_handler(StarletteHTTPException)
    async def starlette_http_exception_handler(
        request: Request,
        exc: StarletteHTTPException,
    ):
        return JSONResponse(
            status_code=exc.status_code,
            content=_error_payload(
                request,
                _http_error_code(exc.status_code),
                _http_error_message(exc.detail, exc.status_code),
            ),
        )

    @app.exception_handler(Exception)
    async def unhandled_exception_handler(request: Request, exc: Exception):
        return JSONResponse(
            status_code=500,
            content=_error_payload(request, "INTERNAL_ERROR", str(exc)),
        )

    app.include_router(health_router, prefix="/api")
    app.include_router(models_router, prefix="/api")
    app.include_router(arena_router, prefix="/api")
    app.include_router(tasks_router, prefix="/api")

    frontend_dist = _resolve_frontend_dist()
    if frontend_dist is not None:
        assets_dir = frontend_dist / "assets"
        if assets_dir.is_dir():
            app.mount("/assets", StaticFiles(directory=str(assets_dir)), name="assets")

        @app.get("/", include_in_schema=False)
        def serve_frontend_index() -> FileResponse:
            return FileResponse(frontend_dist / "index.html")

        @app.get("/{full_path:path}", include_in_schema=False)
        def serve_frontend_fallback(full_path: str) -> FileResponse:
            if full_path.startswith("api/"):
                raise HTTPException(status_code=404)

            candidate = (frontend_dist / full_path).resolve()
            frontend_root = frontend_dist.resolve()
            if candidate.is_file() and frontend_root in candidate.parents:
                return FileResponse(candidate)

            return FileResponse(frontend_dist / "index.html")

    return app


app = create_app()
