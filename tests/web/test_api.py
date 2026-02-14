from __future__ import annotations

import base64
import importlib
import json
import tempfile
from pathlib import Path
from types import SimpleNamespace

import pytest
from fastapi.testclient import TestClient

from ocr_arena.config import Config
from ocr_arena.web.errors import ArenaAPIError
from ocr_arena.web.stores.task_store import TaskStore


def _fake_result() -> dict:
    return {
        "request_id": "arena_test_req",
        "input": {"filename": "demo.png", "width": 1024, "height": 768},
        "paddle": {
            "name": "PaddleOCR-VL-1.5",
            "latency_ms": 100,
            "confidence_avg": 0.95,
            "text": "hello paddle",
            "boxes": [
                {
                    "index": 0,
                    "label": "text",
                    "score": 0.95,
                    "bbox_2d": [0, 0, 100, 100],
                    "polygon": None,
                }
            ],
            "error": None,
        },
        "glm": {
            "name": "GLM-OCR",
            "latency_ms": 120,
            "confidence_avg": 0.85,
            "text": "hello glm",
            "boxes": [
                {
                    "index": 0,
                    "label": "text",
                    "score": 0.85,
                    "bbox_2d": [0, 0, 100, 100],
                    "polygon": None,
                }
            ],
            "error": None,
        },
        "manifest": None,
    }


class FakeRunner:
    def __init__(self):
        self.result = _fake_result()

    def run_compare(self, **_: dict) -> dict:
        return self.result

    def stop(self) -> None:
        return None


class FakeQueue:
    def __init__(self, task_store: TaskStore, runner: FakeRunner):
        self.task_store = task_store
        self.runner = runner

    def submit(
        self, *, task_id: str, request_id: str, filename: str, file_bytes: bytes
    ) -> None:
        self.task_store.update_task(
            task_id,
            status="running",
            progress=40,
            stage="queued",
            message=f"received {filename}:{len(file_bytes)}",
        )
        self.task_store.mark_succeeded(
            task_id,
            result=self.runner.run_compare(),
            message="done",
        )

    def stop(self) -> None:
        return None


class FakeModelManager:
    def __init__(self, task_store: TaskStore):
        self.task_store = task_store
        self.ready = True
        self.status_payload = {
            "paddle": {"name": "PaddleOCR-VL-1.5", "status": "ready", "progress": 100},
            "glm": {"name": "GLM-OCR", "status": "ready", "progress": 100},
        }

    def get_models_status(self) -> dict:
        return self.status_payload

    def assert_models_ready(self) -> None:
        if not self.ready:
            raise ArenaAPIError(
                status_code=409,
                code="MODEL_NOT_READY",
                message="glm: not_ready",
            )

    def create_download_task(
        self, model_key: str, request_id: str, source: str | None = None
    ) -> str:
        task = self.task_store.create_task(
            task_type="download",
            request_id=request_id,
            model_key=model_key,
        )
        self.task_store.update_task(
            task.task_id,
            status="running",
            progress=50,
            stage="downloading",
            message=source or "modelscope",
        )
        return task.task_id


@pytest.fixture
def client(monkeypatch: pytest.MonkeyPatch) -> tuple[TestClient, SimpleNamespace]:
    web_app_module = importlib.import_module("ocr_arena.web.app")
    cfg = Config()
    task_store = TaskStore(ttl_seconds=1800)
    runner = FakeRunner()
    model_manager = FakeModelManager(task_store)
    queue = FakeQueue(task_store, runner)
    services = SimpleNamespace(
        config=cfg,
        task_store=task_store,
        arena_runner=runner,
        model_manager=model_manager,
        ocr_queue=queue,
    )

    monkeypatch.setattr(web_app_module, "build_services", lambda config: services)
    app = web_app_module.create_app(cfg)
    with TestClient(app) as test_client:
        yield test_client, services


def test_models_status_ready(client: tuple[TestClient, SimpleNamespace]) -> None:
    test_client, _ = client
    response = test_client.get("/api/models/status")
    assert response.status_code == 200
    payload = response.json()
    assert payload["models"]["paddle"]["status"] == "ready"
    assert payload["models"]["glm"]["status"] == "ready"


def test_create_ocr_job_success(client: tuple[TestClient, SimpleNamespace]) -> None:
    test_client, _ = client
    response = test_client.post(
        "/api/arena/jobs",
        files={"file": ("demo.png", b"png-bytes", "image/png")},
    )
    assert response.status_code == 200
    task_id = response.json()["task_id"]

    task_response = test_client.get(f"/api/tasks/{task_id}")
    assert task_response.status_code == 200
    task_payload = task_response.json()
    assert task_payload["status"] == "succeeded"
    assert task_payload["result"]["paddle"]["name"] == "PaddleOCR-VL-1.5"


def test_create_ocr_job_model_not_ready(
    client: tuple[TestClient, SimpleNamespace],
) -> None:
    test_client, services = client
    services.model_manager.ready = False
    response = test_client.post(
        "/api/arena/jobs",
        files={"file": ("demo.png", b"png-bytes", "image/png")},
    )
    assert response.status_code == 409
    payload = response.json()
    assert payload["code"] == "MODEL_NOT_READY"
    assert "request_id" in payload


def test_invalid_upload_rejected(client: tuple[TestClient, SimpleNamespace]) -> None:
    test_client, _ = client
    response = test_client.post(
        "/api/arena/jobs",
        files={"file": ("demo.txt", b"hello", "text/plain")},
    )
    assert response.status_code == 400
    assert response.json()["code"] == "UNSUPPORTED_FILE_TYPE"


def test_missing_task_returns_410(client: tuple[TestClient, SimpleNamespace]) -> None:
    test_client, _ = client
    response = test_client.get("/api/tasks/not_exists")
    assert response.status_code == 410
    payload = response.json()
    assert payload["code"] in {"TASK_NOT_FOUND", "TASK_EXPIRED"}


def test_unknown_api_route_uses_unified_error(
    client: tuple[TestClient, SimpleNamespace],
) -> None:
    test_client, _ = client
    response = test_client.get("/api/not-found")
    assert response.status_code == 404
    payload = response.json()
    assert payload["code"] == "NOT_FOUND"
    assert payload["message"]
    assert "request_id" in payload


def test_validation_error_uses_unified_error(
    client: tuple[TestClient, SimpleNamespace],
) -> None:
    test_client, _ = client
    response = test_client.post("/api/arena/jobs")
    assert response.status_code == 422
    payload = response.json()
    assert payload["code"] == "INVALID_REQUEST"
    assert payload["message"]
    assert "request_id" in payload


def test_sync_ocr_schema(client: tuple[TestClient, SimpleNamespace]) -> None:
    test_client, _ = client
    response = test_client.post(
        "/api/arena/ocr",
        files={"file": ("demo.png", b"png-bytes", "image/png")},
    )
    assert response.status_code == 200
    payload = response.json()
    assert payload["paddle"]["boxes"][0]["score"] == pytest.approx(0.95)
    assert payload["glm"]["text"] == "hello glm"


def test_download_task_created(client: tuple[TestClient, SimpleNamespace]) -> None:
    test_client, _ = client
    response = test_client.post(
        "/api/models/glm/download", json={"source": "modelscope"}
    )
    assert response.status_code == 200
    payload = response.json()
    assert payload["task_type"] == "download"
    task_id = payload["task_id"]

    task_response = test_client.get(f"/api/tasks/{task_id}")
    assert task_response.status_code == 200
    assert task_response.json()["status"] == "running"


def test_get_task_layout_preview(client: tuple[TestClient, SimpleNamespace]) -> None:
    test_client, services = client
    task = services.task_store.create_task(task_type="ocr", request_id="req_layout")

    png_bytes = base64.b64decode(
        "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mP8/x8AAwMCAO7Z8ioAAAAASUVORK5CYII="
    )

    with tempfile.TemporaryDirectory(prefix="ocr_arena_layout_test_") as tmp_dir:
        root = Path(tmp_dir)
        layout_file = root / "layout_page0.png"
        layout_file.write_bytes(png_bytes)

        manifest = root / "manifest.json"
        manifest.write_text(
            json.dumps({"layout": {"files": [str(layout_file)]}}),
            encoding="utf-8",
        )

        services.task_store.mark_succeeded(
            task.task_id,
            result={"manifest": str(manifest)},
            message="done",
        )

        response = test_client.get(f"/api/tasks/{task.task_id}/layout?page=0")

    assert response.status_code == 200
    assert response.headers.get("content-type", "").startswith("image/")
