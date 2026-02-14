from __future__ import annotations

import shlex
import subprocess
from pathlib import Path
from threading import Thread
from typing import Dict, List, Optional, Tuple

from ...config import Config
from ...utils.logging import get_logger
from ..errors import ArenaAPIError
from ..stores.task_store import TaskStore

logger = get_logger(__name__)

MODEL_NAME_MAPPING = {
    "paddle": "PaddleOCR-VL-1.5",
    "glm": "GLM-OCR",
}


class ModelManager:
    def __init__(self, config: Config, task_store: TaskStore):
        self._config = config
        self._task_store = task_store

    def _model_config(self, model_key: str):
        if model_key == "paddle":
            return self._config.pipeline.paddle_ocr_backend
        if model_key == "glm":
            return self._config.pipeline.glm_ocr_backend
        raise ArenaAPIError(
            status_code=400,
            code="INVALID_MODEL_KEY",
            message=f"Unsupported model_key: {model_key}",
        )

    def _resolve_model_dir(self, model_dir: str) -> Path:
        raw = Path(model_dir).expanduser()
        if raw.is_absolute():
            return raw.resolve()
        return (Path.cwd() / raw).resolve()

    def _required_files_status(
        self, model_key: str
    ) -> Tuple[Path, List[str], List[str]]:
        model_config = self._model_config(model_key)
        if model_config is None or not bool(getattr(model_config, "enabled", False)):
            raise ArenaAPIError(
                status_code=404,
                code="MODEL_DISABLED",
                message=f"Model is disabled in config: {model_key}",
            )

        required_files = [
            str(item).strip()
            for item in list(getattr(model_config, "required_model_files", []) or [])
            if str(item).strip()
        ]
        if not required_files:
            raise ArenaAPIError(
                status_code=500,
                code="MODEL_CONFIG_ERROR",
                message=f"required_model_files is empty for model: {model_key}",
            )

        model_dir = self._resolve_model_dir(str(getattr(model_config, "model_dir", "")))
        existing = []
        for filename in required_files:
            if (model_dir / filename).is_file():
                existing.append(filename)
        return model_dir, required_files, existing

    def get_models_status(self) -> Dict[str, Dict]:
        result: Dict[str, Dict] = {}
        for model_key in ("paddle", "glm"):
            name = MODEL_NAME_MAPPING[model_key]
            try:
                model_dir, required_files, existing_files = self._required_files_status(
                    model_key
                )
                active_download = self._task_store.latest_download_task(model_key)
                if active_download is not None and active_download.status in {
                    "queued",
                    "running",
                }:
                    result[model_key] = {
                        "name": name,
                        "status": "downloading",
                        "progress": int(active_download.progress),
                        "message": active_download.message,
                    }
                    continue

                if not model_dir.exists():
                    result[model_key] = {
                        "name": name,
                        "status": "not_ready",
                        "progress": 0,
                        "message": f"model_dir not found: {model_dir}",
                    }
                    continue

                if len(existing_files) == len(required_files):
                    result[model_key] = {
                        "name": name,
                        "status": "ready",
                        "progress": 100,
                        "message": None,
                    }
                    continue

                progress = int((len(existing_files) / len(required_files)) * 100)
                missing = [
                    item for item in required_files if item not in existing_files
                ]
                result[model_key] = {
                    "name": name,
                    "status": "not_ready",
                    "progress": progress,
                    "message": f"missing files: {', '.join(missing)}",
                }
            except ArenaAPIError as exc:
                result[model_key] = {
                    "name": name,
                    "status": "error",
                    "progress": 0,
                    "message": exc.message,
                }
            except Exception as exc:
                result[model_key] = {
                    "name": name,
                    "status": "error",
                    "progress": 0,
                    "message": str(exc),
                }
        return result

    def assert_models_ready(self) -> None:
        statuses = self.get_models_status()
        not_ready = [key for key, val in statuses.items() if val["status"] != "ready"]
        if not_ready:
            details = "; ".join(
                f"{key}: {statuses[key].get('message') or statuses[key]['status']}"
                for key in not_ready
            )
            raise ArenaAPIError(
                status_code=409,
                code="MODEL_NOT_READY",
                message=details,
            )

    def build_download_command(
        self,
        model_key: str,
        source: str,
        files: Optional[List[str]] = None,
    ) -> List[str]:
        model_config = self._model_config(model_key)
        if model_config is None:
            raise ArenaAPIError(
                status_code=404,
                code="MODEL_NOT_FOUND",
                message=f"Unknown model config: {model_key}",
            )

        required_files = files or [
            str(item).strip()
            for item in list(getattr(model_config, "required_model_files", []) or [])
            if str(item).strip()
        ]
        model_dir = self._resolve_model_dir(str(getattr(model_config, "model_dir", "")))
        model_dir.mkdir(parents=True, exist_ok=True)

        if source == "modelscope":
            repo_id = str(getattr(model_config, "modelscope_repo_id", "")).strip()
            return [
                "modelscope",
                "download",
                "--model",
                repo_id,
                *required_files,
                "--local_dir",
                str(model_dir),
            ]

        if source == "huggingface":
            repo_id = str(getattr(model_config, "huggingface_repo_id", "")).strip()
            return [
                "hf",
                "download",
                repo_id,
                *required_files,
                "--local-dir",
                str(model_dir),
            ]

        raise ArenaAPIError(
            status_code=400,
            code="INVALID_DOWNLOAD_SOURCE",
            message=f"Unsupported download source: {source}",
        )

    def _run_download_task(
        self,
        task_id: str,
        model_key: str,
        source: str,
        request_id: str,
    ) -> None:
        try:
            model_dir, required_files, existing_files = self._required_files_status(
                model_key
            )
            self._task_store.update_task(
                task_id,
                status="running",
                stage="prepare",
                progress=int((len(existing_files) / len(required_files)) * 100),
                message=f"target dir: {model_dir}",
            )

            total = len(required_files)
            done = 0

            for file_name in required_files:
                target_file = model_dir / file_name
                if target_file.is_file():
                    done += 1
                    self._task_store.update_task(
                        task_id,
                        stage="downloading",
                        progress=int((done / total) * 100),
                        message=f"already exists: {file_name}",
                    )
                    continue

                command = self.build_download_command(
                    model_key, source, files=[file_name]
                )
                self._task_store.update_task(
                    task_id,
                    stage="downloading",
                    progress=int((done / total) * 100),
                    message=f"downloading {file_name}",
                )
                logger.info("Running download command: %s", shlex.join(command))

                proc = subprocess.run(
                    command,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    text=True,
                    check=False,
                )
                if proc.returncode != 0:
                    stderr_tail = (proc.stderr or proc.stdout or "").strip()[-400:]
                    self._task_store.mark_failed(
                        task_id,
                        code="DOWNLOAD_FAILED",
                        message=(
                            f"download failed for {file_name}. exit={proc.returncode}. "
                            f"detail={stderr_tail}"
                        ),
                        request_id=request_id,
                    )
                    return
                done += 1
                self._task_store.update_task(
                    task_id,
                    stage="downloading",
                    progress=int((done / total) * 100),
                    message=f"downloaded {file_name}",
                )

            _, required_files_final, existing_files_final = self._required_files_status(
                model_key
            )
            missing = [
                item
                for item in required_files_final
                if item not in existing_files_final
            ]
            if missing:
                self._task_store.mark_failed(
                    task_id,
                    code="MODEL_VALIDATION_FAILED",
                    message=f"missing required files after download: {', '.join(missing)}",
                    request_id=request_id,
                )
                return

            self._task_store.mark_succeeded(
                task_id,
                result={
                    "model_key": model_key,
                    "source": source,
                    "model_dir": str(model_dir),
                },
                message="download completed",
            )
        except ArenaAPIError as exc:
            self._task_store.mark_failed(
                task_id,
                code=exc.code,
                message=exc.message,
                request_id=request_id,
            )
        except Exception as exc:
            self._task_store.mark_failed(
                task_id,
                code="DOWNLOAD_FAILED",
                message=str(exc),
                request_id=request_id,
            )

    def create_download_task(
        self,
        model_key: str,
        request_id: str,
        source: Optional[str] = None,
    ) -> str:
        if self._task_store.has_active_download(model_key):
            raise ArenaAPIError(
                status_code=409,
                code="DOWNLOAD_IN_PROGRESS",
                message=f"Download already running for model: {model_key}",
            )

        model_config = self._model_config(model_key)
        if model_config is None or not bool(getattr(model_config, "enabled", False)):
            raise ArenaAPIError(
                status_code=404,
                code="MODEL_DISABLED",
                message=f"Model is disabled in config: {model_key}",
            )

        statuses = self.get_models_status()
        if statuses.get(model_key, {}).get("status") == "ready":
            raise ArenaAPIError(
                status_code=409,
                code="MODEL_ALREADY_READY",
                message=f"Model already ready: {model_key}",
            )

        resolved_source = source or str(
            getattr(model_config, "download_source", "modelscope")
        )
        task = self._task_store.create_task(
            task_type="download",
            request_id=request_id,
            model_key=model_key,
            task_id=f"dl_{model_key}_{self._task_store._gen_task_id('task')}",
        )
        thread = Thread(
            target=self._run_download_task,
            args=(task.task_id, model_key, resolved_source, request_id),
            daemon=True,
        )
        thread.start()
        return task.task_id
