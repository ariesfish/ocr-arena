from __future__ import annotations

import json
import shutil
import tempfile
import time
from copy import deepcopy
from pathlib import Path
from threading import Event, Lock, Thread
from typing import Any, Callable, Dict, List, Optional, Tuple

from ...config import Config
from ...ocr_pipeline import OCRPipeline
from ...parser_result import PipelineResult
from ...utils.logging import get_logger
from .result_mapper import map_backend_result

logger = get_logger(__name__)


class PipelineArenaRunner:
    def __init__(self, config: Config):
        self._config = config
        self._pipeline: Optional[OCRPipeline] = None
        self._pipeline_lock = Lock()
        self._timeout_seconds = int(config.web.model_timeout_seconds)

    def stop(self) -> None:
        with self._pipeline_lock:
            if self._pipeline is not None:
                self._pipeline.stop()
                self._pipeline = None

    def _ensure_pipeline(self) -> OCRPipeline:
        with self._pipeline_lock:
            if self._pipeline is None:
                self._pipeline = OCRPipeline(self._config.pipeline)
            if not self._pipeline._started:
                self._pipeline.start()
            return self._pipeline

    @staticmethod
    def _safe_model_payload(
        regions: List[Tuple[Any, Dict[str, Any], str, int]], page_count: int
    ) -> List[List[Dict[str, Any]]]:
        payload: List[List[Dict[str, Any]]] = [[] for _ in range(page_count)]
        per_page_index: List[int] = [0 for _ in range(page_count)]
        for _img, info, _task, page_idx in regions:
            page_slot = payload[page_idx]
            page_item = {
                "index": per_page_index[page_idx],
                "label": str(info.get("label", "text")),
                "content": "",
                "bbox_2d": info.get("bbox_2d"),
                "score": info.get("score"),
                "polygon": info.get("polygon"),
            }
            per_page_index[page_idx] += 1
            page_slot.append(page_item)
        return payload

    @staticmethod
    def _run_with_timeout(
        fn: Callable[[], Any],
        timeout_seconds: int,
    ) -> Tuple[bool, Optional[Any], Optional[BaseException], float]:
        done = Event()
        holder: Dict[str, Any] = {}

        def _target() -> None:
            try:
                holder["result"] = fn()
            except BaseException as exc:
                holder["error"] = exc
            finally:
                done.set()

        started_at = time.perf_counter()
        thread = Thread(target=_target, daemon=True)
        thread.start()
        finished = done.wait(timeout=timeout_seconds)
        elapsed_ms = (time.perf_counter() - started_at) * 1000.0

        if not finished:
            return False, None, TimeoutError("model execution timeout"), elapsed_ms

        if "error" in holder:
            return True, None, holder["error"], elapsed_ms
        return True, holder.get("result"), None, elapsed_ms

    def run_compare(
        self,
        source_path: Path,
        request_id: str,
        task_id: Optional[str] = None,
        persist_artifacts: bool = True,
        progress_callback: Optional[Callable[[int, str, Optional[str]], None]] = None,
    ) -> Dict[str, Any]:
        pipeline = self._ensure_pipeline()
        source_path = source_path.expanduser().resolve()
        if not source_path.is_file():
            raise FileNotFoundError(f"Input file does not exist: {source_path}")

        workspace: Optional[Dict[str, Path]] = None
        copied_source = source_path
        resolved_task_id = task_id or f"sync_{request_id[-8:]}"

        if persist_artifacts:
            resolved_task_id, workspace = pipeline._prepare_task_workspace(
                source_path=source_path,
                task_id=task_id,
                output_root_dir=self._config.pipeline.output.base_output_dir,
            )
            copied_source = workspace["input"] / source_path.name
            shutil.copy2(source_path, copied_source)

        if progress_callback is not None:
            progress_callback(10, "input_ready", None)

        with tempfile.TemporaryDirectory(
            prefix="ocr_arena_sync_layout_"
        ) as temp_layout_root:
            layout_dir = (
                str(workspace["layout"])
                if workspace is not None
                else str(Path(temp_layout_root).resolve())
            )

            if progress_callback is not None:
                progress_callback(20, "layout_detection", None)

            _, pages, regions = pipeline._load_source_and_regions(
                source=str(copied_source),
                save_layout_visualization=True,
                layout_vis_output_dir=layout_dir,
            )

            if progress_callback is not None:
                progress_callback(35, "ocr_prepare", f"regions={len(regions)}")

            backend_keys = list(pipeline._ocr_backends.keys())
            if "paddle" not in backend_keys or "glm" not in backend_keys:
                raise ValueError("Both GLM and Paddle backends must be enabled.")

            recognition_by_backend: Dict[str, List[Tuple[int, Dict[str, Any]]]] = {}
            latency_ms: Dict[str, Optional[int]] = {}
            backend_errors: Dict[str, str] = {}
            safe_json = self._safe_model_payload(regions, len(pages))

            should_parallel = pipeline._should_parallel_compare(backend_keys)
            if should_parallel:
                workers: Dict[str, Dict[str, Any]] = {}
                for key in backend_keys:
                    workers[key] = {}

                    def _build_runner(model_key: str) -> Callable[[], Any]:
                        return lambda: pipeline._run_backend_regions(model_key, regions)

                    done = Event()
                    holder: Dict[str, Any] = {}

                    def _target(
                        run_fn: Callable[[], Any], output: Dict[str, Any], event: Event
                    ) -> None:
                        try:
                            output["result"] = run_fn()
                        except BaseException as exc:
                            output["error"] = exc
                        finally:
                            event.set()

                    started_at = time.perf_counter()
                    thread = Thread(
                        target=_target,
                        args=(_build_runner(key), holder, done),
                        daemon=True,
                    )
                    thread.start()
                    workers[key] = {
                        "holder": holder,
                        "done": done,
                        "started_at": started_at,
                    }

                for idx, key in enumerate(backend_keys):
                    worker = workers[key]
                    done_event = worker["done"]
                    finished = done_event.wait(timeout=self._timeout_seconds)
                    elapsed = int((time.perf_counter() - worker["started_at"]) * 1000.0)
                    latency_ms[key] = elapsed
                    if not finished:
                        backend_errors[key] = (
                            f"{key} inference timeout after {self._timeout_seconds}s"
                        )
                        recognition_by_backend[key] = []
                    elif "error" in worker["holder"]:
                        backend_errors[key] = str(worker["holder"]["error"])
                        recognition_by_backend[key] = []
                    else:
                        recognition_by_backend[key] = worker["holder"]["result"]

                    if progress_callback is not None:
                        progress_callback(
                            50 + (idx + 1) * 20,
                            f"{key}_inference",
                            backend_errors.get(key),
                        )
            else:
                for idx, key in enumerate(backend_keys):
                    finished, output, error, elapsed_ms = self._run_with_timeout(
                        lambda current_key=key: pipeline._run_backend_regions(
                            current_key,
                            regions,
                        ),
                        timeout_seconds=self._timeout_seconds,
                    )
                    latency_ms[key] = int(elapsed_ms)
                    if not finished and isinstance(error, TimeoutError):
                        backend_errors[key] = (
                            f"{key} inference timeout after {self._timeout_seconds}s"
                        )
                        recognition_by_backend[key] = []
                    elif error is not None:
                        backend_errors[key] = str(error)
                        recognition_by_backend[key] = []
                    else:
                        recognition_by_backend[key] = output or []

                    if progress_callback is not None:
                        progress_callback(
                            50 + (idx + 1) * 20,
                            f"{key}_inference",
                            backend_errors.get(key),
                        )

                    if idx < len(backend_keys) - 1:
                        try:
                            pipeline._ocr_backends[key].stop()
                        except Exception as exc:
                            logger.warning(
                                "Failed to stop backend %s after serial inference: %s",
                                key,
                                exc,
                            )

            if progress_callback is not None:
                progress_callback(92, "formatting", None)

            mapped_results: Dict[str, Dict[str, Any]] = {}
            model_artifacts: Dict[str, Dict[str, Any]] = {}

            for order_idx, key in enumerate(backend_keys):
                if key in backend_errors:
                    json_payload = deepcopy(safe_json)
                    markdown_payload = ""
                else:
                    grouped = pipeline._group_recognition_results(
                        recognition_results=recognition_by_backend[key],
                        page_count=len(pages),
                    )
                    json_str, markdown_payload = pipeline.result_formatter.process(
                        grouped
                    )
                    json_payload = json.loads(json_str)

                mapped_results[key] = map_backend_result(
                    model_key=key,
                    json_result=json_payload,
                    markdown_result=markdown_payload,
                    latency_ms=latency_ms.get(key),
                    error=backend_errors.get(key),
                )

                if workspace is not None:
                    model_output_root = workspace["results"] / key
                    model_output_root.mkdir(parents=True, exist_ok=True)
                    PipelineResult(
                        json_result=json_payload,
                        markdown_result=markdown_payload,
                        original_images=[str(copied_source)],
                        layout_vis_dir=(
                            str(workspace["layout"]) if order_idx == 0 else None
                        ),
                        layout_image_indices=list(range(len(pages))),
                    ).save(
                        output_dir=str(model_output_root),
                        save_layout_visualization=False,
                    )
                    result_dir = model_output_root / copied_source.stem
                    model_artifacts[key] = {
                        "result_dir": str(result_dir.resolve()),
                        "json": str(
                            (result_dir / f"{copied_source.stem}.json").resolve()
                        ),
                        "markdown": str(
                            (result_dir / f"{copied_source.stem}.md").resolve()
                        ),
                        "imgs_dir": str((result_dir / "imgs").resolve()),
                    }

            manifest_path: Optional[str] = None
            if workspace is not None:
                layout_files = pipeline._collect_layout_files(workspace["layout"])
                manifest = {
                    "task_id": resolved_task_id,
                    "task_dir": str(workspace["task"].resolve()),
                    "input": {
                        "original_path": str(source_path),
                        "copied_path": str(copied_source.resolve()),
                    },
                    "layout": {
                        "dir": str(workspace["layout"].resolve()),
                        "files": layout_files,
                    },
                    "models": model_artifacts,
                }
                manifest_file = workspace["meta"] / "manifest.json"
                manifest_file.write_text(
                    json.dumps(manifest, ensure_ascii=False, indent=2),
                    encoding="utf-8",
                )
                manifest_path = str(manifest_file.resolve())

            width = int(pages[0].width) if pages else None
            height = int(pages[0].height) if pages else None

            return {
                "request_id": request_id,
                "input": {
                    "filename": source_path.name,
                    "width": width,
                    "height": height,
                },
                "paddle": mapped_results.get("paddle")
                or map_backend_result(
                    model_key="paddle",
                    json_result=[],
                    markdown_result="",
                    latency_ms=None,
                    error="paddle backend missing",
                ),
                "glm": mapped_results.get("glm")
                or map_backend_result(
                    model_key="glm",
                    json_result=[],
                    markdown_result="",
                    latency_ms=None,
                    error="glm backend missing",
                ),
                "manifest": manifest_path,
            }
