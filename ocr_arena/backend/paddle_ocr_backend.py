"""PaddleOCR-VL backend based on transformers."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Dict, List, Optional

from PIL import Image

from ..utils.logging import get_logger, get_profiler
from .base import BaseBackend

if TYPE_CHECKING:
    from ..config import PaddleOCRBackendConfig, PageLoaderConfig

logger = get_logger(__name__)


_DEFAULT_TASK_TYPE_MAPPING = {
    "text": "ocr",
    "ocr": "ocr",
    "table": "table",
    "formula": "formula",
    "image": "chart",
    "chart": "chart",
    "spotting": "spotting",
    "seal": "seal",
}

class PaddleOCRBackend(BaseBackend):
    """Run OCR with a local PaddleOCR-VL processor/model pair."""

    name = "PaddleOCR-VL"

    def __init__(
        self,
        config: "PaddleOCRBackendConfig",
        page_loader_config: Optional["PageLoaderConfig"] = None,
    ):
        super().__init__(
            max_new_tokens=int(config.max_new_tokens),
            skip_special_tokens=bool(config.skip_special_tokens),
            page_loader_config=page_loader_config,
        )

        self.model_dir = config.model_dir
        self.required_model_files = frozenset(
            item for item in (str(p).strip() for p in config.required_model_files) if item
        )
        self.torch_dtype = config.torch_dtype
        self.device = config.device
        self.trust_remote_code = bool(config.trust_remote_code)
        self.profiler = get_profiler(__name__)

        self.spotting_upscale_threshold = int(config.spotting_upscale_threshold)
        self.default_max_pixels = int(config.default_max_pixels)
        self.spotting_max_pixels = int(config.spotting_max_pixels)

        raw_mapping = config.task_type_mapping or {}
        self.task_type_mapping: Dict[str, str] = {
            **_DEFAULT_TASK_TYPE_MAPPING,
            **{str(k): str(v) for k, v in raw_mapping.items()},
        }

    def _normalize_task_type(self, task_type: str) -> str:
        mapped = self.task_type_mapping.get(str(task_type), str(task_type))
        mapped = str(mapped).strip().lower()
        if mapped in {"text", "doc", "document"}:
            return "ocr"
        if mapped not in {"ocr", "table", "formula", "chart", "spotting", "seal"}:
            return "ocr"
        return mapped

    def _load_processor_and_model(self) -> None:
        """Load Paddle processor/model with compatibility patch for config schema."""
        from transformers import AutoConfig, AutoModelForImageTextToText, AutoProcessor
        import torch

        self._torch = torch
        model_dir = self._resolve_model_dir()
        runtime_device = self._resolve_device_strategy()

        logger.info(
            "Loading local %s backend from %s (runtime_device=%s)",
            self.name,
            model_dir,
            runtime_device,
        )

        self._processor = AutoProcessor.from_pretrained(
            str(model_dir),
            trust_remote_code=self.trust_remote_code,
        )

        model_config = AutoConfig.from_pretrained(
            str(model_dir),
            trust_remote_code=self.trust_remote_code,
        )
        if not hasattr(model_config, "text_config"):
            setattr(model_config, "text_config", model_config)

        self._model = AutoModelForImageTextToText.from_pretrained(
            pretrained_model_name_or_path=str(model_dir),
            config=model_config,
            torch_dtype=self._resolve_torch_dtype(runtime_device),
            trust_remote_code=self.trust_remote_code,
        )
        self._model.eval()

        self._runtime_device = runtime_device
        if runtime_device is not None:
            self._model = self._model.to(runtime_device)

        self._input_device = self._resolve_input_device()

        logger.info(
            "Local %s backend ready (runtime_device=%s, input_device=%s)",
            self.name,
            self._runtime_device,
            self._input_device,
        )

    def _resize_for_spotting(self, image: Image.Image) -> Image.Image:
        width, height = image.size
        if width >= self.spotting_upscale_threshold or height >= self.spotting_upscale_threshold:
            return image

        try:
            resample_filter = Image.Resampling.LANCZOS
        except AttributeError:
            resample_filter = Image.LANCZOS
        return image.resize((width * 2, height * 2), resample_filter)

    def _prepare_messages_for_task(
        self,
        messages: List[Dict[str, Any]],
        paddle_task: str,
    ) -> List[Dict[str, Any]]:
        if paddle_task != "spotting":
            return messages

        prepared_messages: List[Dict[str, Any]] = []
        for msg in messages:
            content = msg.get("content")
            if not isinstance(content, list):
                prepared_messages.append(msg)
                continue

            new_content: List[Any] = []
            for item in content:
                if (
                    isinstance(item, dict)
                    and item.get("type") == "image"
                    and isinstance(item.get("image"), Image.Image)
                ):
                    new_item = dict(item)
                    new_item["image"] = self._resize_for_spotting(item["image"])
                    new_content.append(new_item)
                else:
                    new_content.append(item)

            prepared_messages.append({**msg, "content": new_content})

        return prepared_messages

    def _process_inputs(
        self,
        messages: List[Dict[str, Any]],
        task_type: Optional[str] = None,
    ) -> Any:
        if task_type is None or not str(task_type).strip():
            raise ValueError("PaddleOCRBackend requires explicit task_type.")

        paddle_task = self._normalize_task_type(task_type)
        prepared_messages = self._prepare_messages_for_task(messages, paddle_task)
        max_pixels = (
            self.spotting_max_pixels
            if paddle_task == "spotting"
            else self.default_max_pixels
        )

        image_processor = getattr(self._processor, "image_processor", None)
        shortest_edge = getattr(image_processor, "min_pixels", None)

        with self.profiler.measure("paddle_ocr_apply_chat_template"):
            base_kwargs: Dict[str, Any] = {
                "add_generation_prompt": True,
                "tokenize": True,
                "return_dict": True,
                "return_tensors": "pt",
            }

            if shortest_edge is not None:
                try:
                    return self._processor.apply_chat_template(
                        prepared_messages,
                        **base_kwargs,
                        images_kwargs={
                            "size": {
                                "shortest_edge": shortest_edge,
                                "longest_edge": max_pixels,
                            }
                        },
                    )
                except TypeError:
                    logger.debug(
                        "Processor does not support images_kwargs in apply_chat_template; "
                        "falling back to default image sizing."
                    )

            return self._processor.apply_chat_template(
                prepared_messages,
                **base_kwargs,
            )
