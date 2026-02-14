"""GLM-OCR backend client based on transformers."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Dict, List, Optional

from ..utils.logging import get_logger, get_profiler
from .base import BaseBackend

if TYPE_CHECKING:
    from ..config import GLMOCRBackendConfig, PageLoaderConfig

logger = get_logger(__name__)


class GLMOCRBackend(BaseBackend):
    """Run OCR with a local GLM-OCR processor/model pair."""

    name = "GLM-OCR"

    def __init__(
        self,
        config: "GLMOCRBackendConfig",
        page_loader_config: Optional["PageLoaderConfig"] = None,
    ):
        """Initialize GLM-OCR backend.

        Args:
            config: GLMOCRBackendConfig instance.
            page_loader_config: Optional PageLoaderConfig for prompts.
        """
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

    def _process_inputs(
        self,
        messages: List[Dict[str, Any]],
        task_type: Optional[str] = None,
    ) -> Any:
        """Process messages into GLM model inputs."""
        _ = task_type
        with self.profiler.measure("glm_ocr_apply_chat_template"):
            inputs = self._processor.apply_chat_template(
                messages,
                tokenize=True,
                add_generation_prompt=True,
                return_dict=True,
                return_tensors="pt",
            )
        return inputs

    def _generate_from_inputs(self, inputs: Any) -> Any:
        """Run GLM model generation."""
        with self._torch.inference_mode():
            with self.profiler.measure("glm_ocr_generate"):
                generated_ids = self._model.generate(
                    **inputs,
                    max_new_tokens=self.max_new_tokens,
                )
        return generated_ids
