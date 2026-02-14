"""OCR Backend base classes."""

from __future__ import annotations

import os
from abc import ABC, abstractmethod
from pathlib import Path
from threading import Lock
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Tuple

from PIL import Image

from ..utils.logging import ProfileLogger, get_logger

if TYPE_CHECKING:
    from ..config import PageLoaderConfig

logger = get_logger(__name__)

# Default OCR prompt
DEFAULT_PROMPT = (
    "Recognize the text in the image and output in Markdown format. "
    "Preserve the original layout (headings/paragraphs/tables/formulas). "
    "Do not fabricate content that does not exist in the image."
)


class BaseBackend(ABC):
    """Base class for OCR backends.

    Defines a unified interface for OCR processing with various backends
    (GLM-OCR, PaddleOCR-VL, etc.).
    """

    # Backend name for identification
    name: str = "base"
    required_model_files: frozenset[str] = frozenset()

    def __init__(
        self,
        max_new_tokens: int = 4096,
        skip_special_tokens: bool = True,
        page_loader_config: Optional["PageLoaderConfig"] = None,
    ):
        """Initialize backend.

        Args:
            max_new_tokens: Maximum tokens to generate.
            skip_special_tokens: Whether to skip special tokens in output.
            page_loader_config: Optional page loader config for prompts.
        """
        self.max_new_tokens = max_new_tokens
        self.skip_special_tokens = skip_special_tokens

        # Shared config state used by base helpers and subclass implementations.
        # Keep values pre-populated by subclasses before super().__init__().
        self.model_dir: str = str(getattr(self, "model_dir", "") or "")
        self.required_model_files: frozenset[str] = frozenset(
            getattr(self, "required_model_files", self.required_model_files) or ()
        )
        self.torch_dtype: str = str(getattr(self, "torch_dtype", "bfloat16") or "bfloat16")
        self.device: str = str(getattr(self, "device", "auto") or "auto")
        self.trust_remote_code: bool = bool(getattr(self, "trust_remote_code", False))

        # Task prompt from PageLoaderConfig
        if page_loader_config is not None:
            self.task_prompt_mapping = getattr(page_loader_config, "task_prompt_mapping", None)
            self.default_prompt = getattr(page_loader_config, "default_prompt", DEFAULT_PROMPT)
        else:
            self.task_prompt_mapping = None
            self.default_prompt = DEFAULT_PROMPT

        self._started = False
        self._start_error: Optional[str] = None
        # Model generation is serialized because concurrent generate calls on
        # one model instance are unstable and can corrupt outputs.
        self._generate_lock = Lock()

        # Shared runtime state used by base helpers and subclasses.
        self._processor: Any = None
        self._model: Any = None
        self._torch: Any = None
        self._input_device: Optional[str] = None
        self._runtime_device: Optional[str] = None
        self.profiler: ProfileLogger

    def _is_valid_model_dir(self, model_dir: Path) -> bool:
        """Check whether a directory contains a usable model checkpoint."""
        if not model_dir.is_dir():
            return False

        required_files = set(self.required_model_files)
        file_names = {p.name for p in model_dir.iterdir() if p.is_file()}
        return required_files.issubset(file_names)

    def _candidate_model_dirs(self) -> List[Path]:
        """Build candidate model directories from config path hints."""
        raw_model_dir = getattr(self, "model_dir", None)
        if not isinstance(raw_model_dir, str) or not raw_model_dir.strip():
            raise ValueError(f"{self.__class__.__name__} missing valid model_dir.")

        raw = Path(raw_model_dir).expanduser()
        if raw.is_absolute():
            return [raw]

        candidates = [
            (Path.cwd() / raw),
            (Path(__file__).resolve().parent / raw),
        ]

        dedup: List[Path] = []
        seen: set[str] = set()
        for path in candidates:
            resolved = path.resolve()
            key = str(resolved)
            if key not in seen:
                seen.add(key)
                dedup.append(resolved)
        return dedup

    def _resolve_model_dir(self) -> Path:
        """Resolve usable model directory from config."""
        searched: List[str] = []

        for candidate in self._candidate_model_dirs():
            searched.append(str(candidate))
            if self._is_valid_model_dir(candidate):
                return candidate

        searched_text = ", ".join(searched) if searched else str(getattr(self, "model_dir", ""))
        raise FileNotFoundError(
            f"Cannot find a valid model directory for backend {self.name}. "
            f"Checked: {searched_text}"
        )

    def _resolve_explicit_runtime_device(
        self,
        raw_device: str,
    ) -> Optional[str]:
        """Resolve explicit runtime device strings to a torch device name."""
        normalized = raw_device.lower()

        if normalized in {"", "auto"}:
            return self._resolve_auto_runtime_device()

        if normalized in {"mps", "apple_mps", "metal"}:
            if not self._torch.backends.mps.is_built() or not self._torch.backends.mps.is_available():
                raise RuntimeError(
                    "MPS device requested but unavailable. "
                    "Please check PyTorch MPS support on this machine."
                )
            os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")
            return "mps"

        if normalized == "cpu":
            return "cpu"

        if normalized in {"gpu", "cuda"}:
            if not self._torch.cuda.is_available():
                raise RuntimeError("GPU/CUDA requested but CUDA is unavailable.")
            return "cuda"

        if normalized.startswith("cuda"):
            if not self._torch.cuda.is_available():
                raise RuntimeError(
                    f"CUDA device requested ({raw_device}) but CUDA is unavailable."
                )
            return raw_device

        return None

    def _resolve_auto_runtime_device(self) -> str:
        """Resolve auto runtime device by local hardware availability."""
        if self._torch.cuda.is_available():
            return "cuda"
        if self._torch.backends.mps.is_built() and self._torch.backends.mps.is_available():
            os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")
            return "mps"
        return "cpu"

    def _resolve_torch_dtype(
        self,
        runtime_device: Optional[str],
    ):
        """Resolve torch dtype string config to torch dtype object."""
        raw = str(getattr(self, "torch_dtype", "")).strip().lower()

        if raw in {"", "auto"}:
            raw = "bfloat16"

        mapping = {
            "float16": self._torch.float16,
            "fp16": self._torch.float16,
            "bfloat16": self._torch.bfloat16,
            "bf16": self._torch.bfloat16,
            "float32": self._torch.float32,
            "fp32": self._torch.float32,
        }

        if raw not in mapping:
            raise ValueError(
                f"Unsupported torch_dtype={self.torch_dtype}. "
                f"Use one of: bfloat16, float16, float32."
            )

        dtype = mapping[raw]
        if runtime_device == "mps" and dtype == self._torch.bfloat16:
            logger.warning(
                "bfloat16 on MPS is not stable for all ops; falling back to float16."
            )
            return self._torch.float16

        return dtype

    def _resolve_device_strategy(self) -> Optional[str]:
        """Resolve runtime-device strategy from config device."""
        raw_device = str(self.device).strip()

        runtime_device = self._resolve_explicit_runtime_device(
            raw_device,
        )
        if runtime_device is not None:
            return runtime_device

        raise ValueError(
            f"Unsupported paddle_ocr_backend.device={self.device}. "
            "Use one of: auto, gpu, cuda, cuda:0, mps, cpu."
        )

    def _extract_image_item(self, content_item: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Convert one OpenAI-style image item into backend-specific image item."""
        if "image" in content_item and content_item["image"] is not None:
            return {"type": "image", "image": content_item["image"]}

        if "url" in content_item and content_item["url"]:
            return {"type": "image", "url": content_item["url"]}

        image_url = content_item.get("image_url")
        if isinstance(image_url, dict):
            image_url = image_url.get("url")
        if image_url:
            return {"type": "image", "url": image_url}

        return None

    def _convert_messages(self, request_data: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Convert OpenAI-style messages into backend-specific chat messages.

        Args:
            request_data: OpenAI-compatible request with messages containing images.

        Returns:
            List of converted messages.

        Raises:
            ValueError: If messages are invalid or contain no images.
        """
        messages = request_data.get("messages")
        if not isinstance(messages, list) or not messages:
            raise ValueError("request_data.messages must be a non-empty list.")

        converted: List[Dict[str, Any]] = []
        has_image = False

        for msg in messages:
            role = msg.get("role", "user")
            content = msg.get("content", "")

            if isinstance(content, str):
                converted.append({"role": role, "content": content})
                continue

            if not isinstance(content, list):
                raise ValueError(f"Invalid content type for role={role}.")

            new_content: List[Dict[str, Any]] = []
            for item in content:
                if not isinstance(item, dict):
                    continue

                item_type = item.get("type")
                if item_type == "text":
                    new_content.append({"type": "text", "text": str(item.get("text", ""))})
                elif item_type in {"image", "image_url"}:
                    image_item = self._extract_image_item(item)
                    if image_item is not None:
                        has_image = True
                        new_content.append(image_item)

            converted.append({"role": role, "content": new_content})

        if not has_image:
            raise ValueError("No image found in request_data.messages.")

        return converted

    def _get_prompt(self, task_type: str) -> str:
        """Get prompt text for the given task type.

        Args:
            task_type: Task type (text/table/formula/image, etc.).

        Returns:
            Prompt text for the task.
        """
        if self.task_prompt_mapping:
            prompt = self.task_prompt_mapping.get(task_type)
            if prompt:
                return prompt
        return self.default_prompt

    def _load_processor_and_model(self):
        """Load backend-specific processor and model. Called by start()."""
        from transformers import AutoModelForImageTextToText, AutoProcessor
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

        self._model = AutoModelForImageTextToText.from_pretrained(
            pretrained_model_name_or_path=str(model_dir),
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

    def _get_input_device(self) -> str:
        """Get the device for model inputs."""
        return str(getattr(self, "_input_device", "cpu") or "cpu")

    def _resolve_input_device(self) -> str:
        """Resolve input device from runtime target or model parameters."""
        runtime_device = getattr(self, "_runtime_device", None)
        if runtime_device:
            return str(runtime_device)

        model = getattr(self, "_model", None)
        if model is not None:
            try:
                return str(next(model.parameters()).device)
            except Exception:
                pass

        return "cpu"

    @abstractmethod
    def _process_inputs(
        self,
        messages: List[Dict[str, Any]],
        task_type: Optional[str] = None,
    ) -> Any:
        """Process messages into model inputs.

        Args:
            messages: Backend-specific chat messages.
            task_type: Optional explicit task type from pipeline.

        Returns:
            Backend-specific input format for generation.
        """

    def _generate_from_inputs(self, inputs: Any) -> Any:
        """Run model generation with processed inputs."""
        with self._torch.inference_mode():
            with self.profiler.measure("generate"):
                generated_ids = self._model.generate(
                    **inputs,
                    max_new_tokens=self.max_new_tokens,
                )
        return generated_ids

    def _decode_output(self, generated_ids: Any, prompt_len: int) -> str:
        """Decode generated IDs to text."""
        if hasattr(generated_ids, "ndim") and generated_ids.ndim >= 2:
            generated_ids = generated_ids[0]
        elif isinstance(generated_ids, (list, tuple)) and generated_ids:
            generated_ids = generated_ids[0]

        generated_tail = generated_ids[prompt_len:]

        if hasattr(generated_tail, "tolist"):
            generated_tail = generated_tail.tolist()    #type: ignore

        return self._processor.decode(
            generated_tail,
            skip_special_tokens=self.skip_special_tokens,
        )

    def start(self):
        """Start the backend (load model and processor)."""
        if self._started:
            return

        if self._start_error is not None:
            raise RuntimeError(self._start_error)

        try:
            self._load_processor_and_model()
            self._started = True
            self._start_error = None
            logger.info("Backend %s started successfully.", self.name)
        except Exception as e:
            self._started = False
            self._start_error = f"Backend {self.name} startup failed: {e}"
            raise RuntimeError(self._start_error) from e

    def stop(self):
        """Release backend resources."""
        model = getattr(self, "_model", None)
        torch_obj = getattr(self, "_torch", None)
        input_device = str(getattr(self, "_input_device", "") or "")

        if model is not None and torch_obj is not None:
            if input_device.startswith("cuda"):
                try:
                    torch_obj.cuda.empty_cache()
                except Exception:
                    pass
            if input_device == "mps" and hasattr(torch_obj, "mps"):
                try:
                    torch_obj.mps.empty_cache()
                except Exception:
                    pass

        for attr in ("_model", "_processor", "_input_device", "_runtime_device", "_torch"):
            if hasattr(self, attr):
                setattr(self, attr, None)

        self._started = False
        self._start_error = None
        logger.info("Backend %s stopped.", self.name)

    def _generate(
        self,
        messages: List[Dict[str, Any]],
        task_type: Optional[str] = None,
    ) -> Tuple[Dict[str, Any], int]:
        """Internal method to generate response from messages.

        Args:
            messages: List of messages with image and text content.

        Returns:
            Tuple of (response dict with OpenAI-compatible format, status_code).
        """
        with self._generate_lock:
            # Get input device
            input_device = self._get_input_device()

            # Process inputs
            inputs = self._process_inputs(messages, task_type=task_type)
            if hasattr(inputs, "pop"):
                inputs.pop("token_type_ids", None)

            # Move inputs to correct device
            if input_device and hasattr(inputs, "items"):
                inputs = {
                    key: value.to(input_device) if hasattr(value, "to") else value
                    for key, value in inputs.items()
                }

            # Handle prompt length extraction
            if hasattr(inputs, "__getitem__") and "input_ids" in inputs:
                prompt_len = int(inputs["input_ids"].shape[1])
            else:
                prompt_len = 0

            # Run generation
            generated_ids = self._generate_from_inputs(inputs)

            # Decode output in backend-specific implementation.
            output_text = self._decode_output(generated_ids, prompt_len)

        return {"choices": [{"message": {"content": output_text.strip()}}]}, 200

    def process(
        self,
        image: Image.Image,
        task_type: str = "text",
        prompt_text: Optional[str] = None,
    ) -> Tuple[Dict[str, Any], int]:
        """Process a PIL Image directly.

        Args:
            image: PIL Image to process.
            task_type: Task type (text/table/formula/image, etc.).
            prompt_text: Custom prompt text. If None, uses default/task-specific prompt.

        Returns:
            Tuple of (response dict with OpenAI-compatible format, status_code).
        """
        if not self._started:
            self.start()

        # Get prompt text
        if prompt_text is None:
            prompt_text = self._get_prompt(task_type)

        # Ensure RGB mode
        if image.mode != "RGB":
            image = image.convert("RGB")

        # Build messages directly with PIL Image (backend-specific)
        messages = self._build_image_messages(image, prompt_text)

        return self._generate(messages, task_type=task_type)

    def _build_image_messages(
        self,
        image: Image.Image,
        prompt_text: str,
    ) -> List[Dict[str, Any]]:
        """Build backend-specific messages for a PIL Image.

        Args:
            image: PIL Image to process.
            prompt_text: Prompt text to use.

        Returns:
            List of messages in backend-specific format.
        """
        return [{
            "role": "user",
            "content": [
                {"type": "image", "image": image},
                {"type": "text", "text": prompt_text},
            ],
        }]

    def process_from_request(self, request_data: Dict[str, Any]) -> Tuple[Dict[str, Any], int]:
        """Process OCR request from OpenAI-compatible format.

        Args:
            request_data: OpenAI-compatible request with messages containing images.

        Returns:
            Tuple of (response dict, status_code).
        """
        if not self._started:
            self.start()

        try:
            messages = self._convert_messages(request_data)
            return self._generate(messages, task_type="text")
        except Exception as e:
            logger.exception("Backend %s inference failed: %s", self.name, e)
            return {"error": f"{self.name} inference failed: {str(e)}"}, 500

    @property
    def is_started(self) -> bool:
        """Check if backend is started."""
        return self._started
