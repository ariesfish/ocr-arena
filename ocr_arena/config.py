"""Configuration models and loaders.  """

from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional, Union

import yaml
from pydantic import BaseModel, ConfigDict, Field


class _BaseConfig(BaseModel):
    model_config = ConfigDict(extra="allow")


class ServerConfig(_BaseConfig):
    host: str = "0.0.0.0"
    port: int = 5002
    debug: bool = False


class LoggingConfig(_BaseConfig):
    level: str = "INFO"  # DEBUG, INFO, WARNING, ERROR
    format: Optional[str] = None


class PageLoaderConfig(_BaseConfig):
    max_tokens: int = 16384
    temperature: float = 0.01
    top_p: float = 0.00001
    top_k: int = 1
    repetition_penalty: float = 1.1

    t_patch_size: int = 2
    patch_expand_factor: int = 1
    image_expect_length: int = 6144
    image_format: str = "JPEG"
    min_pixels: int = 112 * 112
    max_pixels: int = 14 * 14 * 4 * 1280

    default_prompt: str = (
        "Recognize the text in the image and output in Markdown format. "
        "Preserve the original layout (headings/paragraphs/tables/formulas). "
        "Do not fabricate content that does not exist in the image."
    )
    task_prompt_mapping: Optional[Dict[str, str]] = None

    pdf_dpi: int = 200
    pdf_max_pages: Optional[int] = None
    pdf_verbose: bool = False


class ResultFormatterConfig(_BaseConfig):
    filter_nested: bool = True
    min_overlap_ratio: float = 0.8
    output_format: str = "both"  # json | markdown | both
    label_visualization_mapping: Dict[str, Any] = Field(default_factory=dict)


class OutputConfig(_BaseConfig):
    """Output storage configuration."""

    base_output_dir: str = "./output"


class LayoutConfig(_BaseConfig):
    model_dir: str = "./models/pp-doclayout-v3"
    threshold: float = 0.4
    threshold_by_class: Optional[Dict[Union[int, str], float]] = None
    id2label: Dict[Union[int, str], str] = Field(default_factory=dict)
    batch_size: int = 8
    workers: int = 1
    cuda_visible_devices: str = "0"
    img_size: Optional[int] = None
    layout_nms: bool = True
    layout_unclip_ratio: Optional[Any] = None
    layout_merge_bboxes_mode: Union[str, Dict[int, str]] = "large"
    label_task_mapping: Optional[Dict[str, Any]] = None


class GLMOCRBackendConfig(_BaseConfig):
    enabled: bool = False
    model_dir: str = ""
    required_model_files: List[str] = Field(min_length=1)
    download_source: Literal["modelscope", "huggingface"] = "modelscope"
    modelscope_repo_id: str = "ZhipuAI/GLM-OCR"
    huggingface_repo_id: str = "zai-org/GLM-OCR"
    max_new_tokens: int = 4096
    torch_dtype: str = "bfloat16"
    device: str = "auto"
    trust_remote_code: bool = False
    skip_special_tokens: bool = True


class PaddleOCRBackendConfig(_BaseConfig):
    """PaddleOCR-VL backend configuration.

    Uses PaddleOCR's Vision-Language model for OCR tasks.
    Compatible with PP-DocLayout-V3 for layout detection.
    """
    enabled: bool = False
    model_dir: str = ""
    required_model_files: List[str] = Field(min_length=1)
    download_source: Literal["modelscope", "huggingface"] = "modelscope"
    modelscope_repo_id: str = "PaddlePaddle/PaddleOCR-VL-1.5"
    huggingface_repo_id: str = "PaddlePaddle/PaddleOCR-VL-1.5"
    max_new_tokens: int = 4096
    skip_special_tokens: bool = True
    trust_remote_code: bool = False

    # Device: auto | gpu | cuda | cuda:0 | mps | cpu
    device: str = ""

    # DType strategy: float16 | bfloat16 | float32
    torch_dtype: str = "bfloat16"

    # Paddle task mapping from pipeline task_type
    task_type_mapping: Optional[Dict[str, str]] = None

    # Spotting-specific image preprocessing
    spotting_upscale_threshold: int = 1500
    default_max_pixels: int = 1280 * 28 * 28
    spotting_max_pixels: int = 2048 * 28 * 28


class PipelineConfig(_BaseConfig):
    enable_layout: bool = True

    page_loader: PageLoaderConfig = Field(default_factory=PageLoaderConfig)
    glm_ocr_backend: Optional[GLMOCRBackendConfig] = None
    paddle_ocr_backend: Optional[PaddleOCRBackendConfig] = None
    result_formatter: ResultFormatterConfig = Field(
        default_factory=ResultFormatterConfig
    )
    output: OutputConfig = Field(default_factory=OutputConfig)
    layout: LayoutConfig = Field(default_factory=LayoutConfig)

    # OCR compare execution mode across two backends (GLM vs Paddle).
    # - auto: parallel only when runtime_device=cuda and GPU memory is sufficient.
    # - serial: always run two OCR backends one by one.
    # - parallel: force parallel intent, but still falls back to serial when
    #   runtime_device / VRAM requirements are not met.
    ocr_execution_mode: Literal["auto", "serial", "parallel"] = "auto"

    # Minimum GPU VRAM requirement to allow dual-backend parallel OCR compare.
    # Effective only when ocr_execution_mode is auto/parallel.
    ocr_parallel_min_vram_gb: float = 8.0

    # Keep one model instance per backend on personal machines.
    # Replicas > 1 are not supported in arena compare path.
    ocr_backend_replicas: Literal[1] = 1

    # Queue sizes for async pipeline.
    page_maxsize: int = 100
    region_maxsize: Optional[int] = None


class WebConfig(_BaseConfig):
    host: str = "127.0.0.1"
    port: int = 8000
    upload_max_mb: int = 10
    allowed_image_suffixes: List[str] = Field(
        default_factory=lambda: ["jpg", "jpeg", "png", "webp", "bmp", "gif"]
    )
    task_ttl_seconds: int = 1800
    model_timeout_seconds: int = 120
    ocr_worker_concurrency: Literal[1] = 1
    task_poll_interval_seconds: float = 1.5


class Config(_BaseConfig):
    """Top-level config model."""

    server: ServerConfig = Field(default_factory=ServerConfig)
    logging: LoggingConfig = Field(default_factory=LoggingConfig)
    pipeline: PipelineConfig = Field(default_factory=PipelineConfig)
    web: WebConfig = Field(default_factory=WebConfig)

    @classmethod
    def default_path(cls) -> str:
        return str(Path(__file__).with_name("config.yaml"))

    @classmethod
    def from_yaml(cls, path: Optional[Union[str, Path]] = None) -> "Config":
        resolved_path = resolve_config_path(path)
        data = yaml.safe_load(resolved_path.read_text(encoding="utf-8")) or {}
        return cls.model_validate(data)

    def to_dict(self) -> Dict[str, Any]:
        return self.model_dump()


def load_config(path: Optional[Union[str, Path]] = None) -> Config:
    """Load config from YAML file.

    Args:
        path: Path to YAML config file. Defaults to config.yaml in the package directory.

    Returns:
        Config instance.
    """
    return Config.from_yaml(path)


def resolve_config_path(path: Optional[Union[str, Path]] = None) -> Path:
    """Resolve config YAML path with practical fallbacks.

    Resolution order:
    1) Explicit ``path`` if provided (relative to cwd, then package dir fallback).
    2) ``OCR_ARENA_CONFIG`` env var (same relative handling).
    3) ``./config.yaml``
    4) ``./ocr_arena/config.yaml``
    5) packaged default ``ocr_arena/config.yaml``
    """

    package_dir = Path(__file__).resolve().parent
    package_default = package_dir / "config.yaml"

    checked: List[Path] = []

    def _append_candidates(raw: Union[str, Path]) -> None:
        candidate = Path(raw).expanduser()
        checked.append(candidate)
        if not candidate.is_absolute():
            checked.append(package_dir / candidate)

    if path is not None:
        _append_candidates(path)
    else:
        env_path = os.getenv("OCR_ARENA_CONFIG")
        if env_path:
            _append_candidates(env_path)

        checked.append(Path.cwd() / "config.yaml")
        checked.append(Path.cwd() / "ocr_arena" / "config.yaml")
        checked.append(package_default)

    dedup: List[Path] = []
    seen: set[str] = set()
    for candidate in checked:
        resolved = candidate.resolve(strict=False)
        key = str(resolved)
        if key in seen:
            continue
        seen.add(key)
        dedup.append(resolved)
        if resolved.is_file():
            return resolved

    searched = ", ".join(str(item) for item in dedup)
    raise FileNotFoundError(f"Config file not found. Checked: {searched}")
