"""OCR Backend module.

This module contains OCR backend implementations for different models.
"""

from .base import BaseBackend
from .glm_ocr_backend import GLMOCRBackend
from .paddle_ocr_backend import PaddleOCRBackend

__all__ = ["BaseBackend", "GLMOCRBackend", "PaddleOCRBackend"]
