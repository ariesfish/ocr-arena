"""Markdown processing and region-crop utilities."""

import ast
import json
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from PIL import Image

from ..utils.image_utils import (
    PYPDFIUM2_AVAILABLE,
    crop_image_region,
    pdf_to_images_pil,
)
from ..utils.logging import get_logger

logger = get_logger(__name__)


def extract_image_refs(markdown_text: str) -> List[Tuple[int, List[int], str]]:
    """Extract image references from Markdown.

    Args:
        markdown_text: Markdown text.

    Returns:
        List of (page_idx, bbox, original_tag).
    """
    # Pattern: ![](page=0,bbox=[57, 199, 884, 444])
    pattern = r"!\[\]\(page=(\d+),bbox=(\[[\d,\s]+\])\)"
    matches = re.finditer(pattern, markdown_text)

    image_refs = []
    for match in matches:
        page_idx = int(match.group(1))
        bbox_str = match.group(2)
        try:
            bbox = ast.literal_eval(bbox_str)
            if not isinstance(bbox, list) or len(bbox) != 4:
                raise ValueError(f"Invalid bbox format: {bbox_str}")
        except (ValueError, SyntaxError) as e:
            logger.warning("Cannot parse bbox %s: %s", bbox_str, e)
            continue
        original_tag = match.group(0)
        image_refs.append((page_idx, bbox, original_tag))

    return image_refs


def _load_original_images(original_images: List[str]) -> List[Image.Image]:
    """Load original image inputs into PIL images (supports PDF)."""
    loaded_images: List[Image.Image] = []

    for img_path in original_images:
        path = Path(img_path)
        suffix = path.suffix.lower()

        if suffix == ".pdf":
            if not PYPDFIUM2_AVAILABLE:
                raise RuntimeError(
                    "PDF support requires pypdfium2. Install: pip install pypdfium2"
                )
            try:
                pdf_images = pdf_to_images_pil(
                    img_path,
                    dpi=200,
                    max_width_or_height=3500,
                )
                loaded_images.extend(pdf_images)
            except Exception as e:
                raise RuntimeError(f"Failed to convert PDF to images: {e}") from e
        else:
            image = Image.open(img_path)
            if image.mode != "RGB":
                image = image.convert("RGB")
            loaded_images.append(image)

    return loaded_images


def _normalize_bbox(bbox: Any) -> Optional[List[int]]:
    """Normalize a bbox value into [x1, y1, x2, y2] int format."""
    if not isinstance(bbox, (list, tuple)) or len(bbox) != 4:
        return None

    try:
        values = [int(round(float(v))) for v in bbox]
    except (TypeError, ValueError):
        return None

    x1, y1, x2, y2 = values
    if x2 <= x1 or y2 <= y1:
        return None

    return [x1, y1, x2, y2]


def _normalize_polygon(polygon: Any) -> Optional[List[List[float]]]:
    """Normalize optional polygon coordinates."""
    if not isinstance(polygon, (list, tuple)) or len(polygon) < 3:
        return None

    normalized: List[List[float]] = []
    for point in polygon:
        if not isinstance(point, (list, tuple)) or len(point) != 2:
            return None
        try:
            px = float(point[0])
            py = float(point[1])
        except (TypeError, ValueError):
            return None
        normalized.append([px, py])

    return normalized


def _sanitize_label(label: str) -> str:
    return re.sub(r"[^0-9A-Za-z_-]+", "_", label).strip("_") or "region"


def _region_image_filename(
    page_idx: int,
    region_idx: int,
    label: str,
    image_prefix: str,
) -> str:
    safe_label = _sanitize_label(label)
    return f"{image_prefix}_page{page_idx}_idx{region_idx}_{safe_label}.jpg"


def _extract_region_refs_from_json(
    json_result: Any,
) -> List[Tuple[int, int, List[int], Optional[List[List[float]]], str]]:
    """Extract (page_idx, region_idx, bbox, polygon, label) from JSON output."""
    if isinstance(json_result, str):
        try:
            json_result = json.loads(json_result)
        except json.JSONDecodeError:
            return []

    if not isinstance(json_result, list):
        return []

    refs: List[Tuple[int, int, List[int], Optional[List[List[float]]], str]] = []

    for page_idx, page_data in enumerate(json_result):
        if isinstance(page_data, dict):
            items = [page_data]
        elif isinstance(page_data, list):
            items = page_data
        else:
            continue

        for region_idx, item in enumerate(items):
            if not isinstance(item, dict):
                continue

            bbox = _normalize_bbox(item.get("bbox_2d"))
            if bbox is None:
                continue

            polygon = _normalize_polygon(item.get("polygon"))
            label = str(item.get("label") or "region")
            refs.append((page_idx, region_idx, bbox, polygon, label))

    return refs


def build_json_region_image_map(
    json_result: Any,
    image_prefix: str = "region",
) -> Dict[Tuple[int, Tuple[int, int, int, int]], List[str]]:
    """Build mapping: (page_idx, bbox) -> exported filename list."""
    refs = _extract_region_refs_from_json(json_result)
    filename_map: Dict[Tuple[int, Tuple[int, int, int, int]], List[str]] = {}

    for page_idx, region_idx, bbox, _, label in refs:
        key = (page_idx, tuple(bbox))
        filename = _region_image_filename(page_idx, region_idx, label, image_prefix)
        filename_map.setdefault(key, []).append(filename)

    return filename_map


def replace_markdown_image_refs_with_json_exports(
    markdown_text: str,
    json_result: Any,
    image_prefix: str = "region",
) -> str:
    """Replace markdown placeholder refs with already-exported JSON crop paths.

    This function only replaces links and does not crop images.
    """
    image_refs = extract_image_refs(markdown_text)
    if not image_refs:
        return markdown_text

    filename_map = build_json_region_image_map(json_result, image_prefix=image_prefix)
    if not filename_map:
        return markdown_text

    result_markdown = markdown_text
    for page_idx, bbox, original_tag in image_refs:
        norm_bbox = _normalize_bbox(bbox)
        if norm_bbox is None:
            continue

        key = (page_idx, tuple(norm_bbox))
        candidates = filename_map.get(key)
        if not candidates:
            continue

        filename = candidates.pop(0)
        relative_path = f"imgs/{filename}"
        new_tag = f"![Image {page_idx}]({relative_path})"
        result_markdown = result_markdown.replace(original_tag, new_tag, 1)

    return result_markdown


def crop_and_replace_images(
    markdown_text: str,
    original_images: List[str],
    output_dir: Path,
    image_prefix: str = "image",
) -> Tuple[str, List[str]]:
    """Crop referenced image regions and replace Markdown tags.

    Args:
        markdown_text: Source Markdown.
        original_images: Original image paths.
        output_dir: Output directory.
        image_prefix: Filename prefix for cropped images.

    Returns:
        (updated_markdown, saved_image_paths)
    """
    image_refs = extract_image_refs(markdown_text)

    if not image_refs:
        return markdown_text, []

    loaded_images = _load_original_images(original_images)

    output_dir.mkdir(parents=True, exist_ok=True)

    result_markdown = markdown_text
    saved_image_paths = []

    for idx, (page_idx, bbox, original_tag) in enumerate(image_refs):
        if page_idx < 0 or page_idx >= len(loaded_images):
            logger.warning(
                "page_idx %d out of range (total %d images), skipping",
                page_idx,
                len(loaded_images),
            )
            continue

        original_image = loaded_images[page_idx]
        try:
            cropped_image = crop_image_region(original_image, bbox)

            image_filename = f"{image_prefix}_page{page_idx}_idx{idx}.jpg"
            image_path = output_dir / image_filename

            cropped_image.save(image_path, quality=95)
            saved_image_paths.append(str(image_path))

            relative_path = f"imgs/{image_filename}"
            new_tag = f"![Image {page_idx}-{idx}]({relative_path})"
            result_markdown = result_markdown.replace(original_tag, new_tag, 1)

        except Exception as e:
            logger.warning(
                "Failed to crop image (page=%d, bbox=%s): %s",
                page_idx,
                bbox,
                e,
            )
            continue

    return result_markdown, saved_image_paths


def export_crops_from_json_result(
    json_result: Any,
    original_images: List[str],
    output_dir: Path,
    image_prefix: str = "region",
) -> List[str]:
    """Export cropped images from JSON bbox/polygon metadata.

    This path does not rely on Markdown image placeholders.

    Args:
        json_result: Structured OCR JSON output.
        original_images: Original image (or PDF) paths.
        output_dir: Directory to save cropped regions.
        image_prefix: Filename prefix.

    Returns:
        List of saved cropped image paths.
    """
    region_refs = _extract_region_refs_from_json(json_result)
    if not region_refs:
        return []

    loaded_images = _load_original_images(original_images)
    if not loaded_images:
        return []

    output_dir.mkdir(parents=True, exist_ok=True)

    saved_image_paths: List[str] = []
    for page_idx, region_idx, bbox, polygon, label in region_refs:
        if page_idx < 0 or page_idx >= len(loaded_images):
            logger.warning(
                "page_idx %d out of range (total %d images), skipping",
                page_idx,
                len(loaded_images),
            )
            continue

        try:
            cropped_image = crop_image_region(loaded_images[page_idx], bbox, polygon)
            image_filename = _region_image_filename(
                page_idx,
                region_idx,
                label,
                image_prefix,
            )
            image_path = output_dir / image_filename
            cropped_image.save(image_path, quality=95)
            saved_image_paths.append(str(image_path))
        except Exception as e:
            logger.warning(
                "Failed to crop JSON region (page=%d, idx=%d, bbox=%s): %s",
                page_idx,
                region_idx,
                bbox,
                e,
            )

    return saved_image_paths
