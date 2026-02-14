from __future__ import annotations

import json
from typing import Any, Dict, Iterable, List, Optional, Tuple

MODEL_NAME_MAPPING = {
    "paddle": "PaddleOCR-VL-1.5",
    "glm": "GLM-OCR",
}


def _to_float(value: Any) -> Optional[float]:
    try:
        if value is None:
            return None
        numeric = float(value)
    except (TypeError, ValueError):
        return None
    return numeric


def _normalize_bbox(raw_bbox: Any) -> Optional[Tuple[float, float, float, float]]:
    if not isinstance(raw_bbox, (list, tuple)) or len(raw_bbox) != 4:
        return None

    values = [_to_float(item) for item in raw_bbox]
    if any(item is None for item in values):
        return None

    x1, y1, x2, y2 = values
    if x1 is None or y1 is None or x2 is None or y2 is None:
        return None
    if x1 >= x2 or y1 >= y2:
        return None

    def _clamp(value: float) -> float:
        return max(0.0, min(1000.0, float(value)))

    return (_clamp(x1), _clamp(y1), _clamp(x2), _clamp(y2))


def _iter_result_items(json_result: Any) -> Iterable[Dict[str, Any]]:
    data = json_result
    if isinstance(json_result, str):
        try:
            data = json.loads(json_result)
        except json.JSONDecodeError:
            return []

    if not isinstance(data, list):
        return []

    items: List[Dict[str, Any]] = []
    for page in data:
        if not isinstance(page, list):
            continue
        for item in page:
            if isinstance(item, dict):
                items.append(item)
    return items


def compute_confidence_avg(boxes: List[Dict[str, Any]]) -> Optional[float]:
    scores = [box["score"] for box in boxes if box.get("score") is not None]
    if not scores:
        return None
    return float(sum(scores) / len(scores))


def map_backend_result(
    model_key: str,
    *,
    json_result: Any,
    markdown_result: str,
    latency_ms: Optional[int],
    error: Optional[str],
) -> Dict[str, Any]:
    mapped_boxes: List[Dict[str, Any]] = []
    text_segments: List[str] = []

    for item in _iter_result_items(json_result):
        bbox = _normalize_bbox(item.get("bbox_2d"))
        if bbox is None:
            continue

        content = item.get("content")
        if isinstance(content, str) and content.strip():
            text_segments.append(content.strip())

        polygon_raw = item.get("polygon")
        polygon: Optional[List[List[float]]] = None
        if isinstance(polygon_raw, list):
            polygon = []
            for point in polygon_raw:
                if not isinstance(point, (list, tuple)) or len(point) != 2:
                    continue
                x = _to_float(point[0])
                y = _to_float(point[1])
                if x is None or y is None:
                    continue
                polygon.append([x, y])
            if not polygon:
                polygon = None

        mapped_boxes.append(
            {
                "index": int(item.get("index", len(mapped_boxes))),
                "label": str(item.get("label", "text")),
                "score": _to_float(item.get("score")),
                "bbox_2d": bbox,
                "polygon": polygon,
            }
        )

    text_value = markdown_result.strip() if isinstance(markdown_result, str) else ""
    if not text_value:
        text_value = "\n".join(text_segments)

    return {
        "name": MODEL_NAME_MAPPING.get(model_key, model_key),
        "latency_ms": latency_ms,
        "confidence_avg": compute_confidence_avg(mapped_boxes),
        "text": text_value,
        "boxes": mapped_boxes,
        "error": error,
    }
