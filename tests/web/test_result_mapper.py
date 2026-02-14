from __future__ import annotations

import pytest

from ocr_arena.web.services.result_mapper import (
    compute_confidence_avg,
    map_backend_result,
)


def test_confidence_avg_from_scores() -> None:
    boxes = [
        {"score": 0.7},
        {"score": 0.9},
        {"score": None},
    ]
    assert compute_confidence_avg(boxes) == pytest.approx(0.8)


def test_map_backend_result_keeps_score_and_bbox() -> None:
    json_result = [
        [
            {
                "index": 0,
                "label": "text",
                "content": "hello",
                "bbox_2d": [10, 20, 200, 300],
                "score": 0.93,
            }
        ]
    ]
    payload = map_backend_result(
        model_key="paddle",
        json_result=json_result,
        markdown_result="hello",
        latency_ms=123,
        error=None,
    )
    assert payload["name"] == "PaddleOCR-VL-1.5"
    assert payload["confidence_avg"] == pytest.approx(0.93)
    assert payload["boxes"][0]["bbox_2d"] == (10.0, 20.0, 200.0, 300.0)
    assert payload["boxes"][0]["score"] == pytest.approx(0.93)
