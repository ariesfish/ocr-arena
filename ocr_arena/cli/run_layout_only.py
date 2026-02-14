"""Run layout detection only and save bbox overlays on original pages."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Dict, List

from PIL import Image, ImageDraw, ImageFont


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments."""
    parser = argparse.ArgumentParser(
        description="Run standalone layout detection only (no OCR calls).",
    )
    parser.add_argument(
        "input",
        type=str,
        help="Local input file path (single image or one PDF)",
    )
    parser.add_argument(
        "--config",
        "-c",
        type=str,
        default=None,
        help="Path to config file (default: ocr_arena/config.yaml)",
    )
    parser.add_argument(
        "--output",
        "-o",
        type=str,
        default="./output_layout_only",
        help="Output directory",
    )
    parser.add_argument(
        "--cpu",
        action="store_true",
        help="Force CPU execution by disabling CUDA visibility",
    )
    parser.add_argument(
        "--log-level",
        type=str,
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Log level",
    )
    parser.add_argument(
        "--font-size-ratio",
        type=float,
        default=0.02,
        help="Text size ratio for bbox annotations (default: 0.02)",
    )
    return parser.parse_args()


def resolve_local_source(raw_input: str) -> Path:
    """Resolve local file path from CLI input."""
    if raw_input.startswith("file://"):
        raw_input = raw_input[7:]

    if raw_input.startswith(("http://", "https://", "data:")):
        raise ValueError("Only local file input is supported.")

    path = Path(raw_input).expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(f"Input file does not exist: {path}")

    allowed_suffix = {".jpg", ".jpeg", ".png", ".bmp", ".gif", ".webp", ".pdf"}
    if path.suffix.lower() not in allowed_suffix:
        raise ValueError(f"Unsupported file type: {path.suffix}")

    return path


def resolve_layout_model_dir(raw_model_dir: str) -> str:
    """Resolve a usable PP-DocLayout model directory.

    Supports passing either the exact model directory or a parent directory
    that contains model subdirectories.
    """
    model_dir = Path(raw_model_dir).expanduser().resolve()

    required_files = {"config.json", "preprocessor_config.json", "model.safetensors"}
    if model_dir.is_dir():
        current_files = {p.name for p in model_dir.iterdir() if p.is_file()}
        if required_files.issubset(current_files):
            return str(model_dir)

    if model_dir.is_dir():
        candidates = []
        for sub in model_dir.iterdir():
            if not sub.is_dir():
                continue
            files = {p.name for p in sub.iterdir() if p.is_file()}
            if required_files.issubset(files):
                candidates.append(sub)

        if candidates:
            # Prefer a doclayout-like directory name when multiple candidates exist.
            candidates.sort(key=lambda p: ("doclayout" not in p.name.lower(), p.name))
            return str(candidates[0])

    return str(model_dir)


def denormalize_layout_regions(
    regions: List[Dict], image_width: int, image_height: int
) -> List[Dict]:
    """Convert normalized layout regions (0-1000) into pixel coordinates."""
    vis_regions: List[Dict] = []
    for region in regions:
        bbox = region.get("bbox_2d")
        if not bbox or len(bbox) != 4:
            continue

        x1 = int(bbox[0] * image_width / 1000)
        y1 = int(bbox[1] * image_height / 1000)
        x2 = int(bbox[2] * image_width / 1000)
        y2 = int(bbox[3] * image_height / 1000)

        polygon_points = []
        polygon = region.get("polygon") or []
        for point in polygon:
            if isinstance(point, (list, tuple)) and len(point) == 2:
                px = int(point[0] * image_width / 1000)
                py = int(point[1] * image_height / 1000)
                polygon_points.append([px, py])

        vis_regions.append(
            {
                "index": int(region.get("index", len(vis_regions))),
                "label": region.get("label", "unknown"),
                "task_type": region.get("task_type", "unknown"),
                "score": float(region.get("score", 1.0)),
                "coordinate": [x1, y1, x2, y2],
                "polygon_points": polygon_points,
            }
        )

    return vis_regions


def save_blue_bbox_visualization(
    image: Image.Image,
    boxes: List[Dict],
    save_path: Path,
    color: tuple = (0, 102, 255),
    text_color: tuple = (255, 0, 0),
    font_size_ratio: float = 0.01,
) -> None:
    """Save layout visualization using blue rectangle outlines only."""
    vis_img = image.copy().convert("RGB")
    draw = ImageDraw.Draw(vis_img)

    line_width = max(2, int(max(vis_img.size) * 0.002))
    img_w, img_h = vis_img.size
    font_size = max(9, int(max(vis_img.size) * font_size_ratio))

    font = None
    try:
        font_path = Path(__file__).resolve().parent / "resources" / "PingFang.ttf"
        if font_path.exists():
            font = ImageFont.truetype(str(font_path), font_size)
    except Exception:
        font = None
    if font is None:
        try:
            font = ImageFont.truetype("DejaVuSans.ttf", font_size)
        except Exception:
            font = ImageFont.load_default()

    for box in boxes:
        coord = box.get("coordinate")
        if not coord or len(coord) != 4:
            continue
        x1, y1, x2, y2 = [int(v) for v in coord]
        draw.rectangle([(x1, y1), (x2, y2)], outline=color, width=line_width)

        # Draw compact metadata: index, task_type, confidence.
        text = (
            f"#{box.get('index', 0)} "
            f"{box.get('label', 'unknown')} "
            f"{box.get('score', 0.0):.2f}"
        )
        bbox_text = draw.textbbox((0, 0), text, font=font)
        tw = bbox_text[2] - bbox_text[0]
        th = bbox_text[3] - bbox_text[1]

        text_x = x1 + 4
        text_y = y1 - th - 6
        if text_y < 0:
            text_y = y1 + 4

        # Keep text fully inside the image canvas.
        text_x = max(0, min(text_x, img_w - tw - 1))
        text_y = max(0, min(text_y, img_h - th - 1))

        draw.text((text_x, text_y), text, fill=text_color, font=font)

    save_path.parent.mkdir(parents=True, exist_ok=True)
    vis_img.save(save_path, quality=95)


def main() -> int:
    """Run layout-only detection and save raw + visualization outputs."""
    args = parse_args()

    if args.cpu:
        os.environ["CUDA_VISIBLE_DEVICES"] = ""

    from ocr_arena.config import load_config
    from ocr_arena.dataloader import PageLoader
    from ocr_arena.layout import PPDocLayoutDetector
    from ocr_arena.utils.logging import configure_logging, get_logger
    configure_logging(level=args.log_level)
    logger = get_logger(__name__)

    source_path = resolve_local_source(args.input)
    output_dir = Path(args.output).expanduser().resolve()
    output_base = output_dir / source_path.stem
    vis_dir = output_base / "layout_vis"
    output_base.mkdir(parents=True, exist_ok=True)
    vis_dir.mkdir(parents=True, exist_ok=True)

    cfg = load_config(args.config)
    cfg.pipeline.layout.model_dir = resolve_layout_model_dir(cfg.pipeline.layout.model_dir)

    loader = PageLoader(cfg.pipeline.page_loader)
    pages = loader.load_pages(str(source_path))

    detector = PPDocLayoutDetector(cfg.pipeline.layout)
    detector.start()
    try:
        layout_results = detector.process(
            pages,
            save_visualization=False,
            visualization_output_dir=None,
            global_start_idx=0,
        )
    finally:
        detector.stop()

    # Save raw normalized layout output.
    raw_json_path = output_base / f"{source_path.stem}.layout.json"
    with open(raw_json_path, "w", encoding="utf-8") as f:
        json.dump(layout_results, f, ensure_ascii=False, indent=2)

    # Draw detected bboxes on original pages.
    for page_idx, (page, regions) in enumerate(zip(pages, layout_results)):
        page_rgb = page.convert("RGB")
        width, height = page_rgb.size
        vis_regions = denormalize_layout_regions(regions, width, height)

        save_name = (
            f"{source_path.stem}.layout.jpg"
            if len(pages) == 1
            else f"{source_path.stem}_page{page_idx}.layout.jpg"
        )
        save_path = vis_dir / save_name

        save_blue_bbox_visualization(
            image=page_rgb,
            boxes=vis_regions,
            save_path=save_path,
            font_size_ratio=args.font_size_ratio,
        )

    summary = {
        "input": str(source_path),
        "pages": len(pages),
        "regions_per_page": [len(items) for items in layout_results],
        "raw_layout_json": str(raw_json_path),
        "visualization_dir": str(vis_dir),
    }
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    logger.info("Layout-only run completed.")

    return 0


if __name__ == "__main__":
    sys.exit(main())
