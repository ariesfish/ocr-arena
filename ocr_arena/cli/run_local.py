"""Run the standalone local layout pipeline on one local image or PDF."""

from __future__ import annotations

import argparse
import json
import sys
import traceback
from pathlib import Path
from typing import Optional

from ocr_arena import load_config
from ocr_arena.ocr_pipeline import OCRPipeline
from ocr_arena.utils.logging import configure_logging, get_logger

logger = get_logger(__name__)


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments."""
    parser = argparse.ArgumentParser(
        description="Run standalone local layout OCR pipeline.",
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
        default=None,
        help="Output directory for result files (default: pipeline.output.base_output_dir)",
    )
    parser.add_argument(
        "--save-layout-visualization",
        action="store_true",
        help="Save layout visualization files",
    )
    parser.add_argument(
        "--layout-vis-dir",
        type=str,
        default=None,
        help="Temporary directory for layout visualization generation",
    )
    parser.add_argument(
        "--stdout",
        action="store_true",
        help="Print result JSON to stdout",
    )
    parser.add_argument(
        "--no-save",
        action="store_true",
        help="Do not save result files",
    )
    parser.add_argument(
        "--log-level",
        type=str,
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Log level",
    )
    parser.add_argument(
        "--print-traceback",
        action="store_true",
        help="Print traceback on failure",
    )
    return parser.parse_args()


def resolve_layout_vis_dir(
    source_path: Path,
    output_dir: Path,
    save_layout_visualization: bool,
    layout_vis_dir: Optional[str],
) -> Optional[str]:
    """Resolve the directory used for layout visualization outputs."""
    if not save_layout_visualization:
        return None
    if layout_vis_dir:
        vis_dir = Path(layout_vis_dir).expanduser().resolve()
    else:
        vis_dir = output_dir / source_path.stem / "_layout_vis_tmp"
    vis_dir.mkdir(parents=True, exist_ok=True)
    return str(vis_dir)


def main() -> int:
    """Run the local pipeline once and save/print results."""
    args = parse_args()
    configure_logging(level=args.log_level)

    try:
        source_path = Path(args.input).expanduser().resolve()
        if not source_path.is_file():
            raise FileNotFoundError(f"Input file does not exist: {source_path}")

        cfg = load_config(args.config)

        output_root = args.output or cfg.pipeline.output.base_output_dir
        output_dir = Path(output_root).expanduser().resolve()
        output_dir.mkdir(parents=True, exist_ok=True)

        vis_dir = resolve_layout_vis_dir(
            source_path=source_path,
            output_dir=output_dir,
            save_layout_visualization=args.save_layout_visualization,
            layout_vis_dir=args.layout_vis_dir,
        )

        with OCRPipeline(cfg.pipeline) as pipeline:
            results = list(
                pipeline.process(
                    str(source_path),
                    save_layout_visualization=args.save_layout_visualization,
                    layout_vis_output_dir=vis_dir,
                )
            )

        if not results:
            raise RuntimeError("Pipeline returned no results.")

        result = results[0]

        if args.stdout:
            print(json.dumps(result.to_dict(), ensure_ascii=False, indent=2))

        if not args.no_save:
            result.save(
                output_dir=str(output_dir),
                save_layout_visualization=args.save_layout_visualization,
            )
            logger.info("Saved result to %s", output_dir)

        return 0

    except Exception as e:
        logger.error("Run failed: %s", e)
        if args.print_traceback:
            traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
