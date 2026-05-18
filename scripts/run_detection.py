#!/usr/bin/env python3
"""
Run the target detection pipeline on point cloud data.

Usage:
    python scripts/run_detection.py --config config/default_config.yaml
    python scripts/run_detection.py --input-dir /path/to/segments --track-csv track.csv
"""

import argparse
import logging
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from pipeline.config import PipelineConfig
from pipeline.run_pipeline import run_pipeline


def main():
    parser = argparse.ArgumentParser(description="Survey target detection pipeline")
    parser.add_argument("--config", type=str, help="Path to YAML config file")
    parser.add_argument("--input-dir", type=str, help="Directory containing PCD/LAZ segments")
    parser.add_argument("--track-csv", type=str, help="Path to track CSV file")
    parser.add_argument("--model-weights", type=str, default="best.pt", help="YOLO model weights")
    parser.add_argument("--templates-dir", type=str, default="templates/", help="Template images directory")
    parser.add_argument("--output-dir", type=str, default="output/", help="Output directory")
    parser.add_argument("--device", type=str, default="cpu", help="Inference device (cpu/cuda)")
    parser.add_argument("--sides", nargs="+", default=["left", "right"], help="Wall sides to process")
    parser.add_argument("-v", "--verbose", action="store_true", help="Verbose logging")

    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    if args.config:
        config = PipelineConfig.from_yaml(args.config)
    else:
        config = PipelineConfig()

    # Override with CLI arguments
    if args.input_dir:
        config.input_dir = args.input_dir
    if args.track_csv:
        config.track_csv = args.track_csv
    if args.model_weights:
        config.detection.model_weights = args.model_weights
    if args.templates_dir:
        config.template.templates_dir = args.templates_dir
    if args.output_dir:
        config.output_dir = args.output_dir
    if args.device:
        config.detection.device = args.device
    if args.sides:
        config.sides = args.sides

    if not config.input_dir or not config.track_csv:
        parser.error("--input-dir and --track-csv are required (or use --config)")

    targets = run_pipeline(config)
    print(f"\nDetected {len(targets)} survey targets.")
    for t in targets:
        print(f"  [{t.side}] station={t.stationing:.1f}m  "
              f"world=({t.world_x:.4f}, {t.world_y:.4f}, {t.world_z:.4f})  "
              f"template={t.template_name} score={t.match_score:.3f}")


if __name__ == "__main__":
    main()
