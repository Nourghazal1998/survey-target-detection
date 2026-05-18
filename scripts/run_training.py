#!/usr/bin/env python3
"""
Train or evaluate a YOLO model for survey target detection.

Usage:
    python scripts/run_training.py train --data-yaml data.yaml
    python scripts/run_training.py evaluate --weights best.pt --data-yaml data.yaml
    python scripts/run_training.py generate --scenes scenes/ --targets targets/ --output training_data/
"""

import argparse
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from training.evaluate import compute_center_error, evaluate_model
from training.generate_synthetic_data import generate_training_dataset
from training.train import train_yolo


def main():
    parser = argparse.ArgumentParser(description="Target detection training utilities")
    subparsers = parser.add_subparsers(dest="command", required=True)

    # Train
    train_parser = subparsers.add_parser("train", help="Train YOLO model")
    train_parser.add_argument("--data-yaml", required=True, help="Dataset YAML")
    train_parser.add_argument("--model", default="yolo11n.pt", help="Base model")
    train_parser.add_argument("--epochs", type=int, default=100)
    train_parser.add_argument("--batch-size", type=int, default=16)
    train_parser.add_argument("--device", default="0")

    # Evaluate
    eval_parser = subparsers.add_parser("evaluate", help="Evaluate model")
    eval_parser.add_argument("--weights", required=True, help="Model weights path")
    eval_parser.add_argument("--data-yaml", required=True, help="Dataset YAML")
    eval_parser.add_argument("--device", default="0")

    # Generate synthetic data
    gen_parser = subparsers.add_parser("generate", help="Generate synthetic training data")
    gen_parser.add_argument("--scenes", required=True, help="Scene images directory")
    gen_parser.add_argument("--targets", required=True, help="Target images directory")
    gen_parser.add_argument("--output", required=True, help="Output directory")
    gen_parser.add_argument("--variants", type=int, default=67, help="Variants per pair")
    gen_parser.add_argument("--image-size", type=int, default=640)

    parser.add_argument("-v", "--verbose", action="store_true")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    if args.command == "train":
        best = train_yolo(
            data_yaml=args.data_yaml,
            model=args.model,
            epochs=args.epochs,
            batch_size=args.batch_size,
            device=args.device,
        )
        print(f"Best weights saved to: {best}")

    elif args.command == "evaluate":
        metrics = evaluate_model(args.weights, args.data_yaml, device=args.device)
        for k, v in metrics.items():
            print(f"  {k}: {v:.4f}")

    elif args.command == "generate":
        generate_training_dataset(
            scenes_dir=args.scenes,
            targets_dir=args.targets,
            output_dir=args.output,
            image_size=args.image_size,
            num_variants=args.variants,
        )


if __name__ == "__main__":
    main()
