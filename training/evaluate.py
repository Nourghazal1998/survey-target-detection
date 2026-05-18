"""
Model evaluation and validation utilities.
"""

import csv
import logging
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)


def evaluate_model(
    model_weights: str,
    data_yaml: str,
    imgsz: int = 640,
    device: str = "0",
) -> dict:
    """Run YOLO validation and return metrics.

    Returns:
        Dict with mAP50, mAP50-95, precision, recall.
    """
    try:
        from ultralytics import YOLO
    except ImportError:
        raise ImportError("Ultralytics is required: pip install ultralytics")

    model = YOLO(model_weights)
    results = model.val(data=data_yaml, imgsz=imgsz, device=device)

    metrics = {
        "mAP50": results.box.map50,
        "mAP50-95": results.box.map,
        "precision": results.box.mp,
        "recall": results.box.mr,
    }

    logger.info(
        f"Evaluation: mAP50={metrics['mAP50']:.3f}, "
        f"P={metrics['precision']:.3f}, R={metrics['recall']:.3f}"
    )
    return metrics


def compute_center_error(
    predictions_csv: str,
    ground_truth_csv: str,
    max_match_dist_m: float = 0.5,
) -> dict:
    """Compute center localization error between predictions and ground truth.

    Matches predicted target centers to GT by nearest-neighbor in 3D,
    then computes error statistics.

    Returns:
        Dict with mean_error_mm, median_error_mm, max_error_mm, matched, missed.
    """

    def _load_centers(path: str) -> np.ndarray:
        centers = []
        with open(path, "r") as f:
            reader = csv.DictReader(f)
            for row in reader:
                centers.append([
                    float(row["world_x"]),
                    float(row["world_y"]),
                    float(row["world_z"]),
                ])
        return np.array(centers) if centers else np.empty((0, 3))

    pred = _load_centers(predictions_csv)
    gt = _load_centers(ground_truth_csv)

    if len(pred) == 0 or len(gt) == 0:
        return {"mean_error_mm": float("nan"), "matched": 0, "missed": len(gt)}

    # Greedy nearest-neighbor matching
    errors = []
    matched_gt = set()

    for p in pred:
        dists = np.linalg.norm(gt - p, axis=1)
        nearest_idx = np.argmin(dists)
        nearest_dist = dists[nearest_idx]

        if nearest_dist < max_match_dist_m and nearest_idx not in matched_gt:
            errors.append(nearest_dist)
            matched_gt.add(nearest_idx)

    errors_mm = np.array(errors) * 1000  # meters → millimeters

    return {
        "mean_error_mm": float(np.mean(errors_mm)) if len(errors_mm) > 0 else float("nan"),
        "median_error_mm": float(np.median(errors_mm)) if len(errors_mm) > 0 else float("nan"),
        "max_error_mm": float(np.max(errors_mm)) if len(errors_mm) > 0 else float("nan"),
        "matched": len(errors),
        "missed": len(gt) - len(matched_gt),
        "false_positives": len(pred) - len(errors),
    }
