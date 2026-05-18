"""
Target detection using YOLO with SAHI (Sliced Aided Hyper Inference).

Handles large wall projection images by tiling them into manageable slices,
running YOLO on each slice, and merging results with NMS.
"""

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional

import numpy as np

from .config import DetectionConfig

logger = logging.getLogger(__name__)


@dataclass
class Detection:
    """A single target detection in an image."""

    bbox: tuple  # (x1, y1, x2, y2) in pixels
    confidence: float
    class_id: int
    image_path: str

    @property
    def center(self) -> tuple:
        x1, y1, x2, y2 = self.bbox
        return ((x1 + x2) / 2, (y1 + y2) / 2)

    @property
    def width(self) -> float:
        return self.bbox[2] - self.bbox[0]

    @property
    def height(self) -> float:
        return self.bbox[3] - self.bbox[1]


def load_model(weights_path: str, device: str = "cpu"):
    """Load YOLO model for inference."""
    try:
        from ultralytics import YOLO

        model = YOLO(weights_path)
        model.to(device)
        logger.info(f"Loaded YOLO model from {weights_path} on {device}")
        return model
    except ImportError:
        raise ImportError(
            "Ultralytics is required for detection: pip install ultralytics"
        )


def run_sahi_detection(
    image_path: str,
    model,
    config: DetectionConfig,
) -> List[Detection]:
    """Run SAHI sliced inference on a single image.

    Tiles the image into overlapping slices, runs YOLO on each,
    and merges predictions via non-maximum suppression.
    """
    try:
        from sahi import AutoDetectionModel
        from sahi.predict import get_sliced_prediction
    except ImportError:
        raise ImportError("SAHI is required: pip install sahi")

    detection_model = AutoDetectionModel.from_pretrained(
        model_type="yolov8",
        model=model,
        confidence_threshold=config.confidence_threshold,
        device=config.device,
    )

    result = get_sliced_prediction(
        image_path,
        detection_model,
        slice_height=config.slice_height,
        slice_width=config.slice_width,
        overlap_height_ratio=config.slice_overlap_ratio,
        overlap_width_ratio=config.slice_overlap_ratio,
    )

    detections = []
    for pred in result.object_prediction_list:
        bbox = pred.bbox.to_xyxy()
        det = Detection(
            bbox=tuple(bbox),
            confidence=pred.score.value,
            class_id=pred.category.id,
            image_path=image_path,
        )
        detections.append(det)

    return detections


def filter_by_size(
    detections: List[Detection], config: DetectionConfig
) -> List[Detection]:
    """Reject detections with implausible bounding box dimensions."""
    filtered = []
    for det in detections:
        if (
            config.bbox_min_width_px <= det.width <= config.bbox_max_width_px
            and config.bbox_min_height_px <= det.height <= config.bbox_max_height_px
        ):
            filtered.append(det)
        else:
            logger.debug(
                f"Rejected bbox {det.width:.0f}x{det.height:.0f}px "
                f"(limits: {config.bbox_min_width_px}-{config.bbox_max_width_px})"
            )
    return filtered


def filter_by_confidence(
    detections: List[Detection], min_confidence: float = 0.5
) -> List[Detection]:
    """Apply a stricter confidence threshold after initial detection."""
    return [d for d in detections if d.confidence >= min_confidence]


def detect_targets(
    image_path: str,
    model,
    config: DetectionConfig,
) -> List[Detection]:
    """Full detection pipeline for a single image.

    1. Run SAHI sliced inference
    2. Filter by bounding box size
    3. Filter by confidence
    """
    raw = run_sahi_detection(image_path, model, config)
    logger.info(f"Raw detections: {len(raw)}")

    sized = filter_by_size(raw, config)
    logger.info(f"After size filter: {len(sized)}")

    confident = filter_by_confidence(sized)
    logger.info(f"After confidence filter: {len(confident)}")

    return confident
