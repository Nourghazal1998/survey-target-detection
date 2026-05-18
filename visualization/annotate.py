"""
Visualization and annotation tools for detection results.
"""

import logging
from pathlib import Path
from typing import List, Optional, Tuple

import cv2
import numpy as np

logger = logging.getLogger(__name__)


def draw_detections(
    image: np.ndarray,
    detections: List[dict],
    color_confirmed: Tuple[int, int, int] = (0, 0, 255),
    color_rejected: Tuple[int, int, int] = (0, 165, 255),
) -> np.ndarray:
    """Draw detection bounding boxes and centers on an image.

    Args:
        image: Input image (will be copied)
        detections: List of dicts with 'bbox', 'confirmed', optionally 'center', 'label'
        color_confirmed: BGR color for confirmed detections
        color_rejected: BGR color for rejected detections

    Returns:
        Annotated image copy.
    """
    vis = image.copy()
    if len(vis.shape) == 2:
        vis = cv2.cvtColor(vis, cv2.COLOR_GRAY2BGR)

    for det in detections:
        x1, y1, x2, y2 = [int(c) for c in det["bbox"]]
        confirmed = det.get("confirmed", True)
        color = color_confirmed if confirmed else color_rejected
        thickness = 2 if confirmed else 1

        cv2.rectangle(vis, (x1, y1), (x2, y2), color, thickness)

        if "center" in det and confirmed:
            cx, cy = int(det["center"][0]), int(det["center"][1])
            cv2.drawMarker(vis, (cx, cy), (0, 255, 0), cv2.MARKER_CROSS, 15, 2)

        if "label" in det:
            cv2.putText(
                vis, det["label"], (x1, y1 - 5),
                cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1,
            )

    return vis


def create_detection_grid(
    images: List[np.ndarray],
    cols: int = 4,
    cell_size: int = 320,
) -> np.ndarray:
    """Create a grid visualization of multiple detection crops.

    Useful for reviewing detection quality at a glance.
    """
    rows = (len(images) + cols - 1) // cols
    grid = np.zeros((rows * cell_size, cols * cell_size, 3), dtype=np.uint8)

    for i, img in enumerate(images):
        r, c = divmod(i, cols)
        resized = cv2.resize(img, (cell_size, cell_size))
        if len(resized.shape) == 2:
            resized = cv2.cvtColor(resized, cv2.COLOR_GRAY2BGR)
        grid[r * cell_size : (r + 1) * cell_size, c * cell_size : (c + 1) * cell_size] = resized

    return grid
