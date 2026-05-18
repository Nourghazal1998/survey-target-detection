"""
Back-projection from 2D pixel coordinates to 3D world coordinates.

Given the projection geometry metadata for each image, converts detected
target centers from pixel (cx, cy) back to world (X, Y, Z) coordinates.

Also implements deduplication in both 2D (per-image) and 3D (global) space.
"""

import logging
from dataclasses import dataclass
from typing import List

import numpy as np

from .config import DeduplicationConfig
from .generate_projections import ProjectionMetadata
from .refine_detections import RefinedDetection

logger = logging.getLogger(__name__)


@dataclass
class Target3D:
    """A detected target with 3D world coordinates."""

    center_x_px: float
    center_y_px: float
    world_x: float
    world_y: float
    world_z: float
    confidence: float
    template_name: str
    match_score: float
    best_scale: float
    best_angle: float
    image_path: str
    stationing: float
    side: str


def backproject_to_world(
    detection: RefinedDetection,
    metadata: ProjectionMetadata,
) -> Target3D:
    """Convert a 2D pixel detection to 3D world coordinates.

    Uses the projection geometry to reverse the rendering transform:
      u = pixel_x → along-track offset
      v = pixel_y → vertical offset
      d = midpoint of lateral band (depth estimate)
    """
    cx = detection.center_x
    cy = detection.center_y

    # Pixel → physical coordinates
    u = (cx / (metadata.img_width - 1)) * metadata.patch_width_m - metadata.patch_width_m / 2
    v = metadata.height_up_m - (cy / (metadata.img_height - 1)) * (metadata.height_up_m + metadata.height_down_m)
    d_mid = (metadata.min_lateral_m + metadata.max_lateral_m) / 2

    # Physical → world coordinates
    world = (
        metadata.track_position
        + u * metadata.tangent
        + d_mid * metadata.side_direction
        + v * np.array([0.0, 0.0, 1.0])
    )

    return Target3D(
        center_x_px=cx,
        center_y_px=cy,
        world_x=world[0],
        world_y=world[1],
        world_z=world[2],
        confidence=detection.detection.confidence,
        template_name=detection.template_name,
        match_score=detection.match_score,
        best_scale=detection.best_scale,
        best_angle=detection.best_angle,
        image_path=detection.detection.image_path,
        stationing=metadata.stationing,
        side=metadata.side,
    )


def deduplicate_2d(
    detections: List[RefinedDetection],
    max_dist_px: float,
) -> List[RefinedDetection]:
    """Remove duplicate detections within a single image.

    Greedy algorithm: keeps the highest-scored detection, removes
    all others within max_dist_px, then repeats.
    """
    if not detections:
        return []

    sorted_dets = sorted(detections, key=lambda d: d.match_score, reverse=True)
    kept = []

    for det in sorted_dets:
        is_duplicate = False
        for existing in kept:
            dist = np.hypot(
                det.center_x - existing.center_x,
                det.center_y - existing.center_y,
            )
            if dist < max_dist_px:
                is_duplicate = True
                break

        if not is_duplicate:
            kept.append(det)

    logger.info(f"2D dedup: {len(detections)} → {len(kept)}")
    return kept


def deduplicate_3d(
    targets: List[Target3D],
    max_dist_m: float,
) -> List[Target3D]:
    """Remove duplicate targets across overlapping frames in 3D space.

    Same greedy algorithm as 2D, but operating on world coordinates.
    Keeps the highest template-match-scored detection per cluster.
    """
    if not targets:
        return []

    sorted_targets = sorted(targets, key=lambda t: t.match_score, reverse=True)
    kept = []

    for target in sorted_targets:
        pos = np.array([target.world_x, target.world_y, target.world_z])
        is_duplicate = False

        for existing in kept:
            existing_pos = np.array(
                [existing.world_x, existing.world_y, existing.world_z]
            )
            if np.linalg.norm(pos - existing_pos) < max_dist_m:
                is_duplicate = True
                break

        if not is_duplicate:
            kept.append(target)

    logger.info(f"3D dedup: {len(targets)} → {len(kept)}")
    return kept
