"""
Point cloud loading and preprocessing utilities.
"""

import json
import logging
from pathlib import Path
from typing import Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)


def load_pcd_with_metadata(
    pcd_path: str,
) -> Tuple[np.ndarray, Optional[dict]]:
    """Load a PCD file and its JSON metadata sidecar.

    Returns:
        (points_Nx4, metadata_dict_or_None)
    """
    import open3d as o3d

    pcd = o3d.io.read_point_cloud(pcd_path)
    xyz = np.asarray(pcd.points)

    if pcd.has_colors():
        intensity = np.mean(np.asarray(pcd.colors), axis=1)
    else:
        intensity = np.ones(len(xyz))

    points = np.column_stack([xyz, intensity])

    # Try loading JSON sidecar
    json_path = Path(pcd_path + ".json")
    metadata = None
    if json_path.exists():
        with open(json_path, "r") as f:
            metadata = json.load(f)

    return points, metadata


def compute_bounding_box(points: np.ndarray) -> dict:
    """Compute axis-aligned bounding box of a point cloud.

    Returns:
        Dict with 'min', 'max', 'center', 'size' as 3-element lists.
    """
    xyz = points[:, :3]
    bb_min = xyz.min(axis=0)
    bb_max = xyz.max(axis=0)
    return {
        "min": bb_min.tolist(),
        "max": bb_max.tolist(),
        "center": ((bb_min + bb_max) / 2).tolist(),
        "size": (bb_max - bb_min).tolist(),
    }


def filter_by_region(
    points: np.ndarray,
    center: np.ndarray,
    radius: float,
) -> np.ndarray:
    """Filter points within a cylindrical region (XY plane)."""
    xy_dist = np.linalg.norm(points[:, :2] - center[:2], axis=1)
    return points[xy_dist <= radius]
