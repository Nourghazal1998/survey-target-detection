"""
Point cloud downsampling utilities.
"""

import logging
from pathlib import Path

import numpy as np

logger = logging.getLogger(__name__)


def downsample_voxel(points: np.ndarray, voxel_size: float = 0.01) -> np.ndarray:
    """Voxel-grid downsampling of a point cloud.

    Args:
        points: Nx3+ array (at least x, y, z columns)
        voxel_size: Voxel edge length in meters

    Returns:
        Downsampled point array (one point per voxel, mean aggregation).
    """
    if len(points) == 0:
        return points

    xyz = points[:, :3]
    voxel_ids = np.floor(xyz / voxel_size).astype(np.int64)

    # Unique voxels
    _, inverse, counts = np.unique(
        voxel_ids, axis=0, return_inverse=True, return_counts=True
    )

    # Mean aggregation per voxel
    n_voxels = len(counts)
    result = np.zeros((n_voxels, points.shape[1]), dtype=np.float64)
    np.add.at(result, inverse, points)
    result /= counts[:, np.newaxis]

    logger.info(f"Downsampled {len(points):,} → {n_voxels:,} points (voxel={voxel_size}m)")
    return result.astype(points.dtype)


def downsample_random(points: np.ndarray, fraction: float = 0.5) -> np.ndarray:
    """Random subsampling of a point cloud.

    Args:
        points: Nx3+ array
        fraction: Fraction of points to keep (0-1)

    Returns:
        Randomly subsampled point array.
    """
    n = int(len(points) * fraction)
    indices = np.random.choice(len(points), size=n, replace=False)
    return points[indices]
