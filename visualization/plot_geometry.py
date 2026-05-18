"""
Geometry visualization for projection planes and track points.

Generates colored point clouds for visual debugging in CloudCompare
or similar tools.
"""

import logging
from pathlib import Path
from typing import List, Optional

import numpy as np

logger = logging.getLogger(__name__)


def visualize_projection_geometry(
    track_points: np.ndarray,
    frame_positions: np.ndarray,
    wall_points: Optional[np.ndarray] = None,
    output_path: str = "geometry_debug.ply",
):
    """Export a colored point cloud showing the projection geometry.

    Colors:
      - White: Track centerline
      - Yellow: Frame positions (projection origins)
      - Green: Wall points (if provided)

    Args:
        track_points: Nx3 track centerline points
        frame_positions: Mx3 frame origin positions
        wall_points: Optional Kx3 wall surface points
        output_path: Output PLY file path
    """
    points = []
    colors = []

    # Track (white)
    points.append(track_points)
    colors.append(np.full((len(track_points), 3), 255, dtype=np.uint8))

    # Frame positions (yellow)
    points.append(frame_positions)
    colors.append(
        np.tile(np.array([255, 255, 0], dtype=np.uint8), (len(frame_positions), 1))
    )

    # Wall points (green)
    if wall_points is not None and len(wall_points) > 0:
        points.append(wall_points)
        colors.append(
            np.tile(np.array([0, 255, 0], dtype=np.uint8), (len(wall_points), 1))
        )

    all_points = np.vstack(points)
    all_colors = np.vstack(colors)

    _write_ply(all_points, all_colors, output_path)
    logger.info(f"Wrote geometry visualization: {output_path} ({len(all_points):,} points)")


def _write_ply(points: np.ndarray, colors: np.ndarray, path: str):
    """Write a colored point cloud to PLY format."""
    n = len(points)
    header = (
        "ply\n"
        "format ascii 1.0\n"
        f"element vertex {n}\n"
        "property float x\n"
        "property float y\n"
        "property float z\n"
        "property uchar red\n"
        "property uchar green\n"
        "property uchar blue\n"
        "end_header\n"
    )

    with open(path, "w") as f:
        f.write(header)
        for i in range(n):
            x, y, z = points[i]
            r, g, b = colors[i]
            f.write(f"{x:.4f} {y:.4f} {z:.4f} {r} {g} {b}\n")
