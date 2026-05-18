"""
Orthographic wall projection from 3D point clouds.

Implements a sliding-window approach that advances along the track axis,
projecting nearby points onto a metric-preserving 2D image plane.

The coordinate frame at each station point:
  - u: along-track (tangent direction)
  - v: vertical (world Z)
  - d: lateral (perpendicular to track, toward the wall)
"""

import csv
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Generator, List, Optional, Tuple

import numpy as np
from PIL import Image

from .config import ProjectionConfig

logger = logging.getLogger(__name__)


@dataclass
class TrackPoint:
    """A single point along the survey track."""

    stationing: float
    x: float
    y: float
    z: float


@dataclass
class ProjectionMetadata:
    """Metadata describing the geometry of a single wall projection."""

    stationing: float
    side: str
    track_position: np.ndarray
    tangent: np.ndarray
    side_direction: np.ndarray
    patch_width_m: float
    height_up_m: float
    height_down_m: float
    min_lateral_m: float
    max_lateral_m: float
    img_width: int
    img_height: int


def load_track(csv_path: str) -> List[TrackPoint]:
    """Load track centerline from CSV (stationing, x, y, z)."""
    points = []
    with open(csv_path, "r") as f:
        reader = csv.DictReader(f)
        for row in reader:
            points.append(
                TrackPoint(
                    stationing=float(row["stationing"]),
                    x=float(row["x"]),
                    y=float(row["y"]),
                    z=float(row["z"]),
                )
            )
    return sorted(points, key=lambda p: p.stationing)


def compute_tangent(track: List[TrackPoint], idx: int) -> np.ndarray:
    """Compute the tangent direction at a track point (horizontal plane)."""
    if idx == 0:
        dx = track[1].x - track[0].x
        dy = track[1].y - track[0].y
    elif idx == len(track) - 1:
        dx = track[-1].x - track[-2].x
        dy = track[-1].y - track[-2].y
    else:
        dx = track[idx + 1].x - track[idx - 1].x
        dy = track[idx + 1].y - track[idx - 1].y

    tangent = np.array([dx, dy, 0.0])
    norm = np.linalg.norm(tangent)
    if norm < 1e-12:
        return np.array([1.0, 0.0, 0.0])
    return tangent / norm


def compute_side_direction(tangent: np.ndarray, side: str) -> np.ndarray:
    """Compute the lateral direction perpendicular to the track.

    For 'left', rotates tangent 90° counterclockwise in the XY plane.
    For 'right', rotates 90° clockwise.
    """
    if side == "left":
        return np.array([-tangent[1], tangent[0], 0.0])
    else:
        return np.array([tangent[1], -tangent[0], 0.0])


def load_segment_points(segment_path: str) -> np.ndarray:
    """Load point cloud segment and return Nx4 array (x, y, z, intensity).

    Supports PCD and LAZ formats via Open3D and laspy respectively.

    Demo note: In production, this includes JSON-based coordinate
    transforms to align scanner-local coordinates to world frame.
    """
    path = Path(segment_path)

    if path.suffix == ".pcd":
        try:
            import open3d as o3d

            pcd = o3d.io.read_point_cloud(str(path))
            xyz = np.asarray(pcd.points)
            if pcd.has_colors():
                intensity = np.mean(np.asarray(pcd.colors), axis=1)
            else:
                intensity = np.ones(len(xyz))
            return np.column_stack([xyz, intensity])
        except ImportError:
            raise ImportError("Open3D is required for PCD loading: pip install open3d")

    elif path.suffix in (".laz", ".las"):
        try:
            import laspy

            las = laspy.read(str(path))
            xyz = np.column_stack([las.x, las.y, las.z])
            intensity = (
                las.intensity / 65535.0
                if hasattr(las, "intensity")
                else np.ones(len(xyz))
            )
            return np.column_stack([xyz, intensity])
        except ImportError:
            raise ImportError("laspy is required for LAZ loading: pip install laspy")

    else:
        raise ValueError(f"Unsupported point cloud format: {path.suffix}")


def project_wall_patch(
    points: np.ndarray,
    track_pos: np.ndarray,
    tangent: np.ndarray,
    side_dir: np.ndarray,
    config: ProjectionConfig,
) -> Tuple[np.ndarray, float]:
    """Project 3D points onto a 2D wall image.

    Args:
        points: Nx4 array (x, y, z, intensity)
        track_pos: 3D position of the track at this station
        tangent: Along-track unit vector
        side_dir: Lateral unit vector (toward wall)
        config: Projection parameters

    Returns:
        image: Grayscale image as uint8 numpy array
        coverage: Fraction of non-zero pixels
    """
    # Compute local coordinates relative to track position
    relative = points[:, :3] - track_pos

    u = relative @ tangent  # along-track
    d = relative @ side_dir  # lateral (toward wall)
    v = relative[:, 2] - track_pos[2]  # vertical

    # Filter points within the projection volume
    half_w = config.patch_width_m / 2.0
    mask = (
        (np.abs(u) <= half_w)
        & (d >= config.min_lateral_m)
        & (d <= config.max_lateral_m)
        & (v >= -config.height_down_m)
        & (v <= config.height_up_m)
    )

    u_filt = u[mask]
    v_filt = v[mask]
    intensity = points[mask, 3]

    # Map to pixel coordinates
    img_w = config.img_width
    img_h = config.img_height

    x_px = ((u_filt + half_w) / config.patch_width_m * (img_w - 1)).astype(int)
    y_px = ((config.height_up_m - v_filt) / (config.height_up_m + config.height_down_m) * (img_h - 1)).astype(int)

    # Clamp to image bounds
    x_px = np.clip(x_px, 0, img_w - 1)
    y_px = np.clip(y_px, 0, img_h - 1)

    # Render: take maximum intensity per pixel (front-most point)
    image = np.zeros((img_h, img_w), dtype=np.float32)
    np.maximum.at(image, (y_px, x_px), intensity)

    # Normalize to uint8
    if image.max() > 0:
        image = (image / image.max() * 255).astype(np.uint8)
    else:
        image = image.astype(np.uint8)

    # Apply brightness boost
    if config.brightness_boost != 1.0:
        nonzero = image > 0
        boosted = image[nonzero].astype(np.float32) * config.brightness_boost
        image[nonzero] = np.clip(boosted, 0, 255).astype(np.uint8)

    coverage = np.count_nonzero(image) / image.size
    return image, coverage


def apply_hole_fill(image: np.ndarray, coverage: float, mode: str = "auto") -> np.ndarray:
    """Fill holes (zero-valued pixels) in the projection image.

    Demo note: Production uses a more sophisticated coverage-adaptive
    approach with segmentation-aware filling.
    """
    from scipy.ndimage import median_filter

    if mode == "none":
        return image

    if mode == "auto":
        passes = 2 if coverage < 0.6 else 1
    elif mode == "double":
        passes = 2
    else:
        passes = 1

    result = image.copy()
    for _ in range(passes):
        holes = result == 0
        if not np.any(holes):
            break
        filled = median_filter(result, size=3)
        result[holes] = filled[holes]

    return result


def generate_projections(
    config: ProjectionConfig,
    track: List[TrackPoint],
    segment_paths: List[str],
    output_dir: str,
    sides: List[str] = ("left", "right"),
) -> Generator[Tuple[str, ProjectionMetadata], None, None]:
    """Generate wall projection images along the track.

    Implements a sliding window that loads nearby segments and projects
    points at each station interval.

    Yields:
        (image_path, metadata) for each generated projection.
    """
    output = Path(output_dir)
    for side in sides:
        (output / side).mkdir(parents=True, exist_ok=True)

    # Determine station positions
    min_s = track[0].stationing
    max_s = track[-1].stationing
    stations = np.arange(min_s, max_s, config.frame_spacing_m)

    logger.info(f"Generating {len(stations)} frames per side across {max_s - min_s:.0f}m")

    # Load all segment points (demo: in production, uses sliding window with caching)
    all_points = []
    for seg_path in segment_paths:
        pts = load_segment_points(seg_path)
        all_points.append(pts)
        logger.info(f"Loaded {len(pts):,} points from {seg_path}")

    if not all_points:
        logger.warning("No point cloud segments found")
        return

    points = np.vstack(all_points)
    logger.info(f"Total points: {len(points):,}")

    for station in stations:
        # Find nearest track point
        idx = int(np.argmin([abs(tp.stationing - station) for tp in track]))
        tp = track[idx]
        track_pos = np.array([tp.x, tp.y, tp.z])
        tangent = compute_tangent(track, idx)

        for side in sides:
            side_dir = compute_side_direction(tangent, side)

            image, coverage = project_wall_patch(
                points, track_pos, tangent, side_dir, config
            )

            if config.hole_fill_mode != "none":
                image = apply_hole_fill(image, coverage, config.hole_fill_mode)

            # Save image
            filename = f"frame_{station:07.1f}_{side}.png"
            img_path = str(output / side / filename)
            Image.fromarray(image).save(img_path)

            metadata = ProjectionMetadata(
                stationing=station,
                side=side,
                track_position=track_pos,
                tangent=tangent,
                side_direction=side_dir,
                patch_width_m=config.patch_width_m,
                height_up_m=config.height_up_m,
                height_down_m=config.height_down_m,
                min_lateral_m=config.min_lateral_m,
                max_lateral_m=config.max_lateral_m,
                img_width=config.img_width,
                img_height=config.img_height,
            )

            logger.info(f"Generated {filename} (coverage: {coverage:.1%})")
            yield img_path, metadata
