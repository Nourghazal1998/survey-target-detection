"""
End-to-end pipeline orchestrator.

Runs the full detection pipeline:
  1. Generate orthographic wall projections from point clouds
  2. Detect targets using YOLO + SAHI
  3. Refine with template matching
  4. Back-project to 3D and deduplicate

Uses a producer-consumer pattern: generation runs in a background
thread while detection processes images as they become available.
"""

import csv
import logging
import time
from pathlib import Path
from queue import Queue
from threading import Thread
from typing import List, Optional

import cv2
import numpy as np

from .backproject_3d import (
    Target3D,
    backproject_to_world,
    deduplicate_2d,
    deduplicate_3d,
)
from .config import PipelineConfig
from .detect_targets import Detection, detect_targets, load_model
from .generate_projections import (
    ProjectionMetadata,
    generate_projections,
    load_track,
)
from .refine_detections import RefinedDetection, load_templates, refine_detections

logger = logging.getLogger(__name__)

_SENTINEL = None  # Signals end of generation queue


def _run_generation(
    config: PipelineConfig,
    track,
    segment_paths: List[str],
    queue: Queue,
):
    """Producer thread: generates wall projections and puts them on the queue."""
    try:
        for img_path, metadata in generate_projections(
            config.projection,
            track,
            segment_paths,
            config.output_dir,
            config.sides,
        ):
            queue.put((img_path, metadata))
    except Exception as e:
        logger.error(f"Generation error: {e}")
    finally:
        queue.put(_SENTINEL)


def _save_centres(targets: List[Target3D], output_path: str):
    """Write deduplicated target centers to CSV."""
    with open(output_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "image", "cx_px", "cy_px",
            "world_x", "world_y", "world_z",
            "confidence", "template", "score", "scale", "angle",
        ])
        for t in targets:
            writer.writerow([
                Path(t.image_path).name,
                f"{t.center_x_px:.1f}",
                f"{t.center_y_px:.1f}",
                f"{t.world_x:.4f}",
                f"{t.world_y:.4f}",
                f"{t.world_z:.4f}",
                f"{t.confidence:.3f}",
                t.template_name,
                f"{t.match_score:.3f}",
                f"{t.best_scale:.2f}",
                f"{t.best_angle:.1f}",
            ])


def _save_annotated_image(
    image_path: str,
    detections: List[Detection],
    refined: List[RefinedDetection],
    output_dir: str,
):
    """Save image with detection visualizations overlaid."""
    img = cv2.imread(image_path)
    if img is None:
        return

    # Draw raw detections (orange)
    for det in detections:
        x1, y1, x2, y2 = [int(c) for c in det.bbox]
        cv2.rectangle(img, (x1, y1), (x2, y2), (0, 165, 255), 1)

    # Draw refined detections (red boxes + green center cross)
    for ref in refined:
        det = ref.detection
        x1, y1, x2, y2 = [int(c) for c in det.bbox]
        cv2.rectangle(img, (x1, y1), (x2, y2), (0, 0, 255), 2)

        cx, cy = int(ref.center_x), int(ref.center_y)
        cv2.drawMarker(img, (cx, cy), (0, 255, 0), cv2.MARKER_CROSS, 15, 2)

        label = f"{ref.template_name} {ref.match_score:.2f}"
        cv2.putText(img, label, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 1)

    out_path = Path(output_dir) / "annotated" / Path(image_path).name
    out_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out_path), img)


def run_pipeline(config: PipelineConfig) -> List[Target3D]:
    """Run the full detection pipeline.

    Returns:
        List of deduplicated Target3D objects with world coordinates.
    """
    t0 = time.time()
    output = Path(config.output_dir)
    det_dir = output / "detections"
    det_dir.mkdir(parents=True, exist_ok=True)

    # Load resources
    track = load_track(config.track_csv)
    logger.info(f"Loaded track: {len(track)} points, "
                f"{track[0].stationing:.1f}m – {track[-1].stationing:.1f}m")

    model = load_model(config.detection.model_weights, config.detection.device)
    templates = load_templates(config.template.templates_dir)

    # Discover segments
    input_dir = Path(config.input_dir)
    segment_paths = sorted(
        str(p) for p in input_dir.rglob("*.pcd")
    ) + sorted(
        str(p) for p in input_dir.rglob("*.laz")
    )
    logger.info(f"Found {len(segment_paths)} point cloud segments")

    # Producer-consumer: generate projections in background
    queue = Queue(maxsize=config.queue_maxsize)
    gen_thread = Thread(
        target=_run_generation,
        args=(config, track, segment_paths, queue),
        daemon=True,
    )
    gen_thread.start()

    # Consumer: detect targets as images arrive
    all_targets: List[Target3D] = []
    total_raw = 0
    total_refined = 0
    images_processed = 0

    while True:
        item = queue.get()
        if item is _SENTINEL:
            break

        img_path, metadata = item
        images_processed += 1

        # Load image
        image = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        if image is None:
            logger.warning(f"Could not read {img_path}")
            continue

        # Detect
        raw_detections = detect_targets(img_path, model, config.detection)
        total_raw += len(raw_detections)

        # Refine with template matching
        refined = refine_detections(
            raw_detections, image, templates, config.template
        )
        total_refined += len(refined)

        # 2D deduplication within this image
        refined = deduplicate_2d(refined, config.deduplication.max_dist_2d_px)

        # Back-project to 3D
        for det in refined:
            target = backproject_to_world(det, metadata)
            all_targets.append(target)

        # Save annotated image
        _save_annotated_image(
            img_path, raw_detections, refined, str(det_dir)
        )

    gen_thread.join()

    # Global 3D deduplication
    final_targets = deduplicate_3d(
        all_targets, config.deduplication.max_dist_3d_m
    )

    # Save results
    _save_centres(final_targets, str(det_dir / "centres.txt"))

    elapsed = time.time() - t0
    summary = (
        f"Pipeline complete in {elapsed:.1f}s\n"
        f"Images processed: {images_processed}\n"
        f"Raw detections: {total_raw}\n"
        f"After template matching: {total_refined}\n"
        f"Final targets (3D dedup): {len(final_targets)}\n"
    )
    logger.info(summary)
    (det_dir / "run_summary.txt").write_text(summary)

    return final_targets
