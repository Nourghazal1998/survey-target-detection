"""
Template matching refinement for detected targets.

After YOLO provides candidate bounding boxes, this module validates each
detection by matching against known target shape templates at multiple
scales and rotations. This dramatically reduces false positives and
provides sub-pixel center localization.
"""

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Tuple

import cv2
import numpy as np

from .config import TemplateConfig
from .detect_targets import Detection

logger = logging.getLogger(__name__)


@dataclass
class RefinedDetection:
    """A detection refined by template matching."""

    detection: Detection
    center_x: float
    center_y: float
    template_name: str
    match_score: float
    best_scale: float
    best_angle: float


def load_templates(templates_dir: str) -> dict:
    """Load template images from directory.

    Returns:
        Dict mapping template name to grayscale numpy array.
    """
    templates = {}
    tdir = Path(templates_dir)

    if not tdir.exists():
        logger.warning(f"Templates directory not found: {templates_dir}")
        return templates

    for img_path in sorted(tdir.glob("*.png")):
        template = cv2.imread(str(img_path), cv2.IMREAD_GRAYSCALE)
        if template is not None:
            templates[img_path.stem] = template
            logger.info(f"Loaded template: {img_path.stem} ({template.shape})")

    return templates


def _rotate_image(image: np.ndarray, angle: float) -> np.ndarray:
    """Rotate image by angle (degrees) around its center."""
    h, w = image.shape[:2]
    center = (w / 2, h / 2)
    M = cv2.getRotationMatrix2D(center, angle, 1.0)
    return cv2.warpAffine(image, M, (w, h), borderMode=cv2.BORDER_REPLICATE)


def _resize_template(template: np.ndarray, scale: float) -> np.ndarray:
    """Resize template by scale factor."""
    h, w = template.shape[:2]
    new_w = max(1, int(round(w * scale)))
    new_h = max(1, int(round(h * scale)))
    return cv2.resize(template, (new_w, new_h), interpolation=cv2.INTER_LINEAR)


def match_template_in_roi(
    roi_gray: np.ndarray,
    templates: dict,
    config: TemplateConfig,
) -> Optional[Tuple[str, float, float, float, float, float]]:
    """Match templates against an ROI at multiple scales and rotations.

    Args:
        roi_gray: Grayscale ROI image (upscaled for finer search)
        templates: Dict of template name → grayscale image
        config: Template matching parameters

    Returns:
        (template_name, score, center_x, center_y, best_scale, best_angle)
        or None if no match exceeds threshold.
    """
    # Upscale ROI for sub-pixel accuracy
    upscale = config.roi_upscale_factor
    roi_up = cv2.resize(
        roi_gray,
        None,
        fx=upscale,
        fy=upscale,
        interpolation=cv2.INTER_LINEAR,
    )

    best_score = -np.inf
    best_result = None

    scales = np.linspace(
        config.scale_range[0], config.scale_range[1], config.scale_steps
    )

    for tmpl_name, tmpl_img in templates.items():
        for angle in config.rotation_angles:
            rotated = _rotate_image(tmpl_img, angle)

            for scale in scales:
                scaled = _resize_template(rotated, scale * upscale)

                # Skip if template is larger than ROI
                if scaled.shape[0] > roi_up.shape[0] or scaled.shape[1] > roi_up.shape[1]:
                    continue

                result = cv2.matchTemplate(
                    roi_up, scaled, cv2.TM_CCOEFF_NORMED
                )
                _, max_val, _, max_loc = cv2.minMaxLoc(result)

                if max_val > best_score:
                    best_score = max_val
                    # Center of matched region in upscaled coordinates
                    cx = (max_loc[0] + scaled.shape[1] / 2) / upscale
                    cy = (max_loc[1] + scaled.shape[0] / 2) / upscale
                    best_result = (tmpl_name, max_val, cx, cy, scale, angle)

    if best_result is None or best_result[1] < config.match_threshold:
        return None

    return best_result


def refine_detections(
    detections: List[Detection],
    image: np.ndarray,
    templates: dict,
    config: TemplateConfig,
) -> List[RefinedDetection]:
    """Refine detections using template matching.

    For each detection, extracts the ROI from the image and runs
    multi-scale, multi-rotation template matching to validate
    and precisely localize the target center.
    """
    refined = []
    h, w = image.shape[:2]

    for det in detections:
        x1, y1, x2, y2 = [int(c) for c in det.bbox]

        # Expand ROI slightly for template matching context
        pad = int(max(x2 - x1, y2 - y1) * 0.25)
        rx1 = max(0, x1 - pad)
        ry1 = max(0, y1 - pad)
        rx2 = min(w, x2 + pad)
        ry2 = min(h, y2 + pad)

        roi = image[ry1:ry2, rx1:rx2]
        if roi.size == 0:
            continue

        # Convert to grayscale if needed
        if len(roi.shape) == 3:
            roi_gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        else:
            roi_gray = roi

        match = match_template_in_roi(roi_gray, templates, config)

        if match is None:
            logger.debug(f"Template match rejected detection at ({x1},{y1})")
            continue

        tmpl_name, score, cx_roi, cy_roi, scale, angle = match

        # Convert ROI-local center back to image coordinates
        cx_img = rx1 + cx_roi
        cy_img = ry1 + cy_roi

        refined.append(
            RefinedDetection(
                detection=det,
                center_x=cx_img,
                center_y=cy_img,
                template_name=tmpl_name,
                match_score=score,
                best_scale=scale,
                best_angle=angle,
            )
        )

    logger.info(
        f"Template matching: {len(detections)} → {len(refined)} detections"
    )
    return refined
