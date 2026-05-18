"""
Synthetic training data generation.

Creates diverse training images by inserting target shapes into
point-cloud-rendered scene backgrounds with realistic augmentations.

Demo note: Production uses more sophisticated insertion logic including
brightness matching, segmentation-aware hole filling, and ballast
detection.
"""

import logging
import random
from pathlib import Path
from typing import List, Optional, Tuple

import cv2
import numpy as np
from PIL import Image

logger = logging.getLogger(__name__)


# --- Augmentation functions ---


def augment_rotation(image: np.ndarray, max_angle: float = 5.0) -> np.ndarray:
    """Random small rotation."""
    angle = random.uniform(-max_angle, max_angle)
    h, w = image.shape[:2]
    M = cv2.getRotationMatrix2D((w / 2, h / 2), angle, 1.0)
    return cv2.warpAffine(image, M, (w, h), borderMode=cv2.BORDER_REFLECT)


def augment_brightness(image: np.ndarray, range_: Tuple[float, float] = (0.75, 1.25)) -> np.ndarray:
    """Random brightness scaling."""
    factor = random.uniform(*range_)
    return np.clip(image.astype(np.float32) * factor, 0, 255).astype(np.uint8)


def augment_noise(image: np.ndarray, max_std: float = 1.3) -> np.ndarray:
    """Additive Gaussian noise."""
    std = random.uniform(0, max_std)
    noise = np.random.normal(0, std, image.shape).astype(np.float32)
    return np.clip(image.astype(np.float32) + noise, 0, 255).astype(np.uint8)


def augment_contrast(image: np.ndarray, range_: Tuple[float, float] = (0.9, 1.1)) -> np.ndarray:
    """Random contrast adjustment."""
    factor = random.uniform(*range_)
    mean = image.mean()
    return np.clip((image.astype(np.float32) - mean) * factor + mean, 0, 255).astype(np.uint8)


def augment_clahe(image: np.ndarray, max_clip: float = 1.0) -> np.ndarray:
    """CLAHE (Contrast Limited Adaptive Histogram Equalization)."""
    clip = random.uniform(0, max_clip)
    clahe = cv2.createCLAHE(clipLimit=clip, tileGridSize=(8, 8))
    return clahe.apply(image)


def apply_augmentations(
    image: np.ndarray,
    rotation_prob: float = 0.9,
    brightness_prob: float = 0.9,
    noise_prob: float = 0.1,
    contrast_prob: float = 0.15,
    clahe_prob: float = 0.12,
) -> np.ndarray:
    """Apply a randomized augmentation chain."""
    if random.random() < rotation_prob:
        image = augment_rotation(image)
    if random.random() < brightness_prob:
        image = augment_brightness(image)
    if random.random() < noise_prob:
        image = augment_noise(image)
    if random.random() < contrast_prob:
        image = augment_contrast(image)
    if random.random() < clahe_prob:
        image = augment_clahe(image)
    return image


# --- Target insertion ---


def insert_target(
    scene: np.ndarray,
    target: np.ndarray,
    position: Tuple[int, int],
    brightness_scale: float = 0.92,
) -> Tuple[np.ndarray, Tuple[int, int, int, int]]:
    """Insert a target shape into a scene image.

    Matches target brightness to local scene statistics for realism.

    Returns:
        (modified_scene, bbox) where bbox is (cx, cy, w, h) in pixels.
    """
    scene = scene.copy()
    th, tw = target.shape[:2]
    px, py = position

    # Ensure target fits within scene
    x1 = max(0, px - tw // 2)
    y1 = max(0, py - th // 2)
    x2 = min(scene.shape[1], x1 + tw)
    y2 = min(scene.shape[0], y1 + th)

    crop_w = x2 - x1
    crop_h = y2 - y1
    if crop_w <= 0 or crop_h <= 0:
        return scene, (px, py, 0, 0)

    target_crop = target[:crop_h, :crop_w]

    # Match brightness to local scene median
    local_region = scene[y1:y2, x1:x2]
    local_median = np.median(local_region[local_region > 0]) if np.any(local_region > 0) else 128
    target_median = np.median(target_crop[target_crop > 0]) if np.any(target_crop > 0) else 128

    if target_median > 0:
        scale = (local_median / target_median) * brightness_scale
        adjusted = np.clip(target_crop.astype(np.float32) * scale, 0, 255).astype(np.uint8)
    else:
        adjusted = target_crop

    # Insert (non-zero pixels only, preserving scene background)
    mask = adjusted > 0
    scene[y1:y2, x1:x2][mask] = adjusted[mask]

    bbox = (px, py, crop_w, crop_h)
    return scene, bbox


# --- Dataset generation ---


def generate_training_dataset(
    scenes_dir: str,
    targets_dir: str,
    output_dir: str,
    image_size: int = 640,
    num_variants: int = 67,
    negative_ratio: float = 0.25,
):
    """Generate a synthetic training dataset.

    For each scene × target combination, generates multiple augmented
    variants and corresponding YOLO-format labels.

    Args:
        scenes_dir: Directory of background scene images (rendered from point clouds)
        targets_dir: Directory of target shape images (circles, diamonds, chess patterns)
        output_dir: Output directory for images/ and labels/
        image_size: Output image size (square)
        num_variants: Number of augmented variants per scene-target pair
        negative_ratio: Fraction of output that should be negative (no target) samples
    """
    out = Path(output_dir)
    (out / "images").mkdir(parents=True, exist_ok=True)
    (out / "labels").mkdir(parents=True, exist_ok=True)

    scenes = sorted(Path(scenes_dir).glob("*.png"))
    targets = sorted(Path(targets_dir).glob("*.png"))

    if not scenes:
        logger.warning(f"No scene images found in {scenes_dir}")
        return
    if not targets:
        logger.warning(f"No target images found in {targets_dir}")
        return

    logger.info(f"Generating from {len(scenes)} scenes × {len(targets)} targets × {num_variants} variants")

    count = 0
    for scene_path in scenes:
        scene = cv2.imread(str(scene_path), cv2.IMREAD_GRAYSCALE)
        if scene is None:
            continue

        scene = cv2.resize(scene, (image_size, image_size))

        for target_path in targets:
            target_img = cv2.imread(str(target_path), cv2.IMREAD_GRAYSCALE)
            if target_img is None:
                continue

            for v in range(num_variants):
                # Random position within center region
                margin = image_size // 4
                px = random.randint(margin, image_size - margin)
                py = random.randint(margin, image_size - margin)

                # Random target scale
                scale = random.uniform(0.5, 1.5)
                th, tw = target_img.shape[:2]
                scaled_target = cv2.resize(
                    target_img,
                    (max(1, int(tw * scale)), max(1, int(th * scale))),
                )

                # Insert and augment
                result, bbox = insert_target(scene, scaled_target, (px, py))
                result = apply_augmentations(result)

                # Save image
                img_name = f"{scene_path.stem}_{target_path.stem}_{v:03d}.png"
                cv2.imwrite(str(out / "images" / img_name), result)

                # Save YOLO label (class 0, normalized bbox)
                cx_norm = bbox[0] / image_size
                cy_norm = bbox[1] / image_size
                w_norm = bbox[2] / image_size
                h_norm = bbox[3] / image_size

                label_name = img_name.replace(".png", ".txt")
                with open(out / "labels" / label_name, "w") as f:
                    f.write(f"0 {cx_norm:.6f} {cy_norm:.6f} {w_norm:.6f} {h_norm:.6f}\n")

                count += 1

    # Generate negatives (scenes without targets)
    num_negatives = int(count * negative_ratio)
    for i in range(num_negatives):
        scene_path = random.choice(scenes)
        scene = cv2.imread(str(scene_path), cv2.IMREAD_GRAYSCALE)
        if scene is None:
            continue

        scene = cv2.resize(scene, (image_size, image_size))
        scene = apply_augmentations(scene)

        img_name = f"negative_{i:04d}.png"
        cv2.imwrite(str(out / "images" / img_name), scene)
        # Empty label file (no targets)
        (out / "labels" / img_name.replace(".png", ".txt")).touch()

    logger.info(f"Generated {count} positives + {num_negatives} negatives")
