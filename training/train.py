"""
YOLO model training wrapper.

Configures and launches YOLOv11 training using the Ultralytics API
on the generated synthetic training dataset.
"""

import logging
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)


def train_yolo(
    data_yaml: str,
    model: str = "yolo11n.pt",
    epochs: int = 100,
    batch_size: int = 16,
    imgsz: int = 640,
    patience: int = 20,
    device: str = "0",
    project: str = "runs/train",
    name: str = "target_detection",
    resume: bool = False,
) -> str:
    """Train a YOLO model for target detection.

    Args:
        data_yaml: Path to dataset YAML (train/val/test splits)
        model: Pretrained model path or YOLO variant name
        epochs: Maximum training epochs
        batch_size: Batch size
        imgsz: Input image size
        patience: Early stopping patience
        device: CUDA device(s) or 'cpu'
        project: Output project directory
        name: Run name
        resume: Resume from last checkpoint

    Returns:
        Path to best weights file.
    """
    try:
        from ultralytics import YOLO
    except ImportError:
        raise ImportError("Ultralytics is required: pip install ultralytics")

    yolo = YOLO(model)

    results = yolo.train(
        data=data_yaml,
        epochs=epochs,
        batch=batch_size,
        imgsz=imgsz,
        patience=patience,
        device=device,
        project=project,
        name=name,
        resume=resume,
        verbose=True,
    )

    best_path = Path(project) / name / "weights" / "best.pt"
    logger.info(f"Training complete. Best weights: {best_path}")
    return str(best_path)
