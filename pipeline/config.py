"""
Pipeline configuration parameters.

Centralizes all tunable parameters for the detection pipeline.
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional

import numpy as np


@dataclass
class ProjectionConfig:
    """Parameters controlling point cloud → image projection."""

    frame_spacing_m: float = 5.0
    px_per_m: float = 230.0

    min_lateral_m: float = 0.5
    max_lateral_m: float = 12.0
    height_up_m: float = 4.0
    height_down_m: float = 1.5

    brightness_boost: float = 1.25
    enable_floor_clipping: bool = True
    hole_fill_mode: str = "auto"  # "none", "single", "double", "auto"

    generation_workers: int = 4

    @property
    def patch_width_m(self) -> float:
        overlap_fraction = 0.1
        return self.frame_spacing_m * (1.0 + overlap_fraction)

    @property
    def img_width(self) -> int:
        return int(round(self.patch_width_m * self.px_per_m))

    @property
    def img_height(self) -> int:
        total_height = self.height_up_m + self.height_down_m
        return int(round(total_height * self.px_per_m))


@dataclass
class DetectionConfig:
    """Parameters controlling YOLO + SAHI detection."""

    model_weights: str = "best.pt"
    device: str = "cpu"
    confidence_threshold: float = 0.35

    slice_width: int = 640
    slice_height: int = 640
    slice_overlap_ratio: float = 0.1

    bbox_min_width_px: int = 19
    bbox_max_width_px: int = 65
    bbox_min_height_px: int = 19
    bbox_max_height_px: int = 65


@dataclass
class TemplateConfig:
    """Parameters controlling template matching refinement."""

    templates_dir: str = "templates/"
    rotation_angles: List[float] = field(
        default_factory=lambda: list(range(0, 100, 10))
    )
    scale_range: tuple = (0.6, 1.6)
    scale_steps: int = 9
    match_threshold: float = 0.44
    roi_upscale_factor: float = 2.5


@dataclass
class DeduplicationConfig:
    """Parameters controlling deduplication."""

    max_dist_2d_px: float = 50.0
    max_dist_3d_m: float = 0.20


@dataclass
class PipelineConfig:
    """Top-level configuration combining all sub-configs."""

    projection: ProjectionConfig = field(default_factory=ProjectionConfig)
    detection: DetectionConfig = field(default_factory=DetectionConfig)
    template: TemplateConfig = field(default_factory=TemplateConfig)
    deduplication: DeduplicationConfig = field(default_factory=DeduplicationConfig)

    input_dir: str = ""
    track_csv: str = ""
    output_dir: str = "output/"
    sides: List[str] = field(default_factory=lambda: ["left", "right"])

    queue_maxsize: int = 8

    @classmethod
    def from_yaml(cls, path: str) -> "PipelineConfig":
        """Load configuration from a YAML file."""
        import yaml

        with open(path, "r") as f:
            raw = yaml.safe_load(f)

        config = cls()
        for section_name, section_data in raw.items():
            if hasattr(config, section_name) and isinstance(section_data, dict):
                sub = getattr(config, section_name)
                for k, v in section_data.items():
                    if hasattr(sub, k):
                        setattr(sub, k, v)
            elif hasattr(config, section_name):
                setattr(config, section_name, section_data)
        return config
