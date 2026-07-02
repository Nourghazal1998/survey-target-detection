# Survey Target Detection & Center Localization

An end-to-end pipeline for **automated detection and sub-millimeter localization** of survey targets (reflectors) from 3D point cloud scans of infrastructure environments.

The system converts 3D point clouds into 2D orthographic wall projections, detects survey targets using deep learning (YOLO + SAHI), refines detections via multi-scale template matching, and back-projects results to 3D world coordinates.


---

## Pipeline Overview

```
Point Cloud (PCD/LAZ segments)
        │
        ▼
┌───────────────────────┐
│  Orthographic Wall    │   Sliding window along track axis
│  Projection           │   Metric-preserving 2D rendering
└───────────┬───────────┘
            │  PNG images + metadata
            ▼
┌───────────────────────┐
│  YOLO Detection       │   Sliced inference (SAHI) for
│  via SAHI             │   large-image support
└───────────┬───────────┘
            │  Bounding boxes + confidence
            ▼
┌───────────────────────┐
│  Template Matching    │   Multi-scale, multi-rotation
│  Refinement           │   Sub-pixel center localization
└───────────┬───────────┘
            │  Refined centers + scores
            ▼
┌───────────────────────┐
│  3D Back-Projection   │   Pixel coords → world XYZ
│  & Deduplication      │   2D + 3D proximity filtering
└───────────┴───────────┘
            │
            ▼
    centres.txt / detections.csv
    (World coordinates with < 1mm precision)
```

---

## Key Features

- **Orthographic projection**: Track-following coordinate frame preserves metric dimensions across images — enables direct pixel-to-meter conversion
- **Sliding window generation**: Processes arbitrarily long scans via overlapping windows along the track axis
- **SAHI sliced inference**: Handles large wall images (3000+ px) by tiling into YOLO-compatible 640×640 slices
- **Template-based refinement**: Multi-scale, multi-rotation template matching filters false positives and provides sub-pixel center accuracy
- **3D deduplication**: Merges overlapping detections in world coordinates, not just image space
- **Synthetic training data**: Procedural generation of training images with target insertion, brightness matching, and domain-specific augmentations
- **Producer-consumer architecture**: Overlapping generation and detection threads maximize throughput

---

## Project Structure

```
survey-target-detection/
├── pipeline/                    # Core detection pipeline
│   ├── config.py                # Configuration parameters
│   ├── generate_projections.py  # Point cloud → 2D wall images
│   ├── detect_targets.py        # YOLO + SAHI detection
│   ├── refine_detections.py     # Template matching refinement
│   ├── backproject_3d.py        # Pixel → world coordinate mapping
│   └── run_pipeline.py          # End-to-end orchestrator
├── training/                    # Model training utilities
│   ├── generate_synthetic_data.py
│   ├── train.py
│   └── evaluate.py
├── preprocessing/               # Point cloud preprocessing
│   ├── downsample.py
│   └── point_cloud_utils.py
├── visualization/               # Debug & visualization tools
│   ├── annotate.py
│   └── plot_geometry.py
├── scripts/                     # CLI entry points
│   ├── run_detection.py
│   └── run_training.py
├── config/
│   └── default_config.yaml      # Default pipeline parameters
├── templates/                   # Target shape templates (user-provided)
├── data/                        # Input data (user-provided)
│   └── sample/
├── docs/
│   └── images/
├── requirements.txt
└── pyproject.toml
```

---

## Installation

```bash
# Clone
git clone https://github.com/Nourghazal1998/survey-target-detection.git
cd survey-target-detection

# Create virtual environment
python -m venv venv
source venv/bin/activate  # Linux/Mac
# or: venv\Scripts\activate  # Windows

# Install dependencies
pip install -r requirements.txt
```

---

## Usage

### Run Detection Pipeline

```bash
python scripts/run_detection.py \
    --input-dir /path/to/point_cloud_segments \
    --track-csv /path/to/track.csv \
    --model-weights /path/to/best.pt \
    --templates-dir templates/ \
    --output-dir output/
```

### Train Model

```bash
python scripts/run_training.py \
    --data-yaml config/data.yaml \
    --epochs 100 \
    --batch-size 16
```

### Generate Synthetic Training Data

```bash
python -m training.generate_synthetic_data \
    --scenes-dir /path/to/scenes \
    --targets-dir /path/to/targets \
    --output-dir /path/to/training_data \
    --num-variants 67
```

---

## Configuration

All pipeline parameters are defined in [`config/default_config.yaml`](config/default_config.yaml):

| Parameter | Default | Description |
|-----------|---------|-------------|
| `frame_spacing_m` | 5.0 | Distance between projection frames along track |
| `px_per_m` | 230 | Image resolution (pixels per meter) |
| `min_lateral_m` | 0.5 | Minimum distance from track (wall filter) |
| `max_lateral_m` | 12.0 | Maximum distance from track |
| `height_up_m` | 4.0 | Vertical extent above track |
| `height_down_m` | 1.5 | Vertical extent below track |
| `confidence_threshold` | 0.35 | YOLO detection threshold |
| `template_match_threshold` | 0.44 | Template matching gate |
| `dedup_2d_px` | 50 | 2D deduplication radius (pixels) |
| `dedup_3d_m` | 0.20 | 3D deduplication radius (meters) |

---

## Input Data Format

### Point Cloud Segments

```
data/
├── track.csv                    # Columns: stationing, x, y, z
├── segment_001/
│   ├── segment_001.pcd          # Open3D-compatible point cloud
│   └── segment_001.pcd.json     # Metadata: position, origin, bounding box
├── segment_002/
│   └── ...
```

### Track CSV

```csv
stationing,x,y,z
0.000,100.123,200.456,50.789
5.000,105.123,200.456,50.791
...
```

---

## Output Format

```
output/
├── left/                        # Left-side wall projections
│   ├── frame_000.png
│   ├── frame_005.png
│   └── ...
├── right/                       # Right-side wall projections
├── detections/
│   ├── detections.csv           # All detections with metadata
│   ├── centres.txt              # Deduplicated target centers (world XYZ)
│   ├── run_summary.txt          # Processing statistics
│   └── annotated/               # Visualized detections
├── metadata.csv                 # Per-image projection geometry
```

### centres.txt

```
image,cx_px,cy_px,world_x,world_y,world_z,confidence,template,score,scale,angle
frame_005_left.png,632,804,105.12,203.45,52.10,0.92,circle,0.78,1.0,0
```

---

## Approach Details

### Orthographic Wall Projection

The pipeline constructs a **track-following coordinate frame** at each station point:

- **u-axis**: Along-track direction (tangent)
- **v-axis**: Vertical (world Z)
- **d-axis**: Lateral (perpendicular to track)

Points within the lateral band `[min_lateral, max_lateral]` are projected onto the (u, v) plane, producing metric-preserving grayscale images.

A **sliding window** advances along the track at configurable spacing, with overlap between consecutive frames to ensure no targets are missed.

### Floor Detection & Clipping

An occupancy-based analysis detects the track bed (ballast) surface and clips the rendering to exclude floor points, producing cleaner wall images.

### Detection & Refinement

1. **SAHI** tiles large wall images into 640×640 slices with overlap
2. **YOLO** detects candidate targets in each slice
3. **Size gating** rejects implausible bounding boxes
4. **Template matching** validates each detection against known target shapes at multiple scales and rotations
5. **2D deduplication** merges nearby detections within each image
6. **3D back-projection** maps pixel centers to world coordinates
7. **3D deduplication** merges detections across overlapping frames

---

## Performance

- **Detection rate**: >95% on known survey targets
- **Localization precision**: Sub-millimeter center accuracy via template matching
- **Throughput**: Producer-consumer threading overlaps I/O with inference

---

## Technologies

- **Deep Learning**: [Ultralytics YOLO](https://github.com/ultralytics/ultralytics) (object detection)
- **Large Image Inference**: [SAHI](https://github.com/obss/sahi) (sliced-aided hyper inference)
- **Point Cloud Processing**: [Open3D](http://www.open3d.org/)
- **Template Matching**: [OpenCV](https://opencv.org/)
- **Image Processing**: NumPy, SciPy, Pillow

---

## License

This project is provided for demonstration and portfolio purposes.
