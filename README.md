# Vehicle Counter

Count vehicles crossing a virtual line in a traffic video using YOLO object detection and centroid tracking.

## Technologies

- **Python 3.8+**
- **OpenCV** — video I/O, drawing, display
- **Ultralytics YOLO** (YOLOv8 nano) — real-time object detection
- **NumPy** — distance calculations for tracking

## Setup

```bash
pip install -r requirements.txt
```

Place your traffic video at `input/traffic.mp4`.

## Usage

```bash
python main.py
```

- A window opens showing the annotated video in real time.
- Press **Q** to stop early.
- The processed video is saved to `output/counted_traffic.mp4`.
- Final counts are printed to the terminal when processing finishes.

## How the counting works

1. **Detection** — Each frame is passed through YOLOv8 to detect cars, motorcycles, buses, and trucks.
2. **Centroid tracking** — The center point of every bounding box is computed. A simple tracker matches centroids frame-to-frame using nearest-neighbor distance, assigning a stable ID to each vehicle.
3. **Line crossing** — A horizontal counting line is drawn at 60 % of the frame height. When a tracked centroid moves from above the line to below it between consecutive frames, that vehicle's class counter increments.
4. **Once-only counting** — Each ID is counted at most once, so a vehicle lingering near the line won't inflate the total.

## Example output

```
========================================
  VEHICLE COUNT RESULTS
========================================
  Cars:        42
  Motorcycles: 7
  Buses:       3
  Trucks:      11
  ─────────────────────
  Total:       63
========================================
```

## Configuration

Edit the constants at the top of `main.py`:

| Variable | Default | Description |
|---|---|---|
| `MODEL_NAME` | `yolov8n.pt` | YOLO model weight file |
| `LINE_POSITION` | `0.60` | Counting line as fraction of frame height |
| `CONFIDENCE_THRESHOLD` | `0.35` | Minimum detection confidence |
| `MAX_DISAPPEAR` | `30` | Frames before a lost track is dropped |
| `MAX_MATCH_DISTANCE` | `80` | Max pixel distance for centroid matching |
