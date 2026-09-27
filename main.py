"""
Vehicle Counter — counts vehicles crossing a virtual line in a traffic video.

Uses YOLO for detection and simple centroid tracking to assign IDs and
detect line crossings.
"""

import cv2
import numpy as np
from ultralytics import YOLO
from collections import defaultdict

# ── Config ────────────────────────────────────────────────────────────────────

INPUT_VIDEO = "input/traffic.mp4"
OUTPUT_VIDEO = "output/counted_traffic.mp4"
MODEL_NAME = "yolov8n.pt"  # lightweight; swap to yolov8s.pt for better accuracy

# COCO class IDs for the vehicles we care about
VEHICLE_CLASSES = {
    2: "car",
    3: "motorcycle",
    5: "bus",
    7: "truck",
}

LINE_POSITION = 0.60  # counting line at 60 % of frame height
CONFIDENCE_THRESHOLD = 0.35
MAX_DISAPPEAR = 30  # frames before a tracked object is dropped
MAX_MATCH_DISTANCE = 80  # max pixel distance for centroid matching

# Colors (BGR)
COLOR_LINE = (0, 0, 255)
COLOR_BOX = (0, 255, 0)
COLOR_TEXT = (255, 255, 255)
COLOR_COUNTER_BG = (50, 50, 50)


# ── Simple centroid tracker ──────────────────────────────────────────────────

class CentroidTracker:
    """Lightweight frame-by-frame centroid tracker."""

    def __init__(self, max_disappear=MAX_DISAPPEAR, max_distance=MAX_MATCH_DISTANCE):
        self.next_id = 0
        self.objects = {}        # id -> (cx, cy)
        self.classes = {}        # id -> class name
        self.disappeared = {}    # id -> frames since last seen
        self.max_disappear = max_disappear
        self.max_distance = max_distance

    def _register(self, centroid, class_name):
        oid = self.next_id
        self.objects[oid] = centroid
        self.classes[oid] = class_name
        self.disappeared[oid] = 0
        self.next_id += 1
        return oid

    def _deregister(self, oid):
        del self.objects[oid]
        del self.classes[oid]
        del self.disappeared[oid]

    def update(self, detections):
        """
        detections: list of (cx, cy, class_name)
        Returns dict  {id: (cx, cy)}  and  self.classes  for labels.
        """
        # No detections — age existing objects
        if len(detections) == 0:
            for oid in list(self.disappeared):
                self.disappeared[oid] += 1
                if self.disappeared[oid] > self.max_disappear:
                    self._deregister(oid)
            return self.objects

        input_centroids = np.array([(d[0], d[1]) for d in detections])
        input_classes = [d[2] for d in detections]

        # First frame or all previous objects lost
        if len(self.objects) == 0:
            for i in range(len(detections)):
                self._register(input_centroids[i], input_classes[i])
            return self.objects

        object_ids = list(self.objects.keys())
        object_centroids = np.array(list(self.objects.values()))

        # Pairwise distance matrix
        dists = np.linalg.norm(
            object_centroids[:, np.newaxis] - input_centroids[np.newaxis, :],
            axis=2,
        )

        # Greedy matching: sort by distance, assign closest pairs first
        rows = dists.min(axis=1).argsort()
        cols = dists.argmin(axis=1)[rows]

        used_rows = set()
        used_cols = set()

        for row, col in zip(rows, cols):
            if row in used_rows or col in used_cols:
                continue
            if dists[row, col] > self.max_distance:
                continue

            oid = object_ids[row]
            self.objects[oid] = input_centroids[col]
            self.classes[oid] = input_classes[col]
            self.disappeared[oid] = 0
            used_rows.add(row)
            used_cols.add(col)

        # Age unmatched existing objects
        for row in range(len(object_ids)):
            if row not in used_rows:
                oid = object_ids[row]
                self.disappeared[oid] += 1
                if self.disappeared[oid] > self.max_disappear:
                    self._deregister(oid)

        # Register unmatched new detections
        for col in range(len(input_centroids)):
            if col not in used_cols:
                self._register(input_centroids[col], input_classes[col])

        return self.objects


# ── Main pipeline ────────────────────────────────────────────────────────────

def main():
    # Load model
    model = YOLO(MODEL_NAME)

    # Open video
    cap = cv2.VideoCapture(INPUT_VIDEO)
    if not cap.isOpened():
        print(f"Error: cannot open {INPUT_VIDEO}")
        return

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    # Counting line y-coordinate
    line_y = int(height * LINE_POSITION)

    # Video writer
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(OUTPUT_VIDEO, fourcc, fps, (width, height))

    tracker = CentroidTracker()

    # Counters
    counts = defaultdict(int)  # class_name -> count
    counted_ids = set()        # IDs that already crossed the line
    prev_positions = {}        # id -> previous cy

    frame_num = 0

    print(f"Processing {INPUT_VIDEO}  ({width}x{height} @ {fps:.1f} fps, {total_frames} frames)")
    print(f"Counting line at y={line_y} ({LINE_POSITION*100:.0f}% of height)")
    print("Press Q in the video window to stop.\n")

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame_num += 1

        # ── Detect ────────────────────────────────────────────────────────
        results = model(frame, conf=CONFIDENCE_THRESHOLD, verbose=False)[0]

        detections = []  # (cx, cy, class_name)

        for box in results.boxes:
            cls_id = int(box.cls[0])
            if cls_id not in VEHICLE_CLASSES:
                continue

            x1, y1, x2, y2 = map(int, box.xyxy[0])
            cx = (x1 + x2) // 2
            cy = (y1 + y2) // 2
            class_name = VEHICLE_CLASSES[cls_id]
            detections.append((cx, cy, class_name))

        # ── Track ─────────────────────────────────────────────────────────
        objects = tracker.update(detections)

        # ── Count crossings & draw ────────────────────────────────────────
        for oid, (cx, cy) in objects.items():
            class_name = tracker.classes[oid]

            # Check line crossing (top → bottom)
            if oid in prev_positions and oid not in counted_ids:
                prev_cy = prev_positions[oid]
                if prev_cy < line_y <= cy:
                    counts[class_name] += 1
                    counted_ids.add(oid)

            prev_positions[oid] = cy

            # Draw centroid
            cv2.circle(frame, (int(cx), int(cy)), 4, COLOR_BOX, -1)

        # Draw bounding boxes from current detections (aligned to YOLO output)
        for box in results.boxes:
            cls_id = int(box.cls[0])
            if cls_id not in VEHICLE_CLASSES:
                continue
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            cx = (x1 + x2) // 2
            cy = (y1 + y2) // 2
            class_name = VEHICLE_CLASSES[cls_id]

            # Find the tracker ID for this detection
            best_id = None
            best_dist = float("inf")
            for oid, (ocx, ocy) in objects.items():
                d = ((cx - ocx) ** 2 + (cy - ocy) ** 2) ** 0.5
                if d < best_dist:
                    best_dist = d
                    best_id = oid

            cv2.rectangle(frame, (x1, y1), (x2, y2), COLOR_BOX, 2)
            label = f"{class_name} #{best_id}" if best_id is not None else class_name
            label_y = max(y1 - 8, 15)
            cv2.putText(frame, label, (x1, label_y),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, COLOR_BOX, 2)

        # ── Counting line ─────────────────────────────────────────────────
        cv2.line(frame, (0, line_y), (width, line_y), COLOR_LINE, 2)
        cv2.putText(frame, "Counting Line", (10, line_y - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, COLOR_LINE, 2)

        # ── Counter overlay ───────────────────────────────────────────────
        total = sum(counts.values())
        overlay_lines = [
            f"Total: {total}",
            f"Cars: {counts['car']}",
            f"Motorcycles: {counts['motorcycle']}",
            f"Buses: {counts['bus']}",
            f"Trucks: {counts['truck']}",
        ]

        y_offset = 30
        for i, text in enumerate(overlay_lines):
            ty = y_offset + i * 30
            # Background rectangle for readability
            (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)
            cv2.rectangle(frame, (8, ty - th - 5), (18 + tw, ty + 5), COLOR_COUNTER_BG, -1)
            cv2.putText(frame, text, (12, ty),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, COLOR_TEXT, 2)

        # ── Progress ──────────────────────────────────────────────────────
        progress = f"Frame {frame_num}/{total_frames}"
        cv2.putText(frame, progress, (width - 220, height - 15),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, COLOR_TEXT, 1)

        # ── Write & show ──────────────────────────────────────────────────
        writer.write(frame)
        cv2.imshow("Vehicle Counter", frame)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            print("Stopped by user.")
            break

    # ── Cleanup ───────────────────────────────────────────────────────────
    cap.release()
    writer.release()
    cv2.destroyAllWindows()

    # ── Final summary ─────────────────────────────────────────────────────
    total = sum(counts.values())
    print("=" * 40)
    print("  VEHICLE COUNT RESULTS")
    print("=" * 40)
    print(f"  Cars:        {counts['car']}")
    print(f"  Motorcycles: {counts['motorcycle']}")
    print(f"  Buses:       {counts['bus']}")
    print(f"  Trucks:      {counts['truck']}")
    print(f"  ─────────────────────")
    print(f"  Total:       {total}")
    print("=" * 40)
    print(f"\nOutput saved to {OUTPUT_VIDEO}")


if __name__ == "__main__":
    main()
