import cv2
import json
import time
from pathlib import Path
from collections import Counter
from ultralytics import YOLO

VIDEO = Path("data/validation/urban_intersection_01.mp4")
MODEL = Path("yolo11n_openvino_model")
OUTPUT = Path("output/intersection_tracking/openvino")

def main():
    if not VIDEO.exists():
        raise FileNotFoundError(VIDEO)

    if not MODEL.exists():
        raise FileNotFoundError(MODEL)

    if OUTPUT.exists():
        raise FileExistsError(OUTPUT)

    OUTPUT.mkdir(parents=True)

    cap = cv2.VideoCapture(str(VIDEO))

    if not cap.isOpened():
        raise RuntimeError("Cannot open input video.")

    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    writer = cv2.VideoWriter(
        str(OUTPUT / "annotated_tracking.mp4"),
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps,
        (width, height)
    )

    if not writer.isOpened():
        cap.release()
        raise RuntimeError("Cannot create output video.")

    print("Loading PyTorch YOLO model...")
    model = YOLO(str(MODEL), task="detect")

    frames = []
    unique_ids = set()
    class_counts = Counter()
    total_detections = 0
    frames_with_detections = 0
    frame_number = 0

    start = time.perf_counter()

    try:
        while True:
            success, frame = cap.read()

            if not success:
                break

            frame_number += 1

            results = model.track(
                frame,
                persist=True,
                verbose=False,
                device="intel:gpu"
            )

            detections = []

            if results and results[0].boxes is not None:
                boxes = results[0].boxes
                coordinates = boxes.xyxy.cpu().numpy()
                classes = boxes.cls.int().cpu().tolist()
                confidences = boxes.conf.cpu().tolist()

                ids = (
                    boxes.id.int().cpu().tolist()
                    if boxes.id is not None
                    else [None] * len(classes)
                )

                for box, class_id, confidence, track_id in zip(
                    coordinates, classes, confidences, ids
                ):
                    detections.append({
                        "track_id": track_id,
                        "class_id": class_id,
                        "confidence": round(float(confidence), 5),
                        "bbox": [round(float(x), 2) for x in box]
                    })

                    class_counts[class_id] += 1

                    if track_id is not None:
                        unique_ids.add(track_id)

            total_detections += len(detections)

            if detections:
                frames_with_detections += 1

            frames.append({
                "frame_number": frame_number,
                "detection_count": len(detections),
                "detections": detections
            })

            annotated = results[0].plot()
            writer.write(annotated)

            if frame_number % 100 == 0:
                print(
                    f"Processed {frame_number}/{total_frames} "
                    f"| Detections: {total_detections} "
                    f"| IDs: {len(unique_ids)}"
                )

    finally:
        cap.release()
        writer.release()

    elapsed = time.perf_counter() - start

    summary = {
        "experiment": "Urban Intersection PyTorch Tracking",
        "video": str(VIDEO),
        "model": str(MODEL),
        "device": "cpu",
        "resolution": [width, height],
        "source_fps": fps,
        "expected_frames": total_frames,
        "processed_frames": frame_number,
        "total_detections": total_detections,
        "frames_with_detections": frames_with_detections,
        "detection_rate_percent": (
            frames_with_detections / frame_number * 100
            if frame_number else 0
        ),
        "unique_track_ids": sorted(unique_ids),
        "unique_track_count": len(unique_ids),
        "class_detection_counts": dict(class_counts),
        "processing_seconds": elapsed,
        "processing_fps": (
            frame_number / elapsed if elapsed else 0
        )
    }

    with open(OUTPUT / "tracking_detections.json", "w") as f:
        json.dump({"frames": frames}, f, indent=2)

    with open(OUTPUT / "tracking_summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    print("\nPYTORCH TRACKING COMPLETED")
    print(json.dumps(summary, indent=2))
    print(f"\nResults saved in: {OUTPUT}")

if __name__ == "__main__":
    main()
