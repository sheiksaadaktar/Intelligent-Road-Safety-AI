import csv
import json
from pathlib import Path

import cv2

ROOT = Path(__file__).resolve().parent.parent

VIDEO = ROOT / "data/tracking_test.mp4"
COMPARISON = ROOT / "output/velocity_experiment_consolidated/velocity_experiment_comparison.json"

DETECTION_FILES = {
    "pytorch": ROOT / "output/backend_tracking_audit/pytorch_detections.json",
    "openvino": ROOT / "output/backend_tracking_audit/openvino_detections.json",
}

OUTPUT = ROOT / "output/ground_truth_review"
FRAMES_PER_EVENT = 5


def load_json(path):
    if not path.exists():
        raise FileNotFoundError(f"Missing input file: {path}")
    with path.open("r", encoding="utf-8-sig") as handle:
        return json.load(handle)


def frame_samples(event):
    start = event["start_frame"]
    last_risk = event["last_risk_frame"]
    end = event["end_frame"]

    midpoint = (start + last_risk) // 2

    candidates = [
        start,
        min(start + 5, last_risk),
        midpoint,
        last_risk,
        end,
    ]

    return sorted(set(candidates))


def draw_detections(image, detections, target_pair, backend):
    target_ids = set(target_pair)

    for detection in detections:
        track_id = detection["track_id"]
        x1, y1, x2, y2 = [
            int(round(value)) for value in detection["bbox"]
        ]

        if track_id in target_ids:
            color = (0, 0, 255) if track_id == target_pair[0] else (0, 220, 0)
            thickness = 4
        else:
            color = (150, 150, 150)
            thickness = 1

        cv2.rectangle(image, (x1, y1), (x2, y2), color, thickness)

        label = f"{backend} ID {track_id} {detection['confidence']:.2f}"
        cv2.putText(
            image,
            label,
            (x1, max(22, y1 - 8)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55 if track_id in target_ids else 0.35,
            color,
            2 if track_id in target_ids else 1,
            cv2.LINE_AA,
        )

    return image


def main():
    for path in [VIDEO, COMPARISON, *DETECTION_FILES.values()]:
        if not path.exists():
            raise FileNotFoundError(f"Required input missing: {path}")

    comparison = load_json(COMPARISON)

    detection_data = {
        backend: load_json(path)
        for backend, path in DETECTION_FILES.items()
    }

    frame_maps = {}
    for backend, data in detection_data.items():
        frame_maps[backend] = {
            frame["frame_number"]: frame["detections"]
            for frame in data["frames"]
        }

    cap = cv2.VideoCapture(str(VIDEO))
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {VIDEO}")

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)

    OUTPUT.mkdir(parents=True, exist_ok=False)

    rows = []
    image_count = 0

    for backend in ("pytorch", "openvino"):
        events = comparison["backends"][backend][
            "stage_summaries"
        ]["visibility_v3"]["confirmed_events"]

        backend_dir = OUTPUT / backend
        backend_dir.mkdir()

        for index, event in enumerate(events, start=1):
            pair = event["pair"]
            event_id = f"{backend.upper()}_EVENT_{index:02d}"
            event_dir = backend_dir / event_id
            event_dir.mkdir()

            samples = frame_samples(event)

            for frame_number in samples:
                if not 1 <= frame_number <= total_frames:
                    continue

                cap.set(cv2.CAP_PROP_POS_FRAMES, frame_number - 1)
                ok, image = cap.read()

                if not ok:
                    print(f"WARNING: Could not read frame {frame_number}")
                    continue

                detections = frame_maps[backend].get(frame_number, [])

                image = draw_detections(
                    image.copy(),
                    detections,
                    pair,
                    backend,
                )

                header = (
                    f"{event_id} | Pair {pair[0]}-{pair[1]} | "
                    f"Frame {frame_number} | "
                    f"Peak {event['peak_risk']}"
                )

                cv2.rectangle(image, (0, 0), (1920, 42), (20, 20, 20), -1)
                cv2.putText(
                    image,
                    header,
                    (15, 29),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.75,
                    (255, 255, 255),
                    2,
                    cv2.LINE_AA,
                )

                image_path = event_dir / f"frame_{frame_number:04d}.jpg"
                if not cv2.imwrite(
                    str(image_path),
                    image,
                    [cv2.IMWRITE_JPEG_QUALITY, 92],
                ):
                    raise RuntimeError(f"Could not write {image_path}")

                image_count += 1

            rows.append({
                "event_id": event_id,
                "backend": backend,
                "track_pair": f"{pair[0]}-{pair[1]}",
                "peak_risk": event["peak_risk"],
                "start_frame": event["start_frame"],
                "last_risk_frame": event["last_risk_frame"],
                "last_visible_frame": event["last_visible_frame"],
                "end_frame": event["end_frame"],
                "end_reason": event["end_reason"],
                "reviewed_collision_or_near_collision": "",
                "observed_interaction": "",
                "relative_motion": "",
                "occlusion_or_disappearance": "",
                "track_identity_correct": "",
                "review_confidence": "",
                "notes": "",
            })

    cap.release()

    csv_path = OUTPUT / "manual_event_annotations.csv"
    with csv_path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    readme = """GROUND-TRUTH REVIEW PACKAGE

Purpose:
Manually review the seven confirmed events from each tracking backend.

Folders:
- pytorch/: frames annotated with PyTorch track IDs.
- openvino/: frames annotated with OpenVINO track IDs.

Images:
Each event folder contains up to five selected frames:
- Event start
- Five frames after start
- Approximate midpoint
- Last valid risk frame
- Lifecycle end frame

Bounding boxes:
- Red: first track in the event pair.
- Green: second track in the event pair.
- Grey: other detections from the same backend.

Important:
The bounding boxes are algorithm outputs, not ground truth.
Use the original video to judge what actually happens.
Track IDs are local to each backend and must not be compared directly.

CSV annotation fields:
- reviewed_collision_or_near_collision: yes / no / uncertain
- observed_interaction: describe what the objects visibly do
- relative_motion: approaching / separating / crossing / parallel / unclear
- occlusion_or_disappearance: describe any visibility interruption
- track_identity_correct: yes / no / uncertain
- review_confidence: high / medium / low
- notes: free-text observations

Do not infer a collision from bounding-box overlap alone.
A single still image may be insufficient to establish motion.
Review the corresponding video sequence before assigning labels.

Frame numbers are 1-based.
Video FPS: approximately 59.94.
"""

    (OUTPUT / "README.txt").write_text(readme, encoding="utf-8")

    print("\nGROUND-TRUTH REVIEW PACKAGE CREATED")
    print(f"Output directory: {OUTPUT}")
    print(f"Event annotation rows: {len(rows)}")
    print(f"Extracted images: {image_count}")
    print(f"Video frames: {total_frames}")
    print(f"Video FPS: {fps:.6f}")
    print(f"CSV: {csv_path}")
    print(f"README: {OUTPUT / 'README.txt'}")


if __name__ == "__main__":
    main()
