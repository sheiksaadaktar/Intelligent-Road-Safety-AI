import json
import csv
import cv2
import numpy as np
from collections import defaultdict, deque
from pathlib import Path

FPS_VIDEO = Path("data/tracking_test.mp4")
INPUTS = {
    "pytorch": Path("output/backend_tracking_audit/pytorch_detections.json"),
    "openvino": Path("output/backend_tracking_audit/openvino_detections.json"),
}
OUTPUT_DIR = Path("output/velocity_controlled_comparison")

HISTORY_SIZE = 5
MAX_REASONABLE_VELOCITY = 800.0
MAX_VELOCITY_GAP_FRAMES = 3

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

cap = cv2.VideoCapture(str(FPS_VIDEO))
if not cap.isOpened():
    raise RuntimeError(f"Could not open video: {FPS_VIDEO}")

fps = cap.get(cv2.CAP_PROP_FPS)
cap.release()

if not fps or fps <= 0:
    raise RuntimeError("Could not determine video FPS.")

print(f"Video FPS: {fps:.6f}")
print(f"Maximum accepted velocity: {MAX_REASONABLE_VELOCITY} px/sec")
print(f"Experimental maximum frame gap: {MAX_VELOCITY_GAP_FRAMES}")
print(f"Output directory: {OUTPUT_DIR}")

all_summaries = {}
all_detail_rows = []
all_track_rows = []

for backend, input_path in INPUTS.items():
    print(f"\nProcessing {backend}...")

    if not input_path.exists():
        raise FileNotFoundError(input_path)

    with input_path.open("r", encoding="utf-8") as f:
        data = json.load(f)

    previous_positions = {}
    previous_frames = {}

    original_histories = defaultdict(lambda: deque(maxlen=HISTORY_SIZE))
    experimental_histories = defaultdict(lambda: deque(maxlen=HISTORY_SIZE))

    counters = {
        "detections": 0,
        "original_velocity_samples": 0,
        "experimental_velocity_samples": 0,
        "original_rejected_samples": 0,
        "experimental_rejected_samples": 0,
        "experimental_gap_resets": 0,
        "experimental_corrected_gaps": 0,
        "gaps_greater_than_one": 0,
    }

    gap_counts = defaultdict(int)
    per_track = defaultdict(lambda: {
        "detections": 0,
        "original_samples": 0,
        "experimental_samples": 0,
        "gap_resets": 0,
        "corrected_gaps": 0,
        "original_smoothed_speeds": [],
        "experimental_smoothed_speeds": [],
        "smoothed_speed_differences": [],
    })

    comparable_smoothed_differences = []

    def mean_velocity(history):
        if not history:
            return None
        return np.mean(np.array(history, dtype=float), axis=0)

    for frame_record in data["frames"]:
        frame_number = int(frame_record["frame_number"])

        for detection in frame_record.get("detections", []):
            track_id = int(detection["track_id"])
            x1, y1, x2, y2 = map(float, detection["bbox"])

            # Bottom-centre position, matching the production risk logic.
            position = np.array([(x1 + x2) / 2.0, y2], dtype=float)

            counters["detections"] += 1
            per_track[track_id]["detections"] += 1

            frame_gap = None
            original_velocity = None
            experimental_velocity = None

            if track_id in previous_positions:
                frame_gap = frame_number - previous_frames[track_id]
                gap_counts[str(frame_gap)] += 1

                if frame_gap > 1:
                    counters["gaps_greater_than_one"] += 1

                displacement = position - previous_positions[track_id]

                # Original calculation: displacement multiplied by FPS,
                # regardless of how many frames elapsed.
                original_velocity = displacement * fps
                original_speed = float(np.linalg.norm(original_velocity))

                if original_speed <= MAX_REASONABLE_VELOCITY:
                    original_histories[track_id].append(original_velocity)
                    counters["original_velocity_samples"] += 1
                    per_track[track_id]["original_samples"] += 1
                else:
                    counters["original_rejected_samples"] += 1

                # Experimental calculation: account for elapsed frames.
                if 1 <= frame_gap <= MAX_VELOCITY_GAP_FRAMES:
                    experimental_velocity = displacement / (frame_gap / fps)
                    experimental_speed = float(np.linalg.norm(experimental_velocity))

                    if experimental_speed <= MAX_REASONABLE_VELOCITY:
                        experimental_histories[track_id].append(experimental_velocity)
                        counters["experimental_velocity_samples"] += 1
                        per_track[track_id]["experimental_samples"] += 1
                    else:
                        counters["experimental_rejected_samples"] += 1

                    if frame_gap > 1:
                        counters["experimental_corrected_gaps"] += 1
                        per_track[track_id]["corrected_gaps"] += 1

                else:
                    experimental_histories[track_id].clear()
                    counters["experimental_gap_resets"] += 1
                    per_track[track_id]["gap_resets"] += 1

            # Update the previous observation after both calculations.
            previous_positions[track_id] = position
            previous_frames[track_id] = frame_number

            original_smoothed = mean_velocity(original_histories[track_id])
            experimental_smoothed = mean_velocity(experimental_histories[track_id])

            original_smoothed_speed = (
                float(np.linalg.norm(original_smoothed))
                if original_smoothed is not None else None
            )
            experimental_smoothed_speed = (
                float(np.linalg.norm(experimental_smoothed))
                if experimental_smoothed is not None else None
            )

            difference = None
            if original_smoothed is not None and experimental_smoothed is not None:
                difference = float(np.linalg.norm(
                    original_smoothed - experimental_smoothed
                ))
                comparable_smoothed_differences.append(difference)
                per_track[track_id]["smoothed_speed_differences"].append(difference)

            if original_smoothed_speed is not None:
                per_track[track_id]["original_smoothed_speeds"].append(
                    original_smoothed_speed
                )

            if experimental_smoothed_speed is not None:
                per_track[track_id]["experimental_smoothed_speeds"].append(
                    experimental_smoothed_speed
                )

            all_detail_rows.append({
                "backend": backend,
                "frame": frame_number,
                "track_id": track_id,
                "frame_gap": frame_gap,
                "original_raw_speed": (
                    round(float(np.linalg.norm(original_velocity)), 4)
                    if original_velocity is not None else None
                ),
                "experimental_raw_speed": (
                    round(float(np.linalg.norm(experimental_velocity)), 4)
                    if experimental_velocity is not None else None
                ),
                "original_smoothed_speed": (
                    round(original_smoothed_speed, 4)
                    if original_smoothed_speed is not None else None
                ),
                "experimental_smoothed_speed": (
                    round(experimental_smoothed_speed, 4)
                    if experimental_smoothed_speed is not None else None
                ),
                "smoothed_velocity_vector_difference": (
                    round(difference, 4) if difference is not None else None
                ),
                "experimental_history_length": len(experimental_histories[track_id]),
            })

    track_summaries = []

    for track_id, values in sorted(per_track.items()):
        original_speeds = values["original_smoothed_speeds"]
        experimental_speeds = values["experimental_smoothed_speeds"]
        differences = values["smoothed_speed_differences"]

        row = {
            "backend": backend,
            "track_id": track_id,
            "detections": values["detections"],
            "original_velocity_samples": values["original_samples"],
            "experimental_velocity_samples": values["experimental_samples"],
            "experimental_gap_resets": values["gap_resets"],
            "experimental_corrected_gaps": values["corrected_gaps"],
            "original_mean_smoothed_speed": (
                round(float(np.mean(original_speeds)), 4)
                if original_speeds else None
            ),
            "experimental_mean_smoothed_speed": (
                round(float(np.mean(experimental_speeds)), 4)
                if experimental_speeds else None
            ),
            "mean_smoothed_vector_difference": (
                round(float(np.mean(differences)), 4)
                if differences else None
            ),
        }

        track_summaries.append(row)
        all_track_rows.append(row)

    summary = {
        "backend": backend,
        "video_fps": fps,
        "frames": len(data["frames"]),
        "detections": counters["detections"],
        "unique_tracks": len(per_track),
        "original_velocity_samples": counters["original_velocity_samples"],
        "experimental_velocity_samples": counters["experimental_velocity_samples"],
        "original_rejected_samples": counters["original_rejected_samples"],
        "experimental_rejected_samples": counters["experimental_rejected_samples"],
        "experimental_gap_resets": counters["experimental_gap_resets"],
        "experimental_corrected_gaps": counters["experimental_corrected_gaps"],
        "observations_with_gap_greater_than_one": counters["gaps_greater_than_one"],
        "frame_gap_distribution": dict(sorted(
            gap_counts.items(), key=lambda item: int(item[0])
        )),
        "comparable_smoothed_velocity_observations": len(
            comparable_smoothed_differences
        ),
        "mean_smoothed_velocity_vector_difference": (
            round(float(np.mean(comparable_smoothed_differences)), 4)
            if comparable_smoothed_differences else None
        ),
        "max_smoothed_velocity_vector_difference": (
            round(float(np.max(comparable_smoothed_differences)), 4)
            if comparable_smoothed_differences else None
        ),
        "per_track": track_summaries,
    }

    all_summaries[backend] = summary

    print(f"Frames: {summary['frames']}")
    print(f"Detections: {summary['detections']}")
    print(f"Unique tracks: {summary['unique_tracks']}")
    print(f"Original velocity samples: {summary['original_velocity_samples']}")
    print(f"Experimental velocity samples: {summary['experimental_velocity_samples']}")
    print(f"Experimental gap resets: {summary['experimental_gap_resets']}")
    print(f"Experimental corrected gaps: {summary['experimental_corrected_gaps']}")
    print(
        "Mean smoothed velocity-vector difference:",
        summary["mean_smoothed_velocity_vector_difference"]
    )

# Save machine-readable summary.
summary_path = OUTPUT_DIR / "velocity_comparison_summary.json"
with summary_path.open("w", encoding="utf-8") as f:
    json.dump({
        "methodology": {
            "original": "displacement * FPS",
            "experimental": (
                "displacement / (frame_gap / FPS); reset history "
                f"when frame_gap > {MAX_VELOCITY_GAP_FRAMES}"
            ),
            "position": "bounding-box bottom-centre",
            "history_size": HISTORY_SIZE,
            "max_reasonable_velocity_px_per_sec": MAX_REASONABLE_VELOCITY,
            "risk_thresholds_changed": False,
            "yolo_rerun": False,
        },
        "backends": all_summaries,
    }, f, indent=2)

# Save per-observation detail.
detail_path = OUTPUT_DIR / "velocity_observation_details.csv"
if all_detail_rows:
    with detail_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(all_detail_rows[0].keys()))
        writer.writeheader()
        writer.writerows(all_detail_rows)

# Save per-track summary.
track_path = OUTPUT_DIR / "velocity_track_summary.csv"
if all_track_rows:
    with track_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(all_track_rows[0].keys()))
        writer.writeheader()
        writer.writerows(all_track_rows)

print("\nCONTROLLED VELOCITY COMPARISON COMPLETE")
print(f"Summary: {summary_path}")
print(f"Observation details: {detail_path}")
print(f"Track summary: {track_path}")
print("\nProduction files were not modified.")
