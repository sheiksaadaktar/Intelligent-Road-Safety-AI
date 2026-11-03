import json
import cv2
import numpy as np

from pathlib import Path
from collections import defaultdict, deque, Counter
from itertools import combinations

from risk_analysis import (
    RiskEventManager,
    bottom_center,
    smooth_velocity,
    calculate_trajectory_metrics,
    calculate_risk,
)

from safety_config import (
    HISTORY_SIZE,
    MAX_REASONABLE_VELOCITY,
    MAX_REASONABLE_RELATIVE_VELOCITY,
)

from compare_velocity_events import summarize_event


VIDEO = Path("data/tracking_test.mp4")

INPUTS = {
    "pytorch": Path("output/backend_tracking_audit/pytorch_detections.json"),
    "openvino": Path("output/backend_tracking_audit/openvino_detections.json"),
}

OUTPUT = Path("output/velocity_warmup_comparison")
MAX_GAP_FRAMES = 3

CONFIGURATIONS = [
    ("original", 1),
    ("gap_aware_1", 1),
    ("gap_aware_3", 3),
    ("gap_aware_5", 5),
]


def run_configuration(data, backend, method, minimum_history, fps):
    history = defaultdict(lambda: deque(maxlen=HISTORY_SIZE))
    previous_positions = {}
    previous_frames = {}

    manager = RiskEventManager(fps=fps)

    counts = Counter()
    risk_counts = Counter()

    for frame_record in data["frames"]:
        frame_number = int(frame_record["frame_number"])
        counts["frames"] += 1

        current_positions = {}

        for detection in frame_record.get("detections", []):
            track_id = int(detection["track_id"])

            position = bottom_center(
                np.asarray(detection["bbox"], dtype=np.float32)
            )

            current_positions[track_id] = position
            counts["detections"] += 1

            if track_id in previous_positions:
                frame_gap = frame_number - previous_frames[track_id]

                displacement = (
                    position - previous_positions[track_id]
                )

                if method == "original":
                    velocity = displacement * fps

                    if (
                        float(np.linalg.norm(velocity))
                        <= MAX_REASONABLE_VELOCITY
                    ):
                        history[track_id].append(velocity)

                elif method == "gap_aware":
                    if 1 <= frame_gap <= MAX_GAP_FRAMES:
                        velocity = displacement / (frame_gap / fps)

                        if (
                            float(np.linalg.norm(velocity))
                            <= MAX_REASONABLE_VELOCITY
                        ):
                            history[track_id].append(velocity)

                        if frame_gap > 1:
                            counts["corrected_gaps"] += 1

                    else:
                        history[track_id].clear()
                        counts["gap_resets"] += 1

                else:
                    raise ValueError(f"Unknown method: {method}")

            previous_positions[track_id] = position
            previous_frames[track_id] = frame_number

        observed_pairs = set()
        track_ids = list(current_positions.keys())

        for id_1, id_2 in combinations(track_ids, 2):
            counts["pairs_seen"] += 1

            if len(history[id_1]) < minimum_history:
                counts["insufficient_history"] += 1
                continue

            if len(history[id_2]) < minimum_history:
                counts["insufficient_history"] += 1
                continue

            velocity_1 = smooth_velocity(history[id_1])
            velocity_2 = smooth_velocity(history[id_2])

            relative_velocity = velocity_2 - velocity_1
            relative_speed = float(np.linalg.norm(relative_velocity))

            if relative_speed > MAX_REASONABLE_RELATIVE_VELOCITY:
                counts["relative_speed_rejections"] += 1
                continue

            metrics = calculate_trajectory_metrics(
                current_positions[id_1],
                velocity_1,
                current_positions[id_2],
                velocity_2,
            )

            risk, score = calculate_risk(metrics)

            pair = tuple(sorted((id_1, id_2)))
            observed_pairs.add(pair)

            manager.observe(
                frame_number,
                pair,
                risk,
                score,
                metrics,
            )

            counts["evaluated_pairs"] += 1
            risk_counts[risk] += 1

        manager.mark_missing_pairs(frame_number, observed_pairs)

    manager.finalize_all()

    events = [
        summarize_event(event)
        for event in manager.completed_events
    ]

    events.sort(
        key=lambda event: (
            event["start_frame"],
            event["pair"],
        )
    )

    return {
        "backend": backend,
        "configuration": method,
        "minimum_history": minimum_history,
        "counts": dict(counts),
        "risk_counts": dict(risk_counts),
        "confirmed_event_count": len(events),
        "confirmed_events": events,
    }


def main():
    if not VIDEO.exists():
        raise FileNotFoundError(VIDEO)

    cap = cv2.VideoCapture(str(VIDEO))

    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {VIDEO}")

    fps = float(cap.get(cv2.CAP_PROP_FPS))
    cap.release()

    if not np.isfinite(fps) or fps <= 0:
        raise RuntimeError(f"Invalid video FPS: {fps}")

    OUTPUT.mkdir(parents=True, exist_ok=False)

    report = {
        "video": str(VIDEO),
        "fps": fps,
        "history_capacity": HISTORY_SIZE,
        "max_gap_frames": MAX_GAP_FRAMES,
        "configurations": {},
    }

    for backend, input_path in INPUTS.items():
        print(f"\n========== {backend.upper()} ==========")

        if not input_path.exists():
            raise FileNotFoundError(input_path)

        data = json.loads(input_path.read_text(encoding="utf-8"))

        report["configurations"][backend] = {}

        for name, minimum_history in CONFIGURATIONS:
            method = (
                "original"
                if name == "original"
                else "gap_aware"
            )

            result = run_configuration(
                data,
                backend,
                method,
                minimum_history,
                fps,
            )

            report["configurations"][backend][name] = result

            print(f"\n{name}")
            print(
                "Minimum history:",
                minimum_history,
                "| Evaluated pairs:",
                result["counts"].get("evaluated_pairs", 0),
                "| Insufficient history:",
                result["counts"].get("insufficient_history", 0),
            )
            print("Risk counts:", result["risk_counts"])
            print(
                "Confirmed events:",
                result["confirmed_event_count"],
            )

            for event in result["confirmed_events"]:
                print(
                    "  Pair:", event["pair"],
                    "| Frames:", event["start_frame"],
                    "-", event["last_frame"],
                    "| Risk:", event["peak_risk"],
                    "| Score:", event["peak_score"],
                    "| Strong:", event["strong_observations"],
                )

    report_path = OUTPUT / "warmup_comparison.json"

    report_path.write_text(
        json.dumps(report, indent=2, allow_nan=False),
        encoding="utf-8",
    )

    print("\n========== COMPARISON COMPLETE ==========")
    print("Report:", report_path)
    print("Production files were not modified.")
    print("YOLO inference was not run.")


if __name__ == "__main__":
    main()
