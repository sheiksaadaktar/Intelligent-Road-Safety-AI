import json
import cv2
import numpy as np

from pathlib import Path
from collections import defaultdict, deque, Counter
from itertools import combinations

# Import the ACTUAL production implementation.
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

VIDEO = Path("data/tracking_test.mp4")

INPUTS = {
    "pytorch": Path("output/backend_tracking_audit/pytorch_detections.json"),
    "openvino": Path("output/backend_tracking_audit/openvino_detections.json"),
}

OUTPUT = Path("output/velocity_event_comparison")

MAX_GAP_FRAMES = 3


def finite_or_none(value):
    if isinstance(value, (int, float, np.number)):
        value = float(value)
        return value if np.isfinite(value) else None
    return value


def summarize_event(event):
    return {
        "pair": list(event["pair"]),
        "start_frame": event["start_frame"],
        "last_frame": event["last_frame"],
        "duration_seconds": finite_or_none(event["duration_seconds"]),
        "frames_seen": event["frames_seen"],
        "strong_observations": event["strong_observations"],
        "strong_frames": list(event["strong_frames"]),
        "peak_risk": event["peak_risk"],
        "peak_score": event["peak_score"],
        "min_distance": finite_or_none(event["min_distance"]),
        "min_predicted_distance": finite_or_none(
            event["min_predicted_distance"]
        ),
        "max_approach_speed": finite_or_none(
            event["max_approach_speed"]
        ),
        "min_tca": finite_or_none(event["min_tca"]),
        "max_convergence": finite_or_none(event["max_convergence"]),
    }


def run_method(data, backend, method, fps):
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

        # Match production object extraction and velocity updates.
        for detection in frame_record.get("detections", []):
            track_id = int(detection["track_id"])
            position = bottom_center(
                np.asarray(detection["bbox"], dtype=np.float32)
            )

            current_positions[track_id] = position
            counts["detections"] += 1

            if track_id in previous_positions:
                frame_gap = (
                    frame_number - previous_frames[track_id]
                )

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

                elif method == "experimental":
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

        # This set is deliberately method-specific.
        # It follows production: only pairs that reach risk evaluation
        # are considered observed by RiskEventManager.
        observed_pairs = set()

        track_ids = list(current_positions.keys())

        for id_1, id_2 in combinations(track_ids, 2):
            counts["pairs_seen"] += 1

            if not history[id_1] or not history[id_2]:
                counts["no_velocity_history"] += 1
                continue

            velocity_1 = smooth_velocity(history[id_1])
            velocity_2 = smooth_velocity(history[id_2])

            relative_velocity = velocity_2 - velocity_1

            relative_speed = float(
                np.linalg.norm(relative_velocity)
            )

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

        # Same production missing-pair handling.
        manager.mark_missing_pairs(
            frame_number,
            observed_pairs,
        )

    # Same production end-of-video finalization.
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
        "method": method,
        "counts": dict(counts),
        "risk_counts": dict(risk_counts),
        "confirmed_event_count": len(events),
        "confirmed_events": events,
    }


def compare_event_sets(original, experimental):
    original_events = original["confirmed_events"]
    experimental_events = experimental["confirmed_events"]

    original_pairs = Counter(
        tuple(event["pair"]) for event in original_events
    )
    experimental_pairs = Counter(
        tuple(event["pair"]) for event in experimental_events
    )

    common_pairs = sorted(
        set(original_pairs) & set(experimental_pairs)
    )

    return {
        "original_event_count": len(original_events),
        "experimental_event_count": len(experimental_events),
        "event_count_difference": (
            len(experimental_events) - len(original_events)
        ),
        "original_event_pairs": [
            list(pair) for pair in sorted(original_pairs)
        ],
        "experimental_event_pairs": [
            list(pair) for pair in sorted(experimental_pairs)
        ],
        "pairs_present_in_both_methods": [
            list(pair) for pair in common_pairs
        ],
        "pairs_only_in_original": [
            list(pair)
            for pair in sorted(set(original_pairs) - set(experimental_pairs))
        ],
        "pairs_only_in_experimental": [
            list(pair)
            for pair in sorted(set(experimental_pairs) - set(original_pairs))
        ],
        "pair_event_count_differences": {
            str(pair): {
                "original": original_pairs[pair],
                "experimental": experimental_pairs[pair],
            }
            for pair in sorted(set(original_pairs) | set(experimental_pairs))
            if original_pairs[pair] != experimental_pairs[pair]
        },
    }


def main():
    OUTPUT.mkdir(parents=True, exist_ok=False)

    cap = cv2.VideoCapture(str(VIDEO))
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {VIDEO}")

    fps = float(cap.get(cv2.CAP_PROP_FPS))
    cap.release()

    if not fps or fps <= 0:
        raise RuntimeError(f"Invalid video FPS: {fps}")

    report = {
        "methodology": {
            "fps": fps,
            "inputs": {
                name: str(path) for name, path in INPUTS.items()
            },
            "position": "bounding-box bottom-centre",
            "original_velocity": "displacement * FPS",
            "experimental_velocity": (
                "displacement / (frame_gap / FPS)"
            ),
            "experimental_max_gap_frames": MAX_GAP_FRAMES,
            "history_size": HISTORY_SIZE,
            "max_reasonable_velocity": MAX_REASONABLE_VELOCITY,
            "max_relative_velocity": MAX_REASONABLE_RELATIVE_VELOCITY,
            "risk_calculation": "imported directly from src/risk_analysis.py",
            "event_manager": "imported directly from src/risk_analysis.py",
            "event_confirmation": "production RiskEventManager",
            "missing_pair_handling": (
                "production mark_missing_pairs() using method-specific "
                "observed_pairs"
            ),
            "risk_thresholds_changed": False,
            "production_files_modified": False,
            "yolo_rerun": False,
        },
        "backends": {},
    }

    for backend, input_path in INPUTS.items():
        print(f"\n========== {backend.upper()} ==========")

        if not input_path.exists():
            raise FileNotFoundError(input_path)

        with input_path.open("r", encoding="utf-8") as file:
            data = json.load(file)

        original = run_method(
            data, backend, "original", fps
        )

        experimental = run_method(
            data, backend, "experimental", fps
        )

        comparison = compare_event_sets(
            original,
            experimental,
        )

        report["backends"][backend] = {
            "original": original,
            "experimental": experimental,
            "comparison": comparison,
        }

        for result in (original, experimental):
            print(f"\nMethod: {result['method']}")
            print("Frames:", result["counts"].get("frames", 0))
            print("Detections:", result["counts"].get("detections", 0))
            print("Evaluated pairs:", result["counts"].get("evaluated_pairs", 0))
            print("No velocity history:", result["counts"].get("no_velocity_history", 0))
            print("Relative speed rejections:", result["counts"].get("relative_speed_rejections", 0))
            print("Risk counts:", result["risk_counts"])
            print("Confirmed events:", result["confirmed_event_count"])

            for index, event in enumerate(result["confirmed_events"], 1):
                print(
                    f"  Event {index}: pair={event['pair']} "
                    f"frames={event['start_frame']}-{event['last_frame']} "
                    f"risk={event['peak_risk']} "
                    f"score={event['peak_score']} "
                    f"strong_frames={event['strong_frames']}"
                )

        print("\nEvent comparison:")
        print(json.dumps(comparison, indent=2))

    report_path = OUTPUT / "velocity_event_comparison.json"

    with report_path.open("w", encoding="utf-8") as file:
        json.dump(report, file, indent=2, allow_nan=False)

    print("\n========== REPLAY COMPLETE ==========")
    print("Report:", report_path)
    print("Production files were not modified.")
    print("YOLO inference was not run.")


if __name__ == "__main__":
    main()
