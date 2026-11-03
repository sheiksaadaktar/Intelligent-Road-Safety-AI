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

OUTPUT = Path("output/velocity_gap_policy_comparison")

POLICIES = [
    {"name": "original", "mode": "original", "max_gap": None},
    {"name": "reset_1", "mode": "reset", "max_gap": 1},
    {"name": "reset_3", "mode": "reset", "max_gap": 3},
    {"name": "reset_5", "mode": "reset", "max_gap": 5},
    {"name": "reset_8", "mode": "reset", "max_gap": 8},
    {"name": "retain_3", "mode": "retain", "max_gap": 3},
]

MINIMUM_HISTORY = 1


def update_velocity(
    track_id,
    position,
    frame_number,
    history,
    previous_positions,
    previous_frames,
    policy,
    counts,
    fps,
):
    if track_id in previous_positions:
        frame_gap = frame_number - previous_frames[track_id]
        displacement = position - previous_positions[track_id]

        if policy["mode"] == "original":
            velocity = displacement * fps

            if float(np.linalg.norm(velocity)) <= MAX_REASONABLE_VELOCITY:
                history[track_id].append(velocity)
            else:
                counts["velocity_limit_rejections"] += 1

        elif frame_gap <= 0:
            history[track_id].clear()
            counts["invalid_gap_resets"] += 1

        elif frame_gap <= policy["max_gap"]:
            velocity = displacement / (frame_gap / fps)

            if float(np.linalg.norm(velocity)) <= MAX_REASONABLE_VELOCITY:
                history[track_id].append(velocity)

                if frame_gap > 1:
                    counts["corrected_gaps"] += 1
            else:
                counts["velocity_limit_rejections"] += 1

        elif policy["mode"] == "reset":
            history[track_id].clear()
            counts["gap_resets"] += 1
            counts["over_limit_gaps"] += 1

        elif policy["mode"] == "retain":
            # Keep previous valid samples, but do not append a velocity
            # calculated across this long gap.
            counts["retained_history_gaps"] += 1
            counts["over_limit_gaps"] += 1

        else:
            raise ValueError(f"Unknown policy: {policy}")

    previous_positions[track_id] = position
    previous_frames[track_id] = frame_number


def run_policy(data, backend, policy, fps):
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

            update_velocity(
                track_id,
                position,
                frame_number,
                history,
                previous_positions,
                previous_frames,
                policy,
                counts,
                fps,
            )

        observed_pairs = set()
        track_ids = list(current_positions.keys())

        for id_1, id_2 in combinations(track_ids, 2):
            counts["pairs_seen"] += 1

            if len(history[id_1]) < MINIMUM_HISTORY:
                counts["insufficient_history"] += 1
                continue

            if len(history[id_2]) < MINIMUM_HISTORY:
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
        key=lambda event: (event["start_frame"], event["pair"])
    )

    return {
        "backend": backend,
        "policy": policy,
        "minimum_history": MINIMUM_HISTORY,
        "counts": dict(counts),
        "risk_counts": dict(risk_counts),
        "confirmed_event_count": len(events),
        "confirmed_events": events,
    }


def compare_to_baseline(baseline, candidate):
    base_events = {
        tuple(event["pair"]): event
        for event in baseline["confirmed_events"]
    }

    candidate_events = {
        tuple(event["pair"]): event
        for event in candidate["confirmed_events"]
    }

    common = sorted(set(base_events) & set(candidate_events))

    event_differences = []

    for pair in common:
        old = base_events[pair]
        new = candidate_events[pair]

        event_differences.append({
            "pair": list(pair),
            "baseline_start": old["start_frame"],
            "candidate_start": new["start_frame"],
            "start_delta": new["start_frame"] - old["start_frame"],
            "baseline_end": old["last_frame"],
            "candidate_end": new["last_frame"],
            "end_delta": new["last_frame"] - old["last_frame"],
            "baseline_risk": old["peak_risk"],
            "candidate_risk": new["peak_risk"],
            "baseline_score": old["peak_score"],
            "candidate_score": new["peak_score"],
            "baseline_strong": old["strong_observations"],
            "candidate_strong": new["strong_observations"],
        })

    baseline_risks = baseline["risk_counts"]
    candidate_risks = candidate["risk_counts"]

    risk_deltas = {
        risk: candidate_risks.get(risk, 0) - baseline_risks.get(risk, 0)
        for risk in ["LOW", "MEDIUM", "HIGH", "CRITICAL"]
    }

    return {
        "baseline_event_count": len(base_events),
        "candidate_event_count": len(candidate_events),
        "missing_event_pairs": [
            list(pair) for pair in sorted(set(base_events) - set(candidate_events))
        ],
        "additional_event_pairs": [
            list(pair) for pair in sorted(set(candidate_events) - set(base_events))
        ],
        "common_event_details": event_differences,
        "risk_count_deltas": risk_deltas,
        "evaluated_pair_delta": (
            candidate["counts"].get("evaluated_pairs", 0)
            - baseline["counts"].get("evaluated_pairs", 0)
        ),
    }


def main():
    if not VIDEO.exists():
        raise FileNotFoundError(VIDEO)

    for input_path in INPUTS.values():
        if not input_path.exists():
            raise FileNotFoundError(input_path)

    cap = cv2.VideoCapture(str(VIDEO))

    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {VIDEO}")

    fps = float(cap.get(cv2.CAP_PROP_FPS))
    cap.release()

    if not np.isfinite(fps) or fps <= 0:
        raise RuntimeError(f"Invalid video FPS: {fps}")

    OUTPUT.mkdir(parents=True, exist_ok=False)

    report = {
        "methodology": {
            "video": str(VIDEO),
            "fps": fps,
            "history_capacity": HISTORY_SIZE,
            "minimum_history": MINIMUM_HISTORY,
            "position": "bounding-box bottom-centre",
            "gap_aware_velocity": "displacement / (frame_gap / FPS)",
            "original_velocity": "displacement * FPS",
            "risk_calculation": "production calculate_trajectory_metrics and calculate_risk",
            "event_manager": "production RiskEventManager",
            "risk_thresholds_changed": False,
            "production_files_modified": False,
            "yolo_inference_run": False,
        },
        "backends": {},
    }

    for backend, input_path in INPUTS.items():
        print(f"\n========== {backend.upper()} ==========")

        data = json.loads(input_path.read_text(encoding="utf-8"))
        backend_results = {}

        for policy in POLICIES:
            result = run_policy(data, backend, policy, fps)
            backend_results[policy["name"]] = result

            counts = result["counts"]

            print(f"\nPolicy: {policy['name']}")
            print("Evaluated pairs:", counts.get("evaluated_pairs", 0))
            print("Insufficient history:", counts.get("insufficient_history", 0))
            print("Corrected gaps:", counts.get("corrected_gaps", 0))
            print("Gap resets:", counts.get("gap_resets", 0))
            print("Retained-history gaps:", counts.get("retained_history_gaps", 0))
            print("Velocity limit rejections:", counts.get("velocity_limit_rejections", 0))
            print("Risk counts:", result["risk_counts"])
            print("Confirmed events:", result["confirmed_event_count"])

            for event in result["confirmed_events"]:
                print(
                    f"  Pair {event['pair']} "
                    f"| {event['start_frame']}-{event['last_frame']} "
                    f"| {event['peak_risk']} "
                    f"| score {event['peak_score']} "
                    f"| strong {event['strong_observations']}"
                )

        baseline = backend_results["original"]
        comparisons = {}

        for policy in POLICIES[1:]:
            name = policy["name"]
            comparisons[name] = compare_to_baseline(
                baseline,
                backend_results[name],
            )

        report["backends"][backend] = {
            "policies": backend_results,
            "comparisons_to_original": comparisons,
        }

    report_path = OUTPUT / "gap_policy_comparison.json"

    report_path.write_text(
        json.dumps(report, indent=2, allow_nan=False),
        encoding="utf-8",
    )

    print("\n========== EXPERIMENT COMPLETE ==========")
    print("Report:", report_path)
    print("Production files were not modified.")
    print("YOLO inference was not run.")


if __name__ == "__main__":
    main()
