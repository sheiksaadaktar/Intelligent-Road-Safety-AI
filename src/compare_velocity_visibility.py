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
    EVENT_GAP_TOLERANCE,
)

from compare_velocity_events import summarize_event


VIDEO = Path("data/tracking_test.mp4")

INPUTS = {
    "pytorch": Path("output/backend_tracking_audit/pytorch_detections.json"),
    "openvino": Path("output/backend_tracking_audit/openvino_detections.json"),
}

BASELINE_REPORT = Path(
    "output/velocity_gap_policy_comparison/gap_policy_comparison.json"
)

OUTPUT = Path("output/velocity_visibility_experiment")

MAX_VELOCITY_GAP = 3
MINIMUM_HISTORY = 1


def update_velocity(
    track_id,
    position,
    frame_number,
    history,
    previous_positions,
    previous_frames,
    counts,
    fps,
):
    if track_id in previous_positions:
        frame_gap = frame_number - previous_frames[track_id]
        displacement = position - previous_positions[track_id]

        if frame_gap <= 0:
            history[track_id].clear()
            counts["invalid_gap_resets"] += 1

        elif frame_gap <= MAX_VELOCITY_GAP:
            velocity = displacement / (frame_gap / fps)

            if float(np.linalg.norm(velocity)) <= MAX_REASONABLE_VELOCITY:
                history[track_id].append(velocity)

                if frame_gap > 1:
                    counts["corrected_gaps"] += 1
            else:
                counts["velocity_limit_rejections"] += 1

        else:
            history[track_id].clear()
            counts["gap_resets"] += 1

    previous_positions[track_id] = position
    previous_frames[track_id] = frame_number


def run_visibility_policy(data, backend, fps):
    history = defaultdict(lambda: deque(maxlen=HISTORY_SIZE))
    previous_positions = {}
    previous_frames = {}

    manager = RiskEventManager(fps=fps)

    counts = Counter()
    risk_counts = Counter()

    last_risk_frame = {}

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
                counts,
                fps,
            )

        track_ids = list(current_positions.keys())

        visible_pairs = {
            tuple(sorted(pair))
            for pair in combinations(track_ids, 2)
        }

        counts["visible_pairs"] += len(visible_pairs)

        # Update visibility independently from risk evaluation.
        # A pair that remains visible is not treated as missing
        # merely because its velocity is temporarily unavailable.

        for pair in list(manager.active_events.keys()):
            event = manager.active_events.get(pair)

            if event is None:
                continue

            if pair in visible_pairs:
                visibility_gap = frame_number - event["last_frame"]

                if visibility_gap > EVENT_GAP_TOLERANCE:
                    event["end_reason"] = "disappearance_timeout"
                    manager._finalize_event(pair)
                    counts["events_ended_after_disappearance"] += 1
                else:
                    # last_frame now represents the latest visible frame.
                    # This does not increment frames_seen or add risk evidence.
                    event["last_frame"] = frame_number

            else:
                visibility_gap = frame_number - event["last_frame"]

                if visibility_gap > EVENT_GAP_TOLERANCE:
                    event["end_reason"] = "disappearance_timeout"
                    manager._finalize_event(pair)
                    counts["events_ended_after_disappearance"] += 1

        evaluated_pairs = set()

        for id_1, id_2 in combinations(track_ids, 2):
            pair = tuple(sorted((id_1, id_2)))
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

            manager.observe(
                frame_number,
                pair,
                risk,
                score,
                metrics,
            )

            evaluated_pairs.add(pair)
            last_risk_frame[pair] = frame_number

            counts["evaluated_pairs"] += 1
            risk_counts[risk] += 1

        # Visible but unevaluable pairs do not produce LOW-risk evidence.
        # However, an event cannot remain unassessable indefinitely.

        for pair in list(manager.active_events.keys()):
            if pair not in visible_pairs:
                continue

            last_valid_risk = last_risk_frame.get(pair)

            if last_valid_risk is None:
                continue

            unevaluated_gap = frame_number - last_valid_risk

            if (
                unevaluated_gap > EVENT_GAP_TOLERANCE
                and pair not in evaluated_pairs
            ):
                event = manager.active_events.get(pair)

                if event is not None:
                    event["end_reason"] = "visible_but_unassessable_timeout"
                    manager._finalize_event(pair)
                    counts["events_ended_unassessable"] += 1

        counts["visible_but_unevaluated_pairs"] += (
            len(visible_pairs - evaluated_pairs)
        )

    # Any events still active at the end of the video are finalized.
    for pair in list(manager.active_events.keys()):
        event = manager.active_events.get(pair)

        if event is not None:
            event["end_reason"] = "end_of_video"

    manager.finalize_all()

    events = []

    for raw_event in manager.completed_events:
        event = summarize_event(raw_event)
        event["end_reason"] = raw_event.get(
            "end_reason",
            "end_of_video",
        )
        events.append(event)

    events.sort(
        key=lambda event: (event["start_frame"], event["pair"])
    )

    return {
        "backend": backend,
        "policy": "reset_3_visibility",
        "minimum_history": MINIMUM_HISTORY,
        "max_velocity_gap": MAX_VELOCITY_GAP,
        "counts": dict(counts),
        "risk_counts": dict(risk_counts),
        "confirmed_event_count": len(events),
        "confirmed_events": events,
    }


def compare_events(baseline, candidate):
    baseline_events = {
        tuple(event["pair"]): event
        for event in baseline["confirmed_events"]
    }

    candidate_events = {
        tuple(event["pair"]): event
        for event in candidate["confirmed_events"]
    }

    common_pairs = sorted(
        set(baseline_events) & set(candidate_events)
    )

    details = []

    for pair in common_pairs:
        old = baseline_events[pair]
        new = candidate_events[pair]

        details.append({
            "pair": list(pair),
            "baseline_start": old["start_frame"],
            "candidate_start": new["start_frame"],
            "start_delta": (
                new["start_frame"] - old["start_frame"]
            ),
            "baseline_end": old["last_frame"],
            "candidate_end": new["last_frame"],
            "end_delta": (
                new["last_frame"] - old["last_frame"]
            ),
            "baseline_risk": old["peak_risk"],
            "candidate_risk": new["peak_risk"],
            "baseline_score": old["peak_score"],
            "candidate_score": new["peak_score"],
            "baseline_strong": old["strong_observations"],
            "candidate_strong": new["strong_observations"],
            "candidate_end_reason": new.get("end_reason"),
        })

    risk_deltas = {
        risk: (
            candidate["risk_counts"].get(risk, 0)
            - baseline["risk_counts"].get(risk, 0)
        )
        for risk in ["LOW", "MEDIUM", "HIGH", "CRITICAL"]
    }

    return {
        "baseline_event_count": len(baseline_events),
        "candidate_event_count": len(candidate_events),
        "missing_event_pairs": [
            list(pair)
            for pair in sorted(
                set(baseline_events) - set(candidate_events)
            )
        ],
        "additional_event_pairs": [
            list(pair)
            for pair in sorted(
                set(candidate_events) - set(baseline_events)
            )
        ],
        "risk_count_deltas": risk_deltas,
        "evaluated_pair_delta": (
            candidate["counts"].get("evaluated_pairs", 0)
            - baseline["counts"].get("evaluated_pairs", 0)
        ),
        "common_event_details": details,
    }


def main():
    if not VIDEO.exists():
        raise FileNotFoundError(VIDEO)

    if not BASELINE_REPORT.exists():
        raise FileNotFoundError(BASELINE_REPORT)

    for input_path in INPUTS.values():
        if not input_path.exists():
            raise FileNotFoundError(input_path)

    cap = cv2.VideoCapture(str(VIDEO))

    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {VIDEO}")

    fps = float(cap.get(cv2.CAP_PROP_FPS))
    cap.release()

    if not np.isfinite(fps) or fps <= 0:
        raise RuntimeError(f"Invalid FPS: {fps}")

    baseline_report = json.loads(
        BASELINE_REPORT.read_text(encoding="utf-8")
    )

    OUTPUT.mkdir(parents=True, exist_ok=False)

    report = {
        "methodology": {
            "video": str(VIDEO),
            "fps": fps,
            "velocity_policy": (
                "Correct displacement using actual frame gap "
                "for gaps up to 3; clear history for longer gaps."
            ),
            "visibility_policy": (
                "Track visible pairs independently of risk-evaluated pairs."
            ),
            "unassessable_timeout_frames": EVENT_GAP_TOLERANCE,
            "disappearance_timeout_frames": EVENT_GAP_TOLERANCE,
            "low_risk_evidence": (
                "Only valid risk evaluations count as LOW observations."
            ),
            "event_manager": "Production RiskEventManager",
            "risk_functions": "Production risk calculation functions",
            "production_files_modified": False,
            "yolo_inference_run": False,
        },
        "backends": {},
    }

    for backend, input_path in INPUTS.items():
        print(f"\n========== {backend.upper()} ==========")

        data = json.loads(
            input_path.read_text(encoding="utf-8")
        )

        candidate = run_visibility_policy(data, backend, fps)

        baseline = (
            baseline_report["backends"][backend]
            ["policies"]["original"]
        )

        comparison = compare_events(baseline, candidate)

        report["backends"][backend] = {
            "candidate": candidate,
            "comparison_to_original": comparison,
        }

        print("\nCandidate counts:")
        for name, value in candidate["counts"].items():
            print(f"  {name}: {value}")

        print("\nRisk counts:", candidate["risk_counts"])
        print("Confirmed events:", candidate["confirmed_event_count"])

        print("\nComparison with original:")
        print("  Missing event pairs:", comparison["missing_event_pairs"])
        print("  Additional event pairs:", comparison["additional_event_pairs"])
        print("  Risk-count deltas:", comparison["risk_count_deltas"])
        print("  Evaluated-pair delta:", comparison["evaluated_pair_delta"])

        print("\nEvent details:")
        for event in candidate["confirmed_events"]:
            print(
                f"  Pair {event['pair']} "
                f"| {event['start_frame']}-{event['last_frame']} "
                f"| {event['peak_risk']} "
                f"| score {event['peak_score']} "
                f"| strong {event['strong_observations']} "
                f"| ended: {event['end_reason']}"
            )

    report_path = OUTPUT / "visibility_experiment_report.json"

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
