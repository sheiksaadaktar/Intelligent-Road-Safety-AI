import csv
import json
import cv2
import numpy as np
from collections import defaultdict, deque, Counter
from pathlib import Path
from itertools import combinations

VIDEO = Path("data/tracking_test.mp4")
INPUTS = {
    "pytorch": Path("output/backend_tracking_audit/pytorch_detections.json"),
    "openvino": Path("output/backend_tracking_audit/openvino_detections.json"),
}
OUTPUT = Path("output/velocity_risk_comparison")

HISTORY_SIZE = 5
MAX_VELOCITY = 800.0
MAX_RELATIVE_VELOCITY = 1600.0
MAX_GAP_FRAMES = 3

# Exact risk configuration from src/safety_config.py
MAX_RISK_DISTANCE = 320.0

CRITICAL_DISTANCE = 100.0
HIGH_DISTANCE = 170.0
MEDIUM_DISTANCE = 260.0

CRITICAL_PREDICTED_DISTANCE = 30.0
HIGH_PREDICTED_DISTANCE = 70.0
MEDIUM_PREDICTED_DISTANCE = 150.0

MIN_APPROACH_SPEED = 20.0
HIGH_APPROACH_SPEED = 80.0
CRITICAL_APPROACH_SPEED = 160.0

CRITICAL_TCA = 0.6
HIGH_TCA = 1.2
MEDIUM_TCA = 2.5
MAX_TCA = 5.0

MIN_CONVERGENCE = 0.30
STRONG_CONVERGENCE = 0.75

OUTPUT.mkdir(parents=True, exist_ok=True)

cap = cv2.VideoCapture(str(VIDEO))
if not cap.isOpened():
    raise RuntimeError(f"Cannot open video: {VIDEO}")
FPS = cap.get(cv2.CAP_PROP_FPS)
cap.release()

if not FPS or FPS <= 0:
    raise RuntimeError("Invalid video FPS.")

def smooth_velocity(history):
    if not history:
        return np.zeros(2, dtype=np.float32)
    return np.mean(np.asarray(history, dtype=np.float32), axis=0)

def calculate_trajectory_metrics(position_1, velocity_1, position_2, velocity_2):
    position_1 = np.asarray(position_1, dtype=np.float32)
    position_2 = np.asarray(position_2, dtype=np.float32)
    velocity_1 = np.asarray(velocity_1, dtype=np.float32)
    velocity_2 = np.asarray(velocity_2, dtype=np.float32)

    r = position_2 - position_1
    v = velocity_2 - velocity_1

    distance_now = float(np.linalg.norm(r))
    relative_speed = float(np.linalg.norm(v))

    if distance_now <= 1e-6:
        return {
            "distance": 0.0,
            "relative_speed": relative_speed,
            "approach_speed": relative_speed,
            "convergence": 1.0,
            "tca": 0.0,
            "predicted_distance": 0.0,
            "converging": True,
        }

    r_unit = r / distance_now
    approach_speed = -float(np.dot(r_unit, v))

    if relative_speed > 1e-6:
        convergence = approach_speed / relative_speed
        convergence = float(np.clip(convergence, -1.0, 1.0))
    else:
        convergence = 0.0

    velocity_squared = float(np.dot(v, v))

    if velocity_squared <= 1e-6:
        tca = float("inf")
        predicted_distance = distance_now
    else:
        tca = -float(np.dot(r, v)) / velocity_squared
        if tca < 0:
            tca = 0.0
        closest_position = r + (v * tca)
        predicted_distance = float(np.linalg.norm(closest_position))

    converging = (
        approach_speed >= MIN_APPROACH_SPEED
        and convergence >= MIN_CONVERGENCE
        and tca <= MAX_TCA
    )

    return {
        "distance": distance_now,
        "relative_speed": relative_speed,
        "approach_speed": approach_speed,
        "convergence": convergence,
        "tca": tca,
        "predicted_distance": predicted_distance,
        "converging": converging,
    }

def calculate_risk(metrics):
    distance = metrics["distance"]
    predicted_distance = metrics["predicted_distance"]
    approach_speed = metrics["approach_speed"]
    tca = metrics["tca"]
    convergence = metrics["convergence"]
    converging = metrics["converging"]

    if distance > MAX_RISK_DISTANCE:
        return "LOW", 0

    if not converging:
        return "LOW", 0

    score = 0

    if distance <= CRITICAL_DISTANCE:
        score += 30
    elif distance <= HIGH_DISTANCE:
        score += 20
    elif distance <= MEDIUM_DISTANCE:
        score += 10

    if predicted_distance <= CRITICAL_PREDICTED_DISTANCE:
        score += 30
    elif predicted_distance <= HIGH_PREDICTED_DISTANCE:
        score += 25
    elif predicted_distance <= MEDIUM_PREDICTED_DISTANCE:
        score += 15

    if approach_speed >= CRITICAL_APPROACH_SPEED:
        score += 20
    elif approach_speed >= HIGH_APPROACH_SPEED:
        score += 15
    elif approach_speed >= MIN_APPROACH_SPEED:
        score += 5

    if tca <= CRITICAL_TCA:
        score += 25
    elif tca <= HIGH_TCA:
        score += 20
    elif tca <= MEDIUM_TCA:
        score += 10

    if convergence >= STRONG_CONVERGENCE:
        score += 15
    elif convergence >= 0.60:
        score += 8
    else:
        score += 3

    critical_trajectory = (
        (
            predicted_distance <= CRITICAL_PREDICTED_DISTANCE
            and tca <= CRITICAL_TCA
        )
        or (
            distance <= CRITICAL_DISTANCE
            and approach_speed >= CRITICAL_APPROACH_SPEED
            and convergence >= STRONG_CONVERGENCE
        )
        or (
            predicted_distance <= CRITICAL_PREDICTED_DISTANCE
            and approach_speed >= CRITICAL_APPROACH_SPEED
            and convergence >= STRONG_CONVERGENCE
        )
        or (
            tca <= 0.30
            and approach_speed >= CRITICAL_APPROACH_SPEED
            and convergence >= STRONG_CONVERGENCE
        )
    )

    if critical_trajectory and score >= 70:
        return "CRITICAL", score

    high_trajectory = (
        (
            predicted_distance <= HIGH_PREDICTED_DISTANCE
            and tca <= HIGH_TCA
        )
        or (
            distance <= HIGH_DISTANCE
            and approach_speed >= HIGH_APPROACH_SPEED
            and convergence >= MIN_CONVERGENCE
        )
        or (
            predicted_distance <= MEDIUM_PREDICTED_DISTANCE
            and approach_speed >= HIGH_APPROACH_SPEED
            and convergence >= STRONG_CONVERGENCE
        )
    )

    if high_trajectory and score >= 50:
        return "HIGH", score

    medium_trajectory = (
        (
            distance <= MEDIUM_DISTANCE
            and approach_speed >= MIN_APPROACH_SPEED
        )
        or (
            predicted_distance <= MEDIUM_PREDICTED_DISTANCE
            and tca <= MEDIUM_TCA
        )
        or (
            tca <= MEDIUM_TCA
            and convergence >= STRONG_CONVERGENCE
        )
    )

    if medium_trajectory and score >= 25:
        return "MEDIUM", score

    return "LOW", 0

def evaluate_pair(position_1, velocity_1, position_2, velocity_2):
    relative_velocity = np.asarray(velocity_2) - np.asarray(velocity_1)
    relative_speed = float(np.linalg.norm(relative_velocity))

    if relative_speed > MAX_RELATIVE_VELOCITY:
        return {
            "status": "REJECTED_RELATIVE_SPEED",
            "relative_speed": relative_speed,
        }

    metrics = calculate_trajectory_metrics(
        position_1, velocity_1, position_2, velocity_2
    )
    risk, score = calculate_risk(metrics)

    return {
        "status": "EVALUATED",
        "relative_speed": relative_speed,
        "risk": risk,
        "score": score,
        **metrics,
    }

all_rows = []
backend_summaries = {}

for backend, input_path in INPUTS.items():
    print(f"\nProcessing {backend.upper()}")

    with input_path.open("r", encoding="utf-8") as f:
        data = json.load(f)

    original_history = defaultdict(lambda: deque(maxlen=HISTORY_SIZE))
    experimental_history = defaultdict(lambda: deque(maxlen=HISTORY_SIZE))

    previous_positions = {}
    previous_frames = {}

    counts = {
        "frames": 0,
        "detections": 0,
        "pairs_seen": 0,
        "original_evaluated": 0,
        "experimental_evaluated": 0,
        "original_relative_speed_rejections": 0,
        "experimental_relative_speed_rejections": 0,
        "experimental_gap_resets": 0,
        "experimental_corrected_gaps": 0,
    }

    original_risks = Counter()
    experimental_risks = Counter()
    transitions = Counter()

    for frame_record in data["frames"]:
        frame_number = int(frame_record["frame_number"])
        counts["frames"] += 1

        current_positions = {}

        for detection in frame_record.get("detections", []):
            track_id = int(detection["track_id"])
            x1, y1, x2, y2 = map(float, detection["bbox"])

            position = np.asarray(
                [(x1 + x2) / 2.0, y2],
                dtype=np.float32,
            )
            current_positions[track_id] = position
            counts["detections"] += 1

            if track_id in previous_positions:
                frame_gap = frame_number - previous_frames[track_id]
                displacement = position - previous_positions[track_id]

                # Original production calculation.
                original_velocity = displacement * FPS
                if float(np.linalg.norm(original_velocity)) <= MAX_VELOCITY:
                    original_history[track_id].append(original_velocity)

                # Experimental frame-aware calculation and history reset.
                if 1 <= frame_gap <= MAX_GAP_FRAMES:
                    experimental_velocity = displacement / (frame_gap / FPS)
                    if float(np.linalg.norm(experimental_velocity)) <= MAX_VELOCITY:
                        experimental_history[track_id].append(
                            experimental_velocity
                        )
                    if frame_gap > 1:
                        counts["experimental_corrected_gaps"] += 1
                else:
                    experimental_history[track_id].clear()
                    counts["experimental_gap_resets"] += 1

            previous_positions[track_id] = position
            previous_frames[track_id] = frame_number

        track_ids = sorted(current_positions.keys())

        for id_1, id_2 in combinations(track_ids, 2):
            counts["pairs_seen"] += 1

            if not original_history[id_1] or not original_history[id_2]:
                original_result = {"status": "NO_VELOCITY_HISTORY"}
            else:
                original_result = evaluate_pair(
                    current_positions[id_1],
                    smooth_velocity(original_history[id_1]),
                    current_positions[id_2],
                    smooth_velocity(original_history[id_2]),
                )

            if not experimental_history[id_1] or not experimental_history[id_2]:
                experimental_result = {"status": "NO_VELOCITY_HISTORY"}
            else:
                experimental_result = evaluate_pair(
                    current_positions[id_1],
                    smooth_velocity(experimental_history[id_1]),
                    current_positions[id_2],
                    smooth_velocity(experimental_history[id_2]),
                )

            if original_result["status"] == "EVALUATED":
                counts["original_evaluated"] += 1
                original_risks[original_result["risk"]] += 1
            elif original_result["status"] == "REJECTED_RELATIVE_SPEED":
                counts["original_relative_speed_rejections"] += 1

            if experimental_result["status"] == "EVALUATED":
                counts["experimental_evaluated"] += 1
                experimental_risks[experimental_result["risk"]] += 1
            elif experimental_result["status"] == "REJECTED_RELATIVE_SPEED":
                counts["experimental_relative_speed_rejections"] += 1

            if (
                original_result["status"] == "EVALUATED"
                and experimental_result["status"] == "EVALUATED"
            ):
                transitions[
                    (original_result["risk"], experimental_result["risk"])
                ] += 1

            row = {
                "backend": backend,
                "frame": frame_number,
                "id_1": id_1,
                "id_2": id_2,
                "original_status": original_result["status"],
                "experimental_status": experimental_result["status"],
            }

            metric_names = (
                "relative_speed",
                "distance",
                "approach_speed",
                "convergence",
                "tca",
                "predicted_distance",
                "score",
                "risk",
            )

            for name in metric_names:
                ov = original_result.get(name)
                ev = experimental_result.get(name)

                row[f"original_{name}"] = (
                    round(float(ov), 6)
                    if isinstance(ov, (int, float, np.number))
                    and np.isfinite(ov) else ov
                )
                row[f"experimental_{name}"] = (
                    round(float(ev), 6)
                    if isinstance(ev, (int, float, np.number))
                    and np.isfinite(ev) else ev
                )

            all_rows.append(row)

    backend_summaries[backend] = {
        **counts,
        "original_risk_counts": dict(original_risks),
        "experimental_risk_counts": dict(experimental_risks),
        "risk_transition_counts": {
            f"{a}_TO_{b}": n for (a, b), n in sorted(transitions.items())
        },
        "matched_classification_observations": sum(transitions.values()),
        "classification_agreements": sum(
            n for (a, b), n in transitions.items() if a == b
        ),
        "classification_changes": sum(
            n for (a, b), n in transitions.items() if a != b
        ),
    }

    print("Frames:", counts["frames"])
    print("Detections:", counts["detections"])
    print("Pairs seen:", counts["pairs_seen"])
    print("Original evaluated:", counts["original_evaluated"])
    print("Experimental evaluated:", counts["experimental_evaluated"])
    print("Original risk counts:", dict(original_risks))
    print("Experimental risk counts:", dict(experimental_risks))
    print("Risk transitions:", backend_summaries[backend]["risk_transition_counts"])
    print("Experimental gap resets:", counts["experimental_gap_resets"])
    print("Experimental corrected gaps:", counts["experimental_corrected_gaps"])

csv_path = OUTPUT / "pairwise_risk_comparison.csv"
if all_rows:
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(all_rows[0].keys()))
        writer.writeheader()
        writer.writerows(all_rows)

summary_path = OUTPUT / "risk_comparison_summary.json"
with summary_path.open("w", encoding="utf-8") as f:
    json.dump({
        "methodology": {
            "fps": FPS,
            "position": "bounding-box bottom-centre",
            "original_velocity": "displacement * FPS",
            "experimental_velocity": "displacement / (frame_gap / FPS)",
            "experimental_history_reset_gap": MAX_GAP_FRAMES,
            "history_size": HISTORY_SIZE,
            "risk_logic": "copied from supplied production risk_analysis.py",
            "risk_thresholds_changed": False,
            "event_confirmation_included": False,
            "yolo_rerun": False,
        },
        "backends": backend_summaries,
    }, f, indent=2)

print("\nPAIRWISE RISK COMPARISON COMPLETE")
print("Summary:", summary_path)
print("Pairwise CSV:", csv_path)
print("Production files were not modified.")
