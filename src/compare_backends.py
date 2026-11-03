import json
import time
from collections import defaultdict, deque, Counter
from pathlib import Path

import cv2
import numpy as np
from ultralytics import YOLO

from risk_analysis import (
    MODEL_PATH,
    HISTORY_SIZE,
    MAX_REASONABLE_VELOCITY,
    MAX_REASONABLE_RELATIVE_VELOCITY,
    bottom_center,
    smooth_velocity,
    calculate_trajectory_metrics,
    calculate_risk,
)

from live_alert_pipeline import LiveAlertPipeline
from live_incident_lifecycle import LiveIncidentLifecycle


VIDEO_SOURCE = "data/tracking_test.mp4"
OPENVINO_MODEL_PATH = "yolo11n_openvino_model"

OUTPUT_ROOT = Path("output/backend_comparison")


def save_json(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    with open(path, "w", encoding="utf-8") as file:
        json.dump(data, file, indent=2, default=str)

    return path


def run_backend(name, model_path, device, output_dir):
    print("")
    print("=" * 72)
    print(f"RUNNING BACKEND: {name}")
    print("=" * 72)
    print(f"Model: {model_path}")
    print(f"Device: {device}")
    print(f"Video: {VIDEO_SOURCE}")

    output_dir.mkdir(parents=True, exist_ok=True)

    if name == "OpenVINO":
        model = YOLO(model_path, task="detect")
    else:
        model = YOLO(model_path)

    capture = cv2.VideoCapture(VIDEO_SOURCE)

    if not capture.isOpened():
        raise RuntimeError(
            f"Could not open video: {VIDEO_SOURCE}"
        )

    fps = capture.get(cv2.CAP_PROP_FPS)

    if fps <= 0:
        fps = 30.0

    source_width = int(
        capture.get(cv2.CAP_PROP_FRAME_WIDTH)
    )

    source_height = int(
        capture.get(cv2.CAP_PROP_FRAME_HEIGHT)
    )

    print(f"Resolution: {source_width}x{source_height}")
    print(f"Source FPS: {fps:.2f}")

    alert_pipeline = LiveAlertPipeline(
        fps=fps,
        output_path=output_dir / "live_alerts.json",
    )

    incident_lifecycle = LiveIncidentLifecycle(
        fps=fps,
        output_path=output_dir / "live_incidents.json",
    )

    velocity_history = defaultdict(
        lambda: deque(maxlen=HISTORY_SIZE)
    )

    previous_positions = {}

    frame_tracking = []
    risk_observations = Counter()
    unique_track_ids = set()

    total_detections = 0
    frames_with_detections = 0
    frame_number = 0

    tracking_kwargs = {
        "device": device,
    }

    start_time = time.perf_counter()

    try:
        while True:
            success, frame = capture.read()

            if not success:
                break

            frame_number += 1

            results = model.track(
                frame,
                persist=True,
                verbose=False,
                **tracking_kwargs,
            )

            detections = []
            current_frame_ids = []

            if (
                results
                and results[0].boxes is not None
                and results[0].boxes.id is not None
            ):
                boxes = results[0].boxes

                track_ids = (
                    boxes.id.int().cpu().tolist()
                )

                class_ids = (
                    boxes.cls.int().cpu().tolist()
                )

                coordinates = (
                    boxes.xyxy.cpu().numpy()
                )

                for track_id, class_id, box in zip(
                    track_ids,
                    class_ids,
                    coordinates,
                ):
                    center = bottom_center(box)

                    if track_id in previous_positions:
                        previous_position = (
                            previous_positions[track_id]
                        )

                        displacement = (
                            center - previous_position
                        )

                        velocity = displacement * fps

                        velocity_magnitude = float(
                            np.linalg.norm(velocity)
                        )

                        if (
                            velocity_magnitude
                            <= MAX_REASONABLE_VELOCITY
                        ):
                            velocity_history[track_id].append(
                                velocity
                            )

                    previous_positions[track_id] = center

                    smoothed_velocity = smooth_velocity(
                        velocity_history[track_id]
                    )

                    detections.append({
                        "id": track_id,
                        "class_id": class_id,
                        "box": box,
                        "position": center,
                        "velocity": smoothed_velocity,
                    })

                    current_frame_ids.append(track_id)
                    unique_track_ids.add(track_id)

            total_detections += len(detections)

            if detections:
                frames_with_detections += 1

            frame_tracking.append({
                "frame": frame_number,
                "detection_count": len(detections),
                "track_ids": sorted(current_frame_ids),
            })

            observed_pairs = set()

            for i in range(len(detections)):
                for j in range(i + 1, len(detections)):
                    first = detections[i]
                    second = detections[j]

                    if (
                        first["id"] not in velocity_history
                        or second["id"] not in velocity_history
                    ):
                        continue

                    velocity_one = smooth_velocity(
                        velocity_history[first["id"]]
                    )

                    velocity_two = smooth_velocity(
                        velocity_history[second["id"]]
                    )

                    if (
                        velocity_one is None
                        or velocity_two is None
                    ):
                        continue

                    relative_velocity = (
                        velocity_two - velocity_one
                    )

                    relative_speed = float(
                        np.linalg.norm(relative_velocity)
                    )

                    if (
                        relative_speed
                        > MAX_REASONABLE_RELATIVE_VELOCITY
                    ):
                        continue

                    metrics = calculate_trajectory_metrics(
                        first["position"],
                        velocity_one,
                        second["position"],
                        velocity_two,
                    )

                    risk, score = calculate_risk(metrics)

                    pair = (
                        first["id"],
                        second["id"],
                    )

                    normalized_pair = tuple(sorted(pair))
                    observed_pairs.add(normalized_pair)

                    risk_observations[risk] += 1

                    alert_pipeline.observe(
                        frame_number=frame_number,
                        pair=pair,
                        risk=risk,
                        score=score,
                        metrics=metrics,
                    )

            alert_pipeline.mark_missing_pairs(
                frame_number,
                observed_pairs,
            )

            active_events = (
                alert_pipeline.event_manager.active_events
            )

            completed_events = (
                alert_pipeline.event_manager.completed_events
            )

            incident_lifecycle.process_alerts(
                alerts=alert_pipeline.alerts,
                active_events=active_events,
                frame_number=frame_number,
            )

            incident_lifecycle.resolve_completed_events(
                completed_events=completed_events,
                frame_number=frame_number,
            )

            if frame_number % 100 == 0:
                print(
                    f"Processed {frame_number} frames | "
                    f"Detections: {total_detections} | "
                    f"Alerts: {len(alert_pipeline.alerts)}"
                )

    finally:
        capture.release()

    processing_seconds = (
        time.perf_counter() - start_time
    )

    alert_pipeline.finalize()

    incident_lifecycle.resolve_completed_events(
        completed_events=(
            alert_pipeline.event_manager.completed_events
        ),
        frame_number=frame_number,
    )

    incident_lifecycle.resolve_all(
        frame_number=frame_number,
    )

    alert_output = alert_pipeline.save(
        source_video=VIDEO_SOURCE
    )

    incident_output = incident_lifecycle.save(
        source_video=VIDEO_SOURCE
    )

    alert_statistics = alert_pipeline.statistics()
    incident_statistics = incident_lifecycle.statistics()

    processing_fps = (
        frame_number / processing_seconds
        if processing_seconds > 0
        else 0
    )

    summary = {
        "backend": name,
        "model": model_path,
        "device": device,
        "source_video": VIDEO_SOURCE,
        "source_resolution": {
            "width": source_width,
            "height": source_height,
        },
        "source_fps": fps,
        "frames_processed": frame_number,
        "total_detections": total_detections,
        "frames_with_detections": frames_with_detections,
        "detection_rate_percent": (
            frames_with_detections / frame_number * 100
            if frame_number else 0
        ),
        "unique_track_ids": sorted(unique_track_ids),
        "unique_track_count": len(unique_track_ids),
        "risk_observations": dict(risk_observations),
        "alert_count": len(alert_pipeline.alerts),
        "incident_count": len(incident_lifecycle.incidents),
        "alert_statistics": alert_statistics,
        "incident_statistics": incident_statistics,
        "processing_seconds": processing_seconds,
        "average_processing_fps": processing_fps,
        "average_frame_time_ms": (
            processing_seconds / frame_number * 1000
            if frame_number else 0
        ),
        "output_files": {
            "alerts": str(alert_output),
            "incidents": str(incident_output),
        },
    }

    save_json(
        output_dir / "frame_tracking.json",
        frame_tracking,
    )

    save_json(
        output_dir / "backend_summary.json",
        summary,
    )

    print("")
    print(f"Backend: {name}")
    print(f"Frames processed: {frame_number}")
    print(f"Total detections: {total_detections}")
    print(f"Frames with detections: {frames_with_detections}")
    print(f"Unique track IDs: {len(unique_track_ids)}")
    print(f"Confirmed alerts: {len(alert_pipeline.alerts)}")
    print(f"Incidents: {len(incident_lifecycle.incidents)}")
    print(f"Processing time: {processing_seconds:.2f}s")
    print(f"Average processing FPS: {processing_fps:.2f}")
    print(f"Results: {output_dir}")

    return summary


def main():
    print("=" * 72)
    print("INTELLIGENT ROAD SAFETY AI")
    print("ISOLATED BACKEND COMPARISON")
    print("=" * 72)

    if not Path(VIDEO_SOURCE).exists():
        raise FileNotFoundError(VIDEO_SOURCE)

    if not Path(OPENVINO_MODEL_PATH, "yolo11n.xml").exists():
        raise FileNotFoundError(
            "OpenVINO XML model is missing."
        )

    if not Path(OPENVINO_MODEL_PATH, "yolo11n.bin").exists():
        raise FileNotFoundError(
            "OpenVINO BIN model is missing."
        )

    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)

    pytorch_summary = run_backend(
        name="PyTorch",
        model_path=MODEL_PATH,
        device="cpu",
        output_dir=OUTPUT_ROOT / "pytorch",
    )

    openvino_summary = run_backend(
        name="OpenVINO",
        model_path=OPENVINO_MODEL_PATH,
        device="intel:gpu",
        output_dir=OUTPUT_ROOT / "openvino",
    )

    comparison = {
        "comparison_version": "Backend-Comparison-1.0",
        "source_video": VIDEO_SOURCE,
        "pytorch": pytorch_summary,
        "openvino": openvino_summary,
        "comparison": {
            "same_frame_count": (
                pytorch_summary["frames_processed"]
                == openvino_summary["frames_processed"]
            ),
            "frame_count_difference": (
                openvino_summary["frames_processed"]
                - pytorch_summary["frames_processed"]
            ),
            "detection_count_difference": (
                openvino_summary["total_detections"]
                - pytorch_summary["total_detections"]
            ),
            "alert_count_difference": (
                openvino_summary["alert_count"]
                - pytorch_summary["alert_count"]
            ),
            "incident_count_difference": (
                openvino_summary["incident_count"]
                - pytorch_summary["incident_count"]
            ),
            "processing_fps_speedup": (
                openvino_summary["average_processing_fps"]
                / pytorch_summary["average_processing_fps"]
                if pytorch_summary["average_processing_fps"] > 0
                else None
            ),
        },
    }

    comparison_path = save_json(
        OUTPUT_ROOT / "comparison_summary.json",
        comparison,
    )

    print("")
    print("=" * 72)
    print("BACKEND COMPARISON COMPLETE")
    print("=" * 72)
    print(
        "Same frame count:",
        comparison["comparison"]["same_frame_count"],
    )
    print(
        "PyTorch FPS:",
        f'{pytorch_summary["average_processing_fps"]:.2f}',
    )
    print(
        "OpenVINO FPS:",
        f'{openvino_summary["average_processing_fps"]:.2f}',
    )
    print(
        "OpenVINO FPS speedup:",
        comparison["comparison"]["processing_fps_speedup"],
    )
    print(
        "Alert count difference:",
        comparison["comparison"]["alert_count_difference"],
    )
    print(
        "Incident count difference:",
        comparison["comparison"]["incident_count_difference"],
    )
    print(f"Comparison report: {comparison_path}")
    print("")
    print("Production monitor and production JSON files were not modified.")


if __name__ == "__main__":
    main()
