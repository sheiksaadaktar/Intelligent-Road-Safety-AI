import json
import time
from pathlib import Path
from collections import Counter, defaultdict

import cv2
import numpy as np
from ultralytics import YOLO

from safety_config import MODEL_PATH, VIDEO_PATH


# ------------------------------------------------------------
# Configuration
# ------------------------------------------------------------

OPENVINO_MODEL_PATH = "yolo11n_openvino_model"
IOU_THRESHOLD = 0.50

OUTPUT_DIR = Path("output/backend_tracking_audit")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# ------------------------------------------------------------
# Helpers
# ------------------------------------------------------------

def save_json(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as file:
        json.dump(data, file, indent=2, allow_nan=False)


def extract_tracked_detections(result):
    detections = []
    boxes = result.boxes

    if boxes is None or len(boxes) == 0 or boxes.id is None:
        return detections, 0

    xyxy = boxes.xyxy.cpu().numpy()
    track_ids = boxes.id.int().cpu().tolist()
    class_ids = boxes.cls.int().cpu().tolist()
    confidences = boxes.conf.cpu().numpy().tolist()

    for box, track_id, class_id, confidence in zip(
        xyxy, track_ids, class_ids, confidences
    ):
        detections.append({
            "track_id": int(track_id),
            "class_id": int(class_id),
            "confidence": float(confidence),
            "bbox": [float(value) for value in box],
        })

    return detections, len(boxes)


def run_backend(backend_name, model_path, device):
    print("\n" + "=" * 68)
    print(f"RUNNING {backend_name.upper()}")
    print(f"Model: {model_path}")
    print(f"Device: {device}")
    print("=" * 68)

    if backend_name == "openvino":
        model = YOLO(model_path, task="detect")
    else:
        model = YOLO(model_path)

    capture = cv2.VideoCapture(str(VIDEO_PATH))

    if not capture.isOpened():
        raise RuntimeError(f"Could not open video: {VIDEO_PATH}")

    expected_frames = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    records = []
    total_boxes = 0
    total_tracked = 0
    frames_with_detections = 0
    start_time = time.perf_counter()
    frame_number = 0

    while True:
        success, frame = capture.read()
        if not success:
            break

        frame_number += 1

        results = model.track(
            frame,
            persist=True,
            verbose=False,
            device=device,
        )

        result = results[0]
        detections, box_count = extract_tracked_detections(result)

        total_boxes += box_count
        total_tracked += len(detections)

        if detections:
            frames_with_detections += 1

        records.append({
            "frame_number": frame_number,
            "box_count": int(box_count),
            "tracked_detection_count": len(detections),
            "detections": detections,
        })

        if frame_number % 100 == 0:
            elapsed = time.perf_counter() - start_time
            print(
                f"{backend_name}: frame {frame_number}/{expected_frames} | "
                f"tracked detections: {total_tracked} | "
                f"elapsed: {elapsed:.1f}s"
            )

    capture.release()

    elapsed = time.perf_counter() - start_time
    unique_ids = sorted({
        detection["track_id"]
        for frame in records
        for detection in frame["detections"]
    })

    summary = {
        "backend": backend_name,
        "model": str(model_path),
        "device": device,
        "video": str(VIDEO_PATH),
        "expected_frames": expected_frames,
        "processed_frames": frame_number,
        "total_detector_boxes": total_boxes,
        "total_tracked_detections": total_tracked,
        "frames_with_tracked_detections": frames_with_detections,
        "unique_track_ids": len(unique_ids),
        "track_ids": unique_ids,
        "elapsed_seconds": round(elapsed, 3),
        "processing_fps": round(frame_number / elapsed, 3) if elapsed else 0,
    }

    save_json(
        OUTPUT_DIR / f"{backend_name}_detections.json",
        {
            "summary": summary,
            "frames": records,
        },
    )

    print(f"\n{backend_name.upper()} COMPLETE")
    print(f"Frames processed: {frame_number}")
    print(f"Detector boxes: {total_boxes}")
    print(f"Tracked detections: {total_tracked}")
    print(f"Unique track IDs: {len(unique_ids)}")
    print(f"Elapsed: {elapsed:.2f}s")
    print(f"Processing FPS: {summary['processing_fps']}")

    return records, summary


def calculate_iou(box_a, box_b):
    ax1, ay1, ax2, ay2 = box_a
    bx1, by1, bx2, by2 = box_b

    intersection_x1 = max(ax1, bx1)
    intersection_y1 = max(ay1, by1)
    intersection_x2 = min(ax2, bx2)
    intersection_y2 = min(ay2, by2)

    intersection_width = max(0.0, intersection_x2 - intersection_x1)
    intersection_height = max(0.0, intersection_y2 - intersection_y1)
    intersection = intersection_width * intersection_height

    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    union = area_a + area_b - intersection

    if union <= 0:
        return 0.0

    return intersection / union


# ------------------------------------------------------------
# Frame-by-frame detection association
# ------------------------------------------------------------

def audit_tracking_identity(pytorch_records, openvino_records):
    print("\n" + "=" * 68)
    print("MATCHING PYTORCH AND OPENVINO DETECTIONS")
    print(f"Same-frame, same-class IoU threshold: {IOU_THRESHOLD}")
    print("=" * 68)

    pytorch_by_frame = {
        frame["frame_number"]: frame["detections"]
        for frame in pytorch_records
    }
    openvino_by_frame = {
        frame["frame_number"]: frame["detections"]
        for frame in openvino_records
    }

    all_frames = sorted(set(pytorch_by_frame) | set(openvino_by_frame))

    pair_counts = Counter()
    pytorch_track_appearances = Counter()
    openvino_track_appearances = Counter()

    total_pytorch_detections = 0
    total_openvino_detections = 0
    total_matches = 0
    iou_values = []
    frames_with_matches = 0
    per_frame_results = []

    for frame_number in all_frames:
        p_detections = pytorch_by_frame.get(frame_number, [])
        o_detections = openvino_by_frame.get(frame_number, [])

        total_pytorch_detections += len(p_detections)
        total_openvino_detections += len(o_detections)

        for detection in p_detections:
            pytorch_track_appearances[detection["track_id"]] += 1

        for detection in o_detections:
            openvino_track_appearances[detection["track_id"]] += 1

        candidates = []

        for p_index, p_detection in enumerate(p_detections):
            for o_index, o_detection in enumerate(o_detections):
                if p_detection["class_id"] != o_detection["class_id"]:
                    continue

                iou = calculate_iou(
                    p_detection["bbox"],
                    o_detection["bbox"],
                )

                if iou >= IOU_THRESHOLD:
                    candidates.append((iou, p_index, o_index))

        # Greedy one-to-one matching, highest IoU first.
        candidates.sort(key=lambda item: item[0], reverse=True)

        used_pytorch = set()
        used_openvino = set()
        frame_matches = []

        for iou, p_index, o_index in candidates:
            if p_index in used_pytorch or o_index in used_openvino:
                continue

            used_pytorch.add(p_index)
            used_openvino.add(o_index)

            p_detection = p_detections[p_index]
            o_detection = o_detections[o_index]

            p_id = p_detection["track_id"]
            o_id = o_detection["track_id"]

            pair_counts[(p_id, o_id)] += 1
            iou_values.append(iou)

            frame_matches.append({
                "pytorch_track_id": p_id,
                "openvino_track_id": o_id,
                "class_id": p_detection["class_id"],
                "iou": round(float(iou), 4),
            })

        total_matches += len(frame_matches)

        if frame_matches:
            frames_with_matches += 1

        per_frame_results.append({
            "frame_number": frame_number,
            "pytorch_tracked_detections": len(p_detections),
            "openvino_tracked_detections": len(o_detections),
            "matched_detections": len(frame_matches),
            "unmatched_pytorch": len(p_detections) - len(frame_matches),
            "unmatched_openvino": len(o_detections) - len(frame_matches),
            "matches": frame_matches,
        })

        if frame_number % 100 == 0:
            print(
                f"Compared frame {frame_number}/{len(all_frames)} | "
                f"matches so far: {total_matches}"
            )

    # Summarize the dominant counterpart for every track ID.
    pytorch_counterparts = defaultdict(Counter)
    openvino_counterparts = defaultdict(Counter)

    for (p_id, o_id), count in pair_counts.items():
        pytorch_counterparts[p_id][o_id] += count
        openvino_counterparts[o_id][p_id] += count

    pytorch_mapping = []

    for p_id in sorted(pytorch_track_appearances):
        counterparts = pytorch_counterparts[p_id]
        matched_count = sum(counterparts.values())

        if counterparts:
            dominant_o_id, dominant_count = counterparts.most_common(1)[0]
            purity = dominant_count / matched_count
        else:
            dominant_o_id = None
            dominant_count = 0
            purity = 0.0

        pytorch_mapping.append({
            "pytorch_track_id": p_id,
            "pytorch_track_appearances": pytorch_track_appearances[p_id],
            "matched_appearances": matched_count,
            "dominant_openvino_track_id": dominant_o_id,
            "dominant_match_count": dominant_count,
            "mapping_purity": round(purity, 4),
            "all_openvino_counterparts": [
                {
                    "openvino_track_id": o_id,
                    "matched_frames": count,
                }
                for o_id, count in counterparts.most_common()
            ],
        })

    openvino_mapping = []

    for o_id in sorted(openvino_track_appearances):
        counterparts = openvino_counterparts[o_id]
        matched_count = sum(counterparts.values())

        if counterparts:
            dominant_p_id, dominant_count = counterparts.most_common(1)[0]
            purity = dominant_count / matched_count
        else:
            dominant_p_id = None
            dominant_count = 0
            purity = 0.0

        openvino_mapping.append({
            "openvino_track_id": o_id,
            "openvino_track_appearances": openvino_track_appearances[o_id],
            "matched_appearances": matched_count,
            "dominant_pytorch_track_id": dominant_p_id,
            "dominant_match_count": dominant_count,
            "mapping_purity": round(purity, 4),
            "all_pytorch_counterparts": [
                {
                    "pytorch_track_id": p_id,
                    "matched_frames": count,
                }
                for p_id, count in counterparts.most_common()
            ],
        })

    pytorch_coverage = (
        total_matches / total_pytorch_detections
        if total_pytorch_detections else 0.0
    )
    openvino_coverage = (
        total_matches / total_openvino_detections
        if total_openvino_detections else 0.0
    )

    audit_summary = {
        "iou_threshold": IOU_THRESHOLD,
        "matching_method": (
            "Same frame, same class, greedy one-to-one matching "
            "in descending IoU order"
        ),
        "frames_compared": len(all_frames),
        "frames_with_at_least_one_match": frames_with_matches,
        "total_pytorch_tracked_detections": total_pytorch_detections,
        "total_openvino_tracked_detections": total_openvino_detections,
        "matched_detection_pairs": total_matches,
        "unmatched_pytorch_detections": (
            total_pytorch_detections - total_matches
        ),
        "unmatched_openvino_detections": (
            total_openvino_detections - total_matches
        ),
        "pytorch_detection_match_coverage": round(pytorch_coverage, 4),
        "openvino_detection_match_coverage": round(openvino_coverage, 4),
        "mean_matched_iou": (
            round(float(np.mean(iou_values)), 4)
            if iou_values else None
        ),
        "median_matched_iou": (
            round(float(np.median(iou_values)), 4)
            if iou_values else None
        ),
        "pytorch_unique_track_ids": len(pytorch_track_appearances),
        "openvino_unique_track_ids": len(openvino_track_appearances),
        "distinct_track_id_pairs_observed": len(pair_counts),
        "note": (
            "This is a detection-association audit, not ground-truth "
            "object-tracking accuracy. Mapping purity measures how often "
            "a track ID's matched detections correspond to its dominant "
            "counterpart in the other backend."
        ),
    }

    mapping_output = {
        "summary": audit_summary,
        "pytorch_to_openvino_track_mapping": pytorch_mapping,
        "openvino_to_pytorch_track_mapping": openvino_mapping,
        "track_pair_cooccurrence_counts": [
            {
                "pytorch_track_id": p_id,
                "openvino_track_id": o_id,
                "matched_frames": count,
            }
            for (p_id, o_id), count in pair_counts.most_common()
        ],
        "per_frame_matching": per_frame_results,
    }

    save_json(
        OUTPUT_DIR / "identity_mapping.json",
        mapping_output,
    )

    print("\nIDENTITY AUDIT SUMMARY")
    print("-" * 50)
    print(f"Frames compared: {len(all_frames)}")
    print(f"Matched detection pairs: {total_matches}")
    print(f"PyTorch match coverage: {pytorch_coverage:.1%}")
    print(f"OpenVINO match coverage: {openvino_coverage:.1%}")
    print(
        "Mean matched IoU: "
        f"{audit_summary['mean_matched_iou']}"
    )
    print(
        "Distinct track-ID pairs: "
        f"{len(pair_counts)}"
    )

    return audit_summary


# ------------------------------------------------------------
# Main
# ------------------------------------------------------------

def main():
    print("INTELLIGENT ROAD SAFETY AI")
    print("BACKEND TRACKING-IDENTITY AUDIT")
    print("-" * 68)
    print(f"Video: {VIDEO_PATH}")
    print(f"PyTorch model: {MODEL_PATH}")
    print(f"OpenVINO model: {OPENVINO_MODEL_PATH}")
    print(f"IoU threshold: {IOU_THRESHOLD}")
    print(f"Output directory: {OUTPUT_DIR}")

    pytorch_records, pytorch_summary = run_backend(
        backend_name="pytorch",
        model_path=MODEL_PATH,
        device="cpu",
    )

    openvino_records, openvino_summary = run_backend(
        backend_name="openvino",
        model_path=OPENVINO_MODEL_PATH,
        device="intel:gpu",
    )

    identity_summary = audit_tracking_identity(
        pytorch_records,
        openvino_records,
    )

    final_summary = {
        "pytorch": pytorch_summary,
        "openvino": openvino_summary,
        "identity_audit": identity_summary,
    }

    save_json(
        OUTPUT_DIR / "audit_summary.json",
        final_summary,
    )

    print("\n" + "=" * 68)
    print("TRACKING-IDENTITY AUDIT COMPLETED")
    print("=" * 68)
    print(f"Results saved in: {OUTPUT_DIR.resolve()}")
    print("  - pytorch_detections.json")
    print("  - openvino_detections.json")
    print("  - identity_mapping.json")
    print("  - audit_summary.json")


if __name__ == "__main__":
    main()
