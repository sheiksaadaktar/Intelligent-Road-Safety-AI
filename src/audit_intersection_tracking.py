import json
from pathlib import Path
from collections import Counter, defaultdict

import numpy as np

PYTORCH_FILE = Path(
    "output/intersection_tracking/pytorch/tracking_detections.json"
)
OPENVINO_FILE = Path(
    "output/intersection_tracking/openvino/tracking_detections.json"
)
OUTPUT_DIR = Path("output/intersection_tracking/audit")
IOU_THRESHOLD = 0.50


def load_records(path):
    with path.open("r", encoding="utf-8-sig") as file:
        data = json.load(file)

    if "frames" not in data:
        raise ValueError(f"Missing frames in {path}")

    records = data["frames"]

    for frame in records:
        if "frame_number" not in frame or "detections" not in frame:
            raise ValueError(f"Invalid frame record in {path}")

        for detection in frame["detections"]:
            if detection["track_id"] is None:
                raise ValueError(
                    f"Untracked detection found in {path}, "
                    f"frame {frame['frame_number']}"
                )

    return records


def calculate_iou(a, b):
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b

    x1 = max(ax1, bx1)
    y1 = max(ay1, by1)
    x2 = min(ax2, bx2)
    y2 = min(ay2, by2)

    intersection = max(0.0, x2 - x1) * max(0.0, y2 - y1)

    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)

    union = area_a + area_b - intersection

    return intersection / union if union > 0 else 0.0


def audit(pytorch_records, openvino_records):
    p_by_frame = {
        frame["frame_number"]: frame["detections"]
        for frame in pytorch_records
    }
    o_by_frame = {
        frame["frame_number"]: frame["detections"]
        for frame in openvino_records
    }

    all_frames = sorted(set(p_by_frame) | set(o_by_frame))

    pair_counts = Counter()
    p_appearances = Counter()
    o_appearances = Counter()
    iou_values = []
    per_frame = []

    total_p = 0
    total_o = 0
    total_matches = 0
    frames_with_matches = 0

    for frame_number in all_frames:
        p_detections = p_by_frame.get(frame_number, [])
        o_detections = o_by_frame.get(frame_number, [])

        total_p += len(p_detections)
        total_o += len(o_detections)

        for detection in p_detections:
            p_appearances[detection["track_id"]] += 1

        for detection in o_detections:
            o_appearances[detection["track_id"]] += 1

        candidates = []

        for pi, pd in enumerate(p_detections):
            for oi, od in enumerate(o_detections):
                if pd["class_id"] != od["class_id"]:
                    continue

                iou = calculate_iou(pd["bbox"], od["bbox"])

                if iou >= IOU_THRESHOLD:
                    candidates.append((iou, pi, oi))

        candidates.sort(key=lambda item: item[0], reverse=True)

        used_p = set()
        used_o = set()
        matches = []

        for iou, pi, oi in candidates:
            if pi in used_p or oi in used_o:
                continue

            used_p.add(pi)
            used_o.add(oi)

            pd = p_detections[pi]
            od = o_detections[oi]

            p_id = pd["track_id"]
            o_id = od["track_id"]

            pair_counts[(p_id, o_id)] += 1
            iou_values.append(iou)

            matches.append({
                "pytorch_track_id": p_id,
                "openvino_track_id": o_id,
                "class_id": pd["class_id"],
                "iou": round(float(iou), 5)
            })

        total_matches += len(matches)

        if matches:
            frames_with_matches += 1

        per_frame.append({
            "frame_number": frame_number,
            "pytorch_detections": len(p_detections),
            "openvino_detections": len(o_detections),
            "matched": len(matches),
            "unmatched_pytorch": len(p_detections) - len(matches),
            "unmatched_openvino": len(o_detections) - len(matches),
            "matches": matches
        })

        if frame_number % 200 == 0:
            print(
                f"Compared frame {frame_number}/{len(all_frames)} "
                f"| Matches: {total_matches}"
            )

    p_counterparts = defaultdict(Counter)
    o_counterparts = defaultdict(Counter)

    for (p_id, o_id), count in pair_counts.items():
        p_counterparts[p_id][o_id] += count
        o_counterparts[o_id][p_id] += count

    p_mapping = []

    for p_id in sorted(p_appearances):
        counterparts = p_counterparts[p_id]
        matched = sum(counterparts.values())

        dominant_id, dominant_count = (
            counterparts.most_common(1)[0]
            if counterparts else (None, 0)
        )

        p_mapping.append({
            "pytorch_track_id": p_id,
            "appearances": p_appearances[p_id],
            "matched_appearances": matched,
            "dominant_openvino_id": dominant_id,
            "mapping_purity": round(
                dominant_count / matched, 5
            ) if matched else 0.0,
            "counterparts": [
                {
                    "openvino_track_id": track_id,
                    "matched_frames": count
                }
                for track_id, count in counterparts.most_common()
            ]
        })

    o_mapping = []

    for o_id in sorted(o_appearances):
        counterparts = o_counterparts[o_id]
        matched = sum(counterparts.values())

        dominant_id, dominant_count = (
            counterparts.most_common(1)[0]
            if counterparts else (None, 0)
        )

        o_mapping.append({
            "openvino_track_id": o_id,
            "appearances": o_appearances[o_id],
            "matched_appearances": matched,
            "dominant_pytorch_id": dominant_id,
            "mapping_purity": round(
                dominant_count / matched, 5
            ) if matched else 0.0,
            "counterparts": [
                {
                    "pytorch_track_id": track_id,
                    "matched_frames": count
                }
                for track_id, count in counterparts.most_common()
            ]
        })

    summary = {
        "iou_threshold": IOU_THRESHOLD,
        "matching_method": (
            "Same-frame, same-class, greedy one-to-one matching "
            "in descending IoU order"
        ),
        "frames_compared": len(all_frames),
        "frames_with_matches": frames_with_matches,
        "pytorch_detections": total_p,
        "openvino_detections": total_o,
        "matched_detection_pairs": total_matches,
        "unmatched_pytorch": total_p - total_matches,
        "unmatched_openvino": total_o - total_matches,
        "pytorch_match_coverage": round(
            total_matches / total_p, 5
        ) if total_p else 0.0,
        "openvino_match_coverage": round(
            total_matches / total_o, 5
        ) if total_o else 0.0,
        "mean_matched_iou": round(
            float(np.mean(iou_values)), 5
        ) if iou_values else None,
        "median_matched_iou": round(
            float(np.median(iou_values)), 5
        ) if iou_values else None,
        "pytorch_unique_track_ids": len(p_appearances),
        "openvino_unique_track_ids": len(o_appearances),
        "distinct_track_id_pairs": len(pair_counts),
        "note": (
            "This is a detection-association audit, not ground-truth "
            "tracking accuracy. Greedy matching is not guaranteed "
            "to find the globally optimal assignment."
        )
    }

    result = {
        "summary": summary,
        "pytorch_to_openvino_mapping": p_mapping,
        "openvino_to_pytorch_mapping": o_mapping,
        "track_pair_cooccurrence": [
            {
                "pytorch_track_id": p_id,
                "openvino_track_id": o_id,
                "matched_frames": count
            }
            for (p_id, o_id), count in pair_counts.most_common()
        ],
        "per_frame_matching": per_frame
    }

    return result


def main():
    if not PYTORCH_FILE.exists():
        raise FileNotFoundError(PYTORCH_FILE)

    if not OPENVINO_FILE.exists():
        raise FileNotFoundError(OPENVINO_FILE)

    if OUTPUT_DIR.exists():
        raise FileExistsError(OUTPUT_DIR)

    print("INTERSECTION TRACKING CONSISTENCY AUDIT")
    print("-" * 60)
    print(f"PyTorch input: {PYTORCH_FILE}")
    print(f"OpenVINO input: {OPENVINO_FILE}")
    print(f"IoU threshold: {IOU_THRESHOLD}")

    pytorch_records = load_records(PYTORCH_FILE)
    openvino_records = load_records(OPENVINO_FILE)

    if len(pytorch_records) != len(openvino_records):
        raise ValueError(
            "Frame counts differ; inspect inputs before comparison."
        )

    result = audit(pytorch_records, openvino_records)

    OUTPUT_DIR.mkdir(parents=True)

    output_file = OUTPUT_DIR / "intersection_tracking_audit.json"

    with output_file.open("w", encoding="utf-8") as file:
        json.dump(result, file, indent=2, allow_nan=False)

    summary = result["summary"]

    print("\nAUDIT SUMMARY")
    print("-" * 50)
    for key, value in summary.items():
        print(f"{key}: {value}")

    print(f"\nFull report saved to: {output_file}")


if __name__ == "__main__":
    main()
