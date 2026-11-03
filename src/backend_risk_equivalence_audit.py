import json
from pathlib import Path
from collections import defaultdict


# ============================================================
# BACKEND RISK-EQUIVALENCE AUDIT
# Read-only comparison of existing PyTorch and OpenVINO results
# ============================================================

BASE = Path("output/backend_comparison")
IDENTITY_FILE = Path("output/backend_tracking_audit/identity_mapping.json")
OUTPUT_DIR = Path("output/backend_risk_equivalence_audit")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

FILES = {
    "pytorch_alerts": BASE / "pytorch" / "live_alerts.json",
    "pytorch_incidents": BASE / "pytorch" / "live_incidents.json",
    "openvino_alerts": BASE / "openvino" / "live_alerts.json",
    "openvino_incidents": BASE / "openvino" / "live_incidents.json",
}

ALERT_METRICS = [
    "min_distance_pixels",
    "min_predicted_distance_pixels",
    "min_tca_seconds",
    "max_approach_speed_pixels_per_second",
    "max_convergence",
]

INCIDENT_METRICS = ALERT_METRICS

FRAME_FIELDS = [
    ("event", "start_frame"),
    ("event", "last_frame"),
]

INCIDENT_FRAME_FIELDS = [
    ("lifecycle", "confirmed_frame"),
    ("lifecycle", "active_frame"),
    ("lifecycle", "resolved_frame"),
    ("event_timing", "start_frame"),
    ("event_timing", "last_observed_frame"),
]


# ------------------------------------------------------------
# File and data helpers
# ------------------------------------------------------------

def load_json(path):
    if not path.exists():
        raise FileNotFoundError(f"Required file not found: {path}")

    with path.open("r", encoding="utf-8") as file:
        return json.load(file)


def get_nested(data, *keys):
    value = data

    for key in keys:
        if not isinstance(value, dict):
            return None
        value = value.get(key)

    return value


def save_json(path, data):
    with path.open("w", encoding="utf-8") as file:
        json.dump(data, file, indent=2, allow_nan=False)


def normalize_pair(pair):
    if not isinstance(pair, (list, tuple)) or len(pair) != 2:
        return None

    return tuple(sorted((int(pair[0]), int(pair[1]))))


def interval(record, kind):
    if kind == "alert":
        start = get_nested(record, "event", "start_frame")
        end = get_nested(record, "event", "last_frame")
    else:
        start = get_nested(record, "event_timing", "start_frame")
        end = get_nested(record, "event_timing", "last_observed_frame")

    if start is None or end is None:
        return None

    return int(start), int(end)


def interval_overlap(a, b):
    if a is None or b is None:
        return 0

    start = max(a[0], b[0])
    end = min(a[1], b[1])

    if end < start:
        return 0

    return end - start + 1


def interval_iou(a, b):
    if a is None or b is None:
        return None

    overlap = interval_overlap(a, b)
    union = (
        max(a[1], b[1]) - min(a[0], b[0]) + 1
    )

    return overlap / union if union > 0 else None


def numeric_comparison(left, right):
    if left is None or right is None:
        return {
            "pytorch": left,
            "openvino": right,
            "absolute_difference": None,
            "relative_difference_percent": None,
        }

    left = float(left)
    right = float(right)
    difference = right - left

    relative = (
        abs(difference) / abs(left) * 100
        if left != 0
        else None
    )

    return {
        "pytorch": left,
        "openvino": right,
        "openvino_minus_pytorch": difference,
        "absolute_difference": abs(difference),
        "relative_difference_percent": relative,
    }


def compare_metric_group(left, right, metric_names):
    result = {}

    for metric in metric_names:
        result[metric] = numeric_comparison(
            left.get(metric),
            right.get(metric),
        )

    return result


def compare_frame_fields(left, right, fields):
    result = {}

    for group, field in fields:
        left_value = get_nested(left, group, field)
        right_value = get_nested(right, group, field)

        result[f"{group}.{field}"] = numeric_comparison(
            left_value,
            right_value,
        )

    return result


# ------------------------------------------------------------
# Track-ID mapping
# ------------------------------------------------------------

def build_track_mappings(identity_data):
    p_to_o = {}
    o_to_p = {}

    for item in identity_data["pytorch_to_openvino_track_mapping"]:
        p_id = int(item["pytorch_track_id"])
        o_id = item["dominant_openvino_track_id"]

        if o_id is None:
            continue

        # Only use fully consistent mappings for pair comparison.
        if float(item["mapping_purity"]) < 1.0:
            continue

        p_to_o[p_id] = int(o_id)

    for item in identity_data["openvino_to_pytorch_track_mapping"]:
        o_id = int(item["openvino_track_id"])
        p_id = item["dominant_pytorch_track_id"]

        if p_id is None:
            continue

        if float(item["mapping_purity"]) < 1.0:
            continue

        o_to_p[o_id] = int(p_id)

    return p_to_o, o_to_p


def canonical_pair_for_record(record, backend, o_to_p):
    # Alerts store their pair at record["pair"].
    # Incidents store their pair at record["objects"]["pair"].
    original_pair = record.get("pair")

    if original_pair is None:
        original_pair = get_nested(record, "objects", "pair")

    if not isinstance(original_pair, (list, tuple)) or len(original_pair) != 2:
        return None

    if backend == "pytorch":
        return normalize_pair(original_pair)

    converted = []

    for track_id in original_pair:
        mapped_id = o_to_p.get(int(track_id))

        if mapped_id is None:
            return None

        converted.append(mapped_id)

    return normalize_pair(converted)


def annotate_records(records, backend, kind, o_to_p):
    annotated = []

    for index, record in enumerate(records):
        item = dict(record)
        item["_source_index"] = index
        item["_canonical_pytorch_pair"] = canonical_pair_for_record(
            record,
            backend,
            o_to_p,
        )
        item["_interval"] = interval(record, kind)
        annotated.append(item)

    return annotated


# ------------------------------------------------------------
# Event matching
# ------------------------------------------------------------

def match_events(pytorch_records, openvino_records):
    p_groups = defaultdict(list)
    o_groups = defaultdict(list)

    for record in pytorch_records:
        pair = record["_canonical_pytorch_pair"]
        if pair is not None:
            p_groups[pair].append(record)

    for record in openvino_records:
        pair = record["_canonical_pytorch_pair"]
        if pair is not None:
            o_groups[pair].append(record)

    matches = []
    unmatched_p = []
    unmatched_o = []
    all_pairs = sorted(set(p_groups) | set(o_groups))

    for pair in all_pairs:
        p_items = p_groups.get(pair, [])
        o_items = o_groups.get(pair, [])

        candidates = []

        for p_item in p_items:
            for o_item in o_items:
                overlap = interval_overlap(
                    p_item["_interval"],
                    o_item["_interval"],
                )

                if overlap <= 0:
                    continue

                iou = interval_iou(
                    p_item["_interval"],
                    o_item["_interval"],
                )

                p_start = p_item["_interval"][0]
                o_start = o_item["_interval"][0]

                candidates.append((
                    overlap,
                    iou if iou is not None else 0.0,
                    -abs(p_start - o_start),
                    p_item,
                    o_item,
                ))

        candidates.sort(
            key=lambda candidate: (
                candidate[0],
                candidate[1],
                candidate[2],
            ),
            reverse=True,
        )

        used_p = set()
        used_o = set()

        for overlap, iou, _, p_item, o_item in candidates:
            p_index = p_item["_source_index"]
            o_index = o_item["_source_index"]

            if p_index in used_p or o_index in used_o:
                continue

            used_p.add(p_index)
            used_o.add(o_index)

            matches.append({
                "canonical_pytorch_pair": list(pair),
                "pytorch": p_item,
                "openvino": o_item,
                "overlapping_frames": overlap,
                "event_window_iou": iou,
            })

        unmatched_p.extend(
            item for item in p_items
            if item["_source_index"] not in used_p
        )

        unmatched_o.extend(
            item for item in o_items
            if item["_source_index"] not in used_o
        )

    # Include records whose IDs could not be mapped.
    unmatched_p.extend(
        item for item in pytorch_records
        if item["_canonical_pytorch_pair"] is None
    )
    unmatched_o.extend(
        item for item in openvino_records
        if item["_canonical_pytorch_pair"] is None
    )

    return matches, unmatched_p, unmatched_o


# ------------------------------------------------------------
# Detailed comparisons
# ------------------------------------------------------------

def compare_alert_match(match):
    p = match["pytorch"]
    o = match["openvino"]

    p_risk = p.get("risk", {})
    o_risk = o.get("risk", {})

    p_evidence = p.get("trajectory_evidence", {})
    o_evidence = o.get("trajectory_evidence", {})

    p_confirmation = p.get("confirmation", {})
    o_confirmation = o.get("confirmation", {})

    return {
        "canonical_pytorch_pair": match["canonical_pytorch_pair"],
        "pytorch_pair": p.get("pair"),
        "openvino_pair": o.get("pair"),
        "pytorch_alert_id": p.get("alert_id"),
        "openvino_alert_id": o.get("alert_id"),
        "pytorch_alert_level": p.get("alert_level"),
        "openvino_alert_level": o.get("alert_level"),
        "pytorch_peak_risk": p_risk.get("peak_risk"),
        "openvino_peak_risk": o_risk.get("peak_risk"),
        "risk_classification_agrees": (
            p_risk.get("peak_risk") == o_risk.get("peak_risk")
        ),
        "alert_level_agrees": (
            p.get("alert_level") == o.get("alert_level")
        ),
        "peak_score": numeric_comparison(
            p_risk.get("peak_score"),
            o_risk.get("peak_score"),
        ),
        "event_window": {
            "pytorch": p.get("event"),
            "openvino": o.get("event"),
            "start_frame_difference": numeric_comparison(
                get_nested(p, "event", "start_frame"),
                get_nested(o, "event", "start_frame"),
            ),
            "last_frame_difference": numeric_comparison(
                get_nested(p, "event", "last_frame"),
                get_nested(o, "event", "last_frame"),
            ),
            "overlapping_frames": match["overlapping_frames"],
            "interval_iou": match["event_window_iou"],
        },
        "confirmation": {
            "pytorch": p_confirmation,
            "openvino": o_confirmation,
            "frames_seen": numeric_comparison(
                p_confirmation.get("frames_seen"),
                o_confirmation.get("frames_seen"),
            ),
            "strong_observations": numeric_comparison(
                p_confirmation.get("strong_observations"),
                o_confirmation.get("strong_observations"),
            ),
            "strong_frames_agree": (
                p_confirmation.get("strong_frames")
                == o_confirmation.get("strong_frames")
            ),
        },
        "trajectory_metrics": compare_metric_group(
            p_evidence,
            o_evidence,
            ALERT_METRICS,
        ),
        "states": {
            "pytorch": p.get("state"),
            "openvino": o.get("state"),
            "agree": p.get("state") == o.get("state"),
        },
    }


def compare_incident_match(match):
    p = match["pytorch"]
    o = match["openvino"]

    p_class = p.get("classification", {})
    o_class = o.get("classification", {})

    p_evidence = p.get("trajectory_evidence", {})
    o_evidence = o.get("trajectory_evidence", {})

    return {
        "canonical_pytorch_pair": match["canonical_pytorch_pair"],
        "pytorch_pair": get_nested(p, "objects", "pair"),
        "openvino_pair": get_nested(o, "objects", "pair"),
        "pytorch_incident_id": p.get("incident_id"),
        "openvino_incident_id": o.get("incident_id"),
        "classification": {
            "pytorch_risk": p_class.get("risk"),
            "openvino_risk": o_class.get("risk"),
            "risk_classification_agrees": (
                p_class.get("risk") == o_class.get("risk")
            ),
            "pytorch_alert_level": p_class.get("alert_level"),
            "openvino_alert_level": o_class.get("alert_level"),
            "alert_level_agrees": (
                p_class.get("alert_level")
                == o_class.get("alert_level")
            ),
            "score": numeric_comparison(
                p_class.get("score"),
                o_class.get("score"),
            ),
        },
        "lifecycle": {
            "pytorch": p.get("lifecycle"),
            "openvino": o.get("lifecycle"),
            "state_agrees": (
                get_nested(p, "lifecycle", "state")
                == get_nested(o, "lifecycle", "state")
            ),
            "frame_differences": compare_frame_fields(
                p,
                o,
                INCIDENT_FRAME_FIELDS[:3],
            ),
        },
        "event_timing": {
            "pytorch": p.get("event_timing"),
            "openvino": o.get("event_timing"),
            "frame_differences": compare_frame_fields(
                p,
                o,
                INCIDENT_FRAME_FIELDS[3:],
            ),
            "overlapping_frames": match["overlapping_frames"],
            "interval_iou": match["event_window_iou"],
        },
        "trajectory_metrics": compare_metric_group(
            p_evidence,
            o_evidence,
            INCIDENT_METRICS,
        ),
        "confirmation_evidence": {
            "pytorch": p.get("confirmation_evidence"),
            "openvino": o.get("confirmation_evidence"),
            "strong_frames_agree": (
                get_nested(p, "confirmation_evidence", "strong_frames")
                == get_nested(o, "confirmation_evidence", "strong_frames")
            ),
        },
        "resolution": {
            "pytorch": p.get("resolution"),
            "openvino": o.get("resolution"),
        },
    }


# ------------------------------------------------------------
# Human-readable report
# ------------------------------------------------------------

def print_match_table(title, matches, kind):
    print("\n" + title)
    print("-" * 105)

    if not matches:
        print("No matched events.")
        return

    if kind == "alert":
        print(
            f"{'Pair':<12} {'P alert':<12} {'OV alert':<12} "
            f"{'P risk':<10} {'OV risk':<10} "
            f"{'P start':>8} {'OV start':>8} "
            f"{'P score':>8} {'OV score':>8}"
        )
    else:
        print(
            f"{'Pair':<12} {'P incident':<14} {'OV incident':<14} "
            f"{'P risk':<10} {'OV risk':<10} "
            f"{'P resolved':>10} {'OV resolved':>10} "
            f"{'P score':>8} {'OV score':>8}"
        )

    for match in matches:
        p = match["pytorch"]
        o = match["openvino"]
        pair = str(match["canonical_pytorch_pair"])

        if kind == "alert":
            print(
                f"{pair:<12} "
                f"{str(p.get('alert_id')):<12} "
                f"{str(o.get('alert_id')):<12} "
                f"{str(get_nested(p, 'risk', 'peak_risk')):<10} "
                f"{str(get_nested(o, 'risk', 'peak_risk')):<10} "
                f"{get_nested(p, 'event', 'start_frame'):>8} "
                f"{get_nested(o, 'event', 'start_frame'):>8} "
                f"{get_nested(p, 'risk', 'peak_score'):>8} "
                f"{get_nested(o, 'risk', 'peak_score'):>8}"
            )
        else:
            print(
                f"{pair:<12} "
                f"{str(p.get('incident_id')):<14} "
                f"{str(o.get('incident_id')):<14} "
                f"{str(get_nested(p, 'classification', 'risk')):<10} "
                f"{str(get_nested(o, 'classification', 'risk')):<10} "
                f"{get_nested(p, 'lifecycle', 'resolved_frame'):>10} "
                f"{get_nested(o, 'lifecycle', 'resolved_frame'):>10} "
                f"{get_nested(p, 'classification', 'score'):>8} "
                f"{get_nested(o, 'classification', 'score'):>8}"
            )


def summarize_unmatched(records, backend, kind):
    result = []

    for record in records:
        if kind == "alert":
            pair = record.get("pair")
            record_id = record.get("alert_id")
            start = get_nested(record, "event", "start_frame")
            end = get_nested(record, "event", "last_frame")
        else:
            pair = get_nested(record, "objects", "pair")
            record_id = record.get("incident_id")
            start = get_nested(record, "event_timing", "start_frame")
            end = get_nested(record, "event_timing", "last_observed_frame")

        result.append({
            "backend": backend,
            "record_id": record_id,
            "original_pair": pair,
            "canonical_pytorch_pair": (
                list(record["_canonical_pytorch_pair"])
                if record["_canonical_pytorch_pair"] is not None
                else None
            ),
            "start_frame": start,
            "end_frame": end,
        })

    return result


# ------------------------------------------------------------
# Main audit
# ------------------------------------------------------------

def main():
    print("=" * 78)
    print("INTELLIGENT ROAD SAFETY AI")
    print("BACKEND RISK-EQUIVALENCE AUDIT")
    print("=" * 78)

    identity_data = load_json(IDENTITY_FILE)

    raw = {
        key: load_json(path)
        for key, path in FILES.items()
    }

    p_to_o, o_to_p = build_track_mappings(identity_data)

    print(f"\nVerified PyTorch -> OpenVINO ID mappings: {len(p_to_o)}")
    print(f"Verified OpenVINO -> PyTorch ID mappings: {len(o_to_p)}")

    policy_agrees = (
        raw["pytorch_alerts"].get("confirmation_policy")
        == raw["openvino_alerts"].get("confirmation_policy")
    )

    print(f"Alert confirmation policies identical: {policy_agrees}")

    # Annotate records with their canonical PyTorch object pair.
    p_alerts = annotate_records(
        raw["pytorch_alerts"].get("alerts", []),
        "pytorch",
        "alert",
        o_to_p,
    )
    o_alerts = annotate_records(
        raw["openvino_alerts"].get("alerts", []),
        "openvino",
        "alert",
        o_to_p,
    )

    p_incidents = annotate_records(
        raw["pytorch_incidents"].get("incidents", []),
        "pytorch",
        "incident",
        o_to_p,
    )
    o_incidents = annotate_records(
        raw["openvino_incidents"].get("incidents", []),
        "openvino",
        "incident",
        o_to_p,
    )

    # Match alerts and incidents independently.
    alert_matches, unmatched_p_alerts, unmatched_o_alerts = match_events(
        p_alerts,
        o_alerts,
    )

    incident_matches, unmatched_p_incidents, unmatched_o_incidents = match_events(
        p_incidents,
        o_incidents,
    )

    alert_comparisons = [
        compare_alert_match(match)
        for match in alert_matches
    ]

    incident_comparisons = [
        compare_incident_match(match)
        for match in incident_matches
    ]

    print_match_table(
        "MATCHED ALERTS",
        alert_matches,
        "alert",
    )

    print_match_table(
        "MATCHED INCIDENTS",
        incident_matches,
        "incident",
    )

    unmatched = {
        "alerts": {
            "pytorch": summarize_unmatched(
                unmatched_p_alerts, "pytorch", "alert"
            ),
            "openvino": summarize_unmatched(
                unmatched_o_alerts, "openvino", "alert"
            ),
        },
        "incidents": {
            "pytorch": summarize_unmatched(
                unmatched_p_incidents, "pytorch", "incident"
            ),
            "openvino": summarize_unmatched(
                unmatched_o_incidents, "openvino", "incident"
            ),
        },
    }

    # Count exact classification and lifecycle agreements.
    alert_risk_agreements = sum(
        item["risk_classification_agrees"]
        for item in alert_comparisons
    )
    alert_level_agreements = sum(
        item["alert_level_agrees"]
        for item in alert_comparisons
    )

    incident_risk_agreements = sum(
        item["classification"]["risk_classification_agrees"]
        for item in incident_comparisons
    )
    incident_level_agreements = sum(
        item["classification"]["alert_level_agrees"]
        for item in incident_comparisons
    )
    incident_state_agreements = sum(
        item["lifecycle"]["state_agrees"]
        for item in incident_comparisons
    )

    summary = {
        "matching_method": (
            "OpenVINO object IDs are converted to canonical PyTorch IDs "
            "using the verified 100%-purity tracking mapping. Events are "
            "matched only when their canonical object pair is identical "
            "and their frame intervals overlap."
        ),
        "verified_track_mappings": len(p_to_o),
        "confirmation_policy_identical": policy_agrees,
        "alerts": {
            "pytorch_total": len(p_alerts),
            "openvino_total": len(o_alerts),
            "matched": len(alert_matches),
            "unmatched_pytorch": len(unmatched_p_alerts),
            "unmatched_openvino": len(unmatched_o_alerts),
            "risk_classification_agreements": alert_risk_agreements,
            "alert_level_agreements": alert_level_agreements,
        },
        "incidents": {
            "pytorch_total": len(p_incidents),
            "openvino_total": len(o_incidents),
            "matched": len(incident_matches),
            "unmatched_pytorch": len(unmatched_p_incidents),
            "unmatched_openvino": len(unmatched_o_incidents),
            "risk_classification_agreements": incident_risk_agreements,
            "alert_level_agreements": incident_level_agreements,
            "lifecycle_state_agreements": incident_state_agreements,
        },
        "interpretation_note": (
            "Matched event counts and exact classification agreements are "
            "descriptive comparisons, not proof of safety accuracy. "
            "Unmatched events may reflect backend differences or event "
            "segmentation differences. Pixel-based trajectory metrics "
            "are compared in the same coordinate units."
        ),
    }

    report = {
        "summary": summary,
        "alert_comparisons": alert_comparisons,
        "incident_comparisons": incident_comparisons,
        "unmatched_events": unmatched,
    }

    output_path = OUTPUT_DIR / "risk_equivalence_report.json"
    save_json(output_path, report)

    print("\n" + "=" * 78)
    print("RISK-EQUIVALENCE SUMMARY")
    print("=" * 78)
    print(f"Alert confirmation policies identical: {policy_agrees}")
    print(
        f"Alerts matched: {len(alert_matches)} / "
        f"{len(p_alerts)} PyTorch, {len(o_alerts)} OpenVINO"
    )
    print(
        f"Alert risk classification agreement: "
        f"{alert_risk_agreements}/{len(alert_matches)}"
    )
    print(
        f"Alert level agreement: "
        f"{alert_level_agreements}/{len(alert_matches)}"
    )
    print(
        f"Incidents matched: {len(incident_matches)} / "
        f"{len(p_incidents)} PyTorch, {len(o_incidents)} OpenVINO"
    )
    print(
        f"Incident risk classification agreement: "
        f"{incident_risk_agreements}/{len(incident_matches)}"
    )
    print(
        f"Incident alert-level agreement: "
        f"{incident_level_agreements}/{len(incident_matches)}"
    )
    print(
        f"Incident lifecycle-state agreement: "
        f"{incident_state_agreements}/{len(incident_matches)}"
    )

    print("\nUNMATCHED ALERTS")
    print("-" * 78)
    print("PyTorch:")
    for item in unmatched["alerts"]["pytorch"]:
        print(" ", item)

    print("OpenVINO:")
    for item in unmatched["alerts"]["openvino"]:
        print(" ", item)

    print("\nUNMATCHED INCIDENTS")
    print("-" * 78)
    print("PyTorch:")
    for item in unmatched["incidents"]["pytorch"]:
        print(" ", item)

    print("OpenVINO:")
    for item in unmatched["incidents"]["openvino"]:
        print(" ", item)

    print("\nDetailed report saved to:")
    print(output_path.resolve())


if __name__ == "__main__":
    main()
