import json
from pathlib import Path
from collections import Counter

ROOT = Path(__file__).resolve().parent.parent

INPUTS = {
    "gap_policy": ROOT / "output/velocity_gap_policy_comparison/gap_policy_comparison.json",
    "visibility_v1": ROOT / "output/velocity_visibility_experiment/visibility_experiment_report.json",
    "visibility_v2": ROOT / "output/velocity_visibility_experiment_v2/visibility_v2_report.json",
    "visibility_v3": ROOT / "output/velocity_visibility_experiment_v3/visibility_v3_report.json",
}

OUTPUT_DIR = ROOT / "output/velocity_experiment_consolidated"
JSON_OUTPUT = OUTPUT_DIR / "velocity_experiment_comparison.json"
MARKDOWN_OUTPUT = OUTPUT_DIR / "velocity_experiment_comparison.md"

BACKENDS = ("pytorch", "openvino")
RISK_LEVELS = ("LOW", "MEDIUM", "HIGH", "CRITICAL")


def load_json(path):
    if not path.exists():
        raise FileNotFoundError(f"Required report not found: {path}")
    with path.open("r", encoding="utf-8-sig") as handle:
        return json.load(handle)


def require_mapping(value, label):
    if not isinstance(value, dict):
        raise ValueError(f"Expected JSON object for {label}")
    return value


def event_pair(event):
    pair = event.get("pair")
    if not isinstance(pair, list) or len(pair) != 2:
        raise ValueError(f"Invalid event pair: {event}")
    return tuple(sorted(pair))


def summarize_event(event, lifecycle=False):
    result = {
        "pair": list(event_pair(event)),
        "start_frame": event.get("start_frame"),
        "peak_risk": event.get("peak_risk"),
        "peak_score": event.get("peak_score"),
        "frames_seen": event.get("frames_seen"),
        "strong_observations": event.get("strong_observations"),
    }

    if lifecycle:
        result.update({
            "last_risk_frame": event.get("last_risk_frame"),
            "last_visible_frame": event.get("last_visible_frame"),
            "end_frame": event.get("end_frame"),
            "end_reason": event.get("end_reason"),
        })
    else:
        result.update({
            "last_frame": event.get("last_frame"),
            "duration_seconds": event.get("duration_seconds"),
            "end_reason": event.get("end_reason"),
        })

    return result


def summarize_policy(policy_name, policy_data):
    counts = policy_data.get("counts", {})
    risk_counts = policy_data.get("risk_counts", {})
    events = policy_data.get("confirmed_events", [])

    return {
        "policy": policy_name,
        "frames": counts.get("frames"),
        "detections": counts.get("detections"),
        "pairs_seen": counts.get("pairs_seen"),
        "evaluated_pairs": counts.get("evaluated_pairs"),
        "insufficient_history": counts.get("insufficient_history"),
        "visible_pairs": counts.get("visible_pairs"),
        "visible_but_unevaluated_pairs": counts.get(
            "visible_but_unevaluated_pairs"
        ),
        "gap_resets": counts.get("gap_resets"),
        "corrected_gaps": counts.get("corrected_gaps"),
        "retained_history_gaps": counts.get("retained_history_gaps"),
        "velocity_limit_rejections": counts.get("velocity_limit_rejections"),
        "risk_counts": {
            risk: risk_counts.get(risk, 0)
            for risk in RISK_LEVELS
        },
        "confirmed_event_count": policy_data.get(
            "confirmed_event_count", len(events)
        ),
        "confirmed_events": [
            summarize_event(event)
            for event in events
        ],
    }


def build_event_comparison(baseline, candidate, candidate_name):
    baseline_map = {
        event_pair(event): event
        for event in baseline.get("confirmed_events", [])
    }
    candidate_map = {
        event_pair(event): event
        for event in candidate.get("confirmed_events", [])
    }

    all_pairs = sorted(set(baseline_map) | set(candidate_map))
    comparisons = []

    for pair in all_pairs:
        base = baseline_map.get(pair)
        cand = candidate_map.get(pair)

        if base is None:
            comparisons.append({
                "pair": list(pair),
                "status": "candidate_only",
                "candidate": summarize_event(
                    cand, lifecycle=("end_frame" in cand)
                ),
            })
            continue

        if cand is None:
            comparisons.append({
                "pair": list(pair),
                "status": "baseline_only",
                "baseline": summarize_event(base),
            })
            continue

        base_start = base.get("start_frame")
        cand_start = cand.get("start_frame")
        base_last = base.get("last_frame")
        cand_last_risk = cand.get(
            "last_risk_frame",
            cand.get("last_frame"),
        )

        comparisons.append({
            "pair": list(pair),
            "status": "matched",
            "baseline": summarize_event(base),
            "candidate": summarize_event(
                cand, lifecycle=("end_frame" in cand)
            ),
            "start_frame_delta": (
                cand_start - base_start
                if base_start is not None and cand_start is not None
                else None
            ),
            "last_risk_frame_delta": (
                cand_last_risk - base_last
                if base_last is not None and cand_last_risk is not None
                else None
            ),
            "risk_classification_agrees": (
                base.get("peak_risk") == cand.get("peak_risk")
            ),
            "peak_score_delta": (
                cand.get("peak_score") - base.get("peak_score")
                if base.get("peak_score") is not None
                and cand.get("peak_score") is not None
                else None
            ),
        })

    return {
        "baseline_event_count": len(baseline_map),
        "candidate_event_count": len(candidate_map),
        "matched_pairs": sum(
            item["status"] == "matched" for item in comparisons
        ),
        "baseline_only_pairs": [
            item["pair"] for item in comparisons
            if item["status"] == "baseline_only"
        ],
        "candidate_only_pairs": [
            item["pair"] for item in comparisons
            if item["status"] == "candidate_only"
        ],
        "events": comparisons,
    }


def risk_deltas(baseline, candidate):
    base = baseline.get("risk_counts", {})
    cand = candidate.get("risk_counts", {})
    return {
        risk: cand.get(risk, 0) - base.get(risk, 0)
        for risk in RISK_LEVELS
    }


def markdown_table(headers, rows):
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join("---" for _ in headers) + " |",
    ]
    for row in rows:
        lines.append("| " + " | ".join(str(value) for value in row) + " |")
    return "\n".join(lines)


def main():
    reports = {
        name: load_json(path)
        for name, path in INPUTS.items()
    }

    gap_report = require_mapping(
        reports["gap_policy"].get("backends"),
        "gap policy backends",
    )
    v1_report = require_mapping(
        reports["visibility_v1"].get("backends"),
        "visibility V1 backends",
    )
    v2_results = require_mapping(
        reports["visibility_v2"].get("results"),
        "visibility V2 results",
    )
    v3_results = require_mapping(
        reports["visibility_v3"].get("results"),
        "visibility V3 results",
    )

    consolidated = {
        "title": "Velocity and Visibility Experiment Comparison",
        "source_reports": {
            name: str(path.relative_to(ROOT))
            for name, path in INPUTS.items()
        },
        "methodology": {
            "comparison_video": reports["gap_policy"].get(
                "methodology", {}
            ).get("video"),
            "fps": reports["gap_policy"].get(
                "methodology", {}
            ).get("fps"),
            "risk_thresholds_changed": False,
            "production_files_modified": False,
            "yolo_inference_run": False,
            "event_matching": (
                "Track pairs are matched within each backend using sorted "
                "track-ID pairs. PyTorch IDs are not compared directly "
                "with OpenVINO IDs."
            ),
            "timing_note": (
                "For V3, last_risk_frame is compared with older last_frame. "
                "V3 end_frame is a lifecycle finalization timestamp and is "
                "not treated as equivalent to older last_frame."
            ),
        },
        "backends": {},
    }

    markdown = [
        "# Velocity and Visibility Experiment Comparison",
        "",
        "## Scope and interpretation",
        "",
        "This report consolidates the original velocity baseline, gap policies, "
        "visibility V1, visibility V2 and visibility V3.",
        "",
        "- Track pairs are compared within each backend.",
        "- PyTorch and OpenVINO track IDs are not directly interchangeable.",
        "- Risk counts are evaluated pair instances, not unique incidents.",
        "- V3 `end_frame` is a lifecycle finalization frame. It is not treated "
        "as equivalent to older `last_frame` values.",
        "- These experiments use the same recorded detections and production "
        "risk functions; they do not establish ground-truth prediction accuracy.",
        "",
    ]

    for backend in BACKENDS:
        gap_backend = require_mapping(
            gap_report.get(backend), f"gap policy {backend}"
        )
        policies = require_mapping(
            gap_backend.get("policies"), f"{backend} policies"
        )

        if "original" not in policies or "reset_3" not in policies:
            raise ValueError(
                f"{backend}: expected original and reset_3 policies"
            )

        original = policies["original"]
        policy_summaries = {
            name: summarize_policy(name, policy)
            for name, policy in policies.items()
        }

        v1_backend = require_mapping(
            v1_report.get(backend), f"V1 {backend}"
        )
        v1_candidate = require_mapping(
            v1_backend.get("candidate"), f"V1 candidate {backend}"
        )

        v2_backend = require_mapping(
            v2_results.get(backend), f"V2 {backend}"
        )
        v3_backend = require_mapping(
            v3_results.get(backend), f"V3 {backend}"
        )

        stages = {
            "original": original,
            "reset_3": policies["reset_3"],
            "visibility_v1": v1_candidate,
            "visibility_v2": v2_backend,
            "visibility_v3": v3_backend,
        }

        stage_summaries = {}
        comparisons = {}

        for stage_name, stage_data in stages.items():
            if stage_name == "original":
                stage_summaries[stage_name] = summarize_policy(
                    stage_name, stage_data
                )
            elif stage_name in ("visibility_v2", "visibility_v3"):
                stage_summaries[stage_name] = {
                    "policy": stage_name,
                    "frames": stage_data.get("frames"),
                    "detections": stage_data.get("detections"),
                    "pairs_seen": stage_data.get("visible_pair_instances"),
                    "evaluated_pairs": stage_data.get(
                        "evaluated_pair_instances"
                    ),
                    "insufficient_history": stage_data.get(
                        "no_velocity_history"
                    ),
                    "visible_pairs": stage_data.get(
                        "visible_pair_instances"
                    ),
                    "visible_but_unevaluated_pairs": stage_data.get(
                        "visible_but_unevaluated"
                    ),
                    "gap_resets": stage_data.get("gap_resets"),
                    "corrected_gaps": stage_data.get("corrected_gaps"),
                    "velocity_limit_rejections": stage_data.get(
                        "velocity_limit_rejections"
                    ),
                    "risk_counts": {
                        risk: stage_data.get("risk_counts", {}).get(risk, 0)
                        for risk in RISK_LEVELS
                    },
                    "confirmed_event_count": stage_data.get(
                        "confirmed_event_count"
                    ),
                    "finalized_event_count": stage_data.get(
                        "finalized_event_count"
                    ),
                    "unconfirmed_finalized_count": stage_data.get(
                        "unconfirmed_finalized_count"
                    ),
                    "finalization_reason_counts": stage_data.get(
                        "finalization_reason_counts"
                    ),
                    "confirmed_end_reason_counts": stage_data.get(
                        "confirmed_end_reason_counts"
                    ),
                    "unconfirmed_end_reason_counts": stage_data.get(
                        "unconfirmed_end_reason_counts"
                    ),
                    "confirmed_events": [
                        summarize_event(event, lifecycle=True)
                        for event in stage_data.get("confirmed_events", [])
                    ],
                }
            else:
                stage_summaries[stage_name] = summarize_policy(
                    stage_name, stage_data
                )

            comparisons[stage_name] = {
                "risk_count_delta_vs_original": risk_deltas(
                    original, stage_data
                ),
                "evaluated_pair_delta_vs_original": (
                    stage_data.get("counts", {}).get("evaluated_pairs", 0)
                    - original.get("counts", {}).get("evaluated_pairs", 0)
                    if stage_data.get("counts") is not None
                    else stage_data.get("evaluated_pair_instances", 0)
                    - original.get("counts", {}).get("evaluated_pairs", 0)
                ),
                "event_comparison_vs_original": build_event_comparison(
                    original, stage_data, stage_name
                ),
            }

        # V3 finalization accounting integrity checks.
        v3_reason_counts = v3_backend.get(
            "finalization_reason_counts", {}
        )
        v3_confirmed_reasons = v3_backend.get(
            "confirmed_end_reason_counts", {}
        )
        v3_unconfirmed_reasons = v3_backend.get(
            "unconfirmed_end_reason_counts", {}
        )

        accounting = {
            "all_finalizations_sum": sum(v3_reason_counts.values()),
            "confirmed_reason_sum": sum(v3_confirmed_reasons.values()),
            "unconfirmed_reason_sum": sum(v3_unconfirmed_reasons.values()),
            "reported_finalized_count": v3_backend.get(
                "finalized_event_count"
            ),
            "reported_confirmed_count": v3_backend.get(
                "confirmed_event_count"
            ),
            "reported_unconfirmed_count": v3_backend.get(
                "unconfirmed_finalized_count"
            ),
        }
        accounting["all_finalizations_reconcile"] = (
            accounting["all_finalizations_sum"]
            == accounting["reported_finalized_count"]
        )
        accounting["confirmed_reasons_reconcile"] = (
            accounting["confirmed_reason_sum"]
            == accounting["reported_confirmed_count"]
        )
        accounting["unconfirmed_reasons_reconcile"] = (
            accounting["unconfirmed_reason_sum"]
            == accounting["reported_unconfirmed_count"]
        )

        consolidated["backends"][backend] = {
            "policy_summaries": policy_summaries,
            "stage_summaries": stage_summaries,
            "comparisons_to_original": comparisons,
            "v3_finalization_accounting": accounting,
        }

        markdown.extend([
            f"## {backend.upper()}",
            "",
            "### Stage summary",
            "",
        ])

        rows = []
        for stage_name, summary in stage_summaries.items():
            risk = summary["risk_counts"]
            rows.append([
                stage_name,
                summary.get("evaluated_pairs"),
                risk["LOW"],
                risk["MEDIUM"],
                risk["HIGH"],
                risk["CRITICAL"],
                summary.get("confirmed_event_count"),
            ])

        markdown.append(markdown_table(
            [
                "Stage", "Evaluated pairs", "LOW", "MEDIUM",
                "HIGH", "CRITICAL", "Confirmed events"
            ],
            rows,
        ))
        markdown.append("")

        markdown.extend([
            "### Risk-count deltas versus original",
            "",
        ])

        delta_rows = []
        for stage_name, comparison in comparisons.items():
            delta = comparison["risk_count_delta_vs_original"]
            delta_rows.append([
                stage_name,
                delta["LOW"],
                delta["MEDIUM"],
                delta["HIGH"],
                delta["CRITICAL"],
                comparison["evaluated_pair_delta_vs_original"],
            ])

        markdown.append(markdown_table(
            [
                "Stage", "LOW Δ", "MEDIUM Δ", "HIGH Δ",
                "CRITICAL Δ", "Evaluated pairs Δ"
            ],
            delta_rows,
        ))
        markdown.append("")

        markdown.extend([
            "### Event-pair comparison versus original",
            "",
        ])

        event_rows = []
        for stage_name, comparison in comparisons.items():
            event_comparison = comparison["event_comparison_vs_original"]
            event_rows.append([
                stage_name,
                event_comparison["baseline_event_count"],
                event_comparison["candidate_event_count"],
                event_comparison["matched_pairs"],
                len(event_comparison["baseline_only_pairs"]),
                len(event_comparison["candidate_only_pairs"]),
            ])

        markdown.append(markdown_table(
            [
                "Stage", "Baseline events", "Stage events",
                "Matched pairs", "Baseline only", "Stage only"
            ],
            event_rows,
        ))
        markdown.append("")

        markdown.extend([
            "### V3 finalization accounting",
            "",
            f"- All finalizations reconcile: "
            f"**{accounting['all_finalizations_reconcile']}**",
            f"- Confirmed reasons reconcile: "
            f"**{accounting['confirmed_reasons_reconcile']}**",
            f"- Unconfirmed reasons reconcile: "
            f"**{accounting['unconfirmed_reasons_reconcile']}**",
            f"- Finalized / confirmed / unconfirmed: "
            f"{accounting['reported_finalized_count']} / "
            f"{accounting['reported_confirmed_count']} / "
            f"{accounting['reported_unconfirmed_count']}",
            "",
            "### V3 confirmed event lifecycle",
            "",
        ])

        lifecycle_rows = []
        for event in v3_backend.get("confirmed_events", []):
            pair = event_pair(event)
            lifecycle_rows.append([
                f"{pair[0]}–{pair[1]}",
                event.get("peak_risk"),
                event.get("start_frame"),
                event.get("last_risk_frame"),
                event.get("last_visible_frame"),
                event.get("end_frame"),
                event.get("end_reason"),
            ])

        markdown.append(markdown_table(
            [
                "Pair", "Peak risk", "Start", "Last risk",
                "Last visible", "End frame", "End reason"
            ],
            lifecycle_rows,
        ))
        markdown.append("")

    markdown.extend([
        "## Interpretation notes",
        "",
        "1. Risk-count differences describe changes in evaluated pair instances, "
        "not necessarily changes in real-world collision risk.",
        "2. Event-pair matching uses track IDs within each backend. Track IDs "
        "are not compared across backends.",
        "3. A matched event pair does not prove identical trajectories or "
        "ground-truth tracking accuracy.",
        "4. V3 lifecycle timestamps are kept separate from older event timing "
        "fields. In particular, `end_frame` is not substituted for "
        "`last_frame`.",
        "5. The V3 finalization accounting checks validate internal report "
        "consistency, not detection or risk-prediction accuracy.",
        "",
        "## Production recommendation",
        "",
        "This consolidated report is an offline experiment comparison. "
        "It does not authorize a production change. Before adopting a new "
        "velocity or lifecycle policy, validate it against manually reviewed "
        "ground-truth scenarios, including missed detections, occlusion, "
        "track-ID switches, and known near-collision events.",
        "",
    ])

    OUTPUT_DIR.mkdir(parents=True, exist_ok=False)

    with JSON_OUTPUT.open("w", encoding="utf-8") as handle:
        json.dump(consolidated, handle, indent=2)

    MARKDOWN_OUTPUT.write_text(
        "\n".join(markdown),
        encoding="utf-8",
    )

    print("CONSOLIDATED ANALYSIS COMPLETED")
    print(f"JSON report: {JSON_OUTPUT}")
    print(f"Markdown report: {MARKDOWN_OUTPUT}")

    for backend in BACKENDS:
        accounting = consolidated["backends"][backend][
            "v3_finalization_accounting"
        ]
        print(
            f"{backend.upper()} | "
            f"V3 accounting: "
            f"{accounting['all_finalizations_reconcile']}, "
            f"{accounting['confirmed_reasons_reconcile']}, "
            f"{accounting['unconfirmed_reasons_reconcile']}"
        )


if __name__ == "__main__":
    main()
