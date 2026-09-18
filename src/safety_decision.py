import json
from pathlib import Path


# ============================================================
# V5.6 SAFETY DECISION LAYER
# ============================================================

INPUT_PATH = Path(
    "output/safety_report_v54.json"
)

OUTPUT_PATH = Path(
    "output/safety_decisions_v56.json"
)


# Existing risk levels are translated into
# deterministic system-response categories.
RESPONSE_CATEGORIES = {
    "CRITICAL": "IMMEDIATE_ALERT",
    "HIGH": "PRIORITY_ALERT",
    "MEDIUM": "MONITOR",
}


def load_report():
    """
    Load the V5.4 safety intelligence report.
    """

    if not INPUT_PATH.exists():
        raise FileNotFoundError(
            f"Input report not found: {INPUT_PATH}"
        )

    with open(
        INPUT_PATH,
        "r",
        encoding="utf-8",
    ) as report_file:
        return json.load(report_file)


def get_response_category(peak_risk):
    """
    Convert the existing V5.4 peak risk level into
    a deterministic V5.6 response category.
    """

    risk = str(
        peak_risk
    ).upper()

    return RESPONSE_CATEGORIES.get(
        risk,
        "MONITOR",
    )


def build_decision(event, event_index):
    """
    Build one V5.6 decision record while preserving
    the underlying V5.3/V5.4 trajectory evidence.
    """

    peak_risk = str(
        event.get(
            "peak_risk",
            "MEDIUM",
        )
    ).upper()

    response_category = get_response_category(
        peak_risk
    )

    return {
        "event_index": event_index,

        "pair": event.get(
            "pair",
            [],
        ),

        "start_frame": event.get(
            "start_frame"
        ),

        "last_frame": event.get(
            "last_frame"
        ),

        "duration_seconds": event.get(
            "duration_seconds"
        ),

        "peak_risk": peak_risk,

        "peak_score": event.get(
            "peak_score"
        ),

        "response_category": response_category,

        "trajectory_evidence": {
            "min_distance_pixels": event.get(
                "min_distance"
            ),

            "min_predicted_distance_pixels": event.get(
                "min_predicted_distance"
            ),

            "min_tca_seconds": event.get(
                "min_tca"
            ),

            "max_approach_speed_pixels_per_second": event.get(
                "max_approach_speed"
            ),

            "max_convergence": event.get(
                "max_convergence"
            ),
        },

        "confirmation": {
            "frames_seen": event.get(
                "frames_seen"
            ),

            "strong_observations": event.get(
                "strong_observations"
            ),

            "strong_frames": event.get(
                "strong_frames",
                [],
            ),

            "confirmed": event.get(
                "confirmed",
                False,
            ),
        },
    }


def calculate_decision_statistics(decisions):
    """
    Calculate summary counts for the V5.6 decision layer.
    """

    total = len(decisions)

    response_counts = {
        "IMMEDIATE_ALERT": 0,
        "PRIORITY_ALERT": 0,
        "MONITOR": 0,
    }

    risk_counts = {
        "CRITICAL": 0,
        "HIGH": 0,
        "MEDIUM": 0,
    }

    for decision in decisions:

        response = decision.get(
            "response_category"
        )

        if response in response_counts:
            response_counts[response] += 1

        risk = decision.get(
            "peak_risk"
        )

        if risk in risk_counts:
            risk_counts[risk] += 1

    if total > 0:
        response_percentages = {
            key: (
                value / total
            ) * 100
            for key, value in response_counts.items()
        }
    else:
        response_percentages = {
            key: 0.0
            for key in response_counts
        }

    return {
        "total_decisions": total,

        "response_counts": response_counts,

        "response_percentages": response_percentages,

        "peak_risk_counts": risk_counts,
    }


def build_output(report, decisions):
    """
    Construct the machine-readable V5.6 output.
    """

    statistics = calculate_decision_statistics(
        decisions
    )

    return {
        "version": "V5.6",

        "source_version": report.get(
            "version"
        ),

        "video_path": report.get(
            "video_path"
        ),

        "fps": report.get(
            "fps"
        ),

        "decision_policy": {
            "CRITICAL": "IMMEDIATE_ALERT",
            "HIGH": "PRIORITY_ALERT",
            "MEDIUM": "MONITOR",
        },

        "statistics": statistics,

        "decisions": decisions,
    }


def main():
    print("=" * 70)
    print("V5.6 SAFETY DECISION LAYER")
    print("=" * 70)

    print(
        f"Input: {INPUT_PATH}"
    )

    report = load_report()

    source_events = report.get(
        "events",
        []
    )

    print(
        f"Confirmed events loaded: {len(source_events)}"
    )

    decisions = []

    for index, event in enumerate(
        source_events,
        start=1,
    ):
        decision = build_decision(
            event,
            index,
        )

        decisions.append(
            decision
        )

    output = build_output(
        report,
        decisions,
    )

    OUTPUT_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with open(
        OUTPUT_PATH,
        "w",
        encoding="utf-8",
    ) as output_file:
        json.dump(
            output,
            output_file,
            indent=2,
        )

    statistics = output[
        "statistics"
    ]

    print("")
    print("===== V5.6 SUMMARY =====")

    print(
        "Total decisions:",
        statistics[
            "total_decisions"
        ],
    )

    print(
        "IMMEDIATE_ALERT:",
        statistics[
            "response_counts"
        ][
            "IMMEDIATE_ALERT"
        ],
    )

    print(
        "PRIORITY_ALERT:",
        statistics[
            "response_counts"
        ][
            "PRIORITY_ALERT"
        ],
    )

    print(
        "MONITOR:",
        statistics[
            "response_counts"
        ][
            "MONITOR"
        ],
    )

    print("")
    print(
        f"Safety decisions exported to: {OUTPUT_PATH}"
    )

    print("=" * 70)


if __name__ == "__main__":
    main()
