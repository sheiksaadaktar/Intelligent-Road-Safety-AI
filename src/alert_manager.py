import json
from pathlib import Path


# ============================================================
# V5.7 ALERT MANAGER
# ============================================================

INPUT_PATH = Path("output/safety_decisions_v56.json")
OUTPUT_PATH = Path("output/alerts_v57.json")


ALERT_DEFINITIONS = {
    "IMMEDIATE_ALERT": {
        "priority": 1,
        "state": "ACTIVE",
        "message": "Immediate collision-risk alert",
    },
    "PRIORITY_ALERT": {
        "priority": 2,
        "state": "ACTIVE",
        "message": "Priority collision-risk alert",
    },
    "MONITOR": {
        "priority": 3,
        "state": "MONITORING",
        "message": "Monitor developing road-safety risk",
    },
}


def load_decisions():
    """Load validated V5.6 safety decisions."""

    if not INPUT_PATH.exists():
        raise FileNotFoundError(
            f"V5.6 decision file not found: {INPUT_PATH}"
        )

    with open(
        INPUT_PATH,
        "r",
        encoding="utf-8",
    ) as decision_file:
        return json.load(decision_file)


def get_alert_definition(response_category):
    """Return the V5.7 alert definition for a V5.6 response."""

    category = str(
        response_category
    ).upper()

    return ALERT_DEFINITIONS.get(
        category,
        ALERT_DEFINITIONS["MONITOR"],
    )


def build_alert(decision, alert_index):
    """Convert one V5.6 decision into a V5.7 alert."""

    response_category = str(
        decision.get(
            "response_category",
            "MONITOR",
        )
    ).upper()

    definition = get_alert_definition(
        response_category
    )

    return {
        "alert_id": f"V57-{alert_index:04d}",
        "alert_index": alert_index,
        "event_index": decision.get(
            "event_index"
        ),
        "pair": decision.get(
            "pair",
            [],
        ),
        "alert_level": response_category,
        "priority": definition["priority"],
        "state": definition["state"],
        "message": definition["message"],
        "risk": {
            "peak_risk": decision.get(
                "peak_risk"
            ),
            "peak_score": decision.get(
                "peak_score"
            ),
        },
        "event": {
            "start_frame": decision.get(
                "start_frame"
            ),
            "last_frame": decision.get(
                "last_frame"
            ),
            "duration_seconds": decision.get(
                "duration_seconds"
            ),
        },
        "trajectory_evidence": decision.get(
            "trajectory_evidence",
            {},
        ),
        "confirmation": decision.get(
            "confirmation",
            {},
        ),
    }


def calculate_alert_statistics(alerts):
    """Calculate summary statistics for V5.7 alerts."""

    total = len(alerts)

    alert_counts = {
        "IMMEDIATE_ALERT": 0,
        "PRIORITY_ALERT": 0,
        "MONITOR": 0,
    }

    state_counts = {
        "ACTIVE": 0,
        "MONITORING": 0,
    }

    risk_counts = {
        "CRITICAL": 0,
        "HIGH": 0,
        "MEDIUM": 0,
    }

    for alert in alerts:
        alert_level = alert.get(
            "alert_level"
        )

        if alert_level in alert_counts:
            alert_counts[alert_level] += 1

        state = alert.get(
            "state"
        )

        if state in state_counts:
            state_counts[state] += 1

        peak_risk = (
            alert
            .get("risk", {})
            .get("peak_risk")
        )

        if peak_risk in risk_counts:
            risk_counts[peak_risk] += 1

    if total > 0:
        alert_percentages = {
            key: (
                value / total
            ) * 100
            for key, value in alert_counts.items()
        }
    else:
        alert_percentages = {
            key: 0.0
            for key in alert_counts
        }

    return {
        "total_alerts": total,
        "alert_counts": alert_counts,
        "alert_percentages": alert_percentages,
        "state_counts": state_counts,
        "peak_risk_counts": risk_counts,
    }


def build_output(decision_data, alerts):
    """Build the complete V5.7 machine-readable output."""

    statistics = calculate_alert_statistics(
        alerts
    )

    return {
        "version": "V5.7",
        "source_version": decision_data.get(
            "version"
        ),
        "source_video": decision_data.get(
            "video_path"
        ),
        "fps": decision_data.get(
            "fps"
        ),
        "alert_policy": ALERT_DEFINITIONS,
        "statistics": statistics,
        "alerts": alerts,
    }


def main():
    print("=" * 70)
    print("V5.7 ALERT MANAGER")
    print("=" * 70)

    print(
        f"Input: {INPUT_PATH}"
    )

    decision_data = load_decisions()

    decisions = decision_data.get(
        "decisions",
        [],
    )

    print(
        f"Safety decisions loaded: {len(decisions)}"
    )

    alerts = []

    for index, decision in enumerate(
        decisions,
        start=1,
    ):
        alert = build_alert(
            decision,
            index,
        )

        alerts.append(alert)

    output = build_output(
        decision_data,
        alerts,
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
    print("===== V5.7 SUMMARY =====")
    print(
        "Total alerts:",
        statistics["total_alerts"],
    )
    print(
        "IMMEDIATE_ALERT:",
        statistics["alert_counts"][
            "IMMEDIATE_ALERT"
        ],
    )
    print(
        "PRIORITY_ALERT:",
        statistics["alert_counts"][
            "PRIORITY_ALERT"
        ],
    )
    print(
        "MONITOR:",
        statistics["alert_counts"][
            "MONITOR"
        ],
    )

    print("")
    print(
        "===== V5.7 OUTPUT ====="
    )
    print(
        f"Alerts exported to: {OUTPUT_PATH}"
    )

    print("=" * 70)


if __name__ == "__main__":
    main()
