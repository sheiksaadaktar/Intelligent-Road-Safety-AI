import json
from pathlib import Path
from collections import Counter


# ============================================================
# V5.4 EVENT INTELLIGENCE
# ============================================================

INPUT_PATH = Path(
    "output/confirmed_risk_events_v53.json"
)

OUTPUT_PATH = Path(
    "output/safety_report_v54.json"
)


def load_events():
    """Load confirmed V5.3 risk events."""

    if not INPUT_PATH.exists():
        raise FileNotFoundError(
            f"Input file not found: {INPUT_PATH}"
        )

    with open(
        INPUT_PATH,
        "r",
        encoding="utf-8",
    ) as event_file:

        data = json.load(event_file)

    return data


def calculate_statistics(events, fps):
    """Calculate aggregate statistics from confirmed events."""

    severity_counts = Counter()
    object_counts = Counter()

    durations = []
    distances = []
    predicted_distances = []
    tcas = []
    approach_speeds = []
    convergences = []

    for event in events:

        severity = event.get(
            "peak_risk",
            "UNKNOWN",
        )

        severity_counts[severity] += 1

        pair = event.get(
            "pair",
            [],
        )

        for object_id in pair:
            object_counts[str(object_id)] += 1

        duration = event.get(
            "duration_seconds"
        )

        if duration is not None:
            durations.append(
                float(duration)
            )

        distance = event.get(
            "min_distance"
        )

        if distance is not None:
            distances.append(
                float(distance)
            )

        predicted_distance = event.get(
            "min_predicted_distance"
        )

        if predicted_distance is not None:
            predicted_distances.append(
                float(predicted_distance)
            )

        tca = event.get(
            "min_tca"
        )

        if tca is not None:
            tcas.append(
                float(tca)
            )

        approach_speed = event.get(
            "max_approach_speed"
        )

        if approach_speed is not None:
            approach_speeds.append(
                float(approach_speed)
            )

        convergence = event.get(
            "max_convergence"
        )

        if convergence is not None:
            convergences.append(
                float(convergence)
            )

    total_events = len(events)

    def percentage(count):
        if total_events == 0:
            return 0.0

        return (
            count / total_events
        ) * 100.0

    statistics = {
        "total_confirmed_events": total_events,

        "severity_counts": {
            "CRITICAL": severity_counts.get(
                "CRITICAL",
                0,
            ),
            "HIGH": severity_counts.get(
                "HIGH",
                0,
            ),
            "MEDIUM": severity_counts.get(
                "MEDIUM",
                0,
            ),
        },

        "severity_percentages": {
            "CRITICAL": percentage(
                severity_counts.get(
                    "CRITICAL",
                    0,
                )
            ),
            "HIGH": percentage(
                severity_counts.get(
                    "HIGH",
                    0,
                )
            ),
            "MEDIUM": percentage(
                severity_counts.get(
                    "MEDIUM",
                    0,
                )
            ),
        },

        "duration": {
            "average_seconds": (
                sum(durations) / len(durations)
                if durations
                else 0.0
            ),
            "maximum_seconds": (
                max(durations)
                if durations
                else 0.0
            ),
            "minimum_seconds": (
                min(durations)
                if durations
                else 0.0
            ),
        },

        "distance": {
            "minimum_pixels": (
                min(distances)
                if distances
                else None
            ),
            "average_pixels": (
                sum(distances) / len(distances)
                if distances
                else None
            ),
        },

        "predicted_distance": {
            "minimum_pixels": (
                min(predicted_distances)
                if predicted_distances
                else None
            ),
            "average_pixels": (
                sum(predicted_distances)
                / len(predicted_distances)
                if predicted_distances
                else None
            ),
        },

        "tca": {
            "minimum_seconds": (
                min(tcas)
                if tcas
                else None
            ),
            "average_seconds": (
                sum(tcas) / len(tcas)
                if tcas
                else None
            ),
        },

        "approach_speed": {
            "maximum_pixels_per_second": (
                max(approach_speeds)
                if approach_speeds
                else None
            ),
            "average_pixels_per_second": (
                sum(approach_speeds)
                / len(approach_speeds)
                if approach_speeds
                else None
            ),
        },

        "convergence": {
            "maximum": (
                max(convergences)
                if convergences
                else None
            ),
            "average": (
                sum(convergences)
                / len(convergences)
                if convergences
                else None
            ),
        },

        "object_involvement": dict(
            sorted(
                object_counts.items(),
                key=lambda item: (
                    -item[1],
                    int(item[0]),
                ),
            )
        ),
    }

    return statistics


def build_report(data):
    """Build the V5.4 machine-readable safety report."""

    events = data.get(
        "events",
        [],
    )

    fps = data.get(
        "fps"
    )

    statistics = calculate_statistics(
        events,
        fps,
    )

    report = {
        "version": "V5.4",
        "source_version": data.get(
            "version",
            "UNKNOWN",
        ),
        "video_path": data.get(
            "video_path"
        ),
        "fps": fps,

        "statistics": statistics,

        "events": events,
    }

    return report


def main():

    print()
    print("=" * 70)
    print("V5.4 EVENT INTELLIGENCE")
    print("=" * 70)

    data = load_events()

    print(
        f"Input: {INPUT_PATH}"
    )

    events = data.get(
        "events",
        [],
    )

    print(
        f"Confirmed events loaded: "
        f"{len(events)}"
    )

    report = build_report(
        data
    )

    OUTPUT_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with open(
        OUTPUT_PATH,
        "w",
        encoding="utf-8",
    ) as report_file:

        json.dump(
            report,
            report_file,
            indent=2,
        )

    statistics = report[
        "statistics"
    ]

    print()
    print("===== V5.4 SUMMARY =====")

    print(
        "Total confirmed events:",
        statistics[
            "total_confirmed_events"
        ],
    )

    print(
        "CRITICAL:",
        statistics[
            "severity_counts"
        ]["CRITICAL"],
    )

    print(
        "HIGH:",
        statistics[
            "severity_counts"
        ]["HIGH"],
    )

    print(
        "MEDIUM:",
        statistics[
            "severity_counts"
        ]["MEDIUM"],
    )

    print(
        "Average duration:",
        f"{statistics['duration']['average_seconds']:.3f} s",
    )

    print(
        "Minimum distance:",
        statistics[
            "distance"
        ]["minimum_pixels"],
    )

    print(
        "Minimum predicted distance:",
        statistics[
            "predicted_distance"
        ]["minimum_pixels"],
    )

    print(
        "Minimum TCA:",
        statistics[
            "tca"
        ]["minimum_seconds"],
    )

    print(
        "Maximum approach speed:",
        statistics[
            "approach_speed"
        ]["maximum_pixels_per_second"],
    )

    print()
    print(
        f"Safety report exported to: "
        f"{OUTPUT_PATH}"
    )

    print("=" * 70)


if __name__ == "__main__":
    main()
