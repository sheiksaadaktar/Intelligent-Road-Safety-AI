import json
from collections import Counter
from pathlib import Path


INPUT_PATH = Path(
    "output/live_incidents.json"
)

OUTPUT_PATH = Path(
    "output/safety_analytics.json"
)


def load_incident_data():

    if not INPUT_PATH.exists():

        raise FileNotFoundError(
            f"Incident data not found: {INPUT_PATH}"
        )

    with open(
        INPUT_PATH,
        "r",
        encoding="utf-8",
    ) as file:

        return json.load(file)


def safe_average(values):

    if not values:
        return 0.0

    return sum(values) / len(values)


def analyze_incidents(data):

    incidents = data.get(
        "incidents",
        [],
    )

    risk_counter = Counter()
    alert_counter = Counter()
    state_counter = Counter()
    object_counter = Counter()

    durations = []
    distances = []
    predicted_distances = []
    tca_values = []
    approach_speeds = []
    convergence_values = []

    for incident in incidents:

        classification = incident.get(
            "classification",
            {},
        )

        lifecycle = incident.get(
            "lifecycle",
            {},
        )

        trajectory = incident.get(
            "trajectory_evidence",
            {}

        )

        objects = incident.get(
            "objects",
            {},
        )

        risk = classification.get(
            "risk"
        )

        alert_level = classification.get(
            "alert_level"
        )

        state = lifecycle.get(
            "state"
        )

        if risk:
            risk_counter[risk] += 1

        if alert_level:
            alert_counter[
                alert_level
            ] += 1

        if state:
            state_counter[
                state
            ] += 1

        pair = objects.get(
            "pair",
            [],
        )

        for object_id in pair:

            object_counter[
                str(object_id)
            ] += 1

        event_timing = incident.get(
            "event_timing",
            {},
        )

        duration = event_timing.get(
            "duration_seconds"
        )

        if duration is not None:
            durations.append(
                float(duration)
            )

        distance = trajectory.get(
            "min_distance_pixels"
        )

        if distance is not None:
            distances.append(
                float(distance)
            )

        predicted_distance = trajectory.get(
            "min_predicted_distance_pixels"
        )

        if predicted_distance is not None:
            predicted_distances.append(
                float(predicted_distance)
            )

        tca = trajectory.get(
            "min_tca_seconds"
        )

        if tca is not None:
            tca_values.append(
                float(tca)
            )

        approach_speed = trajectory.get(
            "max_approach_speed_pixels_per_second"
        )

        if approach_speed is not None:
            approach_speeds.append(
                float(approach_speed)
            )

        convergence = trajectory.get(
            "max_convergence"
        )

        if convergence is not None:
            convergence_values.append(
                float(convergence)
            )

    total = len(
        incidents
    )

    return {
        "version": "Safety-Analytics-1.0",

        "source": {
            "file": str(INPUT_PATH),
            "version": data.get(
                "version"
            ),
            "source_video": data.get(
                "source",
                {}
            ).get(
                "video"
            ),
            "fps": data.get(
                "source",
                {}
            ).get(
                "fps"
            ),
        },

        "overview": {
            "total_incidents": total,
            "risk_counts": {
                "CRITICAL": risk_counter.get(
                    "CRITICAL",
                    0,
                ),
                "HIGH": risk_counter.get(
                    "HIGH",
                    0,
                ),
                "MEDIUM": risk_counter.get(
                    "MEDIUM",
                    0,
                ),
            },
            "alert_level_counts": {
                "IMMEDIATE_ALERT": alert_counter.get(
                    "IMMEDIATE_ALERT",
                    0,
                ),
                "PRIORITY_ALERT": alert_counter.get(
                    "PRIORITY_ALERT",
                    0,
                ),
                "MONITOR": alert_counter.get(
                    "MONITOR",
                    0,
                ),
            },
            "lifecycle_state_counts": {
                "CONFIRMED": state_counter.get(
                    "CONFIRMED",
                    0,
                ),
                "ACTIVE": state_counter.get(
                    "ACTIVE",
                    0,
                ),
                "RESOLVED": state_counter.get(
                    "RESOLVED",
                    0,
                ),
            },
        },

        "duration_seconds": {
            "average": safe_average(
                durations
            ),
            "minimum": (
                min(durations)
                if durations
                else 0.0
            ),
            "maximum": (
                max(durations)
                if durations
                else 0.0
            ),
        },

        "trajectory": {
            "minimum_distance_pixels": (
                min(distances)
                if distances
                else 0.0
            ),
            "average_distance_pixels": (
                safe_average(
                    distances
                )
            ),
            "minimum_predicted_distance_pixels": (
                min(predicted_distances)
                if predicted_distances
                else 0.0
            ),
            "average_predicted_distance_pixels": (
                safe_average(
                    predicted_distances
                )
            ),
            "minimum_tca_seconds": (
                min(tca_values)
                if tca_values
                else 0.0
            ),
            "average_tca_seconds": (
                safe_average(
                    tca_values
                )
            ),
            "maximum_approach_speed_pixels_per_second": (
                max(approach_speeds)
                if approach_speeds
                else 0.0
            ),
            "average_approach_speed_pixels_per_second": (
                safe_average(
                    approach_speeds
                )
            ),
            "maximum_convergence": (
                max(convergence_values)
                if convergence_values
                else 0.0
            ),
            "average_convergence": (
                safe_average(
                    convergence_values
                )
            ),
        },

        "object_involvement": dict(
            object_counter
        ),
    }


def save_analytics(analytics):

    OUTPUT_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with open(
        OUTPUT_PATH,
        "w",
        encoding="utf-8",
    ) as file:

        json.dump(
            analytics,
            file,
            indent=2,
        )


def main():

    print("=" * 70)
    print(
        "INTELLIGENT ROAD SAFETY AI"
    )
    print(
        "HISTORICAL SAFETY ANALYTICS"
    )
    print("=" * 70)

    data = load_incident_data()

    analytics = analyze_incidents(
        data
    )

    save_analytics(
        analytics
    )

    overview = analytics[
        "overview"
    ]

    trajectory = analytics[
        "trajectory"
    ]

    duration = analytics[
        "duration_seconds"
    ]

    print(
        f"Total incidents: "
        f"{overview['total_incidents']}"
    )

    print(
        f"Critical: "
        f"{overview['risk_counts']['CRITICAL']}"
    )

    print(
        f"High: "
        f"{overview['risk_counts']['HIGH']}"
    )

    print(
        f"Medium: "
        f"{overview['risk_counts']['MEDIUM']}"
    )

    print(
        f"Immediate alerts: "
        f"{overview['alert_level_counts']['IMMEDIATE_ALERT']}"
    )

    print(
        f"Priority alerts: "
        f"{overview['alert_level_counts']['PRIORITY_ALERT']}"
    )

    print(
        f"Average duration: "
        f"{duration['average']:.3f}s"
    )

    print(
        f"Minimum distance: "
        f"{trajectory['minimum_distance_pixels']:.3f}px"
    )

    print(
        f"Minimum predicted distance: "
        f"{trajectory['minimum_predicted_distance_pixels']:.3f}px"
    )

    print(
        f"Minimum TCA: "
        f"{trajectory['minimum_tca_seconds']:.3f}s"
    )

    print(
        f"Maximum approach speed: "
        f"{trajectory['maximum_approach_speed_pixels_per_second']:.3f}px/s"
    )

    print("")
    print(
        f"Output: {OUTPUT_PATH}"
    )
    print("=" * 70)


if __name__ == "__main__":
    main()