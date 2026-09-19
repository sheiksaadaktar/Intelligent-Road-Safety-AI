import json
from pathlib import Path


INPUT_PATH = Path("output/alerts_v57.json")
OUTPUT_PATH = Path("output/incident_history.json")


def load_alert_data():
    if not INPUT_PATH.exists():
        raise FileNotFoundError(
            f"Input alert file not found: {INPUT_PATH}"
        )

    with open(INPUT_PATH, "r", encoding="utf-8") as file:
        return json.load(file)


def build_incident(alert, source_data, incident_number):
    event = alert.get("event", {})
    risk = alert.get("risk", {})
    trajectory = alert.get("trajectory_evidence", {})
    confirmation = alert.get("confirmation", {})

    incident_id = f"INC-{incident_number:04d}"

    return {
        "incident_id": incident_id,
        "alert_id": alert.get("alert_id"),
        "alert_index": alert.get("alert_index"),
        "event_index": alert.get("event_index"),

        "objects": {
            "pair": alert.get("pair", [])
        },

        "classification": {
            "risk": risk.get("peak_risk"),
            "score": risk.get("peak_score"),
            "alert_level": alert.get("alert_level"),
            "priority": alert.get("priority"),
            "state": alert.get("state"),
            "message": alert.get("message")
        },

        "event_timing": {
            "start_frame": event.get("start_frame"),
            "last_frame": event.get("last_frame"),
            "duration_seconds": event.get("duration_seconds")
        },

        "trajectory_evidence": {
            "min_distance_pixels": trajectory.get(
                "min_distance_pixels"
            ),
            "min_predicted_distance_pixels": trajectory.get(
                "min_predicted_distance_pixels"
            ),
            "min_tca_seconds": trajectory.get(
                "min_tca_seconds"
            ),
            "max_approach_speed_pixels_per_second": trajectory.get(
                "max_approach_speed_pixels_per_second"
            ),
            "max_convergence": trajectory.get(
                "max_convergence"
            )
        },

        "confirmation_evidence": {
            "frames_seen": confirmation.get("frames_seen"),
            "strong_observations": confirmation.get(
                "strong_observations"
            ),
            "strong_frames": confirmation.get("strong_frames", []),
            "confirmed": confirmation.get("confirmed")
        },

        "lifecycle": {
            "current_state": alert.get("state"),
            "acknowledged": False,
            "resolved": False
        },

        "source": {
            "source_version": source_data.get("source_version"),
            "source_video": source_data.get("source_video"),
            "fps": source_data.get("fps")
        }
    }


def build_statistics(incidents):
    total = len(incidents)

    risk_counts = {}
    alert_level_counts = {}
    state_counts = {}
    object_counts = {}

    durations = []

    for incident in incidents:
        classification = incident["classification"]
        timing = incident["event_timing"]

        risk = classification.get("risk")
        alert_level = classification.get("alert_level")
        state = classification.get("state")

        if risk:
            risk_counts[risk] = risk_counts.get(risk, 0) + 1

        if alert_level:
            alert_level_counts[alert_level] = (
                alert_level_counts.get(alert_level, 0) + 1
            )

        if state:
            state_counts[state] = state_counts.get(state, 0) + 1

        pair = incident["objects"].get("pair", [])

        for object_id in pair:
            object_key = str(object_id)
            object_counts[object_key] = (
                object_counts.get(object_key, 0) + 1
            )

        duration = timing.get("duration_seconds")

        if isinstance(duration, (int, float)):
            durations.append(duration)

    average_duration = (
        sum(durations) / len(durations)
        if durations
        else 0.0
    )

    return {
        "total_incidents": total,
        "risk_counts": risk_counts,
        "alert_level_counts": alert_level_counts,
        "state_counts": state_counts,
        "object_involvement": object_counts,
        "average_duration_seconds": average_duration,
        "longest_duration_seconds": (
            max(durations) if durations else 0.0
        ),
        "shortest_duration_seconds": (
            min(durations) if durations else 0.0
        )
    }


def build_history(source_data):
    alerts = source_data.get("alerts", [])

    incidents = []

    for number, alert in enumerate(alerts, start=1):
        incidents.append(
            build_incident(
                alert,
                source_data,
                number
            )
        )

    return {
        "version": "Incident-History-1.0",
        "source_version": source_data.get("version"),
        "source_file": str(INPUT_PATH),
        "source_video": source_data.get("source_video"),
        "fps": source_data.get("fps"),
        "incident_count": len(incidents),
        "statistics": build_statistics(incidents),
        "incidents": incidents
    }


def main():
    print("=" * 70)
    print("INTELLIGENT ROAD SAFETY AI")
    print("INCIDENT HISTORY")
    print("=" * 70)

    print(f"Input: {INPUT_PATH}")

    source_data = load_alert_data()

    history = build_history(source_data)

    OUTPUT_PATH.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    with open(
        OUTPUT_PATH,
        "w",
        encoding="utf-8"
    ) as file:
        json.dump(
            history,
            file,
            indent=2,
            ensure_ascii=False
        )

    print(f"Alerts loaded: {len(source_data.get('alerts', []))}")
    print(f"Incidents recorded: {history['incident_count']}")
    print("")
    print(
        "Incident history exported to: "
        f"{OUTPUT_PATH}"
    )
    print("=" * 70)


if __name__ == "__main__":
    main()
