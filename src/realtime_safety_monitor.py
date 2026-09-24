import cv2
import time
import numpy as np

from collections import defaultdict, deque
from pathlib import Path

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
from runtime_health import RuntimeHealth


VIDEO_SOURCE = "data/tracking_test.mp4"

WINDOW_NAME = (
    "Intelligent Road Safety AI - Real-Time Monitor"
)

DISPLAY_WIDTH = 1280
DISPLAY_HEIGHT = 720
UI_SCALE = 1.45

RISK_ORDER = {
    "LOW": 0,
    "MEDIUM": 1,
    "HIGH": 2,
    "CRITICAL": 3,
}

RISK_COLORS = {
    "LOW": (120, 120, 120),
    "MEDIUM": (0, 180, 255),
    "HIGH": (0, 140, 255),
    "CRITICAL": (0, 0, 255),
}

MAX_DISPLAY_PAIRS = 5
MAX_INCIDENT_HISTORY = 5


def format_tca(value):
    if value is None:
        return "N/A"

    if value >= 99:
        return "N/A"

    return f"{value:.2f}s"


def risk_color(risk):
    return RISK_COLORS.get(
        risk,
        (255, 255, 255),
    )


def lifecycle_color(state):
    if state == "CONFIRMED":
        return (0, 255, 255)

    if state == "ACTIVE":
        return (0, 165, 255)

    if state == "RESOLVED":
        return (120, 255, 120)

    return (220, 220, 220)


def resize_for_display(frame):
    return cv2.resize(
        frame,
        (
            DISPLAY_WIDTH,
            DISPLAY_HEIGHT,
        ),
        interpolation=cv2.INTER_AREA,
    )


def print_live_alert(alert):

    print("")
    print("=" * 70)
    print("CONFIRMED LIVE ROAD SAFETY ALERT")
    print("=" * 70)

    print(
        f"Alert ID: {alert['alert_id']}"
    )

    print(
        f"Pair: ID{alert['pair'][0]} <-> "
        f"ID{alert['pair'][1]}"
    )

    print(
        f"Alert level: "
        f"{alert['alert_level']}"
    )

    print(
        f"Priority: "
        f"{alert['priority']}"
    )

    print(
        f"State: "
        f"{alert['state']}"
    )

    print(
        f"Risk: "
        f"{alert['risk']['peak_risk']}"
    )

    print(
        f"Score: "
        f"{alert['risk']['peak_score']}"
    )

    trajectory = alert[
        "trajectory_evidence"
    ]

    print(
        f"Minimum distance: "
        f"{trajectory['min_distance_pixels']:.2f}px"
    )

    print(
        f"Minimum predicted distance: "
        f"{trajectory['min_predicted_distance_pixels']:.2f}px"
    )

    print(
        f"Maximum approach speed: "
        f"{trajectory['max_approach_speed_pixels_per_second']:.2f}px/s"
    )

    print(
        f"Minimum TCA: "
        f"{trajectory['min_tca_seconds']:.2f}s"
    )

    print(
        f"Maximum convergence: "
        f"{trajectory['max_convergence']:.2f}"
    )

    confirmation = alert[
        "confirmation"
    ]

    print(
        f"Strong observations: "
        f"{confirmation['strong_observations']}"
    )

    print(
        f"Strong frames: "
        f"{confirmation['strong_frames']}"
    )

    print("=" * 70)


def draw_header(
    frame,
    frame_number,
    highest_risk,
    highest_score,
    alert_statistics,
    lifecycle_statistics,
):

    cv2.rectangle(
        frame,
        (0, 0),
        (
            frame.shape[1],
            int(88 * UI_SCALE),
        ),
        (18, 18, 18),
        -1,
    )

    cv2.putText(
        frame,
        "INTELLIGENT ROAD SAFETY AI",
        (
            int(22 * UI_SCALE),
            int(31 * UI_SCALE),
        ),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.75 * UI_SCALE,
        (255, 255, 255),
        2,
        cv2.LINE_AA,
    )

    header_line = (
        f"Frame: {frame_number}   |   "
        f"Highest: {highest_risk}   |   "
        f"Score: {highest_score}   |   "
        f"Live Alerts: "
        f"{alert_statistics['total_alerts']}   |   "
        f"Incidents: "
        f"{lifecycle_statistics['total_incidents']}   |   "
        f"Active: "
        f"{lifecycle_statistics['state_counts']['ACTIVE']}"
    )

    cv2.putText(
        frame,
        header_line,
        (
            int(22 * UI_SCALE),
            int(66 * UI_SCALE),
        ),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.43 * UI_SCALE,
        (205, 205, 205),
        1,
        cv2.LINE_AA,
    )


def draw_live_alert_status(
    frame,
    latest_alert,
):

    x = 20
    y = 105

    width = 520
    height = 82

    cv2.rectangle(
        frame,
        (x, y),
        (x + width, y + height),
        (24, 24, 24),
        -1,
    )

    cv2.rectangle(
        frame,
        (x, y),
        (x + width, y + height),
        (70, 70, 70),
        1,
    )

    cv2.putText(
        frame,
        "LIVE ALERT STATUS",
        (
            x + 15,
            y + 27,
        ),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.47 * UI_SCALE,
        (255, 255, 255),
        1,
        cv2.LINE_AA,
    )

    if latest_alert is None:

        cv2.putText(
            frame,
            "No confirmed alert yet",
            (
                x + 15,
                y + 58,
            ),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.40 * UI_SCALE,
            (160, 160, 160),
            1,
            cv2.LINE_AA,
        )

        return

    pair = latest_alert["pair"]

    risk = latest_alert[
        "risk"
    ]["peak_risk"]

    text = (
        f"{latest_alert['alert_id']}  "
        f"ID{pair[0]} <-> ID{pair[1]}  "
        f"{risk}  "
        f"Score {latest_alert['risk']['peak_score']}"
    )

    cv2.putText(
        frame,
        text,
        (
            x + 15,
            y + 58,
        ),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.40 * UI_SCALE,
        risk_color(risk),
        2,
        cv2.LINE_AA,
    )


def draw_safety_analytics(
    frame,
    incidents,
    lifecycle_statistics,
):

    x = 20
    y = 198

    width = 520
    height = 130

    cv2.rectangle(
        frame,
        (x, y),
        (x + width, y + height),
        (20, 20, 20),
        -1,
    )

    cv2.rectangle(
        frame,
        (x, y),
        (x + width, y + height),
        (65, 65, 65),
        1,
    )

    cv2.putText(
        frame,
        "SAFETY ANALYTICS",
        (
            x + 15,
            y + 27,
        ),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.48 * UI_SCALE,
        (255, 255, 255),
        1,
        cv2.LINE_AA,
    )

    total = lifecycle_statistics[
        "total_incidents"
    ]

    risk_counts = lifecycle_statistics[
        "risk_counts"
    ]

    alert_counts = lifecycle_statistics[
        "alert_level_counts"
    ]

    durations = []

    distances = []

    predicted_distances = []

    for incident in incidents:

        duration = incident.get(
            "event_timing",
            {},
        ).get(
            "duration_seconds"
        )

        if duration is not None:
            durations.append(
                float(duration)
            )

        trajectory = incident.get(
            "trajectory_evidence",
            {},
        )

        distance = trajectory.get(
            "min_distance_pixels"
        )

        if distance is not None:
            distances.append(
                float(distance)
            )

        predicted = trajectory.get(
            "min_predicted_distance_pixels"
        )

        if predicted is not None:
            predicted_distances.append(
                float(predicted)
            )

    if durations:

        average_duration = (
            sum(durations)
            / len(durations)
        )

    else:

        average_duration = 0.0

    minimum_distance = (
        min(distances)
        if distances
        else 0.0
    )

    minimum_predicted = (
        min(predicted_distances)
        if predicted_distances
        else 0.0
    )

    object_involvement = (
        lifecycle_statistics[
            "object_involvement"
        ]
    )

    top_objects = sorted(
        object_involvement.items(),
        key=lambda item: (
            -item[1],
            int(item[0]),
        ),
    )[:3]

    line_one = (
        f"Incidents: {total}    "
        f"Critical: "
        f"{risk_counts.get('CRITICAL', 0)}    "
        f"High: "
        f"{risk_counts.get('HIGH', 0)}"
    )

    line_two = (
        f"Immediate: "
        f"{alert_counts.get('IMMEDIATE_ALERT', 0)}    "
        f"Priority: "
        f"{alert_counts.get('PRIORITY_ALERT', 0)}"
    )

    line_three = (
        f"Avg duration: "
        f"{average_duration:.2f}s    "
        f"Min distance: "
        f"{minimum_distance:.2f}px"
    )

    line_four = (
        f"Min predicted: "
        f"{minimum_predicted:.2f}px    "
        f"Most involved: "
    )

    if top_objects:

        line_four += "  ".join(
            f"ID{object_id}:{count}"
            for object_id, count
            in top_objects
        )

    else:

        line_four += "None"

    lines = [
        line_one,
        line_two,
        line_three,
        line_four,
    ]

    for index, line in enumerate(lines):

        cv2.putText(
            frame,
            line,
            (
                x + 15,
                y + 52 + index * 21,
            ),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.36 * UI_SCALE,
            (205, 205, 205),
            1,
            cv2.LINE_AA,
        )


def draw_incident_history(
    frame,
    incidents,
):

    x = 20
    y = 340

    width = 520
    height = 350

    cv2.rectangle(
        frame,
        (x, y),
        (x + width, y + height),
        (20, 20, 20),
        -1,
    )

    cv2.rectangle(
        frame,
        (x, y),
        (x + width, y + height),
        (65, 65, 65),
        1,
    )

    cv2.putText(
        frame,
        "INCIDENT HISTORY",
        (
            x + 15,
            y + 28,
        ),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.48 * UI_SCALE,
        (255, 255, 255),
        1,
        cv2.LINE_AA,
    )

    if not incidents:

        cv2.putText(
            frame,
            "No incidents recorded",
            (
                x + 15,
                y + 62,
            ),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.40 * UI_SCALE,
            (150, 150, 150),
            1,
            cv2.LINE_AA,
        )

        return

    recent_incidents = incidents[
        -MAX_INCIDENT_HISTORY:
    ]

    recent_incidents = list(
        reversed(
            recent_incidents
        )
    )

    entry_height = 58

    for index, incident in enumerate(
        recent_incidents
    ):

        entry_y = (
            y
            + 50
            + index * entry_height
        )

        state = incident[
            "lifecycle"
        ]["state"]

        risk = incident[
            "classification"
        ]["risk"]

        pair = incident[
            "objects"
        ]["pair"]

        score = incident[
            "classification"
        ]["score"]

        duration = incident.get(
            "event_timing",
            {},
        ).get(
            "duration_seconds",
            0.0,
        )

        state_colour = lifecycle_color(
            state
        )

        cv2.putText(
            frame,
            (
                f"{incident['incident_id']}   "
                f"ID{pair[0]} <-> ID{pair[1]}"
            ),
            (
                x + 15,
                entry_y,
            ),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.38 * UI_SCALE,
            (235, 235, 235),
            1,
            cv2.LINE_AA,
        )

        detail = (
            f"{risk}   "
            f"Score {score}   "
            f"{state}   "
            f"{duration:.2f}s"
        )

        cv2.putText(
            frame,
            detail,
            (
                x + 15,
                entry_y + 23,
            ),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.34 * UI_SCALE,
            state_colour,
            1,
            cv2.LINE_AA,
        )

        if index < len(
            recent_incidents
        ) - 1:

            separator_y = (
                entry_y + 34
            )

            cv2.line(
                frame,
                (
                    x + 15,
                    separator_y,
                ),
                (
                    x + width - 15,
                    separator_y,
                ),
                (45, 45, 45),
                1,
            )


def draw_risk_panel(
    frame,
    risk_pairs,
):

    x = 710
    y = 105

    width = 550
    height = 585

    cv2.rectangle(
        frame,
        (x, y),
        (x + width, y + height),
        (20, 20, 20),
        -1,
    )

    cv2.rectangle(
        frame,
        (x, y),
        (x + width, y + height),
        (65, 65, 65),
        1,
    )

    cv2.putText(
        frame,
        "TRAJECTORY RISK MONITOR",
        (
            x + 15,
            y + 28,
        ),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.48 * UI_SCALE,
        (255, 255, 255),
        1,
        cv2.LINE_AA,
    )

    if not risk_pairs:

        cv2.putText(
            frame,
            "No active trajectory risks",
            (
                x + 15,
                y + 60,
            ),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.40 * UI_SCALE,
            (150, 150, 150),
            1,
            cv2.LINE_AA,
        )

        return

    risk_pairs = sorted(
        risk_pairs,
        key=lambda item: (
            -RISK_ORDER.get(
                item["risk"],
                0,
            ),
            -item["score"],
        ),
    )

    risk_pairs = risk_pairs[
        :MAX_DISPLAY_PAIRS
    ]

    entry_height = 100

    for index, item in enumerate(
        risk_pairs
    ):

        entry_y = (
            y
            + 43
            + index * entry_height
        )

        risk = item[
            "risk"
        ]

        colour = risk_color(
            risk
        )

        pair = item[
            "pair"
        ]

        metrics = item[
            "metrics"
        ]

        cv2.putText(
            frame,
            (
                f"{risk}   "
                f"ID{pair[0]} <-> ID{pair[1]}   "
                f"Score {item['score']}"
            ),
            (
                x + 15,
                entry_y,
            ),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.42 * UI_SCALE,
            colour,
            2,
            cv2.LINE_AA,
        )

        line_one = (
            f"D {metrics['distance']:.1f}px    "
            f"Pred {metrics['predicted_distance']:.1f}px"
        )

        line_two = (
            f"Approach "
            f"{metrics['approach_speed']:.1f}px/s    "
            f"TCA {format_tca(metrics['tca'])}"
        )

        line_three = (
            f"Convergence "
            f"{metrics['convergence']:.2f}"
        )

        cv2.putText(
            frame,
            line_one,
            (
                x + 15,
                entry_y + 25,
            ),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.35 * UI_SCALE,
            (205, 205, 205),
            1,
            cv2.LINE_AA,
        )

        cv2.putText(
            frame,
            line_two,
            (
                x + 15,
                entry_y + 46,
            ),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.35 * UI_SCALE,
            (205, 205, 205),
            1,
            cv2.LINE_AA,
        )

        cv2.putText(
            frame,
            line_three,
            (
                x + 15,
                entry_y + 67,
            ),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.35 * UI_SCALE,
            (205, 205, 205),
            1,
            cv2.LINE_AA,
        )


def print_incident_update(
    incident,
    message,
):

    pair = incident[
        "objects"
    ]["pair"]

    classification = incident[
        "classification"
    ]

    lifecycle = incident[
        "lifecycle"
    ]

    print("")
    print(
        f"INCIDENT {message}"
    )
    print(
        f"Incident ID: "
        f"{incident['incident_id']}"
    )
    print(
        f"Pair: ID{pair[0]} <-> ID{pair[1]}"
    )
    print(
        f"State: "
        f"{lifecycle['state']}"
    )
    print(
        f"Risk: "
        f"{classification['risk']}"
    )
    print(
        f"Score: "
        f"{classification['score']}"
    )


def main():

    runtime_health = RuntimeHealth()

    print(
        f"Model: {MODEL_PATH}"
    )

    print(
        f"Video source: {VIDEO_SOURCE}"
    )

    print(
        f"Display: "
        f"{DISPLAY_WIDTH}x{DISPLAY_HEIGHT}"
    )

    print(
        f"UI scale: {UI_SCALE}"
    )

    model = YOLO(
        MODEL_PATH
    )

    capture = cv2.VideoCapture(
        VIDEO_SOURCE
    )

    runtime_health.mark_source_opened(
        capture.isOpened()
    )

    if not capture.isOpened():

        raise RuntimeError(
            f"Could not open video: "
            f"{VIDEO_SOURCE}"
        )

    source_width = int(
        capture.get(
            cv2.CAP_PROP_FRAME_WIDTH
        )
    )

    source_height = int(
        capture.get(
            cv2.CAP_PROP_FRAME_HEIGHT
        )
    )

    fps = capture.get(
        cv2.CAP_PROP_FPS
    )

    if fps <= 0:
        fps = 30.0

    print(
        f"Source resolution: "
        f"{source_width}x{source_height}"
    )

    print(
        f"Source FPS: {fps:.2f}"
    )

    live_alert_pipeline = (
        LiveAlertPipeline(
            fps=fps
        )
    )

    live_incident_lifecycle = (
        LiveIncidentLifecycle(
            fps=fps
        )
    )

    print(
        "Live alert pipeline: ENABLED"
    )

    print(
        "Temporal confirmation: ENABLED"
    )

    print(
        "Live incident lifecycle: ENABLED"
    )

    print(
        "Lifecycle states: "
        "CONFIRMED -> ACTIVE -> RESOLVED"
    )

    velocity_history = defaultdict(
        lambda: deque(
            maxlen=HISTORY_SIZE
        )
    )

    position_history = defaultdict(
        lambda: deque(
            maxlen=HISTORY_SIZE
        )
    )

    previous_positions = {}

    frame_number = 0
    latest_alert = None

    frames_processed = 0

    while True:

        frame_start_time = time.perf_counter()

        success, frame = capture.read()

        if not success:
            break

        runtime_health.record_frame_read(
            True
        )

        frame_number += 1
        frames_processed += 1

        display_frame = resize_for_display(
            frame
        )

        results = model.track(
            frame,
            persist=True,
            verbose=False,
        )

        detections = []

        if (
            results
            and results[0].boxes is not None
            and results[0].boxes.id is not None
        ):

            boxes = (
                results[0]
                .boxes
            )

            track_ids = (
                boxes.id
                .int()
                .cpu()
                .tolist()
            )

            class_ids = (
                boxes.cls
                .int()
                .cpu()
                .tolist()
            )

            coordinates = (
                boxes.xyxy
                .cpu()
                .numpy()
            )

            for (
                track_id,
                class_id,
                box,
            ) in zip(
                track_ids,
                class_ids,
                coordinates,
            ):

                center = bottom_center(
                    box
                )

                position_history[
                    track_id
                ].append(
                    center
                )

                if (
                    track_id
                    in previous_positions
                ):

                    previous_position = (
                        previous_positions[
                            track_id
                        ]
                    )

                    displacement = (
                        center
                        - previous_position
                    )

                    velocity = (
                        displacement
                        * fps
                    )

                    velocity_magnitude = (
                        float(
                            np.linalg.norm(
                                velocity
                            )
                        )
                    )

                    if (
                        velocity_magnitude
                        <= MAX_REASONABLE_VELOCITY
                    ):

                        velocity_history[
                            track_id
                        ].append(
                            velocity
                        )

                previous_positions[
                    track_id
                ] = center

                smoothed_velocity = (
                    smooth_velocity(
                        velocity_history[
                            track_id
                        ]
                    )
                )

                detections.append(
                    {
                        "id": track_id,
                        "class_id": class_id,
                        "box": box,
                        "position": center,
                        "velocity": (
                            smoothed_velocity
                        ),
                    }
                )

        runtime_health.record_frame_processed(
            has_detections=bool(detections)
        )

        risk_pairs = []
        observed_pairs = set()

        highest_risk = "LOW"
        highest_score = 0

        for i in range(
            len(detections)
        ):

            for j in range(
                i + 1,
                len(detections),
            ):

                first = detections[i]
                second = detections[j]

                if (
                    first["id"]
                    not in velocity_history
                    or second["id"]
                    not in velocity_history
                ):
                    continue

                velocity_one = (
                    smooth_velocity(
                        velocity_history[
                            first["id"]
                        ]
                    )
                )

                velocity_two = (
                    smooth_velocity(
                        velocity_history[
                            second["id"]
                        ]
                    )
                )

                if (
                    velocity_one is None
                    or velocity_two is None
                ):
                    continue

                relative_velocity = (
                    velocity_two
                    - velocity_one
                )

                relative_speed = float(
                    np.linalg.norm(
                        relative_velocity
                    )
                )

                if (
                    relative_speed
                    > MAX_REASONABLE_RELATIVE_VELOCITY
                ):
                    continue

                metrics = (
                    calculate_trajectory_metrics(
                        first["position"],
                        velocity_one,
                        second["position"],
                        velocity_two,
                    )
                )

                risk, score = (
                    calculate_risk(
                        metrics
                    )
                )

                pair = (
                    first["id"],
                    second["id"],
                )

                normalized_pair = tuple(
                    sorted(pair)
                )

                observed_pairs.add(
                    normalized_pair
                )

                live_alert_generated = (
                    live_alert_pipeline.observe(
                        frame_number=frame_number,
                        pair=pair,
                        risk=risk,
                        score=score,
                        metrics=metrics,
                    )
                )

                if live_alert_generated:

                    latest_alert = (
                        live_alert_pipeline.alerts[
                            -1
                        ]
                    )

                    print_live_alert(
                        latest_alert
                    )

                if (
                    RISK_ORDER.get(
                        risk,
                        0,
                    )
                    > RISK_ORDER.get(
                        highest_risk,
                        0,
                    )
                ):

                    highest_risk = risk
                    highest_score = score

                elif (
                    risk == highest_risk
                    and score > highest_score
                ):

                    highest_score = score

                if risk != "LOW":

                    risk_pairs.append(
                        {
                            "pair": pair,
                            "risk": risk,
                            "score": score,
                            "metrics": metrics,
                        }
                    )

                point_one = (
                    first["position"]
                )

                point_two = (
                    second["position"]
                )

                scale_x = (
                    DISPLAY_WIDTH
                    / source_width
                )

                scale_y = (
                    DISPLAY_HEIGHT
                    / source_height
                )

                p1 = (
                    int(
                        point_one[0]
                        * scale_x
                    ),
                    int(
                        point_one[1]
                        * scale_y
                    ),
                )

                p2 = (
                    int(
                        point_two[0]
                        * scale_x
                    ),
                    int(
                        point_two[1]
                        * scale_y
                    ),
                )

                cv2.line(
                    display_frame,
                    p1,
                    p2,
                    risk_color(risk),
                    2,
                    cv2.LINE_AA,
                )

        live_alert_pipeline.mark_missing_pairs(
            frame_number,
            observed_pairs,
        )

        active_events = (
            live_alert_pipeline
            .event_manager
            .active_events
        )

        completed_events = (
            live_alert_pipeline
            .event_manager
            .completed_events
        )

        previous_incident_count = len(
            live_incident_lifecycle
            .incidents
        )

        previous_active_count = len(
            live_incident_lifecycle
            .active_incidents()
        )

        live_incident_lifecycle.process_alerts(
            alerts=(
                live_alert_pipeline
                .alerts
            ),
            active_events=active_events,
            frame_number=frame_number,
        )

        live_incident_lifecycle.resolve_completed_events(
            completed_events=completed_events,
            frame_number=frame_number,
        )

        current_incident_count = len(
            live_incident_lifecycle
            .incidents
        )

        current_active_count = len(
            live_incident_lifecycle
            .active_incidents()
        )

        if (
            current_incident_count
            > previous_incident_count
            or current_active_count
            > previous_active_count
        ):

            if current_incident_count:

                latest_incident = (
                    live_incident_lifecycle
                    .incidents[-1]
                )

                print_incident_update(
                    latest_incident,
                    "UPDATE",
                )

        for detection in detections:

            box = detection[
                "box"
            ]

            track_id = detection[
                "id"
            ]

            x1, y1, x2, y2 = (
                box.astype(int)
            )

            scale_x = (
                DISPLAY_WIDTH
                / source_width
            )

            scale_y = (
                DISPLAY_HEIGHT
                / source_height
            )

            dx1 = int(
                x1 * scale_x
            )

            dy1 = int(
                y1 * scale_y
            )

            dx2 = int(
                x2 * scale_x
            )

            dy2 = int(
                y2 * scale_y
            )

            cv2.rectangle(
                display_frame,
                (
                    dx1,
                    dy1,
                ),
                (
                    dx2,
                    dy2,
                ),
                (80, 220, 80),
                2,
            )

            label = (
                f"ID {track_id}"
            )

            cv2.putText(
                display_frame,
                label,
                (
                    dx1,
                    max(
                        20,
                        dy1 - 8,
                    ),
                ),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.50 * UI_SCALE,
                (80, 220, 80),
                2,
                cv2.LINE_AA,
            )

        alert_statistics = (
            live_alert_pipeline
            .statistics()
        )

        lifecycle_statistics = (
            live_incident_lifecycle
            .statistics()
        )

        draw_header(
            display_frame,
            frame_number,
            highest_risk,
            highest_score,
            alert_statistics,
            lifecycle_statistics,
        )

        draw_live_alert_status(
            display_frame,
            latest_alert,
        )

        draw_safety_analytics(
            display_frame,
            live_incident_lifecycle.incidents,
            lifecycle_statistics,
        )

        draw_incident_history(
            display_frame,
            live_incident_lifecycle.incidents,
        )

        draw_risk_panel(
            display_frame,
            risk_pairs,
        )

        cv2.putText(
            display_frame,
            "Press Q to stop",
            (
                int(20 * UI_SCALE),
                DISPLAY_HEIGHT
                - int(18 * UI_SCALE),
            ),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.38 * UI_SCALE,
            (170, 170, 170),
            1,
            cv2.LINE_AA,
        )

        cv2.imshow(
            WINDOW_NAME,
            display_frame,
        )

        key = (
            cv2.waitKey(1)
            & 0xFF
        )

        if key == ord("q"):
            break

        runtime_health.record_processing_time(
            time.perf_counter()
            - frame_start_time
        )

    capture.release()
    cv2.destroyAllWindows()

    live_alert_pipeline.finalize()

    live_incident_lifecycle.resolve_completed_events(
        completed_events=(
            live_alert_pipeline
            .event_manager
            .completed_events
        ),
        frame_number=frame_number,
    )

    live_incident_lifecycle.resolve_all(
        frame_number=frame_number,
    )

    live_alert_pipeline.save(
        source_video=VIDEO_SOURCE
    )

    live_incident_lifecycle.save(
        source_video=VIDEO_SOURCE
    )

    print("")
    print("=" * 70)
    print("REAL-TIME SAFETY MONITOR SUMMARY")
    print("=" * 70)

    print(
        f"Frames processed: "
        f"{frames_processed}"
    )

    final_alert_statistics = (
        live_alert_pipeline.statistics()
    )

    final_lifecycle_statistics = (
        live_incident_lifecycle.statistics()
    )

    runtime_health.update_alert_count(
        final_alert_statistics[
            "total_alerts"
        ]
    )

    runtime_health.update_incident_counts(
        total_incidents=(
            final_lifecycle_statistics[
                "total_incidents"
            ]
        ),
        resolved_incidents=(
            final_lifecycle_statistics[
                "state_counts"
            ][
                "RESOLVED"
            ]
        ),
    )

    runtime_health.finish()

    print(
        f"Confirmed live alerts: "
        f"{final_alert_statistics['total_alerts']}"
    )

    print(
        f"Immediate alerts: "
        f"{final_alert_statistics['alert_counts']['IMMEDIATE_ALERT']}"
    )

    print(
        f"Priority alerts: "
        f"{final_alert_statistics['alert_counts']['PRIORITY_ALERT']}"
    )

    print(
        f"Monitor alerts: "
        f"{final_alert_statistics['alert_counts']['MONITOR']}"
    )

    print(
        f"Live incidents: "
        f"{final_lifecycle_statistics['total_incidents']}"
    )

    print(
        f"Confirmed incidents: "
        f"{final_lifecycle_statistics['state_counts']['CONFIRMED']}"
    )

    print(
        f"Active incidents: "
        f"{final_lifecycle_statistics['state_counts']['ACTIVE']}"
    )

    print(
        f"Resolved incidents: "
        f"{final_lifecycle_statistics['state_counts']['RESOLVED']}"
    )

    print(
        "Live alert output: "
        "output\\live_alerts.json"
    )

    print(
        "Live incident output: "
        "output\\live_incidents.json"
    )

    health_summary = (
        runtime_health.summary()
    )

    print("")
    print("-" * 70)
    print("RUNTIME HEALTH SUMMARY")
    print("-" * 70)

    print(
        f"Runtime state: "
        f"{health_summary['state']}"
    )

    print(
        f"Source opened: "
        f"{health_summary['source_opened']}"
    )

    print(
        f"Frames read: "
        f"{health_summary['frames_read']}"
    )

    print(
        f"Frame read failures: "
        f"{health_summary['frame_read_failures']}"
    )

    print(
        f"Frames with detections: "
        f"{health_summary['frames_with_detections']}"
    )

    print(
        f"Detection rate: "
        f"{health_summary['detection_rate_percent']:.2f}%"
    )

    print(
        f"Average processing FPS: "
        f"{health_summary['average_processing_fps']:.2f}"
    )

    print(
        f"Confirmed alerts tracked: "
        f"{health_summary['confirmed_alerts']}"
    )

    print(
        f"Total incidents tracked: "
        f"{health_summary['total_incidents']}"
    )

    print(
        f"Resolved incidents tracked: "
        f"{health_summary['resolved_incidents']}"
    )

    print(
        f"Runtime elapsed: "
        f"{health_summary['elapsed_seconds']:.2f}s"
    )

    print("-" * 70)

    print("=" * 70)


if __name__ == "__main__":
    main()



