import cv2
import numpy as np

from collections import defaultdict, deque
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


# ============================================================
# REAL-TIME SAFETY MONITOR
# ============================================================

VIDEO_SOURCE = "data/tracking_test.mp4"

WINDOW_NAME = (
    "Intelligent Road Safety AI - Real-Time Monitor"
)


# ============================================================
# DISPLAY
# ============================================================

DISPLAY_WIDTH = 1280
DISPLAY_HEIGHT = 720

UI_SCALE = 1.45


# ============================================================
# RISK DISPLAY
# ============================================================

RISK_ORDER = {
    "LOW": 0,
    "MEDIUM": 1,
    "HIGH": 2,
    "CRITICAL": 3,
}

RISK_COLORS = {
    "LOW": (180, 180, 180),
    "MEDIUM": (0, 220, 255),
    "HIGH": (0, 140, 255),
    "CRITICAL": (0, 0, 255),
}


# ============================================================
# DISPLAY LIMITS
# ============================================================

MAX_DISPLAY_PAIRS = 5


# ============================================================
# FORMAT HELPERS
# ============================================================

def format_tca(tca):

    if np.isfinite(tca):
        return f"{tca:.2f}s"

    return "N/A"


def risk_color(risk):

    return RISK_COLORS.get(
        risk,
        RISK_COLORS["LOW"],
    )


# ============================================================
# DISPLAY SCALING
# ============================================================

def resize_for_display(frame):

    frame_height, frame_width = frame.shape[:2]

    if (
        frame_width <= DISPLAY_WIDTH
        and frame_height <= DISPLAY_HEIGHT
    ):
        return frame

    scale_x = DISPLAY_WIDTH / frame_width
    scale_y = DISPLAY_HEIGHT / frame_height

    scale = min(
        scale_x,
        scale_y,
    )

    new_width = max(
        1,
        int(frame_width * scale),
    )

    new_height = max(
        1,
        int(frame_height * scale),
    )

    return cv2.resize(
        frame,
        (
            new_width,
            new_height,
        ),
        interpolation=cv2.INTER_AREA,
    )


# ============================================================
# LIVE ALERT CONSOLE
# ============================================================

def print_live_alert(alert):

    print("")
    print("!" * 70)
    print("CONFIRMED LIVE ROAD SAFETY ALERT")
    print(f"Alert ID: {alert['alert_id']}")
    print(f"Objects: ID{alert['pair'][0]} <-> ID{alert['pair'][1]}")
    print(f"Alert level: {alert['alert_level']}")
    print(f"Priority: {alert['priority']}")
    print(f"State: {alert['state']}")
    print(f"Peak risk: {alert['risk']['peak_risk']}")
    print(f"Peak score: {alert['risk']['peak_score']}")

    trajectory = alert["trajectory_evidence"]

    print(
        f"Minimum distance: "
        f"{trajectory['min_distance_pixels']:.2f} px"
    )

    print(
        f"Minimum predicted distance: "
        f"{trajectory['min_predicted_distance_pixels']:.2f} px"
    )

    print(
        f"Maximum approach speed: "
        f"{trajectory['max_approach_speed_pixels_per_second']:.2f} px/s"
    )

    print(
        f"Minimum TCA: "
        f"{trajectory['min_tca_seconds']:.2f} s"
    )

    print(
        f"Maximum convergence: "
        f"{trajectory['max_convergence']:.2f}"
    )

    confirmation = alert["confirmation"]

    print(
        f"Strong observations: "
        f"{confirmation['strong_observations']}"
    )

    print(
        f"Strong frames: "
        f"{confirmation['strong_frames']}"
    )

    print("!" * 70)


# ============================================================
# DRAW HEADER
# ============================================================

def draw_header(
    frame,
    frame_number,
    highest_risk,
    highest_score,
    live_alert_count,
):

    color = risk_color(
        highest_risk
    )

    height, width = frame.shape[:2]

    header_height = 115

    cv2.rectangle(
        frame,
        (0, 0),
        (
            width,
            header_height,
        ),
        (15, 15, 15),
        -1,
    )

    cv2.putText(
        frame,
        "INTELLIGENT ROAD SAFETY AI",
        (25, 38),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.95 * UI_SCALE,
        (255, 255, 255),
        2,
        cv2.LINE_AA,
    )

    cv2.putText(
        frame,
        "REAL-TIME TRAJECTORY SAFETY MONITOR",
        (25, 72),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.68 * UI_SCALE,
        (210, 210, 210),
        2,
        cv2.LINE_AA,
    )

    cv2.putText(
        frame,
        (
            f"Frame: {frame_number}    "
            f"Highest Risk: {highest_risk}    "
            f"Score: {highest_score}    "
            f"Live Alerts: {live_alert_count}"
        ),
        (25, 103),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.58 * UI_SCALE,
        color,
        2,
        cv2.LINE_AA,
    )


# ============================================================
# DRAW RISK PANEL
# ============================================================

def draw_risk_panel(
    frame,
    risk_pairs,
):

    height, width = frame.shape[:2]

    panel_width = 530

    panel_x = (
        width
        - panel_width
        - 20
    )

    panel_y = 135

    visible_pairs = risk_pairs[
        :MAX_DISPLAY_PAIRS
    ]

    entry_height = 108

    panel_height = (
        65
        + max(
            1,
            len(visible_pairs),
        )
        * entry_height
        + 20
    )

    panel_height = min(
        panel_height,
        height - panel_y - 20,
    )

    overlay = frame.copy()

    cv2.rectangle(
        overlay,
        (
            panel_x,
            panel_y,
        ),
        (
            panel_x + panel_width,
            panel_y + panel_height,
        ),
        (15, 15, 15),
        -1,
    )

    cv2.addWeighted(
        overlay,
        0.84,
        frame,
        0.16,
        0,
        frame,
    )

    cv2.putText(
        frame,
        "CURRENT RISK PAIRS",
        (
            panel_x + 20,
            panel_y + 38,
        ),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.68 * UI_SCALE,
        (255, 255, 255),
        2,
        cv2.LINE_AA,
    )

    if not visible_pairs:

        cv2.putText(
            frame,
            "No active trajectory risk",
            (
                panel_x + 20,
                panel_y + 85,
            ),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.58 * UI_SCALE,
            (180, 180, 180),
            1,
            cv2.LINE_AA,
        )

        return

    for index, item in enumerate(
        visible_pairs
    ):

        y = (
            panel_y
            + 72
            + index * entry_height
        )

        risk = item["risk"]
        score = item["score"]

        id1 = item["id1"]
        id2 = item["id2"]

        metrics = item["metrics"]

        color = risk_color(
            risk
        )

        cv2.line(
            frame,
            (
                panel_x + 15,
                y - 10,
            ),
            (
                panel_x + panel_width - 15,
                y - 10,
            ),
            (80, 80, 80),
            1,
        )

        cv2.putText(
            frame,
            (
                f"{risk}   "
                f"ID{id1} <-> ID{id2}   "
                f"SCORE {score}"
            ),
            (
                panel_x + 20,
                y + 20,
            ),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.58 * UI_SCALE,
            color,
            2,
            cv2.LINE_AA,
        )

        cv2.putText(
            frame,
            (
                f"Distance: "
                f"{metrics['distance']:.1f}px    "
                f"Predicted: "
                f"{metrics['predicted_distance']:.1f}px"
            ),
            (
                panel_x + 20,
                y + 48,
            ),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.47 * UI_SCALE,
            (235, 235, 235),
            1,
            cv2.LINE_AA,
        )

        cv2.putText(
            frame,
            (
                f"Approach: "
                f"{metrics['approach_speed']:.1f}px/s    "
                f"TCA: "
                f"{format_tca(metrics['tca'])}"
            ),
            (
                panel_x + 20,
                y + 74,
            ),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.47 * UI_SCALE,
            (225, 225, 225),
            1,
            cv2.LINE_AA,
        )

        cv2.putText(
            frame,
            (
                f"Convergence: "
                f"{metrics['convergence']:.2f}"
            ),
            (
                panel_x + 20,
                y + 98,
            ),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.44 * UI_SCALE,
            (205, 205, 205),
            1,
            cv2.LINE_AA,
        )


# ============================================================
# DRAW LIVE ALERT STATUS
# ============================================================

def draw_live_alert_status(
    frame,
    latest_alert,
):

    if latest_alert is None:
        return

    height, width = frame.shape[:2]

    alert_level = latest_alert["alert_level"]

    if alert_level == "IMMEDIATE_ALERT":
        color = RISK_COLORS["CRITICAL"]
    elif alert_level == "PRIORITY_ALERT":
        color = RISK_COLORS["HIGH"]
    else:
        color = RISK_COLORS["MEDIUM"]

    box_width = 520
    box_height = 82

    x = 20
    y = 135

    overlay = frame.copy()

    cv2.rectangle(
        overlay,
        (x, y),
        (
            x + box_width,
            y + box_height,
        ),
        (10, 10, 10),
        -1,
    )

    cv2.addWeighted(
        overlay,
        0.90,
        frame,
        0.10,
        0,
        frame,
    )

    cv2.rectangle(
        frame,
        (x, y),
        (
            x + box_width,
            y + box_height,
        ),
        color,
        2,
    )

    cv2.putText(
        frame,
        (
            f"{alert_level}  "
            f"{latest_alert['alert_id']}"
        ),
        (
            x + 15,
            y + 31,
        ),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.60 * UI_SCALE,
        color,
        2,
        cv2.LINE_AA,
    )

    cv2.putText(
        frame,
        (
            f"ID{latest_alert['pair'][0]} <-> "
            f"ID{latest_alert['pair'][1]}   "
            f"Score: {latest_alert['risk']['peak_score']}"
        ),
        (
            x + 15,
            y + 62,
        ),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.48 * UI_SCALE,
        (240, 240, 240),
        1,
        cv2.LINE_AA,
    )


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 70)
    print("INTELLIGENT ROAD SAFETY AI")
    print("REAL-TIME SAFETY MONITOR")
    print("=" * 70)

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
        f"UI scale: "
        f"{UI_SCALE}"
    )

    print("")

    # --------------------------------------------------------
    # Load YOLO
    # --------------------------------------------------------

    print(
        "Loading YOLO model..."
    )

    model = YOLO(
        MODEL_PATH
    )

    # --------------------------------------------------------
    # Open video source
    # --------------------------------------------------------

    cap = cv2.VideoCapture(
        VIDEO_SOURCE
    )

    if not cap.isOpened():

        print(
            "ERROR: Could not open video source."
        )

        return

    fps = cap.get(
        cv2.CAP_PROP_FPS
    )

    if fps <= 0:
        fps = 30.0

    source_width = int(
        cap.get(
            cv2.CAP_PROP_FRAME_WIDTH
        )
    )

    source_height = int(
        cap.get(
            cv2.CAP_PROP_FRAME_HEIGHT
        )
    )

    print(
        f"Source resolution: "
        f"{source_width}x{source_height}"
    )

    print(
        f"Source FPS: {fps:.2f}"
    )

    # --------------------------------------------------------
    # Live alert pipeline
    # --------------------------------------------------------

    live_alert_pipeline = LiveAlertPipeline(
        fps=fps
    )

    print(
        "Live alert pipeline: ENABLED"
    )

    print(
        "Temporal confirmation: ENABLED"
    )

    print("")

    print(
        "Starting real-time safety monitor..."
    )

    print(
        "Press Q to stop."
    )

    print("")

    # --------------------------------------------------------
    # Tracking state
    # --------------------------------------------------------

    previous_positions = {}

    velocity_history = defaultdict(
        lambda: deque(
            maxlen=HISTORY_SIZE
        )
    )

    frame_number = 0

    latest_live_alert = None

    # --------------------------------------------------------
    # Window setup
    # --------------------------------------------------------

    cv2.namedWindow(
        WINDOW_NAME,
        cv2.WINDOW_NORMAL,
    )

    cv2.resizeWindow(
        WINDOW_NAME,
        DISPLAY_WIDTH,
        DISPLAY_HEIGHT,
    )

    # --------------------------------------------------------
    # Main loop
    # --------------------------------------------------------

    while True:

        success, frame = cap.read()

        if not success:

            print("")
            print(
                "Video source ended."
            )

            break

        frame_number += 1

        # ----------------------------------------------------
        # YOLO tracking
        # ----------------------------------------------------

        results = model.track(
            frame,
            persist=True,
            verbose=False,
        )

        current_positions = {}

        objects = {}

        # ----------------------------------------------------
        # Extract tracking results
        # ----------------------------------------------------

        if (
            results
            and results[0].boxes is not None
            and results[0].boxes.id is not None
        ):

            result = results[0]

            boxes = (
                result.boxes.xyxy
                .cpu()
                .numpy()
            )

            tracking_ids = (
                result.boxes.id
                .int()
                .cpu()
                .tolist()
            )

            class_ids = (
                result.boxes.cls
                .int()
                .cpu()
                .tolist()
            )

            for box, track_id, class_id in zip(
                boxes,
                tracking_ids,
                class_ids,
            ):

                position = bottom_center(
                    box
                )

                current_positions[
                    track_id
                ] = position

                # ------------------------------------------------
                # Velocity
                # ------------------------------------------------

                if (
                    track_id
                    in previous_positions
                ):

                    displacement = (
                        position
                        - previous_positions[
                            track_id
                        ]
                    )

                    velocity = (
                        displacement
                        * fps
                    )

                    velocity_magnitude = float(
                        np.linalg.norm(
                            velocity
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
                ] = position

                objects[
                    track_id
                ] = {
                    "box": box,
                    "position": position,
                    "class": result.names[
                        class_id
                    ],
                }

        # ----------------------------------------------------
        # Calculate pairwise trajectory risk
        # ----------------------------------------------------

        risk_pairs = []

        observed_pairs = set()

        highest_risk = "LOW"
        highest_score = 0

        ids = sorted(
            current_positions.keys()
        )

        for index, id1 in enumerate(
            ids
        ):

            if not velocity_history[
                id1
            ]:
                continue

            velocity_1 = smooth_velocity(
                velocity_history[id1]
            )

            for id2 in ids[
                index + 1:
            ]:

                if not velocity_history[
                    id2
                ]:
                    continue

                velocity_2 = smooth_velocity(
                    velocity_history[id2]
                )

                # ------------------------------------------------
                # Relative velocity sanity check
                # ------------------------------------------------

                relative_velocity = (
                    velocity_2
                    - velocity_1
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

                pair = tuple(
                    sorted(
                        (
                            id1,
                            id2,
                        )
                    )
                )

                # Every valid pair is observed by the temporal
                # confirmation system, including LOW-risk pairs.
                observed_pairs.add(
                    pair
                )

                # ------------------------------------------------
                # Shared trajectory model
                # ------------------------------------------------

                metrics = (
                    calculate_trajectory_metrics(
                        current_positions[id1],
                        velocity_1,
                        current_positions[id2],
                        velocity_2,
                    )
                )

                risk, score = calculate_risk(
                    metrics
                )

                # ------------------------------------------------
                # Temporal live alert pipeline
                # ------------------------------------------------

                alert_generated = (
                    live_alert_pipeline.observe(
                        frame_number=frame_number,
                        pair=pair,
                        risk=risk,
                        score=score,
                        metrics=metrics,
                    )
                )

                if alert_generated:

                    latest_live_alert = (
                        live_alert_pipeline.alerts[-1]
                    )

                    print_live_alert(
                        latest_live_alert
                    )

                # ------------------------------------------------
                # Highest risk
                # ------------------------------------------------

                if (
                    RISK_ORDER[risk]
                    > RISK_ORDER[
                        highest_risk
                    ]
                ):

                    highest_risk = risk
                    highest_score = score

                elif (
                    risk == highest_risk
                    and score > highest_score
                ):

                    highest_score = score

                # ------------------------------------------------
                # Keep risk pairs for display
                # ------------------------------------------------

                if risk != "LOW":

                    risk_pairs.append(
                        {
                            "id1": id1,
                            "id2": id2,
                            "risk": risk,
                            "score": score,
                            "metrics": metrics,
                            "position_1": (
                                current_positions[id1]
                            ),
                            "position_2": (
                                current_positions[id2]
                            ),
                        }
                    )

        # ----------------------------------------------------
        # Inform temporal engine about missing pairs
        # ----------------------------------------------------

        completed_alerts = (
            live_alert_pipeline.mark_missing_pairs(
                frame_number=frame_number,
                observed_pairs=observed_pairs,
            )
        )

        if completed_alerts:

            latest_live_alert = (
                live_alert_pipeline.alerts[-1]
            )

            print_live_alert(
                latest_live_alert
            )

        # ----------------------------------------------------
        # Sort risk pairs
        # ----------------------------------------------------

        risk_pairs.sort(
            key=lambda item: (
                RISK_ORDER[
                    item["risk"]
                ],
                item["score"],
            ),
            reverse=True,
        )

        # ----------------------------------------------------
        # Draw object boxes
        # ----------------------------------------------------

        for track_id, obj in objects.items():

            x1, y1, x2, y2 = map(
                int,
                obj["box"],
            )

            cv2.rectangle(
                frame,
                (x1, y1),
                (x2, y2),
                (255, 255, 255),
                2,
            )

            cv2.putText(
                frame,
                (
                    f"ID {track_id} "
                    f"{obj['class']}"
                ),
                (
                    x1,
                    max(
                        120,
                        y1 - 8,
                    ),
                ),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.68 * UI_SCALE,
                (255, 255, 255),
                2,
                cv2.LINE_AA,
            )

        # ----------------------------------------------------
        # Draw trajectory lines
        # ----------------------------------------------------

        for item in risk_pairs:

            risk = item["risk"]

            position_1 = item[
                "position_1"
            ]

            position_2 = item[
                "position_2"
            ]

            color = risk_color(
                risk
            )

            point_1 = (
                int(position_1[0]),
                int(position_1[1]),
            )

            point_2 = (
                int(position_2[0]),
                int(position_2[1]),
            )

            cv2.line(
                frame,
                point_1,
                point_2,
                color,
                3,
            )

            midpoint = (
                int(
                    (
                        position_1[0]
                        + position_2[0]
                    )
                    / 2
                ),
                int(
                    (
                        position_1[1]
                        + position_2[1]
                    )
                    / 2
                ),
            )

            cv2.circle(
                frame,
                midpoint,
                7,
                color,
                -1,
            )

        # ----------------------------------------------------
        # Header
        # ----------------------------------------------------

        draw_header(
            frame,
            frame_number,
            highest_risk,
            highest_score,
            len(
                live_alert_pipeline.alerts
            ),
        )

        # ----------------------------------------------------
        # Latest confirmed live alert
        # ----------------------------------------------------

        draw_live_alert_status(
            frame,
            latest_live_alert,
        )

        # ----------------------------------------------------
        # Risk panel
        # ----------------------------------------------------

        draw_risk_panel(
            frame,
            risk_pairs,
        )

        # ----------------------------------------------------
        # Controls
        # ----------------------------------------------------

        cv2.putText(
            frame,
            "Press Q to stop",
            (
                20,
                frame.shape[0] - 20,
            ),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.60 * UI_SCALE,
            (210, 210, 210),
            1,
            cv2.LINE_AA,
        )

        # ----------------------------------------------------
        # Resize ONLY for display
        # ----------------------------------------------------

        display_frame = resize_for_display(
            frame
        )

        # ----------------------------------------------------
        # Display
        # ----------------------------------------------------

        cv2.imshow(
            WINDOW_NAME,
            display_frame,
        )

        key = cv2.waitKey(
            1
        ) & 0xFF

        if key in (
            ord("q"),
            ord("Q"),
        ):

            print("")
            print(
                "Monitor stopped by user."
            )

            break

    # --------------------------------------------------------
    # Finalize live alert pipeline
    # --------------------------------------------------------

    final_alerts = (
        live_alert_pipeline.finalize()
    )

    if final_alerts:

        latest_live_alert = (
            live_alert_pipeline.alerts[-1]
        )

        print_live_alert(
            latest_live_alert
        )

    # --------------------------------------------------------
    # Save live alerts
    # --------------------------------------------------------

    live_alert_output = (
        live_alert_pipeline.save(
            source_video=VIDEO_SOURCE
        )
    )

    # --------------------------------------------------------
    # Cleanup
    # --------------------------------------------------------

    cap.release()

    cv2.destroyAllWindows()

    statistics = (
        live_alert_pipeline.statistics()
    )

    print("")
    print("=" * 70)
    print(
        "REAL-TIME SAFETY MONITOR COMPLETE"
    )
    print(
        f"Frames processed: {frame_number}"
    )
    print(
        f"Confirmed live alerts: "
        f"{statistics['total_alerts']}"
    )
    print(
        f"Immediate alerts: "
        f"{statistics['alert_counts']['IMMEDIATE_ALERT']}"
    )
    print(
        f"Priority alerts: "
        f"{statistics['alert_counts']['PRIORITY_ALERT']}"
    )
    print(
        f"Monitor alerts: "
        f"{statistics['alert_counts']['MONITOR']}"
    )
    print(
        f"Live alert output: "
        f"{live_alert_output}"
    )
    print("=" * 70)


if __name__ == "__main__":
    main()
