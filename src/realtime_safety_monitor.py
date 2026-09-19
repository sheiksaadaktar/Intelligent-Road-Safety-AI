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

# Processing remains at the original video resolution.
# Only the displayed frame is resized.

DISPLAY_WIDTH = 1280
DISPLAY_HEIGHT = 720

# The frame is displayed at approximately 2/3 of the
# original 1920x1080 resolution, so UI text is enlarged
# before display to remain readable.
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
# ALERT COOLDOWN
# ============================================================

ALERT_COOLDOWN_FRAMES = 30


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
# CONSOLE ALERT
# ============================================================

def print_alert(
    frame_number,
    id1,
    id2,
    risk,
    score,
    metrics,
):

    tca_text = format_tca(
        metrics["tca"]
    )

    print("")
    print("!" * 70)
    print("ROAD SAFETY ALERT")
    print(f"Frame: {frame_number}")
    print(f"Objects: ID{id1} <-> ID{id2}")
    print(f"Risk: {risk}")
    print(f"Score: {score}")

    print(
        f"Distance: "
        f"{metrics['distance']:.2f} px"
    )

    print(
        f"Predicted distance: "
        f"{metrics['predicted_distance']:.2f} px"
    )

    print(
        f"Approach speed: "
        f"{metrics['approach_speed']:.2f} px/s"
    )

    print(
        f"TCA: {tca_text}"
    )

    print(
        f"Convergence: "
        f"{metrics['convergence']:.2f}"
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

    # Main title
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

    # Subtitle
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

    # Current status
    cv2.putText(
        frame,
        (
            f"Frame: {frame_number}    "
            f"Highest Risk: {highest_risk}    "
            f"Score: {highest_score}"
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

    # Keep panel inside frame.
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

    # Panel title
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

    # --------------------------------------------------------
    # Risk entries
    # --------------------------------------------------------

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

        # Separator
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

        # Risk + objects + score
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

        # Distance
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

        # Approach + TCA
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

        # Convergence
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

    # --------------------------------------------------------
    # Alert state
    # --------------------------------------------------------

    last_alert_frame = {}

    frame_number = 0

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
                # Keep risk pairs
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

                    # ------------------------------------------------
                    # Console alert
                    # ------------------------------------------------

                    if risk in (
                        "HIGH",
                        "CRITICAL",
                    ):

                        pair = tuple(
                            sorted(
                                (
                                    id1,
                                    id2,
                                )
                            )
                        )

                        previous_alert_frame = (
                            last_alert_frame.get(
                                pair,
                                -10**9,
                            )
                        )

                        if (
                            frame_number
                            - previous_alert_frame
                            >= ALERT_COOLDOWN_FRAMES
                        ):

                            print_alert(
                                frame_number,
                                id1,
                                id2,
                                risk,
                                score,
                                metrics,
                            )

                            last_alert_frame[
                                pair
                            ] = frame_number

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
    # Cleanup
    # --------------------------------------------------------

    cap.release()

    cv2.destroyAllWindows()

    print("")
    print("=" * 70)
    print(
        "REAL-TIME SAFETY MONITOR COMPLETE"
    )
    print(
        f"Frames processed: {frame_number}"
    )
    print("=" * 70)


if __name__ == "__main__":
    main()
