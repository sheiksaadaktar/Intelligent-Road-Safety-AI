import cv2
import os
import numpy as np

from collections import defaultdict, deque

from ultralytics import YOLO

from risk_analysis import (
    MODEL_PATH,
    VIDEO_PATH,
    HISTORY_SIZE,
    MAX_REASONABLE_VELOCITY,
    MAX_REASONABLE_RELATIVE_VELOCITY,
    bottom_center,
    smooth_velocity,
    calculate_trajectory_metrics,
    calculate_risk,
)


# ============================================================
# CONFIGURATION
# ============================================================

OUTPUT_PATH = "output/risk_analysis.mp4"

MAX_DISPLAY_RISK_PAIRS = 6


# ============================================================
# RISK DISPLAY SETTINGS
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
# HELPER FUNCTIONS
# ============================================================

def format_tca(tca):

    if np.isinf(tca):
        return "inf"

    return f"{tca:.2f}s"


def draw_risk_panel(
    frame,
    risk_pairs,
    highest_risk,
    highest_score,
):

    height, width = frame.shape[:2]

    panel_x = width - 600
    panel_y = 25
    panel_width = 570

    header_height = 65
    entry_height = 108

    visible_pairs = risk_pairs[
        :MAX_DISPLAY_RISK_PAIRS
    ]

    panel_height = (
        header_height
        + max(1, len(visible_pairs))
        * entry_height
        + 20
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
        (20, 20, 20),
        -1,
    )

    # Semi-transparent background.
    cv2.addWeighted(
        overlay,
        0.78,
        frame,
        0.22,
        0,
        frame,
    )

    # --------------------------------------------------------
    # Panel header
    # --------------------------------------------------------

    cv2.putText(
        frame,
        "V5.2.3 RISK MONITOR",
        (
            panel_x + 20,
            panel_y + 28,
        ),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.78,
        (255, 255, 255),
        2,
    )

    highest_color = RISK_COLORS[
        highest_risk
    ]

    cv2.putText(
        frame,
        (
            f"Highest: {highest_risk} "
            f"  Score: {highest_score}"
        ),
        (
            panel_x + 20,
            panel_y + 55,
        ),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.62,
        highest_color,
        2,
    )

    # --------------------------------------------------------
    # Risk entries
    # --------------------------------------------------------

    for index, item in enumerate(
        visible_pairs
    ):

        y = (
            panel_y
            + header_height
            + index * entry_height
        )

        risk = item["risk"]
        color = RISK_COLORS[risk]

        id1 = item["id1"]
        id2 = item["id2"]
        metrics = item["metrics"]
        score = item["score"]

        # Separator
        cv2.line(
            frame,
            (
                panel_x + 15,
                y,
            ),
            (
                panel_x + panel_width - 15,
                y,
            ),
            (100, 100, 100),
            1,
        )

        # Risk + IDs
        cv2.putText(
            frame,
            (
                f"{risk}   "
                f"ID{id1} <-> ID{id2}"
            ),
            (
                panel_x + 20,
                y + 25,
            ),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            color,
            2,
        )

        # Current / predicted distance
        cv2.putText(
            frame,
            (
                f"D: {metrics['distance']:.1f} px"
                f"    "
                f"Pred: "
                f"{metrics['predicted_distance']:.1f} px"
            ),
            (
                panel_x + 20,
                y + 49,
            ),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.52,
            (255, 255, 255),
            1,
        )

        # Approach / TCA
        cv2.putText(
            frame,
            (
                f"Approach: "
                f"{metrics['approach_speed']:.1f} px/s"
                f"    "
                f"TCA: "
                f"{format_tca(metrics['tca'])}"
            ),
            (
                panel_x + 20,
                y + 71,
            ),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.52,
            (255, 255, 255),
            1,
        )

        # Convergence / score
        cv2.putText(
            frame,
            (
                f"Convergence: "
                f"{metrics['convergence']:.2f}"
                f"    "
                f"Score: {score}"
            ),
            (
                panel_x + 20,
                y + 93,
            ),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.52,
            (255, 255, 255),
            1,
        )


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 70)
    print(
        "INTELLIGENT ROAD SAFETY AI - "
        "RISK VISUALIZATION V5.2.3"
    )
    print(
        "Clean Risk-Annotation Rendering"
    )
    print("=" * 70)

    # --------------------------------------------------------
    # Load model
    # --------------------------------------------------------

    model = YOLO(MODEL_PATH)

    # --------------------------------------------------------
    # Open video
    # --------------------------------------------------------

    cap = cv2.VideoCapture(VIDEO_PATH)

    if not cap.isOpened():

        print(
            f"ERROR: Could not open video: "
            f"{VIDEO_PATH}"
        )

        return

    fps = cap.get(
        cv2.CAP_PROP_FPS
    )

    if fps <= 0:
        fps = 30.0

    width = int(
        cap.get(
            cv2.CAP_PROP_FRAME_WIDTH
        )
    )

    height = int(
        cap.get(
            cv2.CAP_PROP_FRAME_HEIGHT
        )
    )

    print(
        f"Video FPS: {fps}"
    )

    print(
        f"Video resolution: "
        f"{width} x {height}"
    )

    # --------------------------------------------------------
    # Output directory
    # --------------------------------------------------------

    os.makedirs(
        os.path.dirname(OUTPUT_PATH),
        exist_ok=True,
    )

    # --------------------------------------------------------
    # Video writer
    # --------------------------------------------------------

    fourcc = cv2.VideoWriter_fourcc(
        *"mp4v"
    )

    writer = cv2.VideoWriter(
        OUTPUT_PATH,
        fourcc,
        fps,
        (width, height),
    )

    if not writer.isOpened():

        print(
            "ERROR: Could not create output video."
        )

        cap.release()

        return

    # ========================================================
    # TRACKING HISTORY
    # ========================================================

    previous_positions = {}

    velocity_history = defaultdict(
        lambda: deque(
            maxlen=HISTORY_SIZE
        )
    )

    frame_number = 0

    # ========================================================
    # FRAME LOOP
    # ========================================================

    while True:

        ret, frame = cap.read()

        if not ret:
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

        highest_risk = "LOW"
        highest_score = 0

        risk_pairs = []

        # ----------------------------------------------------
        # Handle missing tracking result
        # ----------------------------------------------------

        if not results:

            cv2.putText(
                frame,
                f"Frame: {frame_number}",
                (30, 45),
                cv2.FONT_HERSHEY_SIMPLEX,
                1.0,
                (255, 255, 255),
                2,
            )

            cv2.putText(
                frame,
                "V5.2.3 TRAJECTORY RISK",
                (30, 85),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.85,
                (255, 255, 255),
                2,
            )

            writer.write(frame)

            continue

        result = results[0]

        # ----------------------------------------------------
        # No tracked objects
        # ----------------------------------------------------

        if (
            result.boxes is None
            or result.boxes.id is None
        ):

            cv2.putText(
                frame,
                f"Frame: {frame_number}",
                (30, 45),
                cv2.FONT_HERSHEY_SIMPLEX,
                1.0,
                (255, 255, 255),
                2,
            )

            cv2.putText(
                frame,
                (
                    "V5.2.3 TRAJECTORY RISK | "
                    "Highest: LOW Score: 0"
                ),
                (30, 85),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (255, 255, 255),
                2,
            )

            writer.write(frame)

            continue

        # ----------------------------------------------------
        # Extract tracking data
        # ----------------------------------------------------

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

        objects = {}

        # ====================================================
        # UPDATE POSITIONS / VELOCITIES
        # ====================================================

        for box, track_id, class_id in zip(
            boxes,
            tracking_ids,
            class_ids,
        ):

            position = bottom_center(
                box
            )

            velocity = np.zeros(
                2,
                dtype=np.float32,
            )

            if track_id in previous_positions:

                velocity = (
                    position
                    - previous_positions[
                        track_id
                    ]
                ) * fps

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

            class_name = result.names[
                class_id
            ]

            objects[
                track_id
            ] = {
                "box": box,
                "position": position,
                "class": class_name,
            }

        # ====================================================
        # DRAW OBJECT BOXES
        # ====================================================

        for track_id, obj in objects.items():

            box = obj["box"]

            x1, y1, x2, y2 = map(
                int,
                box,
            )

            cv2.rectangle(
                frame,
                (x1, y1),
                (x2, y2),
                (255, 255, 255),
                2,
            )

            label = (
                f"ID {track_id} "
                f"{obj['class']}"
            )

            cv2.putText(
                frame,
                label,
                (
                    x1,
                    max(
                        25,
                        y1 - 8,
                    ),
                ),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.65,
                (255, 255, 255),
                2,
            )

        # ====================================================
        # PAIRWISE TRAJECTORY RISK
        # ====================================================

        current_positions = {
            track_id: obj["position"]
            for track_id, obj
            in objects.items()
        }

        ids = sorted(
            current_positions.keys()
        )

        for index, id1 in enumerate(ids):

            for id2 in ids[index + 1:]:

                # ------------------------------------------------
                # Require velocity histories
                # ------------------------------------------------

                if (
                    len(
                        velocity_history[id1]
                    ) == 0
                    or
                    len(
                        velocity_history[id2]
                    ) == 0
                ):

                    continue

                position_1 = (
                    current_positions[id1]
                )

                position_2 = (
                    current_positions[id2]
                )

                velocity_1 = smooth_velocity(
                    velocity_history[id1]
                )

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

                relative_velocity_magnitude = float(
                    np.linalg.norm(
                        relative_velocity
                    )
                )

                if (
                    relative_velocity_magnitude
                    > MAX_REASONABLE_RELATIVE_VELOCITY
                ):

                    continue

                # ------------------------------------------------
                # Exact shared trajectory model
                # ------------------------------------------------

                metrics = (
                    calculate_trajectory_metrics(
                        position_1,
                        velocity_1,
                        position_2,
                        velocity_2,
                    )
                )

                risk, score = calculate_risk(
                    metrics
                )

                # ------------------------------------------------
                # Track highest risk
                # ------------------------------------------------

                if (
                    RISK_ORDER[risk]
                    >
                    RISK_ORDER[highest_risk]
                ):

                    highest_risk = risk
                    highest_score = score

                elif (
                    risk == highest_risk
                    and score > highest_score
                ):

                    highest_score = score

                # ------------------------------------------------
                # Keep non-LOW pairs for panel
                # ------------------------------------------------

                if risk != "LOW":

                    risk_pairs.append(
                        {
                            "risk": risk,
                            "score": score,
                            "id1": id1,
                            "id2": id2,
                            "metrics": metrics,
                            "position_1": position_1,
                            "position_2": position_2,
                        }
                    )

        # ====================================================
        # SORT RISK PAIRS
        # ====================================================

        risk_pairs.sort(
            key=lambda item: (
                RISK_ORDER[
                    item["risk"]
                ],
                item["score"],
            ),
            reverse=True,
        )

        # ====================================================
        # DRAW CONNECTION LINES
        # ====================================================

        for item in risk_pairs:

            risk = item["risk"]

            position_1 = item[
                "position_1"
            ]

            position_2 = item[
                "position_2"
            ]

            color = RISK_COLORS[
                risk
            ]

            center1 = (
                int(position_1[0]),
                int(position_1[1]),
            )

            center2 = (
                int(position_2[0]),
                int(position_2[1]),
            )

            cv2.line(
                frame,
                center1,
                center2,
                color,
                3,
            )

            # Small risk marker at midpoint.
            midpoint = (
                int(
                    (
                        position_1[0]
                        + position_2[0]
                    ) / 2
                ),
                int(
                    (
                        position_1[1]
                        + position_2[1]
                    ) / 2
                ),
            )

            cv2.circle(
                frame,
                midpoint,
                7,
                color,
                -1,
            )

        # ====================================================
        # RISK PANEL
        # ====================================================

        draw_risk_panel(
            frame,
            risk_pairs,
            highest_risk,
            highest_score,
        )

        # ====================================================
        # FRAME INFORMATION
        # ====================================================

        cv2.putText(
            frame,
            f"Frame: {frame_number}",
            (30, 45),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.0,
            (255, 255, 255),
            2,
        )

        cv2.putText(
            frame,
            (
                f"V5.2.3 TRAJECTORY RISK | "
                f"Highest: {highest_risk} "
                f"Score: {highest_score}"
            ),
            (30, 85),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (255, 255, 255),
            2,
        )

        # ====================================================
        # WRITE FRAME
        # ====================================================

        writer.write(frame)

        if frame_number % 50 == 0:

            print(
                f"Processed frame: "
                f"{frame_number}"
            )

    # ========================================================
    # CLEANUP
    # ========================================================

    cap.release()
    writer.release()

    print()
    print("=" * 70)
    print(
        "V5.2.3 RISK VISUALIZATION COMPLETE"
    )
    print(
        f"Output saved to: "
        f"{OUTPUT_PATH}"
    )
    print("=" * 70)


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":

    main()
