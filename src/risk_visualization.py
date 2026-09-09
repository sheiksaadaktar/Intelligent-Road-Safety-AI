import cv2
import math
import os
from ultralytics import YOLO


# ============================================================
# SETTINGS
# ============================================================

MODEL_PATH = "yolo11n.pt"
VIDEO_PATH = "data/tracking_test.mp4"
OUTPUT_PATH = "output/risk_analysis.mp4"

# Distance thresholds in pixels
CRITICAL_DISTANCE = 140
HIGH_DISTANCE = 200
MEDIUM_DISTANCE = 280

# Closing speed thresholds in pixels/sec
HIGH_CLOSING_SPEED = 100
CRITICAL_CLOSING_SPEED = 180

# TTC thresholds in seconds
CRITICAL_TTC = 0.8
HIGH_TTC = 1.5
MEDIUM_TTC = 3.0

# History used for smoothing
HISTORY_SIZE = 5

# Ignore unrealistic one-frame spikes
MAX_REASONABLE_CLOSING_SPEED = 1000

# Minimum sustained closing speed before risk is calculated
MIN_CLOSING_SPEED_FOR_RISK = 20


# ============================================================
# LOAD MODEL
# ============================================================

model = YOLO(MODEL_PATH)


# ============================================================
# OPEN VIDEO
# ============================================================

video = cv2.VideoCapture(VIDEO_PATH)

if not video.isOpened():
    print("ERROR: Could not open input video.")
    exit()

fps = video.get(cv2.CAP_PROP_FPS)

if fps <= 0:
    fps = 30.0

width = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))

print(f"Video FPS: {fps}")
print(f"Video resolution: {width} x {height}")


# ============================================================
# CREATE OUTPUT DIRECTORY
# ============================================================

os.makedirs("output", exist_ok=True)


# ============================================================
# VIDEO WRITER
# ============================================================

fourcc = cv2.VideoWriter_fourcc(*"mp4v")

writer = cv2.VideoWriter(
    OUTPUT_PATH,
    fourcc,
    fps,
    (width, height)
)

if not writer.isOpened():
    print("ERROR: Could not create output video.")
    video.release()
    exit()


# ============================================================
# TRACKING DATA
# ============================================================

previous_positions = {}

movement_history = {}

previous_distances = {}

distance_history = {}

closing_speed_history = {}


# ============================================================
# MAIN LOOP
# ============================================================

frame_number = 0


while True:

    success, frame = video.read()

    if not success:
        break

    frame_number += 1

    results = model.track(
        frame,
        persist=True,
        verbose=False
    )

    if not results:
        writer.write(frame)
        continue

    result = results[0]

    objects = []


    # ========================================================
    # DETECTION + TRACKING + BOXES
    # ========================================================

    if result.boxes is not None and result.boxes.id is not None:

        tracking_ids = (
            result.boxes.id
            .int()
            .cpu()
            .tolist()
        )

        for box, track_id in zip(
            result.boxes,
            tracking_ids
        ):

            class_id = int(box.cls[0])

            class_name = result.names[class_id]

            x1, y1, x2, y2 = map(
                int,
                box.xyxy[0]
            )

            center_x = (x1 + x2) / 2
            center_y = (y1 + y2) / 2

            current_position = (
                center_x,
                center_y
            )


            # ------------------------------------------------
            # VELOCITY
            # ------------------------------------------------

            velocity = 0.0

            if track_id in previous_positions:

                previous_x, previous_y = (
                    previous_positions[track_id]
                )

                movement_x = center_x - previous_x
                movement_y = center_y - previous_y

                movement = math.sqrt(
                    movement_x ** 2 +
                    movement_y ** 2
                )

                if movement > 100:
                    movement = 0.0

                if track_id not in movement_history:
                    movement_history[track_id] = []

                movement_history[track_id].append(
                    movement
                )

                if len(movement_history[track_id]) > HISTORY_SIZE:
                    movement_history[track_id].pop(0)

                smoothed_movement = (
                    sum(movement_history[track_id])
                    /
                    len(movement_history[track_id])
                )

                velocity = smoothed_movement * fps


            previous_positions[track_id] = current_position


            # ------------------------------------------------
            # SAVE OBJECT
            # ------------------------------------------------

            objects.append({
                "id": track_id,
                "class": class_name,
                "center": current_position,
                "box": (x1, y1, x2, y2),
                "velocity": velocity
            })


            # ------------------------------------------------
            # DRAW OBJECT BOX
            # ------------------------------------------------

            cv2.rectangle(
                frame,
                (x1, y1),
                (x2, y2),
                (255, 255, 255),
                2
            )


            # ------------------------------------------------
            # DRAW OBJECT INFORMATION
            # ------------------------------------------------

            object_text = (
                f"ID {track_id} | "
                f"{class_name} | "
                f"{velocity:.0f}px/s"
            )

            text_y = max(25, y1 - 10)

            cv2.putText(
                frame,
                object_text,
                (x1, text_y),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (255, 255, 255),
                2,
                cv2.LINE_AA
            )


    # ========================================================
    # PAIRWISE RISK ANALYSIS
    # ========================================================

    highest_risk = "LOW"

    highest_score = 0


    for i in range(len(objects)):

        for j in range(i + 1, len(objects)):

            object1 = objects[i]
            object2 = objects[j]

            id1 = object1["id"]
            id2 = object2["id"]

            class1 = object1["class"]
            class2 = object2["class"]

            x1, y1 = object1["center"]
            x2, y2 = object2["center"]


            # ------------------------------------------------
            # DISTANCE
            # ------------------------------------------------

            distance = math.sqrt(
                (x1 - x2) ** 2 +
                (y1 - y2) ** 2
            )


            pair = tuple(
                sorted([id1, id2])
            )


            # ------------------------------------------------
            # DISTANCE HISTORY
            # ------------------------------------------------

            if pair not in distance_history:
                distance_history[pair] = []

            distance_history[pair].append(
                distance
            )

            if len(distance_history[pair]) > HISTORY_SIZE:
                distance_history[pair].pop(0)


            # ------------------------------------------------
            # APPROACHING DETECTION
            # ------------------------------------------------

            approaching = False

            if len(distance_history[pair]) >= 3:

                recent_distances = (
                    distance_history[pair]
                )

                decreasing_count = 0

                for k in range(
                    1,
                    len(recent_distances)
                ):

                    if (
                        recent_distances[k]
                        <
                        recent_distances[k - 1]
                    ):
                        decreasing_count += 1

                if decreasing_count >= 2:
                    approaching = True


            # ------------------------------------------------
            # CLOSING SPEED
            # ------------------------------------------------

            closing_speed = 0.0

            if pair in previous_distances:

                previous_distance = (
                    previous_distances[pair]
                )

                distance_change = (
                    previous_distance -
                    distance
                )

                closing_speed = (
                    distance_change * fps
                )


            previous_distances[pair] = distance


            if closing_speed < 0:
                closing_speed = 0


            if (
                closing_speed
                >
                MAX_REASONABLE_CLOSING_SPEED
            ):
                closing_speed = (
                    MAX_REASONABLE_CLOSING_SPEED
                )


            # ------------------------------------------------
            # CLOSING SPEED HISTORY
            # ------------------------------------------------

            if pair not in closing_speed_history:
                closing_speed_history[pair] = []

            closing_speed_history[pair].append(
                closing_speed
            )

            if (
                len(closing_speed_history[pair])
                >
                HISTORY_SIZE
            ):
                closing_speed_history[pair].pop(0)


            smoothed_closing_speed = (
                sum(closing_speed_history[pair])
                /
                len(closing_speed_history[pair])
            )


            # ------------------------------------------------
            # TTC
            # ------------------------------------------------

            if (
                approaching
                and
                smoothed_closing_speed > 0
            ):

                ttc = (
                    distance /
                    smoothed_closing_speed
                )

            else:

                ttc = float("inf")


            # ------------------------------------------------
            # RISK SCORE
            # ------------------------------------------------

            score = 0

            if (
                approaching
                and
                smoothed_closing_speed
                >=
                MIN_CLOSING_SPEED_FOR_RISK
            ):

                # Distance score
                if distance <= CRITICAL_DISTANCE:

                    score += 40

                elif distance <= HIGH_DISTANCE:

                    score += 30

                elif distance <= MEDIUM_DISTANCE:

                    score += 20


                # Closing speed score
                if (
                    smoothed_closing_speed
                    >=
                    CRITICAL_CLOSING_SPEED
                ):

                    score += 30

                elif (
                    smoothed_closing_speed
                    >=
                    HIGH_CLOSING_SPEED
                ):

                    score += 20

                elif smoothed_closing_speed > 0:

                    score += 10


                # TTC score
                if ttc <= CRITICAL_TTC:

                    score += 30

                elif ttc <= HIGH_TTC:

                    score += 20

                elif ttc <= MEDIUM_TTC:

                    score += 10


            # ------------------------------------------------
            # RISK LEVEL
            # ------------------------------------------------

            if score >= 80:

                risk = "CRITICAL"

            elif score >= 60:

                risk = "HIGH"

            elif score >= 30:

                risk = "MEDIUM"

            else:

                risk = "LOW"


            # ------------------------------------------------
            # HIGHEST RISK IN CURRENT FRAME
            # ------------------------------------------------

            risk_order = {
                "LOW": 0,
                "MEDIUM": 1,
                "HIGH": 2,
                "CRITICAL": 3
            }

            if (
                risk_order[risk]
                >
                risk_order[highest_risk]
            ):

                highest_risk = risk
                highest_score = score


            # ------------------------------------------------
            # DRAW PAIR INFORMATION
            # ------------------------------------------------

            if risk != "LOW":

                midpoint_x = int(
                    (x1 + x2) / 2
                )

                midpoint_y = int(
                    (y1 + y2) / 2
                )


                # Draw connection line
                cv2.line(
                    frame,
                    (int(x1), int(y1)),
                    (int(x2), int(y2)),
                    (255, 255, 255),
                    3
                )


                if math.isinf(ttc):

                    ttc_text = "N/A"

                else:

                    ttc_text = f"{ttc:.2f}s"


                # Risk information
                risk_text = (
                    f"{risk} | "
                    f"Dist {distance:.0f}px | "
                    f"Close {smoothed_closing_speed:.0f}px/s | "
                    f"TTC {ttc_text} | "
                    f"Score {score}"
                )


                # Background rectangle
                text_size = cv2.getTextSize(
                    risk_text,
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    2
                )[0]

                text_width = text_size[0]
                text_height = text_size[1]


                background_x1 = max(
                    0,
                    midpoint_x - 10
                )

                background_y1 = max(
                    text_height + 10,
                    midpoint_y - 10
                )

                background_x2 = min(
                    width,
                    midpoint_x + text_width + 10
                )

                background_y2 = min(
                    height,
                    midpoint_y + 10
                )


                cv2.rectangle(
                    frame,
                    (
                        background_x1,
                        background_y1
                    ),
                    (
                        background_x2,
                        background_y2
                    ),
                    (0, 0, 0),
                    -1
                )


                # Risk text
                cv2.putText(
                    frame,
                    risk_text,
                    (
                        background_x1 + 5,
                        background_y2 - 5
                    ),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (255, 255, 255),
                    2,
                    cv2.LINE_AA
                )


                # Terminal output
                print(
                    f"Frame: {frame_number} | "
                    f"{risk} RISK | "
                    f"ID {id1} ({class1}) <-> "
                    f"ID {id2} ({class2}) | "
                    f"Distance: {distance:.2f}px | "
                    f"Approaching: {approaching} | "
                    f"Closing: "
                    f"{smoothed_closing_speed:.2f}px/s | "
                    f"TTC: {ttc_text} | "
                    f"Score: {score}"
                )


    # ========================================================
    # TOP WARNING BANNER
    # ========================================================

    if highest_risk != "LOW":

        warning_text = (
            f"ROAD SAFETY WARNING: "
            f"{highest_risk} RISK"
        )

        cv2.rectangle(
            frame,
            (0, 0),
            (width, 60),
            (0, 0, 0),
            -1
        )

        cv2.putText(
            frame,
            warning_text,
            (25, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.1,
            (255, 255, 255),
            3,
            cv2.LINE_AA
        )

        cv2.putText(
            frame,
            f"Risk Score: {highest_score}",
            (width - 300, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (255, 255, 255),
            2,
            cv2.LINE_AA
        )


    # ========================================================
    # FRAME COUNTER
    # ========================================================

    cv2.putText(
        frame,
        f"Frame: {frame_number}",
        (20, height - 20),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        (255, 255, 255),
        2,
        cv2.LINE_AA
    )


    # ========================================================
    # WRITE FRAME
    # ========================================================

    writer.write(frame)


    if frame_number % 50 == 0:

        print(
            f"Processed frame: {frame_number}"
        )


# ============================================================
# CLEAN UP
# ============================================================

video.release()
writer.release()

print()
print("===================================")
print("Risk visualization complete.")
print(f"Output saved to: {OUTPUT_PATH}")
print("===================================")
