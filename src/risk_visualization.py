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

# Risk thresholds in pixels
HIGH_RISK_DISTANCE = 100
MEDIUM_RISK_DISTANCE = 250

# Velocity thresholds in pixels/sec
HIGH_SPEED = 500
MEDIUM_SPEED = 250

# Number of movement measurements used for smoothing
SMOOTHING_WINDOW = 5


# ============================================================
# CREATE OUTPUT DIRECTORY
# ============================================================

os.makedirs("output", exist_ok=True)


# ============================================================
# LOAD MODEL
# ============================================================

model = YOLO(MODEL_PATH)


# ============================================================
# OPEN VIDEO
# ============================================================

video = cv2.VideoCapture(VIDEO_PATH)

if not video.isOpened():
    print("Could not open the video.")
    exit()

fps = video.get(cv2.CAP_PROP_FPS)

width = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))

print(f"Video FPS: {fps}")
print(f"Video resolution: {width} x {height}")


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
    print("Could not create output video.")
    video.release()
    exit()


# ============================================================
# TRACKING DATA
# ============================================================

previous_positions = {}
movement_history = {}
previous_distances = {}


# ============================================================
# MAIN LOOP
# ============================================================

frame_number = 0

while True:

    success, frame = video.read()

    if not success:
        break

    frame_number += 1

    # --------------------------------------------------------
    # YOLO TRACKING
    # --------------------------------------------------------

    results = model.track(
        frame,
        persist=True,
        verbose=False
    )

    result = results[0]

    objects = []


    # --------------------------------------------------------
    # PROCESS DETECTED OBJECTS
    # --------------------------------------------------------

    if result.boxes.id is not None:

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

                if track_id not in movement_history:
                    movement_history[track_id] = []

                movement_history[track_id].append(
                    movement
                )

                if (
                    len(movement_history[track_id])
                    > SMOOTHING_WINDOW
                ):
                    movement_history[track_id].pop(0)

                smoothed_movement = (
                    sum(movement_history[track_id])
                    / len(movement_history[track_id])
                )

                velocity = smoothed_movement * fps

            else:

                movement_history[track_id] = []


            previous_positions[track_id] = current_position


            # ------------------------------------------------
            # STORE OBJECT
            # ------------------------------------------------

            objects.append({
                "id": track_id,
                "class": class_name,
                "center": current_position,
                "bbox": (x1, y1, x2, y2),
                "velocity": velocity
            })


    # ========================================================
    # DRAW OBJECT INFORMATION
    # ========================================================

    for obj in objects:

        track_id = obj["id"]
        class_name = obj["class"]
        x1, y1, x2, y2 = obj["bbox"]
        velocity = obj["velocity"]

        # Draw bounding box
        cv2.rectangle(
            frame,
            (x1, y1),
            (x2, y2),
            (255, 255, 255),
            2
        )

        # Object label
        label = (
            f"ID {track_id} | "
            f"{class_name} | "
            f"{velocity:.0f} px/s"
        )

        cv2.putText(
            frame,
            label,
            (x1, max(y1 - 10, 20)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (255, 255, 255),
            2
        )


    # ========================================================
    # PAIRWISE RISK ANALYSIS
    # ========================================================

    risk_messages = []

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

            # Consistent pair ID
            pair = tuple(
                sorted([id1, id2])
            )

            approaching = False

            if pair in previous_distances:

                previous_distance = (
                    previous_distances[pair]
                )

                if distance < previous_distance:
                    approaching = True

            previous_distances[pair] = distance


            # ------------------------------------------------
            # RISK LEVEL
            # ------------------------------------------------

            risk = "LOW"

            if (
                approaching
                and distance <= HIGH_RISK_DISTANCE
            ):

                risk = "HIGH"

            elif (
                approaching
                and distance <= MEDIUM_RISK_DISTANCE
            ):

                risk = "MEDIUM"


            # ------------------------------------------------
            # DRAW CONNECTION
            # ------------------------------------------------

            if distance <= MEDIUM_RISK_DISTANCE:

                center1 = (
                    int(x1),
                    int(y1)
                )

                center2 = (
                    int(x2),
                    int(y2)
                )

                cv2.line(
                    frame,
                    center1,
                    center2,
                    (255, 255, 255),
                    2
                )

                # Distance text
                mid_x = int((x1 + x2) / 2)
                mid_y = int((y1 + y2) / 2)

                distance_text = (
                    f"{distance:.0f}px"
                )

                cv2.putText(
                    frame,
                    distance_text,
                    (mid_x, mid_y),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (255, 255, 255),
                    2
                )


            # ------------------------------------------------
            # RISK MESSAGE
            # ------------------------------------------------

            if risk != "LOW":

                risk_messages.append(
                    f"{risk} RISK: "
                    f"ID {id1} <-> ID {id2} "
                    f"({distance:.0f}px)"
                )


    # ========================================================
    # DISPLAY RISK WARNINGS
    # ========================================================

    y_position = 35

    if len(risk_messages) > 0:

        for message in risk_messages:

            cv2.putText(
                frame,
                message,
                (20, y_position),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (255, 255, 255),
                2
            )

            y_position += 30


    # ========================================================
    # FRAME INFORMATION
    # ========================================================

    cv2.putText(
        frame,
        f"Frame: {frame_number}",
        (20, height - 20),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        (255, 255, 255),
        2
    )


    # ========================================================
    # WRITE FRAME
    # ========================================================

    writer.write(frame)


    # --------------------------------------------------------
    # PROGRESS
    # --------------------------------------------------------

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