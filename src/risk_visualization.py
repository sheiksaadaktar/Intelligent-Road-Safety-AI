import cv2
import math
from ultralytics import YOLO

# ============================================================
# SETTINGS
# ============================================================

MODEL_PATH = "yolo11n.pt"
VIDEO_PATH = "data/tracking_test.mp4"

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

# Number of previous measurements used for smoothing
HISTORY_SIZE = 5

# Ignore extremely large one-frame closing-speed spikes
MAX_REASONABLE_CLOSING_SPEED = 1000


# ============================================================
# LOAD MODEL AND VIDEO
# ============================================================

model = YOLO(MODEL_PATH)

video = cv2.VideoCapture(VIDEO_PATH)

if not video.isOpened():
    print("Could not open the video.")
    exit()

fps = video.get(cv2.CAP_PROP_FPS)

print(f"Video FPS: {fps}")


# ============================================================
# TRACKING DATA
# ============================================================

previous_positions = {}

movement_history = {}

previous_distances = {}

closing_speed_history = {}


frame_number = 0


# ============================================================
# MAIN LOOP
# ============================================================

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

    result = results[0]

    objects = []


    # ========================================================
    # OBJECT DETECTION + VELOCITY
    # ========================================================

    if result.boxes.id is not None:

        tracking_ids = result.boxes.id.int().cpu().tolist()

        for box, track_id in zip(result.boxes, tracking_ids):

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


            # Store object information
            objects.append({
                "id": track_id,
                "class": class_name,
                "center": current_position
            })


            # =================================================
            # OBJECT VELOCITY
            # =================================================

            if track_id in previous_positions:

                previous_x, previous_y = previous_positions[track_id]

                movement_x = center_x - previous_x
                movement_y = center_y - previous_y

                movement = math.sqrt(
                    movement_x ** 2 +
                    movement_y ** 2
                )


                if track_id not in movement_history:
                    movement_history[track_id] = []


                movement_history[track_id].append(movement)


                if len(movement_history[track_id]) > HISTORY_SIZE:
                    movement_history[track_id].pop(0)


                smoothed_movement = (
                    sum(movement_history[track_id])
                    / len(movement_history[track_id])
                )


                velocity = smoothed_movement * fps


                print(
                    f"Frame: {frame_number} | "
                    f"ID: {track_id} | "
                    f"{class_name} | "
                    f"Velocity: {velocity:.2f} px/sec"
                )


            previous_positions[track_id] = current_position


    # ========================================================
    # PAIRWISE RISK ANALYSIS
    # ========================================================

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


            # Consistent pair identifier
            pair = tuple(
                sorted([id1, id2])
            )


            # ------------------------------------------------
            # CLOSING SPEED
            # ------------------------------------------------

            closing_speed = 0.0


            if pair in previous_distances:

                previous_distance = previous_distances[pair]

                distance_change = (
                    previous_distance - distance
                )

                closing_speed = distance_change * fps


            previous_distances[pair] = distance


            # ------------------------------------------------
            # FILTER EXTREME ONE-FRAME SPIKES
            # ------------------------------------------------

            if closing_speed < 0:

                closing_speed = 0


            if closing_speed > MAX_REASONABLE_CLOSING_SPEED:

                closing_speed = MAX_REASONABLE_CLOSING_SPEED


            # ------------------------------------------------
            # SMOOTH CLOSING SPEED
            # ------------------------------------------------

            if pair not in closing_speed_history:

                closing_speed_history[pair] = []


            closing_speed_history[pair].append(
                closing_speed
            )


            if len(closing_speed_history[pair]) > HISTORY_SIZE:

                closing_speed_history[pair].pop(0)


            smoothed_closing_speed = (
                sum(closing_speed_history[pair])
                / len(closing_speed_history[pair])
            )


            # ------------------------------------------------
            # TIME TO COLLISION
            # ------------------------------------------------

            if smoothed_closing_speed > 0:

                ttc = (
                    distance /
                    smoothed_closing_speed
                )

            else:

                ttc = float("inf")


            # =================================================
            # RISK SCORE
            # =================================================

            score = 0


            # Distance contribution
            if distance <= CRITICAL_DISTANCE:

                score += 40

            elif distance <= HIGH_DISTANCE:

                score += 30

            elif distance <= MEDIUM_DISTANCE:

                score += 20


            # Closing speed contribution
            if smoothed_closing_speed >= CRITICAL_CLOSING_SPEED:

                score += 30

            elif smoothed_closing_speed >= HIGH_CLOSING_SPEED:

                score += 20

            elif smoothed_closing_speed > 0:

                score += 10


            # TTC contribution
            if ttc <= CRITICAL_TTC:

                score += 30

            elif ttc <= HIGH_TTC:

                score += 20

            elif ttc <= MEDIUM_TTC:

                score += 10


            # =================================================
            # RISK LEVEL
            # =================================================

            if score >= 80:

                risk = "CRITICAL"

            elif score >= 60:

                risk = "HIGH"

            elif score >= 30:

                risk = "MEDIUM"

            else:

                risk = "LOW"


            # =================================================
            # OUTPUT
            # =================================================

            if risk != "LOW":

                print(
                    f"Frame: {frame_number} | "
                    f"{risk} RISK | "
                    f"ID {id1} ({class1}) <-> "
                    f"ID {id2} ({class2}) | "
                    f"Distance: {distance:.2f}px | "
                    f"Closing: {smoothed_closing_speed:.2f}px/s | "
                    f"TTC: {ttc:.2f}s | "
                    f"Score: {score}"
                )


    # ========================================================
    # PROGRESS
    # ========================================================

    if frame_number % 50 == 0:

        print(
            f"Processed frame: {frame_number}"
        )


# ============================================================
# CLEAN UP
# ============================================================

video.release()

print()
print("===================================")
print("Risk analysis complete.")
print("===================================")