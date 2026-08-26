import cv2
import math
from ultralytics import YOLO

# -----------------------------
# SETTINGS
# -----------------------------

MODEL_PATH = "yolo11n.pt"
VIDEO_PATH = "data/tracking_test.mp4"

# Distance below which two objects may be considered close
DANGER_DISTANCE = 180

# -----------------------------
# LOAD MODEL AND VIDEO
# -----------------------------

model = YOLO(MODEL_PATH)

video = cv2.VideoCapture(VIDEO_PATH)

if not video.isOpened():
    print("Could not open the video.")
    exit()

fps = video.get(cv2.CAP_PROP_FPS)

print(f"Video FPS: {fps}")

# -----------------------------
# TRACKING DATA
# -----------------------------

previous_positions = {}
movement_history = {}
previous_distances = {}

frame_number = 0

# -----------------------------
# MAIN LOOP
# -----------------------------

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

    # -----------------------------
    # GET TRACKED OBJECTS
    # -----------------------------

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

            # -----------------------------
            # VELOCITY
            # -----------------------------

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

                # Keep latest 5 measurements
                if len(movement_history[track_id]) > 5:
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

    # -----------------------------
    # PAIRWISE DISTANCE ANALYSIS
    # -----------------------------

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

            distance = math.sqrt(
                (x1 - x2) ** 2 +
                (y1 - y2) ** 2
            )

            # Make a consistent pair ID
            pair = tuple(
                sorted([id1, id2])
            )

            approaching = False

            # Compare with previous frame
            if pair in previous_distances:

                previous_distance = previous_distances[pair]

                if distance < previous_distance:
                    approaching = True

            previous_distances[pair] = distance

            # -----------------------------
            # PRINT DISTANCE
            # -----------------------------

            print(
                f"Frame: {frame_number} | "
                f"Distance | "
                f"ID {id1} ({class1}) <-> "
                f"ID {id2} ({class2}) = "
                f"{distance:.2f} px"
            )

            # -----------------------------
            # RISK DETECTION
            # -----------------------------

            if approaching and distance < DANGER_DISTANCE:

                print(
                    f"⚠️ POTENTIAL RISK | "
                    f"ID {id1} ({class1}) <-> "
                    f"ID {id2} ({class2}) | "
                    f"Distance: {distance:.2f} px | "
                    f"Approaching: YES"
                )

# -----------------------------
# CLEAN UP
# -----------------------------

video.release()

print("\nAnalysis complete.")