import cv2
import math
from ultralytics import YOLO

model = YOLO("yolo11n.pt")

video = cv2.VideoCapture("data/tracking_test.mp4")

if not video.isOpened():
    print("Could not open the video.")
    exit()

fps = video.get(cv2.CAP_PROP_FPS)
print(f"Video FPS: {fps}")

DANGER_DISTANCE = 180
previous_positions = {}
movement_history = {}

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

    result = results[0]

    current_objects = {}

    if result.boxes.id is not None:

        tracking_ids = result.boxes.id.int().cpu().tolist()

        for box, track_id in zip(result.boxes, tracking_ids):

            class_id = int(box.cls[0])
            class_name = result.names[class_id]

            x1, y1, x2, y2 = map(int, box.xyxy[0])

            center_x = (x1 + x2) / 2
            center_y = (y1 + y2) / 2

            current_position = (center_x, center_y)

            # Store the current object's position
            current_objects[track_id] = {
                "class": class_name,
                "center": current_position
            }

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

    # -------------------------------------------------
    # Calculate distance between detected objects
    # -------------------------------------------------

    object_ids = list(current_objects.keys())

    for i in range(len(object_ids)):

        for j in range(i + 1, len(object_ids)):

            id1 = object_ids[i]
            id2 = object_ids[j]

            x1, y1 = current_objects[id1]["center"]
            x2, y2 = current_objects[id2]["center"]

            distance = math.sqrt(
                (x2 - x1) ** 2 +
                (y2 - y1) ** 2
            )

            class1 = current_objects[id1]["class"]
            class2 = current_objects[id2]["class"]

            print(
                f"Frame: {frame_number} | "
                f"Distance | "
                f"ID {id1} ({class1}) <-> "
                f"ID {id2} ({class2}) = "
                f"{distance:.2f} px"
            )

video.release()

print("Distance analysis complete.")