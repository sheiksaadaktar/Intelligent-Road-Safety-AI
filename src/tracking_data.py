import cv2
from ultralytics import YOLO

model = YOLO("yolo11n.pt")

video = cv2.VideoCapture("data/tracking_test.mp4")

if not video.isOpened():
    print("Could not open the video.")
    exit()

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

    if result.boxes.id is not None:
        tracking_ids = result.boxes.id.int().cpu().tolist()

        for box, track_id in zip(result.boxes, tracking_ids):
            class_id = int(box.cls[0])
            class_name = result.names[class_id]

            x1, y1, x2, y2 = map(int, box.xyxy[0])

            center_x = (x1 + x2) / 2
            center_y = (y1 + y2) / 2

            print(
                f"Frame: {frame_number} | "
                f"ID: {track_id} | "
                f"Object: {class_name} | "
                f"Center: ({center_x:.1f}, {center_y:.1f})"
            )

video.release()