import cv2
from ultralytics import YOLO

model = YOLO("yolo11n.pt")

video = cv2.VideoCapture("data/road.mp4")

if not video.isOpened():
    print("Could not open the video.")
    exit()

while True:
    success, frame = video.read()

    if not success:
        break

    results = model(frame)
    result = results[0]

    counts = {
        "car": 0,
        "person": 0,
        "motorcycle": 0,
        "bus": 0,
        "truck": 0
    }

    for box in result.boxes:
        class_id = int(box.cls[0])
        class_name = result.names[class_id]

        if class_name in counts:
            counts[class_name] += 1

    print(
        f"Cars: {counts['car']} | "
        f"People: {counts['person']} | "
        f"Motorcycles: {counts['motorcycle']} | "
        f"Buses: {counts['bus']} | "
        f"Trucks: {counts['truck']}"
    )

    annotated_frame = result.plot()
    cv2.imshow("Object Counter", annotated_frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

video.release()
cv2.destroyAllWindows()