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

    annotated_frame = results[0].plot()

    cv2.imshow("Road Safety AI", annotated_frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

video.release()
cv2.destroyAllWindows()