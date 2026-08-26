import cv2
from ultralytics import YOLO

model = YOLO("yolo11n.pt")

image = cv2.imread("data/road.jpg")

results = model(image)

result = results[0]

object_count = 0

for box in result.boxes:
    class_id = int(box.cls[0])
    confidence = float(box.conf[0])
    x1, y1, x2, y2 = map(int, box.xyxy[0])

    class_name = result.names[class_id]

    object_count += 1

    cv2.rectangle(
        image,
        (x1, y1),
        (x2, y2),
        (0, 255, 0),
        2
    )

    label = f"{class_name} {confidence:.2f}"

    cv2.putText(
        image,
        label,
        (x1, y1 - 10),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        (0, 255, 0),
        2
    )

print("Total objects detected:", object_count)

cv2.imshow("Multi-Object Detection", image)
cv2.waitKey(0)
cv2.destroyAllWindows()