import cv2
from ultralytics import YOLO

model = YOLO("yolo11n.pt")

image = cv2.imread("data/road.jpg")

results = model(image)
result = results[0]

for box in result.boxes:
    class_id = int(box.cls[0])
    confidence = float(box.conf[0])
    x1, y1, x2, y2 = map(int, box.xyxy[0])

    class_name = result.names[class_id]

    # Draw bounding box
    cv2.rectangle(
        image,
        (x1, y1),
        (x2, y2),
        (0, 255, 0),
        2
    )

    # Create label
    label = f"{class_name} {confidence:.2f}"

    # Draw label
    cv2.putText(
        image,
        label,
        (x1, y1 - 10),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        (0, 255, 0),
        2
    )

cv2.imshow("Road Safety Detection", image)
cv2.waitKey(0)
cv2.destroyAllWindows()