import cv2

image = cv2.imread("data/road.jpg")

if image is None:
    print("Could not load the image.")
else:
    print("Image loaded successfully!")
    print("Image dimensions:", image.shape)

    # Draw a rectangle
    cv2.rectangle(image, (100, 100), (300, 250), (0, 255, 0), 3)

    # Draw a circle
    cv2.circle(image, (500, 200), 50, (255, 0, 0), 3)

    # Add text
    cv2.putText(
        image,
        "Road Safety AI",
        (100, 80),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        (0, 0, 255),
        2
    )

    cv2.imshow("Road Safety Image", image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()