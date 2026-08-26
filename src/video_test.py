import cv2

video = cv2.VideoCapture("data/road.mp4")

if not video.isOpened():
    print("Could not open the video.")
else:
    print("Video opened successfully!")

    frame_count = 0

    while True:
        success, frame = video.read()

        if not success:
            break

        frame_count += 1

    fps = video.get(cv2.CAP_PROP_FPS)

    print("Total frames:", frame_count)
    print("FPS:", fps)

    video.release()