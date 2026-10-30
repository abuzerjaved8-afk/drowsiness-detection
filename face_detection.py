"""
Live face detection from the laptop webcam using OpenCV's YuNet DNN face
detector (cv2.FaceDetectorYN), with per-face confidence percentage and an
on-screen FPS counter.

Run:
    python face_detection.py
Press 'q' to quit.
"""

import argparse
import os
import time
import urllib.request

import cv2

MODEL_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "models")
MODEL_PATH = os.path.join(MODEL_DIR, "face_detection_yunet_2023mar.onnx")
MODEL_URL = (
    "https://github.com/opencv/opencv_zoo/raw/main/models/"
    "face_detection_yunet/face_detection_yunet_2023mar.onnx"
)


def download(url, dest):
    print(f"Downloading {os.path.basename(dest)} ...")
    urllib.request.urlretrieve(url, dest)


def load_detector(input_size, conf_threshold):
    os.makedirs(MODEL_DIR, exist_ok=True)
    if not os.path.exists(MODEL_PATH):
        download(MODEL_URL, MODEL_PATH)
    return cv2.FaceDetectorYN_create(
        MODEL_PATH, "", input_size, score_threshold=conf_threshold
    )


def detect(detector, frame):
    _, faces = detector.detect(frame)
    boxes = []
    if faces is None:
        return boxes
    for face in faces:
        x, y, w, h = face[:4].astype(int)
        score = float(face[14])
        boxes.append((x, y, x + w, y + h, score * 100.0))
    return boxes


def main():
    parser = argparse.ArgumentParser(description="Live webcam face detection")
    parser.add_argument("--camera", type=int, default=0, help="Camera index")
    parser.add_argument("--width", type=int, default=640, help="Capture width")
    parser.add_argument("--height", type=int, default=480, help="Capture height")
    parser.add_argument("--conf", type=float, default=0.5, help="Confidence threshold (0-1)")
    parser.add_argument("--max-fps", type=float, default=15.0, help="Cap displayed FPS")
    args = parser.parse_args()

    cap = cv2.VideoCapture(args.camera, cv2.CAP_DSHOW)
    if not cap.isOpened():
        cap = cv2.VideoCapture(args.camera)
    if not cap.isOpened():
        raise RuntimeError("Could not open the webcam. Check that it is connected and not in use.")

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, args.width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, args.height)
    cap.set(cv2.CAP_PROP_FPS, 30)

    actual_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)) or args.width
    actual_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) or args.height
    detector = load_detector((actual_width, actual_height), args.conf)

    min_frame_time = 1.0 / args.max_fps if args.max_fps > 0 else 0.0
    fps = 0.0
    prev_time = time.time()
    backend = "YuNet DNN"
    print(f"Using {backend} detector. Press 'q' in the video window to quit.")

    try:
        while True:
            loop_start = time.time()
            ok, frame = cap.read()
            if not ok:
                print("Failed to read frame from camera.")
                break

            boxes = detect(detector, frame)

            for (x1, y1, x2, y2, pct) in boxes:
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                label = f"{pct:.1f}%"
                label_y = y1 - 10 if y1 - 10 > 10 else y1 + 20
                cv2.putText(frame, label, (x1, label_y), cv2.FONT_HERSHEY_SIMPLEX,
                            0.6, (0, 255, 0), 2)

            now = time.time()
            instant_fps = 1.0 / (now - prev_time) if now > prev_time else 0.0
            fps = fps * 0.9 + instant_fps * 0.1
            prev_time = now

            cv2.putText(frame, f"FPS: {fps:.1f}  Faces: {len(boxes)}  [{backend}]",
                        (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)

            cv2.imshow("Face Detection - Live", frame)

            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

            elapsed = time.time() - loop_start
            if elapsed < min_frame_time:
                time.sleep(min_frame_time - elapsed)
    finally:
        cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
