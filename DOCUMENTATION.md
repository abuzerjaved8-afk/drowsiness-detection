# Project Documentation — Live Face Detection (Drowsiness Detection base)

This document explains what this project currently does, which model it uses,
where that model came from, how the code works end-to-end, and what would
need to be added to turn it into a full drowsiness-detection system.

## 1. What this project actually does right now

Despite the repository name, the code in `face_detection.py` currently
performs **real-time face detection only** — it finds faces in the webcam
feed, draws a box around each one, and shows a confidence percentage and the
live FPS. It does **not yet** detect drowsiness (closed eyes, yawning,
head nodding, etc.). Section 5 explains what would need to be added for
that, and which public datasets are normally used.

Think of this repo as the "camera + face-locator" foundation that a
drowsiness classifier would be built on top of.

## 2. The model: YuNet (`cv2.FaceDetectorYN`)

- **File used:** `models/face_detection_yunet_2023mar.onnx` (~232 KB)
- **What it is:** YuNet is a very small, fast, anchor-free face-detection
  neural network designed for real-time use on CPUs and edge devices. It is
  distributed as part of the official [OpenCV Zoo](https://github.com/opencv/opencv_zoo)
  (`models/face_detection_yunet`), and originates from the
  [libfacedetection](https://github.com/ShiqiYu/libfacedetection) project by
  Shiqi Yu's research group.
- **Dataset it was trained on:** [WIDER FACE](http://shuoyang1213.me/WIDERFACE/),
  a large public benchmark dataset of ~32,000 images and ~393,000 labeled
  faces collected from everyday scenes, with wide variation in scale, pose,
  occlusion, lighting and expression. This project does **not** train the
  model — it loads the already-trained `.onnx` weights published by the
  OpenCV Zoo maintainers.
- **Architecture, in brief:** a lightweight convolutional backbone with a
  small feature-pyramid-style neck and an anchor-free detection head. It
  predicts, for each detected face: a bounding box, 5 facial landmark points
  (right eye, left eye, nose tip, right mouth corner, left mouth corner),
  and a confidence score — all in a single forward pass, which is why it's
  fast enough to run at double-digit FPS on an ordinary laptop CPU.
- **Why this model was chosen over alternatives:**
  - OpenCV's older Haar-cascade face detector (`cv2.CascadeClassifier`) was
    tried first, but it has been **removed entirely** from the OpenCV 5.x
    Python bindings installed by this project's `requirements.txt` — it no
    longer exists in `cv2` at all.
  - The classic DNN "SSD + ResNet10 (Caffe)" face detector was tried next,
    but OpenCV 5.x also dropped its Caffe model importer
    (`cv2.dnn.readNetFromCaffe` no longer exists).
  - YuNet is the detector OpenCV 5.x ships first-class support for
    (`cv2.FaceDetectorYN_create`), it gives a genuine calibrated confidence
    score per face (unlike Haar), and it's small/fast enough to hit the
    10–15 FPS target on CPU-only laptops.
- **Model download:** the `.onnx` file is committed directly into this repo
  under `models/`, so cloning the repo is enough — no internet access is
  required at runtime to fetch it. If it's ever missing, `face_detection.py`
  will automatically re-download it from:
  `https://github.com/opencv/opencv_zoo/raw/main/models/face_detection_yunet/face_detection_yunet_2023mar.onnx`

## 3. Code walkthrough (`face_detection.py`)

1. **CLI arguments** (`argparse`): `--camera` (which webcam index to open),
   `--width` / `--height` (capture resolution), `--conf` (minimum confidence
   0–1 to accept a detection), `--max-fps` (throttle, default 15).
2. **`load_detector(input_size, conf_threshold)`**: ensures the YuNet
   `.onnx` file exists (downloading it if needed), then creates a
   `cv2.FaceDetectorYN` instance sized to the camera's actual capture
   resolution.
3. **Camera setup**: opens `cv2.VideoCapture` (using the DirectShow backend
   on Windows for faster start-up), and sets the requested width/height/FPS.
4. **Main loop**, once per frame:
   - Read a frame from the webcam.
   - `detect(detector, frame)` calls `detector.detect(frame)`, which returns
     an array of faces. Each row is `[x, y, w, h, <5 landmark x/y pairs>, score]`
     — the code takes the box (`x, y, w, h`) and the `score` (index 14),
     converts the score to a percentage, and returns a list of
     `(x1, y1, x2, y2, confidence_%)` boxes.
   - Each box is drawn as a green rectangle with its confidence percentage
     label above it.
   - **FPS** is computed per frame and smoothed with an exponential moving
     average (`fps = fps * 0.9 + instant_fps * 0.1`) so the on-screen number
     doesn't flicker.
   - The annotated frame is shown in a window titled
     `"Face Detection - Live"`.
   - If the loop is running faster than `--max-fps`, it sleeps the
     difference so the displayed frame rate stays near the target instead of
     needlessly maxing out the CPU.
   - Pressing `q` in the window breaks the loop; the camera is released and
     the window closed in a `finally` block either way.

## 4. Running the project

See the main `README`/chat history for full setup, but in short:

```
python -m pip install -r requirements.txt
python face_detection.py
```

Desktop launchers `Start Face Detection.bat` / `Stop Face Detection.bat` are
included for double-click start/stop without a terminal (they track the
process's PID in a local `.facedetection.pid` file so Stop can find and kill
the right process).

## 5. Turning this into actual drowsiness detection (not yet implemented)

Face detection only tells you *where* a face is — not whether its eyes are
closed or it's yawning. To build real drowsiness detection on top of this
foundation, the typical approach is:

1. **Get eye/mouth landmarks per frame.** YuNet already returns 5 coarse
   landmarks (including both eyes), but for drowsiness you generally want a
   denser landmark model (e.g. a 68-point face-landmark model, or
   MediaPipe FaceMesh) to compute the **Eye Aspect Ratio (EAR)** — a simple
   geometric ratio that drops sharply when eyes close.
2. **Classify eye state (open/closed) or compute EAR over time.** If EAR
   stays below a threshold for N consecutive frames, flag "eyes closed /
   drowsy." Alternatively, train a small CNN classifier directly on
   cropped eye images (open vs. closed).
3. **Public datasets commonly used for this step:**
   - **MRL Eye Dataset** — large set of labeled open/closed eye crops,
     good for training a simple open/closed-eye CNN classifier.
   - **NTHU Driver Drowsiness Detection (DDD) Dataset** — driver videos
     labeled with drowsy/alert states, including glasses/no-glasses and
     night/day conditions.
   - **UTA-RLDD (Real-Life Drowsiness Dataset)** — long-form videos of
     real people in alert, low-vigilant, and drowsy states, useful for
     sequence/temporal models rather than single-frame classifiers.
   - **YawDD** — driver videos labeled for yawning detection.
4. **Add temporal logic.** Drowsiness is a state over time (e.g. "eyes
   closed for >1.5 seconds" or "yawned 3 times in 2 minutes"), not a single
   frame, so the pipeline needs a rolling counter/timer, not just
   per-frame classification.
5. **Trigger an alert** (sound, on-screen warning, etc.) once the
   drowsiness condition is met.

None of step 1–5 is implemented yet in this repo — `face_detection.py`
currently stops at "detect and box the face with a confidence score."
