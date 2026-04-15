import cv2
import mediapipe as mp
from mediapipe.tasks import python as mp_tasks
from mediapipe.tasks.python import vision
import pydirectinput
import time
import numpy as np
import os
import urllib.request

MODEL_URL = "https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_full/float16/1/pose_landmarker_full.task"
MODEL_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "pose_landmarker_full.task")

CALIBRATION_DURATION = 3
LEAN_THRESHOLD = 0.055
JUMP_THRESHOLD = 0.11
JUMP_CONFIRM_FRAMES = 3
SQUAT_SHOULDER_DROP = 0.10
SQUAT_CONFIRM_FRAMES = 3
CLAP_DISTANCE_THRESHOLD = 0.12
CLAP_COOLDOWN = 1.5

POSE_CONNECTIONS = [
    (0, 1), (1, 2), (2, 3), (3, 7),
    (0, 4), (4, 5), (5, 6), (6, 8),
    (9, 10),
    (11, 12),
    (11, 13), (13, 15),
    (12, 14), (14, 16),
    (11, 23), (12, 24),
    (23, 24),
    (23, 25), (25, 27),
    (24, 26), (26, 28),
    (15, 17), (15, 19), (15, 21),
    (16, 18), (16, 20), (16, 22),
    (27, 29), (29, 31),
    (28, 30), (30, 32),
]

WINDOW_NAME = "Subway Surfers Controller"

baseline_hip_y = None
baseline_shoulder_y = None
baseline_shoulder_tilt = 0.0
last_lean_dir = "center"
last_clap_time = 0.0
jump_confirm_count = 0
squat_confirm_count = 0
in_jump = False
in_squat = False
debug_values = {}


def download_model():
    if os.path.exists(MODEL_PATH):
        return
    print("Downloading pose model...")
    urllib.request.urlretrieve(MODEL_URL, MODEL_PATH)
    print("Done.")


def get_landmark(landmarks, idx):
    lm = landmarks[idx]
    return lm.x, lm.y


def process_frame(landmarker, frame_rgb, timestamp_ms):
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame_rgb)
    return landmarker.detect_for_video(mp_image, timestamp_ms)


def calibrate(cap, landmarker, start_timestamp_ms):
    global baseline_hip_y, baseline_shoulder_y, baseline_shoulder_tilt

    hip_samples = []
    shoulder_y_samples = []
    shoulder_tilt_samples = []
    start = time.time()
    ts_ms = start_timestamp_ms

    while time.time() - start < CALIBRATION_DURATION:
        ret, frame = cap.read()
        if not ret:
            continue

        frame = cv2.flip(frame, 1)
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        ts_ms += 33
        results = process_frame(landmarker, rgb, ts_ms)

        if results.pose_landmarks:
            lm = results.pose_landmarks[0]
            left_hip = get_landmark(lm, 23)
            right_hip = get_landmark(lm, 24)
            hip_samples.append((left_hip[1] + right_hip[1]) / 2)

            left_shoulder = get_landmark(lm, 11)
            right_shoulder = get_landmark(lm, 12)
            shoulder_y_samples.append((left_shoulder[1] + right_shoulder[1]) / 2)
            shoulder_tilt_samples.append(left_shoulder[1] - right_shoulder[1])

        remaining = CALIBRATION_DURATION - (time.time() - start)
        cv2.putText(frame, f"CALIBRATING... {remaining:.1f}s", (10, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
        cv2.putText(frame, "Stand still and face the camera", (10, 80),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

        cv2.imshow(WINDOW_NAME, frame)
        cv2.waitKey(1)

    baseline_hip_y = np.mean(hip_samples) if hip_samples else 0.6
    baseline_shoulder_y = np.mean(shoulder_y_samples) if shoulder_y_samples else 0.35
    baseline_shoulder_tilt = np.mean(shoulder_tilt_samples) if shoulder_tilt_samples else 0.0

    return ts_ms


def detect_lean(landmarks):
    global last_lean_dir

    tilt = (get_landmark(landmarks, 11)[1] - get_landmark(landmarks, 12)[1]) - baseline_shoulder_tilt
    debug_values["lean_tilt"] = tilt

    if tilt > LEAN_THRESHOLD:
        new_dir = "right"
    elif tilt < -LEAN_THRESHOLD:
        new_dir = "left"
    else:
        new_dir = "center"

    if new_dir != "center" and last_lean_dir == "center":
        pydirectinput.press(new_dir)

    last_lean_dir = new_dir
    return new_dir


def detect_jump(landmarks):
    global in_jump, jump_confirm_count

    hip_mid_y = (get_landmark(landmarks, 23)[1] + get_landmark(landmarks, 24)[1]) / 2
    rise = baseline_hip_y - hip_mid_y
    debug_values["jump_rise"] = rise

    if rise > JUMP_THRESHOLD:
        if not in_jump:
            jump_confirm_count += 1
            if jump_confirm_count >= JUMP_CONFIRM_FRAMES:
                pydirectinput.press("up")
                in_jump = True
                jump_confirm_count = 0
                return True
    else:
        jump_confirm_count = 0
        in_jump = False
    return False


def detect_squat(landmarks):
    global in_squat, squat_confirm_count

    shoulder_y = (get_landmark(landmarks, 11)[1] + get_landmark(landmarks, 12)[1]) / 2
    drop = shoulder_y - baseline_shoulder_y
    debug_values["squat_drop"] = drop

    if drop > SQUAT_SHOULDER_DROP:
        if not in_squat:
            squat_confirm_count += 1
            if squat_confirm_count >= SQUAT_CONFIRM_FRAMES:
                pydirectinput.press("down")
                in_squat = True
                squat_confirm_count = 0
                return True
    else:
        squat_confirm_count = 0
        in_squat = False
    return False


def detect_clap(landmarks):
    global last_clap_time

    now = time.time()
    if now - last_clap_time < CLAP_COOLDOWN:
        return False

    lw = get_landmark(landmarks, 15)
    rw = get_landmark(landmarks, 16)
    dist = ((lw[0] - rw[0]) ** 2 + (lw[1] - rw[1]) ** 2) ** 0.5

    if dist < CLAP_DISTANCE_THRESHOLD:
        pydirectinput.press("space")
        last_clap_time = now
        return True
    return False


def draw_skeleton(frame, landmarks):
    h, w, _ = frame.shape
    points = {}
    for idx, lm in enumerate(landmarks):
        px, py = int(lm.x * w), int(lm.y * h)
        points[idx] = (px, py)

    for a, b in POSE_CONNECTIONS:
        if a in points and b in points:
            cv2.line(frame, points[a], points[b], (80, 80, 200), 2)

    for pt in points.values():
        cv2.circle(frame, pt, 3, (50, 210, 50), -1)


def draw_overlay(frame, lean_dir, action, pose_landmarks):
    h, w, _ = frame.shape

    if lean_dir == "left":
        cv2.putText(frame, "LEAN LEFT", (10, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 230, 0), 2)
    elif lean_dir == "right":
        cv2.putText(frame, "LEAN RIGHT", (w - 250, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 230, 0), 2)

    if pose_landmarks:
        draw_skeleton(frame, pose_landmarks)

    cv2.rectangle(frame, (0, h - 70), (w, h), (0, 0, 0), -1)

    if action:
        cv2.putText(frame, action, (10, h - 38),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)

    lean_tilt = debug_values.get("lean_tilt", 0)
    jump_rise = debug_values.get("jump_rise", 0)
    squat_drop = debug_values.get("squat_drop", 0)
    debug_text = (f"Lean: {lean_tilt:+.3f}/{LEAN_THRESHOLD:.3f}  "
                  f"Jump: {jump_rise:.3f}/{JUMP_THRESHOLD:.2f}  "
                  f"Squat: {squat_drop:.3f}/{SQUAT_SHOULDER_DROP:.2f}")

    cv2.putText(frame, debug_text, (10, h - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.4, (180, 180, 180), 1)

    cv2.putText(frame, "Press 'q' to quit", (w - 180, h - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (200, 200, 200), 1)

    return frame


def main():
    pydirectinput.FAILSAFE = False
    download_model()

    options = vision.PoseLandmarkerOptions(
        base_options=mp_tasks.BaseOptions(model_asset_path=MODEL_PATH),
        running_mode=vision.RunningMode.VIDEO,
        min_pose_detection_confidence=0.5,
        min_tracking_confidence=0.5,
        num_poses=1,
    )

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: cannot open webcam.")
        return

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    with vision.PoseLandmarker.create_from_options(options) as landmarker:
        ts_ms = calibrate(cap, landmarker, 0)

        last_action = ""
        action_display_time = 0.0

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            frame = cv2.flip(frame, 1)
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            ts_ms += 33
            results = process_frame(landmarker, rgb, ts_ms)

            lean_dir = last_lean_dir
            action = ""
            detected_landmarks = None

            if results.pose_landmarks:
                lm = results.pose_landmarks[0]
                detected_landmarks = lm
                lean_dir = detect_lean(lm)

                if detect_jump(lm):
                    action = "JUMP"
                elif detect_squat(lm):
                    action = "DUCK"

            now = time.time()
            if action:
                last_action = action
                action_display_time = now
            elif now - action_display_time > 1.0:
                last_action = ""

            frame = draw_overlay(frame, lean_dir, last_action, detected_landmarks)
            cv2.imshow(WINDOW_NAME, frame)

            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
