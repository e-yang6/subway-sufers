import cv2
import mediapipe as mp
from mediapipe.tasks import python as mp_tasks
from mediapipe.tasks.python import vision
import pydirectinput
import time
import numpy as np
import os
import urllib.request

# ─── Constants ───────────────────────────────────────────────────────────────

# Model
MODEL_URL = "https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_full/float16/1/pose_landmarker_full.task"
MODEL_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "pose_landmarker_full.task")

# Calibration
CALIBRATION_DURATION = 3  # seconds

# Lane detection
LANE_LEFT_BOUNDARY = 1 / 3
LANE_RIGHT_BOUNDARY = 2 / 3

# Jump detection
JUMP_THRESHOLD = 0.13  # normalized units above baseline (higher = less sensitive)
JUMP_COOLDOWN = 1.0  # seconds
JUMP_CONFIRM_FRAMES = 3  # consecutive frames above threshold before triggering

# Squat detection (based on shoulder drop from standing position)
SQUAT_SHOULDER_DROP = 0.07  # how far shoulders must drop (normalized) to count
SQUAT_COOLDOWN = 1.0
SQUAT_CONFIRM_FRAMES = 2  # consecutive frames before triggering

# Clap / hoverboard detection
CLAP_DISTANCE_THRESHOLD = 0.15  # normalized units between wrists
CLAP_COOLDOWN = 1.5

# Key press duration
KEY_DURATION = 0.08  # seconds to hold each key

# Standard MediaPipe 33-landmark pose connections for skeleton drawing
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

# ─── Globals ─────────────────────────────────────────────────────────────────

baseline_ankle_y = None  # average ankle Y when standing
baseline_shoulder_y = None  # average shoulder Y when standing
current_lane = "center"  # left | center | right
last_jump_time = 0
last_squat_time = 0
last_clap_time = 0
jump_confirm_count = 0  # consecutive frames with ankles above threshold
squat_confirm_count = 0  # consecutive frames with shoulders dropped

# Live debug values (shown on overlay)
debug_values = {}


# ─── Helper Functions ────────────────────────────────────────────────────────

def download_model():
    """Download the pose landmarker model if not already present."""
    if os.path.exists(MODEL_PATH):
        return
    print(f"[Setup] Downloading pose landmarker model...")
    urllib.request.urlretrieve(MODEL_URL, MODEL_PATH)
    print(f"[Setup] Model saved to {MODEL_PATH}")


def get_landmark(landmarks, idx):
    """Return (x, y) for a given landmark index (normalized 0-1)."""
    lm = landmarks[idx]
    return lm.x, lm.y


def midpoint(p1, p2):
    """Return the midpoint of two (x, y) tuples."""
    return ((p1[0] + p2[0]) / 2, (p1[1] + p2[1]) / 2)


def press_key(key):
    """Send a quick key press via pydirectinput."""
    pydirectinput.press(key)


def process_frame(landmarker, frame_rgb, timestamp_ms):
    """Run pose detection on an RGB frame and return the result."""
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame_rgb)
    return landmarker.detect_for_video(mp_image, timestamp_ms)


# ─── Calibration ─────────────────────────────────────────────────────────────

def calibrate(cap, landmarker, start_timestamp_ms):
    """Run a 3-second calibration. Player should stand still facing the camera."""
    global baseline_ankle_y, baseline_shoulder_y

    print("[Calibration] Stand still and face the camera...")
    ankle_samples = []
    shoulder_samples = []
    start = time.time()
    ts_ms = start_timestamp_ms

    while time.time() - start < CALIBRATION_DURATION:
        ret, frame = cap.read()
        if not ret:
            continue

        frame = cv2.flip(frame, 1)
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        ts_ms += 33  # ~30fps
        results = process_frame(landmarker, rgb, ts_ms)

        if results.pose_landmarks:
            lm = results.pose_landmarks[0]

            left_ankle = get_landmark(lm, 27)
            right_ankle = get_landmark(lm, 28)
            ankle_mid_y = (left_ankle[1] + right_ankle[1]) / 2
            ankle_samples.append(ankle_mid_y)

            left_shoulder = get_landmark(lm, 11)
            right_shoulder = get_landmark(lm, 12)
            shoulder_y = (left_shoulder[1] + right_shoulder[1]) / 2
            shoulder_samples.append(shoulder_y)

        # Show countdown
        remaining = CALIBRATION_DURATION - (time.time() - start)
        cv2.putText(frame, f"CALIBRATING... {remaining:.1f}s", (10, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
        cv2.putText(frame, "Stand still and face the camera", (10, 80),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        cv2.imshow("Subway Surfers Controller", frame)
        cv2.waitKey(1)

    if ankle_samples:
        baseline_ankle_y = np.mean(ankle_samples)
    else:
        baseline_ankle_y = 0.8

    if shoulder_samples:
        baseline_shoulder_y = np.mean(shoulder_samples)
    else:
        baseline_shoulder_y = 0.35

    print(f"[Calibration] Done. Ankle baseline: {baseline_ankle_y:.3f}, "
          f"Shoulder baseline: {baseline_shoulder_y:.3f}")

    return ts_ms


# ─── Detection Functions ─────────────────────────────────────────────────────

def detect_lane(landmarks, frame_width):
    """Detect which lane the player is in based on hip center X position."""
    global current_lane

    left_hip = get_landmark(landmarks, 23)
    right_hip = get_landmark(landmarks, 24)
    hip_center_x = (left_hip[0] + right_hip[0]) / 2

    if hip_center_x < LANE_LEFT_BOUNDARY:
        new_lane = "left"
    elif hip_center_x > LANE_RIGHT_BOUNDARY:
        new_lane = "right"
    else:
        new_lane = "center"

    if new_lane != current_lane:
        old_lane = current_lane
        current_lane = new_lane
        if new_lane == "left" and old_lane != "left":
            press_key("left")
            print("[Lane] -> LEFT")
        elif new_lane == "right" and old_lane != "right":
            press_key("right")
            print("[Lane] -> RIGHT")
        elif new_lane == "center":
            if old_lane == "left":
                press_key("right")
                print("[Lane] -> CENTER (from left)")
            elif old_lane == "right":
                press_key("left")
                print("[Lane] -> CENTER (from right)")

    return current_lane


def detect_jump(landmarks):
    """Detect a jump: ankles rise above calibrated baseline for several frames."""
    global last_jump_time, jump_confirm_count

    now = time.time()
    if now - last_jump_time < JUMP_COOLDOWN:
        jump_confirm_count = 0
        return False

    left_ankle_y = get_landmark(landmarks, 27)[1]
    right_ankle_y = get_landmark(landmarks, 28)[1]
    ankle_mid_y = (left_ankle_y + right_ankle_y) / 2

    if baseline_ankle_y - ankle_mid_y > JUMP_THRESHOLD:
        jump_confirm_count += 1
        if jump_confirm_count >= JUMP_CONFIRM_FRAMES:
            press_key("up")
            last_jump_time = now
            jump_confirm_count = 0
            print("[Action] JUMP")
            return True
    else:
        jump_confirm_count = 0
    return False


def detect_squat(landmarks):
    """Detect a squat: shoulders drop significantly from standing baseline."""
    global last_squat_time, squat_confirm_count

    now = time.time()
    if now - last_squat_time < SQUAT_COOLDOWN:
        squat_confirm_count = 0
        return False

    left_shoulder_y = get_landmark(landmarks, 11)[1]
    right_shoulder_y = get_landmark(landmarks, 12)[1]
    shoulder_y = (left_shoulder_y + right_shoulder_y) / 2

    # In normalized coords Y=0 is top, so squatting means shoulder Y increases
    drop = shoulder_y - baseline_shoulder_y
    debug_values["squat_drop"] = drop

    if drop > SQUAT_SHOULDER_DROP:
        squat_confirm_count += 1
        if squat_confirm_count >= SQUAT_CONFIRM_FRAMES:
            press_key("down")
            last_squat_time = now
            squat_confirm_count = 0
            print(f"[Action] DUCK / SQUAT (drop={drop:.3f})")
            return True
    else:
        squat_confirm_count = 0
    return False


def detect_clap(landmarks):
    """Detect a clap: both wrists close together."""
    global last_clap_time

    now = time.time()
    if now - last_clap_time < CLAP_COOLDOWN:
        return False

    left_wrist = get_landmark(landmarks, 15)
    right_wrist = get_landmark(landmarks, 16)

    dx = abs(left_wrist[0] - right_wrist[0])
    dy = abs(left_wrist[1] - right_wrist[1])
    distance = (dx ** 2 + dy ** 2) ** 0.5
    debug_values["wrist_dist"] = distance

    if distance < CLAP_DISTANCE_THRESHOLD:
        press_key("space")
        last_clap_time = now
        print(f"[Action] HOVERBOARD (wrist dist={distance:.3f})")
        return True
    return False


# ─── GUI Drawing ─────────────────────────────────────────────────────────────

def draw_pose_skeleton(frame, landmarks):
    """Draw pose landmarks and connections on the frame."""
    h, w, _ = frame.shape

    points = {}
    for idx, lm in enumerate(landmarks):
        px, py = int(lm.x * w), int(lm.y * h)
        points[idx] = (px, py)
        cv2.circle(frame, (px, py), 3, (0, 255, 0), -1)

    for start_idx, end_idx in POSE_CONNECTIONS:
        if start_idx in points and end_idx in points:
            cv2.line(frame, points[start_idx], points[end_idx], (0, 0, 255), 2)


def draw_overlay(frame, lane, action, pose_landmarks):
    """Draw lane lines, pose skeleton, and status info on the frame."""
    h, w, _ = frame.shape

    # Lane divider lines
    x1 = int(w * LANE_LEFT_BOUNDARY)
    x2 = int(w * LANE_RIGHT_BOUNDARY)
    cv2.line(frame, (x1, 0), (x1, h), (255, 255, 0), 2)
    cv2.line(frame, (x2, 0), (x2, h), (255, 255, 0), 2)

    # Highlight current lane
    overlay = frame.copy()
    if lane == "left":
        cv2.rectangle(overlay, (0, 0), (x1, h), (0, 255, 0), -1)
    elif lane == "right":
        cv2.rectangle(overlay, (x2, 0), (w, h), (0, 255, 0), -1)
    else:
        cv2.rectangle(overlay, (x1, 0), (x2, h), (0, 255, 0), -1)
    cv2.addWeighted(overlay, 0.15, frame, 0.85, 0, frame)

    # Lane labels
    cv2.putText(frame, "LEFT", (x1 // 2 - 30, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
    cv2.putText(frame, "CENTER", ((x1 + x2) // 2 - 45, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
    cv2.putText(frame, "RIGHT", (x2 + (w - x2) // 2 - 35, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)

    # Draw pose skeleton
    if pose_landmarks:
        draw_pose_skeleton(frame, pose_landmarks)

    # Status text background
    cv2.rectangle(frame, (0, h - 100), (w, h), (0, 0, 0), -1)

    # Status info
    lane_text = f"Lane: {lane.upper()}"
    action_text = f"Action: {action}" if action else "Action: ---"

    squat_drop = debug_values.get("squat_drop", 0)
    wrist_dist = debug_values.get("wrist_dist", 999)
    debug_text = (f"Squat: {squat_drop:.3f}/{SQUAT_SHOULDER_DROP:.2f}  "
                  f"Wrists: {wrist_dist:.3f}/{CLAP_DISTANCE_THRESHOLD:.2f}")

    cv2.putText(frame, lane_text, (10, h - 75),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
    cv2.putText(frame, action_text, (10, h - 45),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
    cv2.putText(frame, debug_text, (10, h - 15),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (180, 180, 180), 1)

    # Quit hint
    cv2.putText(frame, "Press 'q' to quit", (w - 180, h - 15),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (200, 200, 200), 1)

    return frame


# ─── Main ────────────────────────────────────────────────────────────────────

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
        print("Error: Cannot open webcam.")
        return

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    with vision.PoseLandmarker.create_from_options(options) as landmarker:
        # Calibration phase
        ts_ms = calibrate(cap, landmarker, 0)

        print("\n[Ready] Start playing! Move your body to control the game.\n")

        last_action = ""
        action_display_time = 0

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                print("Error: Cannot read frame.")
                break

            frame = cv2.flip(frame, 1)
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            ts_ms += 33
            results = process_frame(landmarker, rgb, ts_ms)

            lane = current_lane
            action = ""

            detected_landmarks = None
            if results.pose_landmarks:
                lm = results.pose_landmarks[0]
                detected_landmarks = lm

                lane = detect_lane(lm, frame.shape[1])

                if detect_jump(lm):
                    action = "JUMP"
                elif detect_squat(lm):
                    action = "DUCK"
                elif detect_clap(lm):
                    action = "HOVERBOARD"

            # Keep action text visible briefly
            now = time.time()
            if action:
                last_action = action
                action_display_time = now
            elif now - action_display_time > 1.0:
                last_action = ""

            frame = draw_overlay(frame, lane, last_action, detected_landmarks)

            cv2.imshow("Subway Surfers Controller", frame)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    cap.release()
    cv2.destroyAllWindows()
    print("[Exit] Controller stopped.")


if __name__ == "__main__":
    main()
