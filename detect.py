import numpy as np
import mediapipe as mp


def detect_hands_up_down(landmarks):
    """Return one of: 'both_up', 'left_up', 'right_up', 'down'.

    landmarks: pose_landmarks.landmark list
    """
    mp_pose = mp.solutions.pose
    nose = [landmarks[mp_pose.PoseLandmark.NOSE.value].x, landmarks[mp_pose.PoseLandmark.NOSE.value].y]
    left_wrist = [landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].x, landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].y]
    right_wrist = [landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value].x, landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value].y]
    left_hand_up = left_wrist[1] < nose[1]
    right_hand_up = right_wrist[1] < nose[1]
    if left_hand_up and right_hand_up:
        return "both_up"
    elif left_hand_up:
        return "left_up"
    elif right_hand_up:
        return "right_up"
    else:
        return "down"


def detect_jump(prev_nose_y, nose_y, is_jumping):
    """Detect a jump based on nose vertical movement.

    Returns (action, is_jumping) where action is 'jump' or None.
    """
    action = None
    if prev_nose_y is not None:
        if prev_nose_y - nose_y > 0.05:
            is_jumping = True
        elif is_jumping and prev_nose_y - nose_y < -0.05:
            is_jumping = False
            action = "jump"
    return action, is_jumping


def detect_walk_run(left_hip, left_knee, left_ankle, right_hip, right_knee, right_ankle):
    """Very basic walk/run detection using knee angles. Returns True if walk/run detected."""

    def calculate_angle(a, b, c):
        a = np.array(a)
        b = np.array(b)
        c = np.array(c)
        radians = np.arctan2(c[1] - b[1], c[0] - b[0]) - np.arctan2(a[1] - b[1], a[0] - b[0])
        angle = np.abs(radians * 180.0 / np.pi)
        if angle > 180.0:
            angle = 360 - angle
        return angle

    left_knee_angle = calculate_angle(left_hip, left_knee, left_ankle)
    right_knee_angle = calculate_angle(right_hip, right_knee, right_ankle)
    if left_knee_angle < 160 or right_knee_angle < 160:
        return True
    return False


def detect_fist(hand_landmarks, mp_hands):
    """Simple heuristic: finger tips below PIP joints indicates closed fist."""
    if not hand_landmarks:
        return False
    finger_tips = [
        hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_TIP].y,
        hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP].y,
        hand_landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_TIP].y,
        hand_landmarks.landmark[mp_hands.HandLandmark.RING_FINGER_TIP].y,
        hand_landmarks.landmark[mp_hands.HandLandmark.PINKY_TIP].y,
    ]
    pip_joints = [
        hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_PIP].y,
        hand_landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_PIP].y,
        hand_landmarks.landmark[mp_hands.HandLandmark.RING_FINGER_PIP].y,
        hand_landmarks.landmark[mp_hands.HandLandmark.PINKY_PIP].y,
    ]
    return all(finger_tips[i+1] > pip_joints[i] for i in range(len(pip_joints)))
