import mediapipe as mp


def detect_hands_up_down(landmarks):
    mp_pose = mp.solutions.pose
    nose = [landmarks[mp_pose.PoseLandmark.NOSE.value].x,
            landmarks[mp_pose.PoseLandmark.NOSE.value].y]
    left_wrist = [landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].x,
                  landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].y]
    right_wrist = [landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value].x,
                   landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value].y]
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
    action = None
    """Action handler stubs.

    Place your custom behavior in these functions. They are intentionally
    left as no-ops so you can implement side-effects (e.g. send keystrokes,
    trigger callbacks, publish events) appropriate for your application.
    """

def on_both_hands_up(frame, landmarks):
    """Called when both hands are detected up. Implement custom behavior here."""
    pass

def on_left_hand_up(frame, landmarks):
    """Called when left hand is up."""
    pass

def on_right_hand_up(frame, landmarks):
    """Called when right hand is up."""
    pass

def on_hands_down(frame, landmarks):
    """Called when hands are down."""
    pass

def on_jump(frame, landmarks):
    """Called when a jump action is detected."""
    pass

def on_walk_run(frame, landmarks):
    """Called when walking/running is detected."""
    pass

def on_fist_closed(frame, hand_landmarks, label):
    """Called when a fist is closed. `label` is 'Left' or 'Right' as displayed."""
    pass

def on_fist_open(frame, hand_landmarks, label):
    """Called when a fist is open."""
    pass
