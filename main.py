import cv2
import mediapipe as mp
import numpy as np

# Initialize MediaPipe Pose and Hands
mp_pose = mp.solutions.pose
pose = mp_pose.Pose()
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(min_detection_confidence=0.7, min_tracking_confidence=0.7, max_num_hands=2)
mp_drawing = mp.solutions.drawing_utils

# Function to calculate angle between three points
def calculate_angle(a, b, c):
    a = np.array(a)  # First
    b = np.array(b)  # Mid
    c = np.array(c)  # End

    radians = np.arctan2(c[1] - b[1], c[0] - b[0]) - np.arctan2(a[1] - b[1], a[0] - b[0])
    angle = np.abs(radians * 180.0 / np.pi)

    if angle > 180.0:
        angle = 360 - angle
    
    return angle

# Function to detect fist gesture
def is_fist_closed(hand_landmarks):
    if not hand_landmarks:
        return False

    # Get all finger landmark y-coordinates
    finger_tips = [
        hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_TIP].y,
        hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP].y,
        hand_landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_TIP].y,
        hand_landmarks.landmark[mp_hands.HandLandmark.RING_FINGER_TIP].y,
        hand_landmarks.landmark[mp_hands.HandLandmark.PINKY_TIP].y
    ]
    
    pip_joints = [
        hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_PIP].y,
        hand_landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_PIP].y,
        hand_landmarks.landmark[mp_hands.HandLandmark.RING_FINGER_PIP].y,
        hand_landmarks.landmark[mp_hands.HandLandmark.PINKY_PIP].y
    ]

    # A simple heuristic for fist detection: check if finger tips are below PIP joints.
    # This is a basic check and can be improved.
    return all(finger_tips[i+1] > pip_joints[i] for i in range(len(pip_joints)))


# Previous y-coordinate of the nose
prev_nose_y = None
is_jumping = False

cap = cv2.VideoCapture(0)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Flip the frame horizontally for a later selfie-view display
    # and convert the BGR image to RGB.
    frame = cv2.cvtColor(cv2.flip(frame, 1), cv2.COLOR_BGR2RGB)
    
    # To improve performance, optionally mark the image as not writeable to
    # pass by reference.
    frame.flags.writeable = False
    pose_results = pose.process(frame)
    hand_results = hands.process(frame)
    frame.flags.writeable = True

    # Draw the pose annotation on the image.
    frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
    if pose_results.pose_landmarks:
        mp_drawing.draw_landmarks(
            frame, pose_results.pose_landmarks, mp_pose.POSE_CONNECTIONS)
        
        landmarks = pose_results.pose_landmarks.landmark
        
        # Get coordinates
        nose = [landmarks[mp_pose.PoseLandmark.NOSE.value].x, landmarks[mp_pose.PoseLandmark.NOSE.value].y]
        left_wrist = [landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].x, landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].y]
        right_wrist = [landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value].x, landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value].y]
        
        # --- Gesture Recognition ---
        
        # Hands up/down detection
        left_hand_up = left_wrist[1] < nose[1]
        right_hand_up = right_wrist[1] < nose[1]

        if left_hand_up and right_hand_up:
            cv2.putText(frame, "Both Hands Up", (50, 200), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2, cv2.LINE_AA)
        elif left_hand_up:
            cv2.putText(frame, "Left Hand Up", (50, 200), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2, cv2.LINE_AA)
        elif right_hand_up:
            cv2.putText(frame, "Right Hand Up", (50, 200), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2, cv2.LINE_AA)
        else:
            cv2.putText(frame, "Hands Down", (50, 200), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2, cv2.LINE_AA)


        # --- Action Recognition ---
        
        # Jump detection
        if prev_nose_y is not None:
            # Simple jump detection based on vertical movement of the nose
            if prev_nose_y - nose[1] > 0.05: # Threshold for jump detection
                is_jumping = True
            elif is_jumping and prev_nose_y - nose[1] < -0.05:
                is_jumping = False
                print("Action: Jump")
                cv2.putText(frame, "Action: Jump", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)


        prev_nose_y = nose[1]

        # Walk/Run detection (very basic)
        left_hip = [landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].x, landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].y]
        right_hip = [landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value].x, landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value].y]
        left_knee = [landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value].x, landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value].y]
        right_knee = [landmarks[mp_pose.PoseLandmark.RIGHT_KNEE.value].x, landmarks[mp_pose.PoseLandmark.RIGHT_KNEE.value].y]

        left_knee_angle = calculate_angle(left_hip, left_knee, [landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value].x, landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value].y])
        right_knee_angle = calculate_angle(right_hip, right_knee, [landmarks[mp_pose.PoseLandmark.RIGHT_ANKLE.value].x, landmarks[mp_pose.PoseLandmark.RIGHT_ANKLE.value].y])

        if left_knee_angle < 160 or right_knee_angle < 160:
             # This is a very simplistic check. A real system would need to analyze the gait cycle.
            print("Action: Walking/Running")
            cv2.putText(frame, "Action: Walking/Running", (50, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)


    if hand_results.multi_hand_landmarks:
        for i, hand_landmarks in enumerate(hand_results.multi_hand_landmarks):
            mp_drawing.draw_landmarks(
                frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

            # --- Gesture Recognition ---
            hand_label = hand_results.multi_handedness[i].classification[0].label
            
            if is_fist_closed(hand_landmarks):
                gesture = "Fist Closed"
                color = (0, 0, 255)
            else:
                gesture = "Fist Open"
                color = (0, 255, 0)
            
            print(f"{hand_label} Hand: {gesture}")
            if hand_label == "Left":
                cv2.putText(frame, f"Left Hand: {gesture}", (50, 150), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2, cv2.LINE_AA)
            else: # Right
                cv2.putText(frame, f"Right Hand: {gesture}", (400, 150), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2, cv2.LINE_AA)


    cv2.imshow('MediaPipe Feed', frame)

    if cv2.waitKey(10) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()