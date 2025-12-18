import cv2
import mediapipe as mp

from detect import detect_hands_up_down, detect_jump, detect_walk_run, detect_fist
from actions import on_both_hands_up, on_left_hand_up, on_right_hand_up, on_hands_down, on_jump, on_walk_run, on_fist_closed, on_fist_open


# Initialize MediaPipe Pose and Hands
mp_pose = mp.solutions.pose
pose = mp_pose.Pose()
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(min_detection_confidence=0.7,
                       min_tracking_confidence=0.7, max_num_hands=2)
mp_drawing = mp.solutions.drawing_utils


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
        nose = [landmarks[mp_pose.PoseLandmark.NOSE.value].x,
                landmarks[mp_pose.PoseLandmark.NOSE.value].y]
        left_wrist = [landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].x,
                      landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].y]
        right_wrist = [landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value].x,
                       landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value].y]

        # --- Gesture Recognition ---
        # Hands up/down detection
        hands_action = detect_hands_up_down(landmarks)
        if hands_action == "both_up":
            cv2.putText(frame, "Both Hands Up", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 1, cv2.LINE_AA)
            on_both_hands_up(frame, landmarks)
        elif hands_action == "left_up":
            cv2.putText(frame, "Left Hand Up", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 1, cv2.LINE_AA)
            on_left_hand_up(frame, landmarks)
        elif hands_action == "right_up":
            cv2.putText(frame, "Right Hand Up", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 1, cv2.LINE_AA)
            on_right_hand_up(frame, landmarks)
        else:
            cv2.putText(frame, "Hands Down", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 1, cv2.LINE_AA)
            on_hands_down(frame, landmarks)

        # --- Action Recognition ---

        # Jump detection
        jump_action, is_jumping = detect_jump(prev_nose_y, nose[1], is_jumping)
        if jump_action == "jump":
            print("Action: Jump")
            cv2.putText(frame, "Action: Jump", (10, 50),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 1, cv2.LINE_AA)
            on_jump(frame, landmarks)
        prev_nose_y = nose[1]

        # Walk/Run detection (very basic)

        # left_hip = [landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].x,
        #             landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].y]
        # right_hip = [landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value].x,
        #              landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value].y]
        # left_knee = [landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value].x,
        #              landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value].y]
        # right_knee = [landmarks[mp_pose.PoseLandmark.RIGHT_KNEE.value].x,
        #               landmarks[mp_pose.PoseLandmark.RIGHT_KNEE.value].y]
        # left_ankle = [landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value].x,
        #               landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value].y]
        # right_ankle = [landmarks[mp_pose.PoseLandmark.RIGHT_ANKLE.value].x,
        #                landmarks[mp_pose.PoseLandmark.RIGHT_ANKLE.value].y]
        # if detect_walk_run(left_hip, left_knee, left_ankle, right_hip, right_knee, right_ankle):
        #     print("Action: Walking/Running")
        #     cv2.putText(frame, "Action: Walking/Running", (10, 70),
        #                 cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 1, cv2.LINE_AA)
        #     on_walk_run(frame, landmarks)

    if hand_results.multi_hand_landmarks:
        for i, hand_landmarks in enumerate(hand_results.multi_hand_landmarks):
            mp_drawing.draw_landmarks(
                frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

            # --- Gesture Recognition ---
            hand_label = hand_results.multi_handedness[i].classification[0].label
            display_label = "Right" if hand_label == "Left" else "Left"
            if detect_fist(hand_landmarks, mp_hands):
                gesture = "Fist Closed"
                color = (0, 0, 255)
                on_fist_closed(frame, hand_landmarks, display_label)
            else:
                gesture = "Fist Open"
                color = (0, 255, 0)
                on_fist_open(frame, hand_landmarks, display_label)
            # print(f"{display_label} Hand: {gesture}")
            # Draw hand status in small text near top-left/right corners
            if display_label == "Left":
                cv2.putText(frame, f"Left: {gesture}", (10, 110),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 1, cv2.LINE_AA)
            else:  # Right
                cv2.putText(frame, f"Right: {gesture}", (300, 110),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 1, cv2.LINE_AA)

    cv2.imshow('Game control', frame)

    key = cv2.waitKey(1) & 0xFF
    if key in (ord('q'), ord('Q'), 27):
        print("Exiting...")
        break

cap.release()
cv2.destroyAllWindows()
print("Program ended successfully.")
