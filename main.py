import cv2
import mediapipe as mp

# Initialize MediaPipe Hands
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(max_num_hands=1, min_detection_confidence=0.7)
mp_draw = mp.solutions.drawing_utils

# Initialize video capture
cap = cv2.VideoCapture(0)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        continue
        
    # Convert to RGB and process
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(rgb_frame)
    
    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            # Get landmarks
            landmarks = hand_landmarks.landmark
            
            # Check fist state using all finger tips
            wrist = landmarks[0]
            tips = [
                landmarks[4],  # Thumb tip
                landmarks[8],  # Index finger tip
                landmarks[12], # Middle finger tip
                landmarks[16], # Ring finger tip
                landmarks[20]  # Pinky tip
            ]
            
            # Calculate distances from each finger tip to wrist
            distances = [((tip.x - wrist.x)**2 + (tip.y - wrist.y)**2)**0.5 for tip in tips]
            
            avg_distance = sum(distances) / len(distances)
            print(f"Avg finger distance: {avg_distance:.3f}")
            
            # Determine fist state
            if avg_distance < 0.2:  # Threshold for reliable fist detection
                state = "Fist Closed"
                color = (0, 0, 255)  # Red
            else:
                state = "Fist Open"
                color = (0, 255, 0)  # Green
            
            # Draw hand landmarks and state
            mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
            cv2.putText(frame, state, (10, 50), 
                        cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)
    
    # Display frame
    cv2.imshow('Fist Detection', frame)
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()