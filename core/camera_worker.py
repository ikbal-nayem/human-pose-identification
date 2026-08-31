import time

import cv2
import mediapipe as mp
from PySide6.QtCore import QThread, Signal
from PySide6.QtGui import QImage

from core.activities import ACTIVITIES_BY_ID
from core.detector import ActivityDetector
from core.key_actions import KeyActionExecutor


class CameraWorker(QThread):
    frame_ready = Signal(QImage)
    status_changed = Signal(list)  # list[str] of active activity display names
    log_message = Signal(str)
    error = Signal(str)

    def __init__(self, mappings: dict[str, str], camera_index: int = 0, parent=None):
        super().__init__(parent)
        self.mappings = mappings
        self.camera_index = camera_index
        self._running = False

    def stop(self):
        self._running = False

    def run(self):
        self._running = True
        mp_pose = mp.solutions.pose
        mp_hands = mp.solutions.hands
        mp_drawing = mp.solutions.drawing_utils

        pose = mp_pose.Pose()
        hands = mp_hands.Hands(min_detection_confidence=0.7, min_tracking_confidence=0.7, max_num_hands=2)
        detector = ActivityDetector()
        executor = KeyActionExecutor(self.mappings, ACTIVITIES_BY_ID)

        cap = cv2.VideoCapture(self.camera_index, cv2.CAP_DSHOW)
        if not cap.isOpened():
            cap = cv2.VideoCapture(self.camera_index)
        if not cap.isOpened():
            self.error.emit("Could not open the camera.")
            self._running = False
            return

        try:
            while self._running:
                ret, frame = cap.read()
                if not ret:
                    self.error.emit("Lost the camera feed.")
                    break

                frame = cv2.cvtColor(cv2.flip(frame, 1), cv2.COLOR_BGR2RGB)
                frame.flags.writeable = False
                pose_results = pose.process(frame)
                hand_results = hands.process(frame)
                frame.flags.writeable = True

                if pose_results.pose_landmarks:
                    mp_drawing.draw_landmarks(frame, pose_results.pose_landmarks, mp_pose.POSE_CONNECTIONS)
                if hand_results.multi_hand_landmarks:
                    for hand_landmarks in hand_results.multi_hand_landmarks:
                        mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

                now = time.time()
                active, events = detector.process(pose_results.pose_landmarks, hand_results, now)
                executor.update(active, events, now, log=self.log_message.emit)

                names = [ACTIVITIES_BY_ID[a].name for a in sorted(active) if a in ACTIVITIES_BY_ID]
                self.status_changed.emit(names)

                h, w, ch = frame.shape
                qimg = QImage(frame.data, w, h, ch * w, QImage.Format.Format_RGB888)
                self.frame_ready.emit(qimg.copy())
        finally:
            executor.release_all()
            cap.release()
            pose.close()
            hands.close()
            self._running = False
