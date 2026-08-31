"""Turns raw MediaPipe pose/hand landmarks into named activities.

Two kinds of activity come out of :class:`ActivityDetector.process`:

- level activities: true for as long as a pose condition holds (e.g.
  ``both_hands_up``, ``squat``). Meant to be *held* on a mapped key.
- edge events: fire once when a motion is recognised (e.g. ``jump``,
  ``sit_down``, ``wave_left_hand``). Meant to be *tapped* on a mapped key.
"""

import math
import time
from collections import deque

import mediapipe as mp

mp_pose = mp.solutions.pose
mp_hands = mp.solutions.hands
PoseLM = mp_pose.PoseLandmark
HandLM = mp_hands.HandLandmark


def _xy(landmarks, idx):
    lm = landmarks[idx]
    return (lm.x, lm.y)


def _dist(a, b):
    return math.hypot(a[0] - b[0], a[1] - b[1])


def _is_fist_closed(hand_landmarks) -> bool:
    lm = hand_landmarks.landmark
    tips = [HandLM.INDEX_FINGER_TIP, HandLM.MIDDLE_FINGER_TIP, HandLM.RING_FINGER_TIP, HandLM.PINKY_TIP]
    pips = [HandLM.INDEX_FINGER_PIP, HandLM.MIDDLE_FINGER_PIP, HandLM.RING_FINGER_PIP, HandLM.PINKY_PIP]
    return all(lm[t].y > lm[p].y for t, p in zip(tips, pips))


class ActivityDetector:
    WAVE_WINDOW = 1.2
    WAVE_MIN_DIRECTION_CHANGES = 3
    WAVE_MIN_MOVE = 0.005
    WAVE_COOLDOWN = 1.0
    JUMP_COOLDOWN = 0.6
    POSTURE_COOLDOWN = 0.8
    SQUAT_ENTER_RATIO = 0.32
    SQUAT_EXIT_RATIO = 0.42

    def __init__(self):
        self._prev_nose_y = None
        self._jumping = False
        self._last_jump_at = float("-inf")

        self._posture_state = "standing"  # or "squatting"
        self._last_posture_change = float("-inf")

        self._wrist_x_history = {"Left": deque(), "Right": deque()}
        self._last_wave_at = {"Left": float("-inf"), "Right": float("-inf")}

    def process(self, pose_landmarks, hand_results, now: float | None = None):
        """Returns (active: set[str], events: list[str])."""
        now = now if now is not None else time.time()
        active: set[str] = set()
        events: list[str] = []

        if pose_landmarks is not None:
            m = self._metrics(pose_landmarks.landmark)
            active |= self._level_activities(m)

            events += self._jump_event(m["nose"][1], m["shoulder_width"], now)

            posture_active, posture_events = self._update_posture(m["leg_ratio"], now)
            active |= posture_active
            events += posture_events

            events += self._wave_event("Left", m["l_wr"][0], m["left_up"], now)
            events += self._wave_event("Right", m["r_wr"][0], m["right_up"], now)
        else:
            self._prev_nose_y = None

        if hand_results and hand_results.multi_hand_landmarks:
            active |= self._hand_activities(hand_results)

        return active, events

    # ---- landmark metrics ---------------------------------------------
    def _metrics(self, lm):
        nose = _xy(lm, PoseLM.NOSE.value)
        l_sh = _xy(lm, PoseLM.LEFT_SHOULDER.value)
        r_sh = _xy(lm, PoseLM.RIGHT_SHOULDER.value)
        l_wr = _xy(lm, PoseLM.LEFT_WRIST.value)
        r_wr = _xy(lm, PoseLM.RIGHT_WRIST.value)
        l_hip = _xy(lm, PoseLM.LEFT_HIP.value)
        r_hip = _xy(lm, PoseLM.RIGHT_HIP.value)
        l_knee = _xy(lm, PoseLM.LEFT_KNEE.value)
        r_knee = _xy(lm, PoseLM.RIGHT_KNEE.value)
        l_ank = _xy(lm, PoseLM.LEFT_ANKLE.value)
        r_ank = _xy(lm, PoseLM.RIGHT_ANKLE.value)

        shoulder_width = max(_dist(l_sh, r_sh), 1e-6)
        hip_y = (l_hip[1] + r_hip[1]) / 2
        knee_y = (l_knee[1] + r_knee[1]) / 2
        ankle_y = (l_ank[1] + r_ank[1]) / 2
        leg_ratio = (knee_y - hip_y) / max(ankle_y - hip_y, 1e-6)

        return dict(
            nose=nose, l_sh=l_sh, r_sh=r_sh, l_wr=l_wr, r_wr=r_wr,
            l_hip=l_hip, r_hip=r_hip, shoulder_width=shoulder_width,
            leg_ratio=leg_ratio,
            left_up=l_wr[1] < nose[1] - 0.02,
            right_up=r_wr[1] < nose[1] - 0.02,
        )

    # ---- level (continuous) activities ---------------------------------
    def _level_activities(self, m):
        active = set()

        mid_sh_x = (m["l_sh"][0] + m["r_sh"][0]) / 2
        mid_sh_y = (m["l_sh"][1] + m["r_sh"][1]) / 2
        l_offset = m["l_wr"][0] - mid_sh_x
        r_offset = m["r_wr"][0] - mid_sh_x
        is_t_pose = (
            abs(m["l_wr"][1] - mid_sh_y) < 0.4 * m["shoulder_width"]
            and abs(m["r_wr"][1] - mid_sh_y) < 0.4 * m["shoulder_width"]
            and abs(l_offset) > 0.9 * m["shoulder_width"]
            and abs(r_offset) > 0.9 * m["shoulder_width"]
            and l_offset * r_offset < 0  # wrists on opposite sides of the body
        )
        if is_t_pose:
            active.add("t_pose")

        if m["left_up"] and m["right_up"]:
            active.add("both_hands_up")
        elif m["left_up"]:
            active.add("left_hand_up")
        elif m["right_up"]:
            active.add("right_hand_up")
        elif not is_t_pose:
            active.add("hands_down")

        mid_hip_y = (m["l_hip"][1] + m["r_hip"][1]) / 2
        top_y = min(m["l_sh"][1], m["r_sh"][1])
        if (
            m["l_wr"][0] > m["r_sh"][0]
            and m["r_wr"][0] < m["l_sh"][0]
            and top_y < m["l_wr"][1] < mid_hip_y
            and top_y < m["r_wr"][1] < mid_hip_y
        ):
            active.add("arms_crossed")

        mid_hip_x = (m["l_hip"][0] + m["r_hip"][0]) / 2
        lean = (mid_sh_x - mid_hip_x) / m["shoulder_width"]
        if lean < -0.35:
            active.add("lean_left")
        elif lean > 0.35:
            active.add("lean_right")

        return active

    def _hand_activities(self, hand_results):
        active = set()
        for i, hand_landmarks in enumerate(hand_results.multi_hand_landmarks):
            label = hand_results.multi_handedness[i].classification[0].label.lower()
            closed = _is_fist_closed(hand_landmarks)
            active.add(f"{label}_fist_{'closed' if closed else 'open'}")
        return active

    # ---- edge (momentary) events ----------------------------------------
    def _jump_event(self, nose_y, shoulder_width, now):
        events = []
        if self._prev_nose_y is not None:
            delta = self._prev_nose_y - nose_y  # positive: nose moved up
            threshold = 0.6 * shoulder_width
            if delta > threshold:
                self._jumping = True
            elif self._jumping and delta < -threshold * 0.5:
                self._jumping = False
                if now - self._last_jump_at > self.JUMP_COOLDOWN:
                    events.append("jump")
                    self._last_jump_at = now
        self._prev_nose_y = nose_y
        return events

    def _update_posture(self, leg_ratio, now):
        events = []
        if self._posture_state == "standing" and leg_ratio < self.SQUAT_ENTER_RATIO:
            self._posture_state = "squatting"
            if now - self._last_posture_change > self.POSTURE_COOLDOWN:
                events.append("sit_down")
            self._last_posture_change = now
        elif self._posture_state == "squatting" and leg_ratio > self.SQUAT_EXIT_RATIO:
            self._posture_state = "standing"
            if now - self._last_posture_change > self.POSTURE_COOLDOWN:
                events.append("stand_up")
            self._last_posture_change = now

        active = {"squat"} if self._posture_state == "squatting" else set()
        return active, events

    def _wave_event(self, side, wrist_x, hand_up, now):
        hist = self._wrist_x_history[side]
        if not hand_up:
            hist.clear()
            return []

        hist.append((now, wrist_x))
        while hist and now - hist[0][0] > self.WAVE_WINDOW:
            hist.popleft()
        if len(hist) < 4:
            return []

        direction_changes = 0
        trend = None
        prev_x = hist[0][1]
        for _, x in list(hist)[1:]:
            d = x - prev_x
            prev_x = x
            if abs(d) < self.WAVE_MIN_MOVE:
                continue
            cur = "r" if d > 0 else "l"
            if trend is not None and cur != trend:
                direction_changes += 1
            trend = cur

        if (
            direction_changes >= self.WAVE_MIN_DIRECTION_CHANGES
            and now - self._last_wave_at[side] > self.WAVE_COOLDOWN
        ):
            self._last_wave_at[side] = now
            hist.clear()
            return [f"wave_{side.lower()}_hand"]
        return []
