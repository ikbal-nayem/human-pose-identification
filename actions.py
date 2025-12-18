import keyboard


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
    keyboard.press(label.lower())


def on_fist_open(frame, hand_landmarks, label):
    """Called when a fist is open."""
    keyboard.release(label.lower())    
