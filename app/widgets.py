from PySide6.QtCore import Qt, Signal
from PySide6.QtGui import QKeySequence
from PySide6.QtWidgets import QPushButton

_SPECIAL_KEYS = {
    Qt.Key.Key_Escape: "esc",
    Qt.Key.Key_Return: "enter",
    Qt.Key.Key_Enter: "enter",
    Qt.Key.Key_Tab: "tab",
    Qt.Key.Key_Backspace: "backspace",
    Qt.Key.Key_Space: "space",
    Qt.Key.Key_Up: "up",
    Qt.Key.Key_Down: "down",
    Qt.Key.Key_Left: "left",
    Qt.Key.Key_Right: "right",
    Qt.Key.Key_Shift: "shift",
    Qt.Key.Key_Control: "ctrl",
    Qt.Key.Key_Alt: "alt",
    Qt.Key.Key_CapsLock: "caps lock",
    Qt.Key.Key_Delete: "delete",
    Qt.Key.Key_Insert: "insert",
    Qt.Key.Key_Home: "home",
    Qt.Key.Key_End: "end",
    Qt.Key.Key_PageUp: "page up",
    Qt.Key.Key_PageDown: "page down",
}


def qt_key_to_keyboard_name(key: int, text: str) -> str | None:
    """Maps a Qt key event to a name understood by the `keyboard` library."""
    if key in _SPECIAL_KEYS:
        return _SPECIAL_KEYS[key]
    if Qt.Key.Key_F1 <= key <= Qt.Key.Key_F35:
        return f"f{key - Qt.Key.Key_F1 + 1}"
    if text and text.strip():
        return text.lower()
    seq = QKeySequence(key).toString()
    return seq.lower() if seq else None


class KeyCaptureButton(QPushButton):
    """A button that, when clicked, listens for the next key press and
    reports it as a `keyboard`-library-compatible key name."""

    key_captured = Signal(str)

    def __init__(self, current_key: str = "", parent=None):
        super().__init__(parent)
        self.setObjectName("keyCapture")
        self._listening = False
        self._key = ""
        self.setFocusPolicy(Qt.FocusPolicy.StrongFocus)
        self.set_key(current_key)
        self.clicked.connect(self._start_listening)

    def set_key(self, key: str | None):
        self._key = key or ""
        self.setText(self._key.upper() if self._key else "Set key")

    def key(self) -> str:
        return self._key

    def _start_listening(self):
        self._listening = True
        self.setText("Press a key…")
        self.setFocus()

    def keyPressEvent(self, event):
        if not self._listening:
            super().keyPressEvent(event)
            return
        if event.key() == Qt.Key.Key_Escape:
            self._listening = False
            self.set_key(self._key)
            return
        name = qt_key_to_keyboard_name(event.key(), event.text())
        self._listening = False
        if name:
            self.set_key(name)
            self.key_captured.emit(name)
        else:
            self.set_key(self._key)

    def focusOutEvent(self, event):
        if self._listening:
            self._listening = False
            self.set_key(self._key)
        super().focusOutEvent(event)
