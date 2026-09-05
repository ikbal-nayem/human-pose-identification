"""Qt adapter around the motionsense engine.

All detection, filtering, key output and camera handling now live in the SDK.
What is left here is the part that is genuinely application-specific: turning
SDK events into Qt signals, and SDK frames into ``QImage``.

The engine runs on this ``QThread`` (via the blocking ``engine.run``), so every
signal below is emitted from a thread Qt knows about, and the connections to the
GUI are ordinary queued connections.
"""

from motionsense import CameraSource, EngineConfig, MotionEngine, Tuning
from motionsense.bindings import KeyBindings
from motionsense.draw import render
from PySide6.QtCore import QThread, Signal
from PySide6.QtGui import QImage

from core.activities import ACTIVITIES_BY_ID, known, needs_hand_model

#: Vertical swipes overlap with simply raising a hand, so the SDK keeps them off
#: unless asked. The app asks only when the user has actually mapped one.
_VERTICAL_SWIPES = ("swipe_up", "swipe_down")


class CameraWorker(QThread):
    frame_ready = Signal(QImage)
    status_changed = Signal(list)  # list[str] of active activity display names
    log_message = Signal(str)
    error = Signal(str)
    stats_changed = Signal(float, float)  # fps, latency in milliseconds

    def __init__(self, mappings: dict[str, str], camera_index: int = 0, parent=None):
        super().__init__(parent)
        self.mappings = {aid: key for aid, key in mappings.items() if aid in ACTIVITIES_BY_ID}
        self.camera_index = camera_index
        self._engine: MotionEngine | None = None
        self._labels = {a.id: a.name for a in ACTIVITIES_BY_ID.values()}
        self._last_names: list[str] = []
        self._stats_counter = 0

    # ---- control (called from the GUI thread) ------------------------------
    def stop(self):
        engine = self._engine
        if engine is not None:
            # Releases every held activity first, so no mapped key is left down.
            engine.stop()

    def calibrate(self, seconds: float = 2.0):
        """Record the subject's neutral pose so thresholds fit this person."""
        engine = self._engine
        if engine is not None:
            engine.calibrate(seconds)
            self.log_message.emit(f"Calibrating for {seconds:.0f}s - stand still, facing the camera.")

    # ---- worker thread ------------------------------------------------------
    def run(self):
        mapped = known(self.mappings)
        config = EngineConfig(
            # Latency matters more than the last few percent of landmark
            # precision when the output is a keypress.
            preset="fast",
            deliver_frames=True,
            enable_hands=needs_hand_model(mapped),
            tuning=Tuning(
                vertical_swipes=any(i in self.mappings for i in _VERTICAL_SWIPES),
                # The SDK's own defaults (min_on=0.06, min_off=0.10) favour
                # chatter-rejection over reaction time, which is right for a
                # hands-free UI but measures out to ~120ms of dead time before a
                # held pose reaches its mapped key -- clearly perceptible, and
                # the previous, pre-SDK version of this app had no debounce at
                # all. Pressing on the very first qualifying frame recovers
                # that; releasing keeps a little dwell so a pose hovering right
                # at the threshold does not chatter the key.
                level_min_on=0.0,
                level_min_off=0.03,
                posture_min_on=0.0,
                posture_min_off=0.05,
            ),
        )

        engine = MotionEngine(config)
        self._engine = engine

        try:
            bindings = KeyBindings(engine, self.mappings)
        except KeyError as exc:
            self.error.emit(f"Invalid key mapping: {exc}")
            self._engine = None
            return

        engine.on_frame(self._on_frame)
        engine.on_error(lambda exc: self.error.emit(str(exc)))
        for activity_id in self.mappings:
            engine.on(activity_id, self._log_event)
            if ACTIVITIES_BY_ID[activity_id].is_level:
                engine.on_end(activity_id, self._log_event)

        if config.enable_hands:
            self.log_message.emit("Hand tracking enabled (a finger activity is mapped).")

        try:
            engine.run(CameraSource(self.camera_index))
        finally:
            bindings.clear()
            self._engine = None

    # ---- engine callbacks (on this thread) ----------------------------------
    def _on_frame(self, result):
        names = [self._labels[a] for a in sorted(result.active) if a in self._labels]
        if names != self._last_names:
            self._last_names = names
            self.status_changed.emit(names)

        # Report throughput about twice a second rather than every frame; the
        # label cannot usefully change faster than that.
        self._stats_counter += 1
        if self._stats_counter >= 15:
            self._stats_counter = 0
            snapshot = self._engine.snapshot() if self._engine else None
            if snapshot is not None:
                self.stats_changed.emit(snapshot.fps, snapshot.latency * 1000.0)

        canvas = render(result, mirror=True, hud=False)
        if canvas is None:
            return
        height, width, _ = canvas.shape
        # The overlay is BGR; Format_BGR888 lets Qt read it directly instead of
        # paying for a colour conversion on every frame.
        image = QImage(canvas.data, width, height, 3 * width, QImage.Format.Format_BGR888)
        self.frame_ready.emit(image.copy())

    def _log_event(self, event):
        definition = ACTIVITIES_BY_ID.get(event.activity)
        name = definition.name if definition else event.activity
        key = self.mappings.get(event.activity, "")
        if definition is not None and definition.is_edge:
            self.log_message.emit(f"Tap '{key}'  ({name})")
        elif event.phase.value == "start":
            self.log_message.emit(f"Hold '{key}'  ({name})")
        else:
            self.log_message.emit(f"Release '{key}'  ({name})")
