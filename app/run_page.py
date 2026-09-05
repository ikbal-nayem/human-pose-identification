from PySide6.QtCore import Qt
from PySide6.QtGui import QPixmap
from PySide6.QtWidgets import (
    QComboBox,
    QFrame,
    QHBoxLayout,
    QLabel,
    QMessageBox,
    QPushButton,
    QVBoxLayout,
    QWidget,
)

from core.activities import ACTIVITIES_BY_ID
from core.camera_worker import CameraWorker
from db.database import ConfigDatabase


class RunPage(QWidget):
    def __init__(self, db: ConfigDatabase, parent=None):
        super().__init__(parent)
        self.db = db
        self.setObjectName("page")
        self.worker: CameraWorker | None = None
        self._active_key_by_name: dict[str, str] = {}
        self._chip_labels: list[QLabel] = []

        root = QVBoxLayout(self)
        root.setContentsMargins(18, 14, 18, 14)
        root.setSpacing(10)

        title = QLabel("Run")
        title.setObjectName("pageTitle")
        root.addWidget(title)

        controls = QFrame()
        controls.setObjectName("card")
        controls_layout = QHBoxLayout(controls)
        controls_layout.setContentsMargins(14, 10, 14, 10)
        controls_layout.setSpacing(10)

        controls_layout.addWidget(QLabel("Configuration:"))
        self.config_combo = QComboBox()
        self.config_combo.setMinimumWidth(200)
        self.config_combo.currentIndexChanged.connect(self._update_start_enabled)
        controls_layout.addWidget(self.config_combo)
        controls_layout.addStretch(1)

        self.status_label = QLabel("Idle")
        self.status_label.setObjectName("statusBadge")
        controls_layout.addWidget(self.status_label)

        self.calibrate_button = QPushButton("Calibrate")
        self.calibrate_button.setToolTip(
            "Stand still facing the camera for two seconds.\n"
            "Tunes posture thresholds to your body and camera angle."
        )
        self.calibrate_button.setEnabled(False)
        self.calibrate_button.clicked.connect(self._calibrate)
        controls_layout.addWidget(self.calibrate_button)

        self.start_button = QPushButton("▶  Start Tracking")
        self.start_button.setObjectName("startButton")
        self.start_button.clicked.connect(self._toggle_tracking)
        controls_layout.addWidget(self.start_button)

        root.addWidget(controls)

        video_card = QFrame()
        video_card.setObjectName("card")
        video_layout = QVBoxLayout(video_card)
        video_layout.setContentsMargins(10, 10, 10, 10)
        self.video_label = QLabel("Camera preview will appear here")
        self.video_label.setObjectName("videoLabel")
        self.video_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.video_label.setMinimumSize(480, 320)
        video_layout.addWidget(self.video_label)
        root.addWidget(video_card, 1)

        activities_card = QFrame()
        activities_card.setObjectName("card")
        activities_card.setFixedHeight(56)
        activities_layout = QHBoxLayout(activities_card)
        activities_layout.setContentsMargins(14, 6, 14, 6)
        activities_layout.setSpacing(10)

        detected_label = QLabel("DETECTED")
        detected_label.setObjectName("sectionLabel")
        activities_layout.addWidget(detected_label)

        self.chips_layout = QHBoxLayout()
        self.chips_layout.setSpacing(8)
        activities_layout.addLayout(self.chips_layout, 1)

        self.empty_activity_label = QLabel("No activity detected")
        self.empty_activity_label.setObjectName("pageHint")
        self.chips_layout.addWidget(self.empty_activity_label)
        self.chips_layout.addStretch(1)

        root.addWidget(activities_card)

        self.refresh_configs()

    # ---- lifecycle -------------------------------------------------------------
    def refresh_configs(self):
        if self.worker is not None:
            return  # don't change selection mid-run
        current_id = self.config_combo.currentData()
        self.config_combo.blockSignals(True)
        self.config_combo.clear()
        for row in self.db.list_configs():
            self.config_combo.addItem(row["name"], row["id"])
        if current_id is not None:
            idx = self.config_combo.findData(current_id)
            if idx >= 0:
                self.config_combo.setCurrentIndex(idx)
        self.config_combo.blockSignals(False)
        self._update_start_enabled()

    def showEvent(self, event):
        super().showEvent(event)
        self.refresh_configs()

    def _update_start_enabled(self):
        has_config = self.config_combo.count() > 0
        if self.worker is None:
            self.start_button.setEnabled(has_config)
        if not has_config:
            self.start_button.setText("No configurations available")
        elif self.worker is None:
            self.start_button.setText("▶  Start Tracking")

    # ---- detected-activities strip ---------------------------------------
    def _set_detected(self, names: list[str]):
        for chip in self._chip_labels:
            chip.deleteLater()
        self._chip_labels.clear()
        self.empty_activity_label.setVisible(not names)

        for name in names:
            key = self._active_key_by_name.get(name)
            text = f"{name}  →  {key}" if key else name
            chip = QLabel(text)
            chip.setObjectName("activityChip")
            chip.setProperty("bound", bool(key))
            self.chips_layout.insertWidget(self.chips_layout.count() - 2, chip)
            self._chip_labels.append(chip)
            self._refresh_style(chip)

    # ---- start/stop ----------------------------------------------------------
    def _toggle_tracking(self):
        if self.worker is None:
            self._start_tracking()
        else:
            self._stop_tracking()

    def _start_tracking(self):
        config_id = self.config_combo.currentData()
        if config_id is None:
            return
        mappings = self.db.get_mappings(config_id)
        if not mappings:
            reply = QMessageBox.question(
                self, "Empty configuration",
                "This configuration has no keys mapped yet. Start anyway?",
                QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
            )
            if reply != QMessageBox.StandardButton.Yes:
                return

        self._active_key_by_name = {
            ACTIVITIES_BY_ID[aid].name: key for aid, key in mappings.items() if aid in ACTIVITIES_BY_ID
        }

        self.config_combo.setEnabled(False)
        self.start_button.setText("⏹  Stop Tracking")
        self.start_button.setObjectName("stopButton")
        self._refresh_style(self.start_button)
        self.status_label.setText("Starting camera…")

        self.worker = CameraWorker(mappings)
        self.worker.frame_ready.connect(self._on_frame)
        self.worker.status_changed.connect(self._on_status)
        self.worker.stats_changed.connect(self._on_stats)
        self.worker.error.connect(self._on_error)
        self.worker.finished.connect(self._on_worker_finished)
        self.worker.start()
        self.calibrate_button.setEnabled(True)

    def _stop_tracking(self):
        if self.worker is not None:
            self.worker.stop()
            self.status_label.setText("Stopping…")

    def _calibrate(self):
        if self.worker is not None:
            self.worker.calibrate(2.0)
            self.status_label.setText("Calibrating…")

    def _on_worker_finished(self):
        self.worker = None
        self.calibrate_button.setEnabled(False)
        self.config_combo.setEnabled(True)
        self.start_button.setText("▶  Start Tracking")
        self.start_button.setObjectName("startButton")
        self._refresh_style(self.start_button)
        self.status_label.setText("Idle")
        self._set_detected([])
        self.video_label.setText("Camera preview will appear here")
        self.video_label.setPixmap(QPixmap())
        self._update_start_enabled()

    @staticmethod
    def _refresh_style(widget):
        widget.style().unpolish(widget)
        widget.style().polish(widget)

    # ---- worker signal handlers -------------------------------------------
    def _on_frame(self, qimg):
        pixmap = QPixmap.fromImage(qimg)
        scaled = pixmap.scaled(
            self.video_label.size(), Qt.AspectRatioMode.KeepAspectRatio, Qt.TransformationMode.SmoothTransformation
        )
        self.video_label.setPixmap(scaled)

    def _on_stats(self, fps: float, latency_ms: float):
        self.status_label.setText(f"Tracking  ·  {fps:.0f} fps  ·  {latency_ms:.0f} ms")

    def _on_status(self, names: list[str]):
        self._set_detected(names)

    def _on_error(self, message: str):
        QMessageBox.critical(self, "Camera error", message)
