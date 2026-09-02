from PySide6.QtCore import Qt
from PySide6.QtGui import QPixmap
from PySide6.QtWidgets import (
    QComboBox,
    QFrame,
    QHBoxLayout,
    QLabel,
    QListWidget,
    QListWidgetItem,
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

        root = QVBoxLayout(self)
        root.setContentsMargins(28, 24, 28, 24)
        root.setSpacing(16)

        title = QLabel("Run")
        title.setObjectName("pageTitle")
        hint = QLabel("Pick a configuration, then start tracking to map your movements to key presses.")
        hint.setObjectName("pageHint")
        root.addWidget(title)
        root.addWidget(hint)

        controls = QFrame()
        controls.setObjectName("card")
        controls_layout = QHBoxLayout(controls)
        controls_layout.setContentsMargins(16, 14, 16, 14)
        controls_layout.setSpacing(12)

        controls_layout.addWidget(QLabel("Configuration:"))
        self.config_combo = QComboBox()
        self.config_combo.setMinimumWidth(220)
        self.config_combo.currentIndexChanged.connect(self._update_start_enabled)
        controls_layout.addWidget(self.config_combo)
        controls_layout.addStretch(1)

        self.status_label = QLabel("Idle")
        self.status_label.setObjectName("statusBadge")
        controls_layout.addWidget(self.status_label)

        self.start_button = QPushButton("▶  Start Tracking")
        self.start_button.setObjectName("startButton")
        self.start_button.clicked.connect(self._toggle_tracking)
        controls_layout.addWidget(self.start_button)

        root.addWidget(controls)

        body = QHBoxLayout()
        body.setSpacing(16)
        root.addLayout(body, 1)

        video_card = QFrame()
        video_card.setObjectName("card")
        video_layout = QVBoxLayout(video_card)
        video_layout.setContentsMargins(12, 12, 12, 12)
        self.video_label = QLabel("Camera preview will appear here")
        self.video_label.setObjectName("videoLabel")
        self.video_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.video_label.setMinimumSize(640, 480)
        video_layout.addWidget(self.video_label)
        body.addWidget(video_card, 3)

        side_panel = QVBoxLayout()
        side_panel.setSpacing(16)
        body.addLayout(side_panel, 2)

        active_card = QFrame()
        active_card.setObjectName("card")
        active_layout = QVBoxLayout(active_card)
        active_layout.setContentsMargins(16, 16, 16, 16)
        active_title = QLabel("ACTIVE ACTIVITIES")
        active_title.setObjectName("sectionLabel")
        active_layout.addWidget(active_title)
        self.active_label = QLabel("None detected yet")
        self.active_label.setWordWrap(True)
        active_layout.addWidget(self.active_label)
        side_panel.addWidget(active_card)

        mapped_card = QFrame()
        mapped_card.setObjectName("card")
        mapped_layout = QVBoxLayout(mapped_card)
        mapped_layout.setContentsMargins(16, 16, 16, 16)
        mapped_title = QLabel("MAPPED KEYS")
        mapped_title.setObjectName("sectionLabel")
        mapped_layout.addWidget(mapped_title)
        self.mapped_keys_list = QListWidget()
        mapped_layout.addWidget(self.mapped_keys_list, 1)
        side_panel.addWidget(mapped_card, 1)

        self.config_combo.currentIndexChanged.connect(self._update_mapped_keys)

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
        self._update_mapped_keys()

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

    def _update_mapped_keys(self):
        self.mapped_keys_list.clear()
        config_id = self.config_combo.currentData()
        if config_id is None:
            return
        mappings = self.db.get_mappings(config_id)
        if not mappings:
            self.mapped_keys_list.addItem(QListWidgetItem("No keys mapped in this configuration"))
            return
        for activity_id, key in mappings.items():
            activity = ACTIVITIES_BY_ID.get(activity_id)
            name = activity.name if activity else activity_id
            self.mapped_keys_list.addItem(QListWidgetItem(f"{name}  →  {key}"))

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

        self.config_combo.setEnabled(False)
        self.start_button.setText("⏹  Stop Tracking")
        self.start_button.setObjectName("stopButton")
        self._refresh_style(self.start_button)
        self.status_label.setText("Starting camera…")

        self.worker = CameraWorker(mappings)
        self.worker.frame_ready.connect(self._on_frame)
        self.worker.status_changed.connect(self._on_status)
        self.worker.error.connect(self._on_error)
        self.worker.finished.connect(self._on_worker_finished)
        self.worker.start()

    def _stop_tracking(self):
        if self.worker is not None:
            self.worker.stop()
            self.status_label.setText("Stopping…")

    def _on_worker_finished(self):
        self.worker = None
        self.config_combo.setEnabled(True)
        self.start_button.setText("▶  Start Tracking")
        self.start_button.setObjectName("startButton")
        self._refresh_style(self.start_button)
        self.status_label.setText("Idle")
        self.active_label.setText("None detected yet")
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
        if self.status_label.text() != "Tracking":
            self.status_label.setText("Tracking")

    def _on_status(self, names: list[str]):
        self.active_label.setText(", ".join(names) if names else "None detected")

    def _on_error(self, message: str):
        QMessageBox.critical(self, "Camera error", message)
