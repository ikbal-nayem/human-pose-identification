from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QFrame,
    QHBoxLayout,
    QHeaderView,
    QInputDialog,
    QLabel,
    QListWidget,
    QListWidgetItem,
    QMessageBox,
    QPushButton,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)

from core.activities import ACTIVITIES
from db.database import ConfigDatabase
from app.widgets import KeyCaptureButton


class ConfigurationsPage(QWidget):
    def __init__(self, db: ConfigDatabase, parent=None):
        super().__init__(parent)
        self.db = db
        self.setObjectName("page")
        self._current_config_id: int | None = None

        root = QVBoxLayout(self)
        root.setContentsMargins(28, 24, 28, 24)
        root.setSpacing(16)

        title = QLabel("Configurations")
        title.setObjectName("pageTitle")
        hint = QLabel("Create key mappings for each activity, then pick a configuration on the Run page.")
        hint.setObjectName("pageHint")
        root.addWidget(title)
        root.addWidget(hint)

        body = QHBoxLayout()
        body.setSpacing(16)
        root.addLayout(body, 1)

        body.addWidget(self._build_config_list(), 1)
        body.addWidget(self._build_mapping_table(), 3)

        self.refresh_configs()

    # ---- left panel: configuration list ------------------------------------
    def _build_config_list(self) -> QFrame:
        card = QFrame()
        card.setObjectName("card")
        layout = QVBoxLayout(card)
        layout.setContentsMargins(16, 16, 16, 16)
        layout.setSpacing(10)

        label = QLabel("SAVED CONFIGURATIONS")
        label.setObjectName("sectionLabel")
        layout.addWidget(label)

        self.config_list = QListWidget()
        self.config_list.currentItemChanged.connect(self._on_config_selected)
        layout.addWidget(self.config_list, 1)

        buttons = QHBoxLayout()
        new_btn = QPushButton("+ New")
        new_btn.setObjectName("primaryButton")
        new_btn.clicked.connect(self._create_config)
        dup_btn = QPushButton("Duplicate")
        dup_btn.clicked.connect(self._duplicate_config)
        buttons.addWidget(new_btn)
        buttons.addWidget(dup_btn)
        layout.addLayout(buttons)

        buttons2 = QHBoxLayout()
        rename_btn = QPushButton("Rename")
        rename_btn.clicked.connect(self._rename_config)
        delete_btn = QPushButton("Delete")
        delete_btn.setObjectName("dangerButton")
        delete_btn.clicked.connect(self._delete_config)
        buttons2.addWidget(rename_btn)
        buttons2.addWidget(delete_btn)
        layout.addLayout(buttons2)

        return card

    # ---- right panel: activity -> key mapping table ------------------------
    def _build_mapping_table(self) -> QFrame:
        card = QFrame()
        card.setObjectName("card")
        layout = QVBoxLayout(card)
        layout.setContentsMargins(16, 16, 16, 16)
        layout.setSpacing(10)

        label = QLabel("KEY MAPPING")
        label.setObjectName("sectionLabel")
        layout.addWidget(label)

        self.table = QTableWidget(len(ACTIVITIES), 5)
        self.table.setHorizontalHeaderLabels(["Activity", "Category", "Trigger", "Key", ""])
        self.table.verticalHeader().setVisible(False)
        self.table.setEditTriggers(QTableWidget.EditTrigger.NoEditTriggers)
        self.table.setSelectionMode(QTableWidget.SelectionMode.NoSelection)
        header = self.table.horizontalHeader()
        header.setSectionResizeMode(0, QHeaderView.ResizeMode.Stretch)
        header.setSectionResizeMode(1, QHeaderView.ResizeMode.ResizeToContents)
        header.setSectionResizeMode(2, QHeaderView.ResizeMode.ResizeToContents)
        header.setSectionResizeMode(3, QHeaderView.ResizeMode.ResizeToContents)
        header.setSectionResizeMode(4, QHeaderView.ResizeMode.ResizeToContents)

        self._key_buttons: dict[str, KeyCaptureButton] = {}
        for row, activity in enumerate(ACTIVITIES):
            name_item = QTableWidgetItem(activity.name)
            name_item.setToolTip(activity.description)
            self.table.setItem(row, 0, name_item)
            self.table.setItem(row, 1, QTableWidgetItem(activity.category))
            trigger_label = "Hold" if activity.trigger == "level" else "Tap"
            self.table.setItem(row, 2, QTableWidgetItem(trigger_label))

            key_btn = KeyCaptureButton()
            key_btn.key_captured.connect(
                lambda key, aid=activity.id: self._on_key_captured(aid, key)
            )
            self._key_buttons[activity.id] = key_btn
            self.table.setCellWidget(row, 3, key_btn)

            clear_btn = QPushButton("Clear")
            clear_btn.setObjectName("clearKey")
            clear_btn.clicked.connect(lambda _, aid=activity.id: self._on_key_cleared(aid))
            self.table.setCellWidget(row, 4, clear_btn)

        layout.addWidget(self.table, 1)
        return card

    # ---- data flow ------------------------------------------------------------
    def refresh_configs(self, select_id: int | None = None):
        self.config_list.blockSignals(True)
        self.config_list.clear()
        rows = self.db.list_configs()
        selected_row = None
        for row in rows:
            item = QListWidgetItem(row["name"])
            item.setData(Qt.ItemDataRole.UserRole, row["id"])
            self.config_list.addItem(item)
            if select_id is not None and row["id"] == select_id:
                selected_row = item
        self.config_list.blockSignals(False)

        if selected_row is not None:
            self.config_list.setCurrentItem(selected_row)
        elif self.config_list.count() > 0:
            self.config_list.setCurrentRow(0)
        else:
            self._current_config_id = None
            self._load_mappings({})

    def _on_config_selected(self, current: QListWidgetItem, _previous):
        if current is None:
            self._current_config_id = None
            self._load_mappings({})
            return
        self._current_config_id = current.data(Qt.ItemDataRole.UserRole)
        mappings = self.db.get_mappings(self._current_config_id)
        self._load_mappings(mappings)

    def _load_mappings(self, mappings: dict[str, str]):
        for activity_id, button in self._key_buttons.items():
            button.set_key(mappings.get(activity_id, ""))

    def _on_key_captured(self, activity_id: str, key: str):
        if self._current_config_id is None:
            return
        self.db.set_mapping(self._current_config_id, activity_id, key)

    def _on_key_cleared(self, activity_id: str):
        if self._current_config_id is None:
            return
        self.db.clear_mapping(self._current_config_id, activity_id)
        self._key_buttons[activity_id].set_key("")

    # ---- config CRUD actions -----------------------------------------------
    def _create_config(self):
        name, ok = QInputDialog.getText(self, "New Configuration", "Configuration name:")
        if not ok or not name.strip():
            return
        try:
            new_id = self.db.create_config(name)
        except (ValueError, Exception) as exc:
            QMessageBox.warning(self, "Could not create configuration", str(exc))
            return
        self.refresh_configs(select_id=new_id)

    def _duplicate_config(self):
        if self._current_config_id is None:
            return
        base_name = self.db.get_config(self._current_config_id)["name"]
        name, ok = QInputDialog.getText(self, "Duplicate Configuration", "New configuration name:", text=f"{base_name} copy")
        if not ok or not name.strip():
            return
        try:
            new_id = self.db.duplicate_config(self._current_config_id, name)
        except Exception as exc:
            QMessageBox.warning(self, "Could not duplicate configuration", str(exc))
            return
        self.refresh_configs(select_id=new_id)

    def _rename_config(self):
        if self._current_config_id is None:
            return
        current_name = self.db.get_config(self._current_config_id)["name"]
        name, ok = QInputDialog.getText(self, "Rename Configuration", "Configuration name:", text=current_name)
        if not ok or not name.strip():
            return
        try:
            self.db.rename_config(self._current_config_id, name)
        except Exception as exc:
            QMessageBox.warning(self, "Could not rename configuration", str(exc))
            return
        self.refresh_configs(select_id=self._current_config_id)

    def _delete_config(self):
        if self._current_config_id is None:
            return
        name = self.db.get_config(self._current_config_id)["name"]
        reply = QMessageBox.question(
            self, "Delete Configuration", f"Delete '{name}'? This cannot be undone.",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
        )
        if reply != QMessageBox.StandardButton.Yes:
            return
        self.db.delete_config(self._current_config_id)
        self.refresh_configs()
