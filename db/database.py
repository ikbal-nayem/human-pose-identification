import datetime
import os
import sqlite3

DB_PATH = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data", "config.db"
)

# Default key mapping seeded into a fresh "Default" configuration so the
# app is immediately usable out of the box.
DEFAULT_MAPPING = {
    "jump": "space",
    "both_hands_up": "up",
    "squat": "down",
    "left_hand_up": "left",
    "right_hand_up": "right",
}


def _now() -> str:
    return datetime.datetime.now().isoformat(timespec="seconds")


class ConfigDatabase:
    def __init__(self, path: str = DB_PATH):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        self.conn = sqlite3.connect(path, check_same_thread=False)
        self.conn.row_factory = sqlite3.Row
        self.conn.execute("PRAGMA foreign_keys = ON")
        self._init_schema()
        self._ensure_default_configuration()

    def _init_schema(self):
        self.conn.executescript(
            """
            CREATE TABLE IF NOT EXISTS configurations (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                name TEXT NOT NULL UNIQUE,
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL
            );

            CREATE TABLE IF NOT EXISTS key_mappings (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                config_id INTEGER NOT NULL REFERENCES configurations(id) ON DELETE CASCADE,
                activity_id TEXT NOT NULL,
                key TEXT NOT NULL,
                UNIQUE(config_id, activity_id)
            );
            """
        )
        self.conn.commit()

    def _ensure_default_configuration(self):
        if self.list_configs():
            return
        config_id = self.create_config("Default")
        for activity_id, key in DEFAULT_MAPPING.items():
            self.set_mapping(config_id, activity_id, key)

    # ---- configurations --------------------------------------------------
    def list_configs(self):
        cur = self.conn.execute(
            "SELECT id, name, created_at, updated_at FROM configurations ORDER BY name COLLATE NOCASE"
        )
        return cur.fetchall()

    def get_config(self, config_id: int):
        cur = self.conn.execute(
            "SELECT id, name, created_at, updated_at FROM configurations WHERE id = ?", (config_id,)
        )
        return cur.fetchone()

    def create_config(self, name: str) -> int:
        name = name.strip()
        if not name:
            raise ValueError("Configuration name cannot be empty.")
        now = _now()
        cur = self.conn.execute(
            "INSERT INTO configurations (name, created_at, updated_at) VALUES (?, ?, ?)",
            (name, now, now),
        )
        self.conn.commit()
        return cur.lastrowid

    def rename_config(self, config_id: int, new_name: str):
        new_name = new_name.strip()
        if not new_name:
            raise ValueError("Configuration name cannot be empty.")
        self.conn.execute(
            "UPDATE configurations SET name = ?, updated_at = ? WHERE id = ?",
            (new_name, _now(), config_id),
        )
        self.conn.commit()

    def delete_config(self, config_id: int):
        self.conn.execute("DELETE FROM configurations WHERE id = ?", (config_id,))
        self.conn.commit()

    def duplicate_config(self, config_id: int, new_name: str) -> int:
        new_id = self.create_config(new_name)
        for activity_id, key in self.get_mappings(config_id).items():
            self.set_mapping(new_id, activity_id, key)
        return new_id

    # ---- key mappings ------------------------------------------------------
    def get_mappings(self, config_id: int) -> dict[str, str]:
        cur = self.conn.execute(
            "SELECT activity_id, key FROM key_mappings WHERE config_id = ?", (config_id,)
        )
        return {row["activity_id"]: row["key"] for row in cur.fetchall()}

    def set_mapping(self, config_id: int, activity_id: str, key: str):
        self.conn.execute(
            """
            INSERT INTO key_mappings (config_id, activity_id, key) VALUES (?, ?, ?)
            ON CONFLICT(config_id, activity_id) DO UPDATE SET key = excluded.key
            """,
            (config_id, activity_id, key),
        )
        self.conn.execute(
            "UPDATE configurations SET updated_at = ? WHERE id = ?", (_now(), config_id)
        )
        self.conn.commit()

    def clear_mapping(self, config_id: int, activity_id: str):
        self.conn.execute(
            "DELETE FROM key_mappings WHERE config_id = ? AND activity_id = ?",
            (config_id, activity_id),
        )
        self.conn.execute(
            "UPDATE configurations SET updated_at = ? WHERE id = ?", (_now(), config_id)
        )
        self.conn.commit()

    def close(self):
        self.conn.close()
