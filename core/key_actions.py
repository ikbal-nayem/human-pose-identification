"""Translates detected activity state into physical key presses."""

import time

import keyboard


class KeyActionExecutor:
    def __init__(self, mappings: dict[str, str], activities_by_id: dict):
        self.mappings = mappings
        self.activities_by_id = activities_by_id
        self._held_keys: dict[str, set[str]] = {}
        self._prev_level_active: set[str] = set()

    def update(self, active: set[str], events: list[str], now: float, log=None):
        mapped_active = {a for a in active if a in self.mappings}
        newly_active = mapped_active - self._prev_level_active
        newly_inactive = self._prev_level_active - mapped_active

        for activity_id in newly_active:
            key = self.mappings[activity_id]
            holders = self._held_keys.setdefault(key, set())
            if not holders:
                keyboard.press(key)
                if log:
                    log(f"Hold '{key}'  ({self.activities_by_id[activity_id].name})")
            holders.add(activity_id)

        for activity_id in newly_inactive:
            key = self.mappings.get(activity_id)
            holders = self._held_keys.get(key) if key else None
            if holders and activity_id in holders:
                holders.discard(activity_id)
                if not holders:
                    keyboard.release(key)
                    if log:
                        log(f"Release '{key}'")

        self._prev_level_active = mapped_active

        for activity_id in events:
            key = self.mappings.get(activity_id)
            if key:
                keyboard.press(key)
                time.sleep(0.05)
                keyboard.release(key)
                if log:
                    log(f"Tap '{key}'  ({self.activities_by_id[activity_id].name})")

    def release_all(self):
        for key, holders in list(self._held_keys.items()):
            if holders:
                try:
                    keyboard.release(key)
                except Exception:
                    pass
        self._held_keys.clear()
        self._prev_level_active.clear()
