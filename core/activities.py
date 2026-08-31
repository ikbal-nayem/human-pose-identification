import json
import os
from dataclasses import dataclass

ACTIVITIES_PATH = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data", "activities.json"
)


@dataclass(frozen=True)
class Activity:
    id: str
    name: str
    category: str
    trigger: str  # "level" -> hold key while active, "edge" -> tap key once
    description: str


def load_activities(path: str = ACTIVITIES_PATH) -> list[Activity]:
    with open(path, "r", encoding="utf-8") as f:
        raw = json.load(f)
    return [Activity(**item) for item in raw]


ACTIVITIES: list[Activity] = load_activities()
ACTIVITIES_BY_ID: dict[str, Activity] = {a.id: a for a in ACTIVITIES}
CATEGORIES: list[str] = sorted({a.category for a in ACTIVITIES})
