"""The activity catalog, sourced from the motionsense SDK.

The catalog lives in the SDK so that the definitions the UI shows and the
definitions the recognizers implement cannot drift apart -- there is one list,
and it is the one the detector actually uses.

``Activity`` is the SDK's ``ActivityDef``. Its ``trigger`` is a ``str`` enum, so
comparisons like ``activity.trigger == "level"`` work as they always did.
"""

from motionsense import ActivityDef as Activity
from motionsense import Signal, catalog

__all__ = ["ACTIVITIES", "ACTIVITIES_BY_ID", "Activity", "known", "needs_hand_model"]

#: Ordered for display: the SDK returns them grouped by category already.
ACTIVITIES: list[Activity] = list(catalog.all_activities())
ACTIVITIES_BY_ID: dict[str, Activity] = {a.id: a for a in ACTIVITIES}


def needs_hand_model(activity_ids) -> bool:
    """Whether any of these activities requires the hand-landmark model.

    The hand model costs roughly as much per frame as the pose model, so the app
    only turns it on when a finger activity is actually mapped to a key.
    """
    return any(
        Signal.HANDS in definition.requires
        for definition in (ACTIVITIES_BY_ID.get(i) for i in activity_ids)
        if definition is not None
    )


def known(activity_ids) -> list[str]:
    """Filter out ids the catalog no longer defines.

    A saved configuration can outlive an activity id. Dropping unknown ids keeps
    an old configuration loadable instead of failing the whole run.
    """
    return [i for i in activity_ids if i in ACTIVITIES_BY_ID]
