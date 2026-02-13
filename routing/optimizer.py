from __future__ import annotations

from typing import Dict

from django.conf import settings

from .fuel_planning import SimpleFuelPlanner


def plan_stops_and_costs(
    route: Dict,
    stations: Dict,  # kept for compatibility with views, but ignored for now
    mpg: float,
    max_range_miles: float,
    geocode_state_func,
    geocode_station_func,
    start_empty: bool = False,
) -> Dict:
    """Thin wrapper around SimpleFuelPlanner to keep views.py working.

    The heavy lifting is delegated to SimpleFuelPlanner, which currently
    returns a stub plan with no stops. This satisfies the interface expected
    by views.plan_route without re-implementing the old optimizer in full.
    """
    planner = SimpleFuelPlanner(mpg=mpg, max_range_miles=max_range_miles)
    return planner.plan(route, start_empty=start_empty)
