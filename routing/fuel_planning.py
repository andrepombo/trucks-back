from __future__ import annotations

import bisect
import logging
import math
import time
from typing import Dict, List, Tuple

from django.conf import settings

from .utils import cumulative_distances_miles, project_point_onto_polyline_miles
from .models import Station

logger = logging.getLogger("routing")


def _route_bbox(coords: List[List[float]], margin_deg: float) -> Tuple[float, float, float, float]:
    """Return (min_lon, min_lat, max_lon, max_lat) with a margin in degrees."""
    lons = [c[0] for c in coords]
    lats = [c[1] for c in coords]
    return (
        min(lons) - margin_deg,
        min(lats) - margin_deg,
        max(lons) + margin_deg,
        max(lats) + margin_deg,
    )


def _downsample(coords: List[List[float]], cum: List[float], max_pts: int):
    """Return (sampled_coords, sampled_cum) with at most max_pts vertices,
    always keeping the first and last point."""
    n = len(coords)
    if n <= max_pts:
        return coords, cum
    step = max(1, (n - 1) // (max_pts - 1))
    indices = list(range(0, n, step))
    if indices[-1] != n - 1:
        indices.append(n - 1)
    return [coords[i] for i in indices], [cum[i] for i in indices]


class SimpleFuelPlanner:
    """Very simple fuel planner.

    Rules:
    - Only consider stations within CORRIDOR_MAX_DISTANCE_MILES of the route polyline.
    - First stop: may be any station within INITIAL_STOP_RADIUS_MILES of the route start.
    - Always keep enough fuel to reach the next stop or the destination,
      respecting MPG and MAX_RANGE_MILES.
    """

    def __init__(self, *, mpg: float, max_range_miles: float) -> None:
        self.mpg = mpg
        self.max_range_miles = max_range_miles

    def plan(self, route: Dict, start_empty: bool = False) -> Dict:
        geometry = route.get("geometry") or {}
        coords: List[List[float]] = geometry.get("coordinates") or []
        if not coords:
            # No geometry: nothing we can do, just return an empty plan.
            return {
                "route": {
                    "distance_miles": 0.0,
                    "duration_seconds": float(route.get("duration", 0.0)),
                    "geometry": geometry,
                },
                "stops": [],
                "fuel": {
                    "total_gallons": 0.0,
                    "total_cost": 0.0,
                    "mpg": self.mpg,
                    "max_range_miles": self.max_range_miles,
                    "start_empty": bool(start_empty),
                },
            }

        cum = cumulative_distances_miles(coords)
        total_distance_miles = cum[-1] if cum else 0.0

        corridor_limit = float(
            getattr(settings, "CORRIDOR_MAX_DISTANCE_MILES", 5.0) or 5.0
        )
        initial_radius = float(
            getattr(settings, "INITIAL_STOP_RADIUS_MILES", 5.0) or 5.0
        )

        # ── Performance: bounding-box pre-filter ──────────────────────
        # Convert corridor_limit miles to approximate degrees (~1° ≈ 69 mi)
        margin_deg = corridor_limit / 69.0 + 0.15  # generous margin
        bbox = _route_bbox(coords, margin_deg)

        # ── Performance: downsample polyline for projection ───────────
        # For long routes the polyline can have thousands of vertices.
        # Station projection is O(stations × vertices), so cap vertices.
        MAX_PROJ_PTS = 400
        proj_coords, proj_cum = _downsample(coords, cum, MAX_PROJ_PTS)

        # Project stations and build corridor candidates
        origin_lon, origin_lat = coords[0]
        candidates = []
        t_proj_start = time.perf_counter()

        # Fetch only stations inside the bounding box from the DB
        qs = Station.objects.filter(
            lon__isnull=False, lat__isnull=False,
            lon__gte=bbox[0], lon__lte=bbox[2],
            lat__gte=bbox[1], lat__lte=bbox[3],
        )
        bbox_count = 0
        for s in qs.iterator():
            bbox_count += 1
            lon = float(s.lon)
            lat = float(s.lat)
            try:
                mile_on_route, detour_miles = project_point_onto_polyline_miles(
                    proj_coords, proj_cum, [lon, lat]
                )
            except Exception:
                continue
            if detour_miles > corridor_limit:
                continue
            candidates.append(
                {
                    "station": s,
                    "mile_on_route": mile_on_route,
                    "detour_miles": detour_miles,
                    "price": float(s.price),
                    "lon": lon,
                    "lat": lat,
                }
            )

        logger.debug(
            "Station projection: %d in bbox → %d in corridor (%.2fs, polyline pts=%d)",
            bbox_count, len(candidates),
            time.perf_counter() - t_proj_start, len(proj_coords),
        )

        # Sort by position along the route
        candidates.sort(key=lambda c: c["mile_on_route"])
        # Pre-compute sorted mile markers for bisect lookups in the planner loop
        candidate_miles = [c["mile_on_route"] for c in candidates]

        mpg = self.mpg
        max_range = self.max_range_miles
        min_gallons_per_stop = float(
            getattr(settings, "MIN_GALLONS_PER_STOP", 0.0) or 0.0
        )

        current_mile = 0.0
        remaining_range = 0.0 if start_empty else max_range
        # Accounting: track total gallons purchased and their raw cost.
        total_purchased_gallons = 0.0
        total_purchase_cost = 0.0
        stops: List[Dict] = []

        # Helper to compute straight-line distance from origin (for first stop radius)
        def _distance_from_origin(lon: float, lat: float) -> float:
            rad = math.pi / 180.0
            x = (lon - origin_lon) * math.cos((lat + origin_lat) * 0.5 * rad)
            y = (lat - origin_lat)
            return math.hypot(x, y) * 69.0

        first_stop_done = False

        while current_mile + remaining_range + 1e-6 < total_distance_miles:
            # How far we can get from here with current fuel
            fuel_reach = current_mile + remaining_range
            # How far we could get if we had a full tank
            max_reach = min(current_mile + max_range, total_distance_miles)

            # Filter candidates we can physically reach from current_mile
            # Use bisect for O(log N) range lookup instead of scanning all candidates
            lo_idx = bisect.bisect_right(candidate_miles, current_mile + 1e-6)
            hi_idx = bisect.bisect_right(candidate_miles, max_reach + 1e-6)
            reachable = candidates[lo_idx:hi_idx]

            if not reachable:
                # Cannot reach any station, give up
                raise RuntimeError("No reachable fuel station found along the route within range.")

            # For the very first stop when starting empty, allow a radius
            # around origin; otherwise, just use corridor candidates.
            if not first_stop_done and start_empty:
                in_radius = [
                    c
                    for c in reachable
                    if _distance_from_origin(c["lon"], c["lat"]) <= initial_radius
                ]
                pool = in_radius or reachable
            else:
                pool = reachable

            # Sort by price (cheapest first)
            pool.sort(key=lambda c: c["price"])

            # Pre-filter: skip stations where we'd buy less than
            # MIN_GALLONS_PER_STOP, unless:
            #   - it's the only reachable station (safety),
            #   - remaining distance to destination is small enough that
            #     we genuinely need less than min_gallons_per_stop, or
            #   - we can't reach any "worthy" station with current fuel.
            min_purchase_miles = min_gallons_per_stop * mpg if min_gallons_per_stop > 0 else 0.0
            chosen = None
            for c in pool:
                dist_to_c = max(0.0, c["mile_on_route"] - current_mile)
                range_after_driving = remaining_range - dist_to_c
                dist_left_at_c = max(0.0, total_distance_miles - c["mile_on_route"])
                target_range_at_c = min(max_range, dist_left_at_c)
                need_miles_at_c = max(0.0, target_range_at_c - max(0.0, range_after_driving))
                gallons_at_c = need_miles_at_c / mpg if mpg > 0 else 0.0

                # If remaining distance is small, any purchase amount is fine
                # (this is effectively the "last stop" exception).
                if dist_left_at_c <= min_purchase_miles + 1e-6:
                    chosen = c
                    break

                # Accept this station if we'd buy at least min_gallons_per_stop
                if gallons_at_c >= min_gallons_per_stop - 1e-6:
                    chosen = c
                    break

            # Fallback: if no station meets the minimum purchase threshold,
            # pick the farthest reachable station (to delay stopping and
            # accumulate more need). This avoids tiny top-ups.
            if chosen is None:
                # Among stations we can actually reach with current fuel,
                # pick the farthest one; otherwise just take the cheapest.
                reachable_now = [
                    c for c in pool
                    if c["mile_on_route"] <= fuel_reach + 1e-6
                ]
                if reachable_now:
                    chosen = max(reachable_now, key=lambda c: c["mile_on_route"])
                else:
                    # We need fuel to reach any station; pick cheapest
                    chosen = pool[0]

            # Drive there
            dist_to_stop = max(0.0, chosen["mile_on_route"] - current_mile)
            # If we don't have enough fuel to reach it, fill just enough now
            if dist_to_stop - 1e-6 > remaining_range:
                # We need extra miles to reach this stop
                need_miles = min(max_range, dist_to_stop)
                add_miles = max(0.0, need_miles - remaining_range)
                gallons = add_miles / mpg
                price_here = chosen["price"]  # approximate with next stop price
                total_purchased_gallons += gallons
                total_purchase_cost += gallons * price_here
                remaining_range += add_miles

            remaining_range -= dist_to_stop
            current_mile = chosen["mile_on_route"]

            # Decide how much to buy at this stop: fill to max_range or
            # enough to reach destination, whichever is smaller.
            dist_left = max(0.0, total_distance_miles - current_mile)
            target_range = min(max_range, dist_left)
            need_miles = max(0.0, target_range - remaining_range)
            gallons_here = need_miles / mpg if mpg > 0 else 0.0

            # Enforce MIN_GALLONS_PER_STOP: if we'd buy less than the
            # minimum but the tank can hold more, bump up to the minimum.
            if min_gallons_per_stop > 0 and dist_left > min_purchase_miles:
                if 0 < gallons_here < min_gallons_per_stop - 1e-6:
                    # How many gallons can we actually add without exceeding
                    # max_range (i.e. tank capacity)?
                    max_addable_miles = max(0.0, max_range - remaining_range)
                    max_addable_gallons = max_addable_miles / mpg if mpg > 0 else 0.0
                    gallons_here = min(min_gallons_per_stop, max_addable_gallons)
                    need_miles = gallons_here * mpg
            cost_here = gallons_here * chosen["price"]

            total_purchased_gallons += gallons_here
            total_purchase_cost += cost_here
            remaining_range += need_miles

            s = chosen["station"]
            stop_entry = {
                "station_id": s.id,
                "name": s.name,
                "address": s.address,
                "city": s.city,
                "state": s.state,
                "price": float(chosen["price"]),
                "mile_on_route": float(chosen["mile_on_route"]),
                "detour_miles": float(chosen["detour_miles"]),
                "lon": float(chosen["lon"]),
                "lat": float(chosen["lat"]),
                "remaining_gallons_on_arrival": remaining_range / mpg - gallons_here,
                "gallons_purchased": gallons_here,
                "total_gallons_after_refuel": remaining_range / mpg,
                "cost": cost_here,
            }
            if not first_stop_done and start_empty:
                stop_entry["is_pre_trip"] = True
                first_stop_done = True

            stops.append(stop_entry)

        # Trip-level fuel economics.
        total_distance_miles = float(total_distance_miles)
        total_fuel_consumed = total_distance_miles / mpg if mpg > 0 else 0.0
        # If we bought more than we burned, allocate only the consumed share of
        # purchase cost to this trip. Remaining fuel (and cost) stays in tank
        # for the next trip.
        if total_purchased_gallons > 0 and total_fuel_consumed > 0:
            consumption_ratio = min(1.0, total_fuel_consumed / total_purchased_gallons)
            trip_fuel_cost = total_purchase_cost * consumption_ratio
        else:
            trip_fuel_cost = 0.0

        ending_fuel_gallons = max(0.0, total_purchased_gallons - total_fuel_consumed)

        return {
            "route": {
                "distance_miles": total_distance_miles,
                "duration_seconds": float(route.get("duration", 0.0)),
                "geometry": geometry,
            },
            "stops": stops,
            "fuel": {
                # Gallons actually BURNED on this trip based on distance & MPG.
                "total_fuel_consumed": total_fuel_consumed,
                # Gallons we PURCHASED at all stops for this trip.
                "total_purchased_gallons": total_purchased_gallons,
                # Effective trip fuel cost: only the share of purchase cost
                # corresponding to fuel actually consumed on this trip.
                "trip_fuel_cost": trip_fuel_cost,
                # Raw purchase cost (for reference / debugging).
                "total_purchase_cost": total_purchase_cost,
                # Fuel that remains in the tank at the end of this trip.
                "ending_fuel_gallons": ending_fuel_gallons,
                "mpg": self.mpg,
                "max_range_miles": self.max_range_miles,
                "start_empty": bool(start_empty),
            },
        }
