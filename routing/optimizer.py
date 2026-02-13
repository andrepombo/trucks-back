from __future__ import annotations

import math
from typing import Dict, List, Optional, Set, Tuple

from django.conf import settings

from .utils import cumulative_distances_miles, interpolate_point_at_distance, project_point_onto_polyline_miles
from .fuel_planning import FuelPlanner

# Types
# Route dict: {distance: m, duration: s, geometry: {type: LineString, coordinates: [[lon, lat], ...]}}


def plan_stops_and_costs(
    route: Dict,
    stations: List[Dict],
    mpg: float,
    max_range_miles: float,
    geocode_state_func,
    geocode_station_func,
    start_empty: bool = False,
) -> Dict:
    coords: List[List[float]] = route["geometry"]["coordinates"]
    cum = cumulative_distances_miles(coords)
    total_distance_miles = cum[-1]

    # Build a decimated polyline for fast projections (coarse but adequate for planning)
    def _decimate_for_projection(coords: List[List[float]], cum: List[float], step_miles: float = 2.0) -> Tuple[List[List[float]], List[float]]:
        if len(coords) <= 1000 and total_distance_miles <= 500.0:
            return coords, cum
        samples: List[float] = [0.0]
        d = step_miles
        while d < total_distance_miles:
            samples.append(d)
            d += step_miles
        if samples[-1] < total_distance_miles:
            samples.append(total_distance_miles)
        dcoords: List[List[float]] = []
        for s in samples:
            dcoords.append(interpolate_point_at_distance(coords, cum, s))
        return dcoords, samples

    proj_coords, proj_cum = _decimate_for_projection(coords, cum, step_miles=2.0)

    # Determine route states to narrow station pool and capture samples to avoid repeated reverse geocoding later
    route_states, state_samples = _estimate_route_states(coords, cum, geocode_state_func)

    # Create per-state sorted lists by price (ascending)
    state_index = _index_stations_by_state_price(stations, route_states)
    # Also keep an all-states index for final fallback
    all_state_index = _index_stations_by_state_price(stations, set())

    reserve_miles = 0.0
    total_with_reserve = total_distance_miles
    target_stop_count = max(1, math.ceil(total_with_reserve / max_range_miles))

    # Greedy plan: optionally start empty, otherwise start full.
    current_mile = 0.0
    remaining_miles = 0.0 if start_empty else max_range_miles

    selected_stops: List[Dict] = []
    total_gallons = 0.0
    total_cost = 0.0

    # Keep a cache of geocoded/projection results to avoid repeating
    projected_cache: Dict[str, Optional[Dict]] = {}

    planner = FuelPlanner(
        mpg=mpg,
        max_range_miles=max_range_miles,
        total_distance_miles=total_distance_miles,
        target_stop_count=target_stop_count,
        min_gallons_per_stop=float(getattr(settings, "MIN_GALLONS_PER_STOP", 0.0) or 0.0),
        leg_target_miles=float(getattr(settings, "LEG_TARGET_MILES", 450.0) or 450.0),
        find_candidate_stop=_find_candidate_stop,
        find_initial_fuel_stop=_find_initial_fuel_stop,
    )

    # If starting empty, first find a nearby station, then buy strategically (not always full)
    if start_empty:
        (
            current_mile,
            remaining_miles,
            total_gallons,
            total_cost,
        ) = planner.compute_initial_stop(
            current_mile=current_mile,
            remaining_miles=remaining_miles,
            coords=coords,
            cum=cum,
            proj_coords=proj_coords,
            proj_cum=proj_cum,
            state_index=state_index,
            all_state_index=all_state_index,
            projected_cache=projected_cache,
            geocode_station_func=geocode_station_func,
            state_samples=state_samples,
            selected_stops=selected_stops,
            total_gallons=total_gallons,
            total_cost=total_cost,
        )

    while current_mile + remaining_miles < total_distance_miles - 1e-6:
        target_mile = min(current_mile + settings.LEG_TARGET_MILES, total_distance_miles)
        # We can only reach up to current + remaining
        reach_mile = min(current_mile + remaining_miles, total_distance_miles)

        # Bias placement to minimize stop count across all plans
        min_mile_on_route: Optional[float] = None
        prefer_farthest = False
        stops_left_allowed_now = max(0, target_stop_count - len(selected_stops))
        # 1) Ensure it's even possible to finish in the remaining allowed stops: push next stop late enough
        if stops_left_allowed_now > 0:
            need_mile_for_feasible_finish = total_with_reserve - (stops_left_allowed_now * max_range_miles) + 0.01
            if need_mile_for_feasible_finish > current_mile + 1e-6:
                min_mile_on_route = max(min_mile_on_route or 0.0, need_mile_for_feasible_finish)
                prefer_farthest = True
        # 2) Enforce minimum fuel per stop by delaying the next stop until headroom >= MIN_GALLONS_PER_STOP (unless finishing)
        min_gallons_cfg = float(getattr(settings, "MIN_GALLONS_PER_STOP", 0.0) or 0.0)
        if min_gallons_cfg > 0.0:
            headroom_miles = max(0.0, max_range_miles - remaining_miles)
            need_headroom_miles = max(0.0, (min_gallons_cfg * mpg) - headroom_miles)
            if need_headroom_miles > 0.0:
                min_mile_for_min_gallons = current_mile + need_headroom_miles + 1e-3
                min_mile_on_route = max(min_mile_on_route or 0.0, min_mile_for_min_gallons)
                prefer_farthest = True

        candidate = _find_candidate_stop(
            current_mile=current_mile,
            reach_mile=reach_mile,
            target_mile=target_mile,
            coords=proj_coords,
            cum=proj_cum,
            state_index=state_index,
            projected_cache=projected_cache,
            geocode_station_func=geocode_station_func,
            state_samples=state_samples,
            min_mile_on_route=min_mile_on_route,
            prefer_farthest=prefer_farthest,
        )

        if not candidate:
            # Try broader search allowing up to max_range ahead
            reach_mile = min(current_mile + max_range_miles, total_distance_miles)
            candidate = _find_candidate_stop(
                current_mile=current_mile,
                reach_mile=reach_mile,
                target_mile=target_mile,
                coords=proj_coords,
                cum=proj_cum,
                state_index=state_index,
                projected_cache=projected_cache,
                geocode_station_func=geocode_station_func,
                state_samples=state_samples,
                widen_search=True,
                min_mile_on_route=min_mile_on_route,
                prefer_farthest=prefer_farthest,
            )

        if not candidate:
            # Final fallback: use full-resolution polyline and disable city prefilter
            reach_mile = min(current_mile + max_range_miles, total_distance_miles)
            candidate = _find_candidate_stop(
                current_mile=current_mile,
                reach_mile=reach_mile,
                target_mile=target_mile,
                coords=coords,
                cum=cum,
                state_index=all_state_index,
                projected_cache=projected_cache,
                geocode_station_func=geocode_station_func,
                state_samples=state_samples,
                widen_search=True,
                disable_city_prefilter=True,
                min_mile_on_route=min_mile_on_route,
                prefer_farthest=prefer_farthest,
            )
            if not candidate and min_mile_on_route is not None:
                # Try a widened detour window while still honoring the minimum mile
                candidate = _find_candidate_stop(
                    current_mile=current_mile,
                    reach_mile=reach_mile,
                    target_mile=target_mile,
                    coords=coords,
                    cum=cum,
                    state_index=all_state_index,
                    projected_cache=projected_cache,
                    geocode_station_func=geocode_station_func,
                    state_samples=state_samples,
                    widen_search=True,
                    disable_city_prefilter=True,
                    min_mile_on_route=min_mile_on_route,
                    prefer_farthest=True,
                    max_detour_miles=float(getattr(settings, "DETOUR_LIMIT_FALLBACK_MILES", 35.0)),
                )
            # If still none (likely because we can't reach the late-window yet), choose the farthest reachable candidate ignoring the bias
            if not candidate and min_mile_on_route is not None:
                candidate = _find_candidate_stop(
                    current_mile=current_mile,
                    reach_mile=min(current_mile + remaining_miles, total_distance_miles),
                    target_mile=target_mile,
                    coords=coords,
                    cum=cum,
                    state_index=all_state_index,
                    projected_cache=projected_cache,
                    geocode_station_func=geocode_station_func,
                    state_samples=state_samples,
                    widen_search=True,
                    disable_city_prefilter=True,
                    min_mile_on_route=None,
                    prefer_farthest=True,
                )

        if not candidate:
            # If we can reach the destination from here, allow finishing
            if remaining_miles >= (total_distance_miles - current_mile) - 1e-6:
                break
            raise RuntimeError(
                "No reachable fuel station found along the route within range."
            )

        # Move to candidate; if somehow out of range, retry keeping bias within current headroom, else pick farthest reachable
        dist_to_stop = max(0.0, candidate["mile_on_route"] - current_mile)
        if dist_to_stop - 1e-6 > remaining_miles:
            reach_mile = min(current_mile + remaining_miles, total_distance_miles)
            candidate_retry = _find_candidate_stop(
                current_mile=current_mile,
                reach_mile=reach_mile,
                target_mile=target_mile,
                coords=proj_coords,
                cum=proj_cum,
                state_index=state_index,
                projected_cache=projected_cache,
                geocode_station_func=geocode_station_func,
                state_samples=state_samples,
                widen_search=True,
                min_mile_on_route=min_mile_on_route,
            )
            if not candidate_retry:
                # As a last resort, choose the farthest reachable candidate ignoring the late-stop bias
                candidate_retry = _find_candidate_stop(
                    current_mile=current_mile,
                    reach_mile=reach_mile,
                    target_mile=target_mile,
                    coords=coords,
                    cum=cum,
                    state_index=all_state_index,
                    projected_cache=projected_cache,
                    geocode_station_func=geocode_station_func,
                    state_samples=state_samples,
                    widen_search=True,
                    disable_city_prefilter=True,
                    min_mile_on_route=None,
                    prefer_farthest=True,
                )
                if not candidate_retry:
                    raise RuntimeError("Chosen station is out of current remaining range")
            candidate = candidate_retry
            dist_to_stop = max(0.0, candidate["mile_on_route"] - current_mile)
            if dist_to_stop - 1e-6 > remaining_miles:
                raise RuntimeError("Chosen station is out of current remaining range")
        remaining_miles -= dist_to_stop
        current_mile = candidate["mile_on_route"]

        (
            remaining_miles,
            gallons_to_fill,
            cost_here,
            stop_entry,
        ) = planner.decide_next_stop_purchase(
            current_mile=current_mile,
            remaining_miles=remaining_miles,
            candidate=candidate,
            proj_coords=proj_coords,
            proj_cum=proj_cum,
            coords=coords,
            cum=cum,
            state_index=state_index,
            projected_cache=projected_cache,
            geocode_station_func=geocode_station_func,
            state_samples=state_samples,
            selected_stops_count=len(selected_stops),
        )

        total_gallons += gallons_to_fill
        total_cost += cost_here
        selected_stops.append(stop_entry)

    return {
        "route": {
            "distance_miles": total_distance_miles,
            "duration_seconds": float(route.get("duration", 0.0)),
            "geometry": route.get("geometry", {}),
        },
        "stops": selected_stops,
        "fuel": {
            "total_gallons": total_gallons,
            "total_cost": total_cost,
            "mpg": mpg,
            "max_range_miles": max_range_miles,
            "start_empty": bool(start_empty),
        },
    }

def _estimate_route_states(
    coords: List[List[float]], cum: List[float], geocode_state_func
) -> Tuple[Set[str], List[Tuple[float, str]]]:
    states: Set[str] = set()
    snapshots: List[Tuple[float, str]] = []
    total = cum[-1]
    # sample every 200 miles plus start and end
    step = 200.0
    samples = [0.0]
    m = step
    while m < total:
        samples.append(m)
        m += step
    samples.append(total)
    # Cap the total number of samples to avoid excessive reverse geocoding calls
    try:
        from django.conf import settings as _dj_settings  # lazy to avoid hard dep at import
        max_samples_mid = int(getattr(_dj_settings, "ROUTE_STATE_SAMPLE_MAX", 8))
    except Exception:
        max_samples_mid = 8
    # Keep first and last; thin the middle if needed
    if len(samples) > (max_samples_mid + 2):
        mid = samples[1:-1]
        keep = []
        # Evenly pick at most max_samples_mid points across mid
        for i in range(1, max_samples_mid + 1):
            idx = round((i * (len(mid) + 1)) / (max_samples_mid + 1)) - 1
            idx = max(0, min(idx, len(mid) - 1))
            keep.append(mid[idx])
        # De-duplicate while preserving order
        seen = set()
        thin_mid = []
        for v in keep:
            if v not in seen:
                thin_mid.append(v)
                seen.add(v)
        samples = [samples[0]] + thin_mid + [samples[-1]]
    for d in samples:
        p = interpolate_point_at_distance(coords, cum, d)
        try:
            code = geocode_state_func(p[0], p[1])
        except Exception:
            code = None
        if code:
            states.add(code)
            snapshots.append((d, code))
    return states, snapshots


def _index_stations_by_state_price(stations: List[Dict], states: Set[str]) -> Dict[str, List[Dict]]:
    # Filter to states on route if available; if empty fall back to all stations
    pool = [s for s in stations if (not states) or (s.get("state") in states)]
    index: Dict[str, List[Dict]] = {}
    for s in pool:
        st = s.get("state")
        if not st:
            continue
        index.setdefault(st, []).append(s)
    for st in list(index.keys()):
        index[st].sort(key=lambda x: (float(x.get("price", 1e9)), int(x.get("id", 1e9))))
    return index


def _find_candidate_stop(
    current_mile: float,
    reach_mile: float,
    target_mile: float,
    coords: List[List[float]],
    cum: List[float],
    state_index: Dict[str, List[Dict]],
    projected_cache: Dict[str, Optional[Dict]],
    geocode_station_func,
    state_samples: List[Tuple[float, str]],
    widen_search: bool = False,
    disable_city_prefilter: bool = False,
    prefer_nearest_cheaper_price: Optional[float] = None,
    min_mile_on_route: Optional[float] = None,
    prefer_farthest: bool = False,
    prefer_nearest: bool = False,
    max_detour_miles: Optional[float] = None,
) -> Optional[Dict]:
    # Build a prioritized list of state pools using pre-sampled route states
    def nearest_state(mile: float) -> Optional[str]:
        if not state_samples:
            return None
        best_code: Optional[str] = None
        best_diff = float("inf")
        for sm, code in state_samples:
            d = abs(sm - mile)
            if d < best_diff:
                best_diff = d
                best_code = code
        return best_code

    prioritize: List[str] = []
    st_target = nearest_state(target_mile)
    st_mid = nearest_state((current_mile + reach_mile) / 2.0)
    for st in (st_target, st_mid):
        if st and st not in prioritize:
            prioritize.append(st)

    # append remaining states
    for st in state_index.keys():
        if st not in prioritize:
            prioritize.append(st)

    # Limits for search breadth (prefer modest breadth; widened when needed)
    if widen_search and disable_city_prefilter:
        per_state_try = 20
        max_states = min(18, len(prioritize))
    else:
        per_state_try = 6 if not widen_search else 12
        max_states = min(7, len(prioritize)) if not widen_search else min(12, len(prioritize))

    best: Optional[Dict] = None
    # Track the best low-detour candidate within the same reach window.
    # This is used as a realistic "easy" baseline alternative to the chosen stop.
    best_low_detour: Optional[Dict] = None
    low_detour_limit = float(getattr(settings, "DETOUR_LOW_THRESHOLD_MILES", 6.0))
    nearest_cheaper: Optional[Dict] = None
    farthest: Optional[Dict] = None
    nearest: Optional[Dict] = None

    # Lazy import to avoid any potential circular import issues at module load
    from .services import geocode_city_with_cache

    for st in prioritize[:max_states]:
        pool = state_index.get(st, [])
        if not pool:
            continue
        tried = 0
        i = 0
        while tried < per_state_try and i < len(pool):
            station = pool[i]
            i += 1
            sid = str(station.get("id"))

            if sid in projected_cache:
                proj = projected_cache[sid]
            else:
                # Fast prefilter using city centroid to avoid expensive per-station geocoding
                city = station.get("city") or ""
                state = station.get("state") or ""
                pre_ok = True
                city_lonlat: Optional[Tuple[float, float]] = None
                c_mile = c_detour = None
                if not disable_city_prefilter and city and state:
                    city_lonlat = geocode_city_with_cache(city, state)
                    if city_lonlat:
                        c_lon, c_lat = city_lonlat
                        c_mile, c_detour = project_point_onto_polyline_miles(coords, cum, [c_lon, c_lat])
                        window_margin = 60.0 if not widen_search else 120.0
                        detour_thresh = 80.0 if not widen_search else 120.0
                        if not (current_mile - window_margin <= c_mile <= reach_mile + window_margin) or c_detour > detour_thresh:
                            pre_ok = False
                if not pre_ok:
                    projected_cache[sid] = None
                    continue

                # Precise station geocoding (skip if station already has lon/lat)
                lonlat = None
                if station.get("lon") is not None and station.get("lat") is not None:
                    try:
                        lonlat = (float(station["lon"]), float(station["lat"]))
                    except Exception:
                        lonlat = None
                if lonlat is None:
                    try:
                        lonlat = geocode_station_func(station)
                    except Exception:
                        lonlat = None

                if lonlat:
                    lon, lat = lonlat
                    mile_on_route, detour_miles = project_point_onto_polyline_miles(
                        coords, cum, [lon, lat]
                    )
                    proj = {
                        **station,
                        "lon": lon,
                        "lat": lat,
                        "mile_on_route": mile_on_route,
                        "detour_miles": detour_miles,
                    }
                    projected_cache[sid] = proj
                elif city_lonlat:
                    # Fallback to city centroid if station geocode failed
                    c_lon, c_lat = city_lonlat
                    if c_mile is None or c_detour is None:
                        c_mile, c_detour = project_point_onto_polyline_miles(coords, cum, [c_lon, c_lat])
                    proj = {
                        **station,
                        "lon": c_lon,
                        "lat": c_lat,
                        "mile_on_route": c_mile,
                        "detour_miles": float(c_detour),
                    }
                    projected_cache[sid] = proj
                else:
                    projected_cache[sid] = None
                    continue

            tried += 1
            if proj is None:
                continue
            if proj.get("detour_miles") is not None:
                detour_val = float(proj.get("detour_miles", 0.0) or 0.0)
                # Track the best low-detour option inside the same reach window as a baseline
                if (
                    current_mile + 1e-6 < proj["mile_on_route"] <= reach_mile + 1e-6
                    and detour_val <= low_detour_limit
                ):
                    if (best_low_detour is None) or (
                        detour_val < float(best_low_detour.get("detour_miles", float("inf")))
                        or (
                            math.isclose(detour_val, float(best_low_detour.get("detour_miles", float("inf"))), rel_tol=1e-3, abs_tol=1e-3)
                            and float(proj.get("price", float("inf"))) < float(best_low_detour.get("price", float("inf")))
                        )
                    ):
                        best_low_detour = proj
            # Accept if within reachable window and detour under threshold
            if widen_search and disable_city_prefilter:
                detour_limit = float(getattr(settings, "DETOUR_LIMIT_FALLBACK_MILES", 35.0))
            else:
                if not widen_search:
                    detour_limit = float(getattr(settings, "DETOUR_LIMIT_MILES", 12.0))
                else:
                    detour_limit = float(getattr(settings, "DETOUR_LIMIT_WIDEN_MILES", 20.0))
            if max_detour_miles is not None:
                detour_limit = max(detour_limit, float(max_detour_miles))
            if (
                current_mile + 1e-6 < proj["mile_on_route"] <= reach_mile + 1e-6
                and proj["detour_miles"] <= detour_limit
                and (min_mile_on_route is None or proj["mile_on_route"] + 1e-6 >= min_mile_on_route)
            ):
                # Track earliest station that is cheaper than the provided price (if requested)
                if (
                    prefer_nearest_cheaper_price is not None
                    and proj["price"] < prefer_nearest_cheaper_price
                    and (
                        nearest_cheaper is None
                        or proj["mile_on_route"] < nearest_cheaper["mile_on_route"]
                    )
                ):
                    nearest_cheaper = proj
                # Always track overall cheapest as fallback
                if (best is None) or (proj["price"] < best["price"]):
                    best = proj
                # Optionally track farthest acceptable candidate
                if (farthest is None) or (proj["mile_on_route"] > farthest["mile_on_route"]):
                    farthest = proj
                if (nearest is None) or (proj["mile_on_route"] < nearest["mile_on_route"]):
                    nearest = proj
    if prefer_farthest and farthest is not None:
        chosen = farthest
    elif prefer_nearest and nearest is not None:
        chosen = nearest
    else:
        chosen = nearest_cheaper if nearest_cheaper is not None else best
    if chosen is None:
        return None

    baseline_ctx: Optional[Dict] = None
    # Use best_low_detour only if it is strictly easier (lower detour) than the chosen stop
    # and lies within the same search window. Otherwise, fall back to treating the
    # chosen station itself as the baseline (no better easy option).
    try:
        chosen_detour = float(chosen.get("detour_miles", float("inf")) or float("inf"))
    except Exception:
        chosen_detour = float("inf")

    if (
        best_low_detour is not None
        and best_low_detour is not chosen
        and float(best_low_detour.get("detour_miles", float("inf")) or float("inf")) + 1e-6 < chosen_detour
        and current_mile + 1e-6 < float(best_low_detour.get("mile_on_route", -1e9)) <= reach_mile + 1e-6
    ):
        try:
            baseline_ctx = {
                "station_id": best_low_detour.get("id"),
                "name": best_low_detour.get("name"),
                "city": best_low_detour.get("city"),
                "state": best_low_detour.get("state"),
                "price": float(best_low_detour.get("price", 0.0) or 0.0),
                "detour_miles": float(best_low_detour.get("detour_miles", 0.0) or 0.0),
            }
        except Exception:
            baseline_ctx = None

    enriched = dict(chosen)
    if baseline_ctx is not None:
        enriched["_baseline_candidate"] = baseline_ctx
    return enriched


def _find_initial_fuel_stop(
    coords: List[List[float]],
    cum: List[float],
    state_index: Dict[str, List[Dict]],
    projected_cache: Dict[str, Optional[Dict]],
    geocode_station_func,
    state_samples: List[Tuple[float, str]],
) -> Optional[Dict]:
    """Find a good first stop near the route start (mile ~ 0) to fill from empty.

    Strategy: search states nearest to mile 0, evaluate stations whose projection is
    within a small window from start, choose the one with the smallest detour
    (tie-breaker by price).
    """
    # Lazy import; reuse helpers
    from .services import geocode_city_with_cache

    # Prioritize the state nearest to mile 0 and its neighbors in index
    prioritize: List[str] = []
    st_start: Optional[str] = None
    best_diff = float("inf")
    for sm, code in state_samples:
        d = abs(sm - 0.0)
        if d < best_diff:
            best_diff = d
            st_start = code
    if st_start:
        prioritize.append(st_start)
    for st in state_index.keys():
        if st not in prioritize:
            prioritize.append(st)

    # Progressive windows and detour thresholds near start
    windows = [5.0, 10.0, 20.0]
    detours = [15.0, 25.0, 40.0]

    best: Optional[Dict] = None
    best_low_detour: Optional[Dict] = None
    low_detour_limit = float(getattr(settings, "DETOUR_LOW_THRESHOLD_MILES", 6.0))

    for win_miles, detour_limit in zip(windows, detours):
        for st in prioritize[:8]:
            pool = state_index.get(st, [])
            for station in pool[:50]:  # limit per state pass
                sid = str(station.get("id"))
                proj = projected_cache.get(sid)
                if proj is None:
                    # try city prefilter and/or station geocode
                    city = station.get("city") or ""
                    state = station.get("state") or ""
                    lonlat = None
                    if station.get("lon") is not None and station.get("lat") is not None:
                        try:
                            lonlat = (float(station["lon"]), float(station["lat"]))
                        except Exception:
                            lonlat = None
                    if lonlat is None and city and state:
                        city_ll = geocode_city_with_cache(city, state)
                        if city_ll:
                            lonlat = city_ll
                    if lonlat is None:
                        try:
                            lonlat = geocode_station_func(station)
                        except Exception:
                            lonlat = None
                    if not lonlat:
                        projected_cache[sid] = None
                        continue
                    lon, lat = lonlat
                    mile_on_route, detour_miles = project_point_onto_polyline_miles(
                        coords, cum, [lon, lat]
                    )
                    proj = {
                        **station,
                        "lon": lon,
                        "lat": lat,
                        "mile_on_route": mile_on_route,
                        "detour_miles": detour_miles,
                    }
                    projected_cache[sid] = proj

                if proj is None:
                    continue
                # Track a low-detour baseline near the start window
                if proj.get("detour_miles") is not None:
                    detour_val = float(proj.get("detour_miles", 0.0) or 0.0)
                    if detour_val <= low_detour_limit and proj["mile_on_route"] <= win_miles:
                        if (best_low_detour is None) or (
                            detour_val < float(best_low_detour.get("detour_miles", float("inf")))
                            or (
                                math.isclose(detour_val, float(best_low_detour.get("detour_miles", float("inf"))), rel_tol=1e-3, abs_tol=1e-3)
                                and float(proj.get("price", float("inf"))) < float(best_low_detour.get("price", float("inf")))
                            )
                        ):
                            best_low_detour = proj

                if proj["mile_on_route"] <= win_miles and proj["detour_miles"] <= detour_limit:
                    if best is None:
                        best = proj
                    else:
                        # Prefer smaller detour; tie-breaker by lower price
                        if (proj["detour_miles"] < best["detour_miles"]) or (
                            math.isclose(proj["detour_miles"], best["detour_miles"], rel_tol=1e-3, abs_tol=1e-3)
                            and proj["price"] < best["price"]
                        ):
                            best = proj
        if best is not None:
            break
    if best is None:
        return None

    # For the very first stop, use a low-detour candidate near the start as baseline
    # only if it is strictly easier than the chosen stop. Otherwise, treat the chosen
    # station itself as the baseline.
    baseline_ctx: Optional[Dict] = None
    try:
        chosen_detour = float(best.get("detour_miles", float("inf")) or float("inf"))
    except Exception:
        chosen_detour = float("inf")

    if (
        best_low_detour is not None
        and best_low_detour is not best
        and float(best_low_detour.get("detour_miles", float("inf")) or float("inf")) + 1e-6 < chosen_detour
        and float(best_low_detour.get("mile_on_route", 0.0) or 0.0) <= max(windows)
    ):
        try:
            baseline_ctx = {
                "station_id": best_low_detour.get("id"),
                "name": best_low_detour.get("name"),
                "city": best_low_detour.get("city"),
                "state": best_low_detour.get("state"),
                "price": float(best_low_detour.get("price", 0.0) or 0.0),
                "detour_miles": float(best_low_detour.get("detour_miles", 0.0) or 0.0),
            }
        except Exception:
            baseline_ctx = None

    result = dict(best)
    if baseline_ctx is not None:
        result["_baseline_candidate"] = baseline_ctx
    return result
