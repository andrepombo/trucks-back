from __future__ import annotations

import math
from typing import Callable, Dict, List, Optional, Tuple

from django.conf import settings


class FuelPlanner:
    def __init__(
        self,
        *,
        mpg: float,
        max_range_miles: float,
        total_distance_miles: float,
        target_stop_count: int,
        min_gallons_per_stop: float,
        leg_target_miles: float,
        find_candidate_stop: Callable[..., Optional[Dict]],
        find_initial_fuel_stop: Callable[..., Optional[Dict]],
    ) -> None:
        self.mpg = mpg
        self.max_range_miles = max_range_miles
        self.total_distance_miles = total_distance_miles
        self.target_stop_count = target_stop_count
        self.min_gallons_per_stop = min_gallons_per_stop
        self.leg_target_miles = leg_target_miles
        self.find_candidate_stop = find_candidate_stop
        self.find_initial_fuel_stop = find_initial_fuel_stop

    # ------------------------------------------------------------------
    # Initial stop helper
    # ------------------------------------------------------------------
    def compute_initial_stop(
        self,
        *,
        current_mile: float,
        remaining_miles: float,
        coords: List[List[float]],
        cum: List[float],
        proj_coords: List[List[float]],
        proj_cum: List[float],
        state_index: Dict[str, List[Dict]],
        all_state_index: Dict[str, List[Dict]],
        projected_cache: Dict[str, Optional[Dict]],
        geocode_station_func,
        state_samples,
        selected_stops: List[Dict],
        total_gallons: float,
        total_cost: float,
    ) -> Tuple[float, float, float, float]:
        init = self.find_initial_fuel_stop(
            coords=coords,
            cum=cum,
            state_index=all_state_index,
            projected_cache=projected_cache,
            geocode_station_func=geocode_station_func,
            state_samples=state_samples,
        )
        if not init:
            initial_window = float(getattr(settings, "INITIAL_SEARCH_MAX_MILES", 60.0) or 60.0)
            init = self.find_candidate_stop(
                current_mile=0.0,
                reach_mile=min(initial_window, self.total_distance_miles),
                target_mile=min(initial_window, self.total_distance_miles),
                coords=coords,
                cum=cum,
                state_index=all_state_index,
                projected_cache=projected_cache,
                geocode_station_func=geocode_station_func,
                state_samples=state_samples,
                widen_search=True,
                disable_city_prefilter=True,
                prefer_nearest=True,
            )
        if not init:
            raise RuntimeError("No starting fuel station found near the origin.")

        reach_from_here = min(current_mile + self.max_range_miles, self.total_distance_miles)
        late_threshold_mile = max(0.0, self.total_distance_miles - self.max_range_miles + 0.01)

        target_mile = min(current_mile + self.leg_target_miles, self.total_distance_miles)
        next_best = self.find_candidate_stop(
            current_mile=current_mile,
            reach_mile=reach_from_here,
            target_mile=target_mile,
            coords=proj_coords,
            cum=proj_cum,
            state_index=state_index,
            projected_cache=projected_cache,
            geocode_station_func=geocode_station_func,
            state_samples=state_samples,
            widen_search=True,
        )
        if not next_best:
            next_best = self.find_candidate_stop(
                current_mile=current_mile,
                reach_mile=reach_from_here,
                target_mile=target_mile,
                coords=coords,
                cum=cum,
                state_index=all_state_index,
                projected_cache=projected_cache,
                geocode_station_func=geocode_station_func,
                state_samples=state_samples,
                widen_search=True,
                disable_city_prefilter=True,
            )

        cheapest_within_reach = self.find_candidate_stop(
            current_mile=current_mile,
            reach_mile=reach_from_here,
            target_mile=target_mile,
            coords=coords,
            cum=cum,
            state_index=all_state_index,
            projected_cache=projected_cache,
            geocode_station_func=geocode_station_func,
            state_samples=state_samples,
            widen_search=True,
            disable_city_prefilter=True,
        )

        cheaper_ahead = self.find_candidate_stop(
            current_mile=current_mile,
            reach_mile=reach_from_here,
            target_mile=target_mile,
            coords=coords,
            cum=cum,
            state_index=all_state_index,
            projected_cache=projected_cache,
            geocode_station_func=geocode_station_func,
            state_samples=state_samples,
            widen_search=True,
            disable_city_prefilter=True,
            prefer_nearest_cheaper_price=init.get("price", float("inf")),
        )

        cheaper_late = None
        late_best = None
        if self.target_stop_count == 2:
            cheaper_late = self.find_candidate_stop(
                current_mile=current_mile,
                reach_mile=reach_from_here,
                target_mile=target_mile,
                coords=coords,
                cum=cum,
                state_index=all_state_index,
                projected_cache=projected_cache,
                geocode_station_func=geocode_station_func,
                state_samples=state_samples,
                widen_search=True,
                disable_city_prefilter=True,
                prefer_nearest_cheaper_price=init.get("price", float("inf")),
                min_mile_on_route=late_threshold_mile,
            )
            if not cheaper_late:
                late_best = self.find_candidate_stop(
                    current_mile=current_mile,
                    reach_mile=reach_from_here,
                    target_mile=target_mile,
                    coords=coords,
                    cum=cum,
                    state_index=all_state_index,
                    projected_cache=projected_cache,
                    geocode_station_func=geocode_station_func,
                    state_samples=state_samples,
                    widen_search=True,
                    disable_city_prefilter=True,
                    min_mile_on_route=late_threshold_mile,
                )
            if not cheaper_late and not late_best:
                late_best = self.find_candidate_stop(
                    current_mile=current_mile,
                    reach_mile=reach_from_here,
                    target_mile=target_mile,
                    coords=coords,
                    cum=cum,
                    state_index=all_state_index,
                    projected_cache=projected_cache,
                    geocode_station_func=geocode_station_func,
                    state_samples=state_samples,
                    widen_search=True,
                    disable_city_prefilter=True,
                    min_mile_on_route=late_threshold_mile,
                    prefer_farthest=True,
                    max_detour_miles=float(getattr(settings, "DETOUR_LIMIT_FALLBACK_MILES", 35.0)),
                )

        if self.total_distance_miles <= self.max_range_miles + 1e-6:
            need_miles = self.total_distance_miles
        elif self.target_stop_count == 2 and (cheaper_late or late_best):
            chosen = cheaper_late if cheaper_late else late_best
            goal_mile = min(float(chosen.get("mile_on_route", current_mile)), self.total_distance_miles)
            need_miles = max(0.0, goal_mile - current_mile)
            next_best = chosen
        elif self.target_stop_count == 2:
            need_miles = self.max_range_miles
        elif cheapest_within_reach and cheapest_within_reach.get("price", float("inf")) < init.get("price", float("inf")):
            goal_mile = min(float(cheapest_within_reach.get("mile_on_route", current_mile)), self.total_distance_miles)
            need_miles = max(0.0, goal_mile - current_mile)
            next_best = cheapest_within_reach
        elif cheaper_ahead:
            goal_mile = min(float(cheaper_ahead.get("mile_on_route", current_mile)), self.total_distance_miles)
            need_miles = max(0.0, goal_mile - current_mile)
            next_best = cheaper_ahead
        elif next_best and next_best.get("price", float("inf")) < init.get("price", float("inf")):
            goal_mile = min(float(next_best.get("mile_on_route", current_mile)), self.total_distance_miles)
            need_miles = max(0.0, goal_mile - current_mile)
        else:
            dist_left = max(0.0, self.total_distance_miles - current_mile)
            if dist_left <= self.max_range_miles + 1e-6:
                need_miles = dist_left
            else:
                need_miles = self.max_range_miles

        add_miles = max(0.0, need_miles - remaining_miles)
        add_miles_clamped = min(add_miles, self.max_range_miles - remaining_miles)

        finishes_trip = (remaining_miles + add_miles_clamped) >= (self.total_distance_miles - current_mile - 1e-6)
        add_miles_clamped = self._enforce_minimum_stop_size(
            add_miles_clamped=add_miles_clamped,
            remaining_miles=remaining_miles,
            finishes_trip=finishes_trip,
        )
        add_miles_clamped = self._apply_micro_stop_elimination(
            current_mile=current_mile,
            remaining_miles=remaining_miles,
            add_miles_clamped=add_miles_clamped,
            selected_stops_count=len(selected_stops),
            stops_left_allowed_after=max(0, self.target_stop_count - 1),
        )

        remaining_gallons_on_arrival = max(0.0, remaining_miles / self.mpg)
        gallons_to_fill, total_after = self._enforce_tank_capacity(
            remaining_gallons_on_arrival=remaining_gallons_on_arrival,
            add_miles_clamped=add_miles_clamped,
        )

        cost_here = gallons_to_fill * init["price"]
        remaining_miles = min(self.max_range_miles, remaining_miles + add_miles_clamped)

        stop_entry = {
            "station_id": init["id"],
            "name": init["name"],
            "address": init.get("address"),
            "city": init.get("city"),
            "state": init.get("state"),
            "price": init["price"],
            "mile_on_route": float(init.get("mile_on_route", 0.0)),
            "detour_miles": float(init.get("detour_miles", 0.0)),
            "remaining_gallons_on_arrival": remaining_gallons_on_arrival,
            "gallons_purchased": gallons_to_fill,
            "total_gallons_after_refuel": total_after,
            "cost": cost_here,
            "is_pre_trip": True,
        }
        selected_stops.append(stop_entry)

        current_mile = float(init.get("mile_on_route", 0.0))

        return current_mile, remaining_miles, total_gallons + gallons_to_fill, total_cost + cost_here

    # ------------------------------------------------------------------
    # Main loop helper
    # ------------------------------------------------------------------
    def decide_next_stop_purchase(
        self,
        *,
        current_mile: float,
        remaining_miles: float,
        candidate: Dict,
        proj_coords: List[List[float]],
        proj_cum: List[float],
        coords: List[List[float]],
        cum: List[float],
        state_index: Dict[str, List[Dict]],
        projected_cache: Dict[str, Optional[Dict]],
        geocode_station_func,
        state_samples,
        selected_stops_count: int,
    ) -> Tuple[float, float, float, Dict]:
        reach_from_here = min(current_mile + self.max_range_miles, self.total_distance_miles)
        target_mile = min(current_mile + self.leg_target_miles, self.total_distance_miles)
        next_best = self.find_candidate_stop(
            current_mile=current_mile,
            reach_mile=reach_from_here,
            target_mile=target_mile,
            coords=proj_coords,
            cum=proj_cum,
            state_index=state_index,
            projected_cache=projected_cache,
            geocode_station_func=geocode_station_func,
            state_samples=state_samples,
            widen_search=True,
        )

        dist_left = max(0.0, self.total_distance_miles - current_mile)
        stops_left_allowed = max(0, self.target_stop_count - selected_stops_count)
        force_finish = stops_left_allowed <= 1

        if dist_left <= self.max_range_miles + 1e-6:
            need_miles = dist_left
        elif force_finish:
            need_miles = dist_left
        elif next_best and next_best.get("price", float("inf")) < candidate.get("price", float("inf")):
            goal_mile = min(float(next_best.get("mile_on_route", current_mile)), self.total_distance_miles)
            need_miles = max(0.0, goal_mile - current_mile)
        else:
            if dist_left <= self.max_range_miles + 1e-6:
                need_miles = dist_left
            else:
                need_miles = self.max_range_miles

        add_miles = max(0.0, need_miles - remaining_miles)
        add_miles_clamped = min(add_miles, self.max_range_miles - remaining_miles)

        finishes_trip = (remaining_miles + add_miles_clamped) >= (self.total_distance_miles - current_mile - 1e-6)
        add_miles_clamped = self._enforce_minimum_stop_size(
            add_miles_clamped=add_miles_clamped,
            remaining_miles=remaining_miles,
            finishes_trip=finishes_trip,
        )
        add_miles_clamped = self._apply_micro_stop_elimination(
            current_mile=current_mile,
            remaining_miles=remaining_miles,
            add_miles_clamped=add_miles_clamped,
            selected_stops_count=selected_stops_count,
            stops_left_allowed_after=max(0, stops_left_allowed - 1),
        )

        def _stops_needed_after(purchase_miles: float) -> int:
            potential_remaining = remaining_miles + purchase_miles
            reachable = current_mile + potential_remaining
            dist_left_after = max(0.0, self.total_distance_miles - reachable)
            if dist_left_after <= 1e-6:
                return 0
            return int(math.ceil(dist_left_after / self.max_range_miles))

        stops_left_allowed_after = max(0, stops_left_allowed - 1)
        if (not force_finish) and (_stops_needed_after(add_miles_clamped) > stops_left_allowed_after):
            add_miles_clamped = self.max_range_miles - remaining_miles

        remaining_gallons_on_arrival = max(0.0, remaining_miles / self.mpg)
        gallons_to_fill, total_after = self._enforce_tank_capacity(
            remaining_gallons_on_arrival=remaining_gallons_on_arrival,
            add_miles_clamped=add_miles_clamped,
        )

        cost_here = gallons_to_fill * candidate["price"]
        remaining_miles = min(self.max_range_miles, remaining_miles + add_miles_clamped)

        stop_entry = {
            "station_id": candidate["id"],
            "name": candidate["name"],
            "address": candidate.get("address"),
            "city": candidate.get("city"),
            "state": candidate.get("state"),
            "price": candidate["price"],
            "lon": candidate["lon"],
            "lat": candidate["lat"],
            "mile_on_route": candidate["mile_on_route"],
            "detour_miles": candidate["detour_miles"],
            "remaining_gallons_on_arrival": remaining_gallons_on_arrival,
            "gallons_purchased": gallons_to_fill,
            "total_gallons_after_refuel": total_after,
            "cost": cost_here,
        }

        return remaining_miles, gallons_to_fill, cost_here, stop_entry

    # ------------------------------------------------------------------
    # Internal utilities
    # ------------------------------------------------------------------
    def _enforce_minimum_stop_size(
        self,
        *,
        add_miles_clamped: float,
        remaining_miles: float,
        finishes_trip: bool,
    ) -> float:
        if self.min_gallons_per_stop > 0.0 and not finishes_trip:
            min_add_miles = min(
                self.max_range_miles - remaining_miles,
                self.min_gallons_per_stop * self.mpg,
            )
            if add_miles_clamped + 1e-9 < min_add_miles:
                add_miles_clamped = min_add_miles
        return add_miles_clamped

    def _apply_micro_stop_elimination(
        self,
        *,
        current_mile: float,
        remaining_miles: float,
        add_miles_clamped: float,
        selected_stops_count: int,
        stops_left_allowed_after: int,
    ) -> float:
        micro_cap = (self.min_gallons_per_stop * self.mpg) if self.min_gallons_per_stop > 0.0 else 0.0

        micro_stops_left_allowed_after = max(0, (self.target_stop_count - selected_stops_count) - 1)
        if micro_stops_left_allowed_after == 0:
            dist_left_now = max(0.0, self.total_distance_miles - current_mile)
            planned_after = remaining_miles + add_miles_clamped
            shortfall = max(0.0, dist_left_now - planned_after)
            if micro_cap > 0.0 and 0.0 < shortfall <= micro_cap + 1e-6:
                add_miles_clamped = min(self.max_range_miles - remaining_miles, add_miles_clamped + shortfall)

        if stops_left_allowed_after == 0:
            dist_left_now = max(0.0, self.total_distance_miles - current_mile)
            planned_after = remaining_miles + add_miles_clamped
            shortfall = max(0.0, dist_left_now - planned_after)
            if micro_cap > 0.0 and 0.0 < shortfall <= micro_cap + 1e-6:
                add_miles_clamped = min(self.max_range_miles - remaining_miles, add_miles_clamped + shortfall)

        return add_miles_clamped

    def _enforce_tank_capacity(
        self,
        *,
        remaining_gallons_on_arrival: float,
        add_miles_clamped: float,
    ) -> Tuple[float, float]:
        gallons_to_fill = max(0.0, add_miles_clamped / self.mpg)
        max_tank_gallons = self.max_range_miles / self.mpg if self.mpg > 0 else float("inf")
        if max_tank_gallons < float("inf"):
            gallons_to_fill = min(gallons_to_fill, max(0.0, max_tank_gallons - remaining_gallons_on_arrival))
            total_after = min(remaining_gallons_on_arrival + gallons_to_fill, max_tank_gallons)
        else:
            total_after = remaining_gallons_on_arrival + gallons_to_fill
        return gallons_to_fill, total_after
