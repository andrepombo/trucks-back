import json
import time
import logging
from django.http import JsonResponse
from django.views.decorators.csrf import csrf_exempt
from django.conf import settings
from .services import geocode_text, route_directions, reverse_geocode_state_code, geocode_station_with_cache
from .data_loader import load_fuel_prices
from .optimizer import plan_stops_and_costs
from .utils import haversine_miles, cumulative_distances_miles

logger = logging.getLogger("routing")


def health(request):
    try:
        logger.debug("/api/health: loading fuel price CSV")
        stations = load_fuel_prices()
        ok = len(stations) > 0
        logger.debug("/api/health: stations loaded: %d", len(stations))
        return JsonResponse({"status": "ok" if ok else "empty", "stations": len(stations)})
    except Exception as e:
        logger.exception("/api/health: error: %s", e)
        return JsonResponse({"status": "error", "error": str(e)}, status=500)


def _polyline_extra_detour_miles(base_geom, detour_geom) -> float:
    """Approximate extra detour miles of detour_geom relative to base_geom.

    We walk along the detour polyline and, for each vertex, measure the
    distance to the closest vertex on the base polyline. Where this distance
    exceeds a small threshold, we treat that segment as "detour-only" and
    accumulate its length in miles.
    """
    try:
        base_coords = base_geom.get("coordinates") or []
        detour_coords = detour_geom.get("coordinates") or []
        if not base_coords or not detour_coords or len(detour_coords) < 2:
            return 0.0

        # Flatten MultiLineString if needed
        if isinstance(base_coords[0][0], (list, tuple)):
            flat: list = []
            for line in base_coords:
                if isinstance(line, list):
                    flat.extend([pt for pt in line if isinstance(pt, (list, tuple)) and len(pt) >= 2])
            base_coords = flat

        if isinstance(detour_coords[0][0], (list, tuple)):
            flat_d: list = []
            for line in detour_coords:
                if isinstance(line, list):
                    flat_d.extend([pt for pt in line if isinstance(pt, (list, tuple)) and len(pt) >= 2])
            detour_coords = flat_d

        if len(base_coords) < 2 or len(detour_coords) < 2:
            return 0.0

        # Hard safety cap: if either polyline is extremely detailed, skip the
        # O(N^2) comparison to avoid long API latencies.
        MAX_VERTICES = 2000
        if len(base_coords) > MAX_VERTICES or len(detour_coords) > MAX_VERTICES:
            logger.debug(
                "_polyline_extra_detour_miles: skipping detailed detour calc (base=%d, detour=%d)",
                len(base_coords),
                len(detour_coords),
            )
            return 0.0

        # Precompute cumulative distances along detour for segment lengths
        detour_cum = cumulative_distances_miles(detour_coords)

        # Threshold for "near" vs "far" in miles (~0.5 mile lateral offset)
        NEAR_THRESHOLD_MILES = 0.5

        extra_detour = 0.0
        for i in range(1, len(detour_coords)):
            p_prev = detour_coords[i - 1]
            p_cur = detour_coords[i]

            # Distance from current detour point to closest base vertex
            best = float("inf")
            for b in base_coords:
                d = haversine_miles((p_cur[0], p_cur[1]), (b[0], b[1]))
                if d < best:
                    best = d

            if best > NEAR_THRESHOLD_MILES:
                seg_len = max(0.0, detour_cum[i] - detour_cum[i - 1])
                extra_detour += seg_len

        return float(extra_detour)
    except Exception:
        return 0.0


@csrf_exempt
def plan_route(request):
    if request.method != "POST":
        return JsonResponse({"error": "POST required"}, status=405)
    try:
        body = json.loads(request.body.decode("utf-8"))
    except Exception:
        return JsonResponse({"error": "Invalid JSON body"}, status=400)

    start = body.get("start")
    finish = body.get("finish")
    if not start or not finish:
        return JsonResponse({"error": "start and finish are required"}, status=400)

    mpg = float(body.get("mpg", settings.MPG))
    max_range = float(body.get("max_range_miles", settings.MAX_RANGE_MILES))

    logger.debug(
        "/api/plan-route: start='%s' finish='%s' mpg=%.2f max_range=%.1f",
        start,
        finish,
        mpg,
        max_range,
    )

    try:
        t0 = time.perf_counter()
        logger.debug("Geocoding start location ...")
        start_ll = geocode_text(start)
        logger.debug("Geocoding finish location ...")
        finish_ll = geocode_text(finish)
        logger.debug(
            "Geocoding done in %.2fs: start=[%.5f, %.5f] finish=[%.5f, %.5f]",
            time.perf_counter() - t0,
            start_ll[0],
            start_ll[1],
            finish_ll[0],
            finish_ll[1],
        )
    except Exception as e:
        logger.exception("Geocoding failed: %s", e)
        return JsonResponse({"error": f"Geocoding failed: {e}"}, status=502)

    try:
        logger.debug("Requesting Geoapify route ...")
        t1 = time.perf_counter()
        route = route_directions([start_ll, finish_ll])
        logger.debug("Geoapify route received in %.2fs", time.perf_counter() - t1)
    except Exception as e:
        logger.exception("Routing failed: %s", e)
        return JsonResponse({"error": f"Routing failed: {e}"}, status=502)

    try:
        logger.debug("Loading fuel prices CSV ...")
        stations = load_fuel_prices()
        logger.debug("Loaded %d distinct stations", len(stations))
    except Exception as e:
        logger.exception("Fuel price file error: %s", e)
        return JsonResponse({"error": f"Fuel price file error: {e}"}, status=500)

    try:
        logger.debug("Planning fuel stops ...")
        t2 = time.perf_counter()
        plan = plan_stops_and_costs(
            route=route,
            stations=stations,
            mpg=mpg,
            max_range_miles=max_range,
            geocode_state_func=reverse_geocode_state_code,
            geocode_station_func=geocode_station_with_cache,
            start_empty=bool(body.get("start_empty")),
        )

        # Preserve the original base route (A->B) used for fuel planning
        base_route = plan.get("route") or {}

        # Build a waypoint list that goes through the selected fuel stops
        stops = plan.get("stops", [])
        waypoints = [list(start_ll)]
        for s in stops:
            try:
                lon = float(s.get("lon"))
                lat = float(s.get("lat"))
            except (TypeError, ValueError):
                continue
            waypoints.append([lon, lat])
        waypoints.append(list(finish_ll))

        # Ask Geoapify for a true multi‑waypoint route following the chosen stops.
        # If this fails for any reason, we fall back to the original base route.
        detour_route = None
        try:
            if len(waypoints) >= 2:
                logger.debug("Requesting Geoapify route with %d waypoints ...", len(waypoints))
                detour_route = route_directions(waypoints)
                # Ensure we expose duration in a consistent "duration_seconds" field
                if isinstance(detour_route, dict):
                    try:
                        d_s = float(detour_route.get("duration", 0.0) or 0.0)
                    except Exception:
                        d_s = 0.0
                    detour_route.setdefault("duration_seconds", d_s)
        except Exception as e:
            logger.exception("Multi‑waypoint routing failed; falling back to base route: %s", e)

        if isinstance(detour_route, dict):
            # Expose both versions to the frontend: base_route (original A->B)
            # and route (actual path via fuel stops).
            plan["base_route"] = {
                "distance": float(base_route.get("distance", 0.0) or 0.0),
                "duration": float(base_route.get("duration", 0.0) or 0.0),
                "distance_miles": float(base_route.get("distance_miles", 0.0) or 0.0),
                "duration_seconds": float(base_route.get("duration_seconds", 0.0) or 0.0),
                "geometry": base_route.get("geometry") or {},
            }
            plan["route"] = detour_route
            route_info = detour_route
        else:
            # No alternative route; keep only the base route
            plan["base_route"] = base_route
            route_info = base_route

        # Add a concise summary to make the response easier to consume.
        # Use the detour route for total distance if available, and also
        # approximate the extra detour miles relative to the base route.
        fuel_info = plan.get("fuel", {})

        base_geom = (plan.get("base_route") or {}).get("geometry") or {}
        route_geom = (plan.get("route") or {}).get("geometry") or {}

        # Total distance of the actual driven route (with stops), if present.
        detour_total_miles = float(route_info.get("distance_miles", 0.0))
        if detour_total_miles <= 0.0 and route_geom.get("coordinates"):
            try:
                detour_coords = route_geom.get("coordinates") or []
                if detour_coords and isinstance(detour_coords[0][0], (list, tuple)):
                    flat_d = []
                    for line in detour_coords:
                        if isinstance(line, list):
                            flat_d.extend([pt for pt in line if isinstance(pt, (list, tuple)) and len(pt) >= 2])
                    detour_coords = flat_d
                if len(detour_coords) >= 2:
                    detour_cum = cumulative_distances_miles(detour_coords)
                    detour_total_miles = float(detour_cum[-1])
            except Exception:
                detour_total_miles = 0.0

        # Total distance of the original base A->B route.
        base_total_miles = float((plan.get("base_route") or {}).get("distance_miles", 0.0) or 0.0)
        if base_total_miles <= 0.0 and base_geom.get("coordinates"):
            try:
                base_coords = base_geom.get("coordinates") or []
                if base_coords and isinstance(base_coords[0][0], (list, tuple)):
                    flat_b = []
                    for line in base_coords:
                        if isinstance(line, list):
                            flat_b.extend([pt for pt in line if isinstance(pt, (list, tuple)) and len(pt) >= 2])
                    base_coords = flat_b
                if len(base_coords) >= 2:
                    base_cum = cumulative_distances_miles(base_coords)
                    base_total_miles = float(base_cum[-1])
            except Exception:
                base_total_miles = 0.0

        # Approximate extra detour miles (red segments) compared to base.
        detour_extra_miles = 0.0
        try:
            if base_geom and route_geom and base_geom is not route_geom:
                detour_extra_miles = _polyline_extra_detour_miles(base_geom, route_geom)
        except Exception:
            detour_extra_miles = 0.0

        # New fuel accounting from SimpleFuelPlanner
        total_fuel_consumed = float(fuel_info.get("total_fuel_consumed", 0.0) or 0.0)
        total_purchased_gallons = float(fuel_info.get("total_purchased_gallons", 0.0) or 0.0)
        trip_fuel_cost = float(fuel_info.get("trip_fuel_cost", 0.0) or 0.0)
        ending_fuel = float(fuel_info.get("ending_fuel_gallons", 0.0) or 0.0)
        # Use the geometric extra detour miles as the main detour metric.
        total_detour_miles = float(detour_extra_miles)

        # Effective unique distance according to the user's spec:
        # distance_miles = green_base_only - shared_blue + red_detour_only.
        # With:
        #   base_total_miles ~= green_base_only + shared_blue
        #   detour_total_miles ~= shared_blue + red_detour_only
        #   total_detour_miles ~= red_detour_only
        # We derive:
        #   distance_miles = base_total_miles - detour_total_miles + 2 * total_detour_miles
        distance_miles_effective = base_total_miles - detour_total_miles + 2.0 * total_detour_miles

        # The full distance actually driven (with detours) is detour_total_miles.
        total_distance_with_detours = detour_total_miles

        plan["summary"] = {
            "start": start,
            "finish": finish,
            # Effective miles as defined above (green - blue + red)
            "distance_miles": distance_miles_effective,
            "total_detour_miles": total_detour_miles,
            # Total miles actually driven following the multi-waypoint route
            "distance_with_detours_miles": total_distance_with_detours,
            "duration_hours": round(float(route_info.get("duration_seconds", 0.0)) / 3600.0, 2),
            "total_gallons": total_purchased_gallons,
            "total_fuel_consumed": total_fuel_consumed,
            "total_cost": trip_fuel_cost,
            "ending_fuel_gallons": ending_fuel,
            "stops_count": len(stops),
        }
        # Respect response size flags
        if bool(body.get("no_geometry")) and isinstance(plan.get("route"), dict):
            plan["route"]["geometry"] = {"type": "LineString", "coordinates": []}
        if bool(body.get("summary_only")):
            return JsonResponse({"summary": plan.get("summary", {}), "stops": stops})
        # Print only fuel stops at INFO level
        if stops:
            logger.info("Fuel stops (%d):", len(stops))
            for idx, s in enumerate(stops, start=1):
                logger.info(
                    "#%d @ %.1f mi: %s — %s, %s | $%.3f/gal | +%.1f gal = $%.2f",
                    idx,
                    float(s.get("mile_on_route", 0.0)),
                    str(s.get("name", "")).strip(),
                    str(s.get("city", "")).strip(),
                    str(s.get("state", "")).strip(),
                    float(s.get("price", 0.0)),
                    float(s.get("gallons_purchased", 0.0)),
                    float(s.get("cost", 0.0)),
                )
        else:
            logger.info("No fuel stops selected (range/route may not require a stop).")
        # Always print a concise summary line at INFO level
        summ = plan.get("summary", {})
        logger.info(
            "Summary: distance=%.1f mi | duration=%.2f h | stops=%d | gallons=%.1f | cost=$%.2f",
            float(summ.get("distance_miles", 0.0)),
            float(summ.get("duration_hours", 0.0)),
            int(summ.get("stops_count", 0)),
            float(summ.get("total_fuel_consumed", 0.0)),
            float(summ.get("total_cost", 0.0)),
        )
        logger.debug(
            "Planning complete in %.2fs: %d stops, total_cost=$%.2f",
            time.perf_counter() - t2,
            len(plan.get("stops", [])),
            plan.get("fuel", {}).get("trip_fuel_cost", 0.0),
        )
    except Exception as e:
        logger.exception("Planning failed: %s", e)
        return JsonResponse({"error": f"Planning failed: {e}"}, status=500)

    return JsonResponse(plan)
