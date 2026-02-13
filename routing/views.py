import json
import time
import logging
from django.http import JsonResponse
from django.views.decorators.csrf import csrf_exempt
from django.conf import settings
from .services import geocode_text, route_directions, reverse_geocode_state_code, geocode_station_with_cache
from .data_loader import load_fuel_prices
from .optimizer import plan_stops_and_costs

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
        # Add a concise summary to make the response easier to consume
        stops = plan.get("stops", [])
        route_info = plan.get("route", {})
        fuel_info = plan.get("fuel", {})
        distance_miles = float(route_info.get("distance_miles", 0.0))
        mpg_used = float(fuel_info.get("mpg", 1.0) or 1.0)
        total_gallons_purchased = float(fuel_info.get("total_gallons", 0.0))
        total_consumed = distance_miles / mpg_used
        start_empty_flag = bool(fuel_info.get("start_empty"))
        initial_fuel = 0.0 if start_empty_flag else float(fuel_info.get("max_range_miles", 0.0)) / mpg_used
        ending_fuel = initial_fuel + total_gallons_purchased - total_consumed
        total_detour_miles = sum(float(s.get("detour_miles", 0.0) or 0.0) for s in stops)
        total_distance_with_detours = distance_miles + total_detour_miles

        plan["summary"] = {
            "start": start,
            "finish": finish,
            "distance_miles": distance_miles,
            "total_detour_miles": total_detour_miles,
            "distance_with_detours_miles": total_distance_with_detours,
            "duration_hours": round(float(route_info.get("duration_seconds", 0.0)) / 3600.0, 2),
            "total_gallons": total_gallons_purchased,
            "total_fuel_consumed": total_consumed,
            "total_cost": float(fuel_info.get("total_cost", 0.0)),
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
            float(summ.get("total_gallons", 0.0)),
            float(summ.get("total_cost", 0.0)),
        )
        logger.debug(
            "Planning complete in %.2fs: %d stops, total_cost=$%.2f",
            time.perf_counter() - t2,
            len(plan.get("stops", [])),
            plan.get("fuel", {}).get("total_cost", 0.0),
        )
    except Exception as e:
        logger.exception("Planning failed: %s", e)
        return JsonResponse({"error": f"Planning failed: {e}"}, status=500)

    return JsonResponse(plan)
