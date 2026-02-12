 
import json
import os
import logging
from typing import Dict, List, Optional, Tuple

import requests
from django.conf import settings

logger = logging.getLogger("routing")


US_STATE_ABBR = {
    "Alabama": "AL",
    "Alaska": "AK",
    "Arizona": "AZ",
    "Arkansas": "AR",
    "California": "CA",
    "Colorado": "CO",
    "Connecticut": "CT",
    "Delaware": "DE",
    "District of Columbia": "DC",
    "Florida": "FL",
    "Georgia": "GA",
    "Hawaii": "HI",
    "Idaho": "ID",
    "Illinois": "IL",
    "Indiana": "IN",
    "Iowa": "IA",
    "Kansas": "KS",
    "Kentucky": "KY",
    "Louisiana": "LA",
    "Maine": "ME",
    "Maryland": "MD",
    "Massachusetts": "MA",
    "Michigan": "MI",
    "Minnesota": "MN",
    "Mississippi": "MS",
    "Missouri": "MO",
    "Montana": "MT",
    "Nebraska": "NE",
    "Nevada": "NV",
    "New Hampshire": "NH",
    "New Jersey": "NJ",
    "New Mexico": "NM",
    "New York": "NY",
    "North Carolina": "NC",
    "North Dakota": "ND",
    "Ohio": "OH",
    "Oklahoma": "OK",
    "Oregon": "OR",
    "Pennsylvania": "PA",
    "Rhode Island": "RI",
    "South Carolina": "SC",
    "South Dakota": "SD",
    "Tennessee": "TN",
    "Texas": "TX",
    "Utah": "UT",
    "Vermont": "VT",
    "Virginia": "VA",
    "Washington": "WA",
    "West Virginia": "WV",
    "Wisconsin": "WI",
    "Wyoming": "WY",
}


def _request_json(url: str, params: Dict[str, str]) -> Dict:
    r = requests.get(url, params=params, timeout=20)
    r.raise_for_status()
    return r.json()


def geocode_text(query: str) -> List[float]:
    """Return [lon, lat] for a free-form query within US using Geoapify."""
    if not getattr(settings, "GEOAPIFY_API_KEY", ""):
        raise RuntimeError("GEOAPIFY_API_KEY is not set")
    url = settings.GEOAPIFY_GEOCODE_URL
    params = {
        "text": query,
        "limit": 1,
        "filter": "countrycode:us",
        "format": "json",
        "apiKey": settings.GEOAPIFY_API_KEY,
    }
    data = _request_json(url, params)
    # Geoapify may return either a FeatureCollection or a results list depending on 'format'
    if isinstance(data, dict) and data.get("features"):
        feat = data["features"][0]
        geom = feat.get("geometry", {})
        coords = geom.get("coordinates")
        if coords and len(coords) >= 2:
            return [float(coords[0]), float(coords[1])]
    if isinstance(data, dict) and data.get("results"):
        res = data["results"][0]
        return [float(res["lon"]), float(res["lat"])]
    raise RuntimeError(f"No geocode result for: {query}")


def reverse_geocode_state_code(lon: float, lat: float) -> Optional[str]:
    # Normalize inputs to floats; accept (lon, lat) or [lon, lat]
    def normalize(a, b) -> Tuple[float, float]:
        try:
            if isinstance(a, (list, tuple)) and len(a) >= 2 and all(isinstance(v, (int, float)) for v in a[:2]):
                return float(a[0]), float(a[1])
            if isinstance(b, (list, tuple)) and len(b) >= 2 and all(isinstance(v, (int, float)) for v in b[:2]):
                return float(b[0]), float(b[1])
            return float(a), float(b)
        except Exception:
            # Last resort: pick first two numbers if present
            def pick(x, idx):
                if isinstance(x, (list, tuple)) and len(x) > idx:
                    return float(x[idx])
                return float(x)
            return pick(a, 0), pick(b, 1)

    lon_f, lat_f = normalize(lon, lat)
    logger.debug("Reverse geocoding state for lon=%.6f lat=%.6f", lon_f, lat_f)

    if not getattr(settings, "GEOAPIFY_API_KEY", ""):
        raise RuntimeError("GEOAPIFY_API_KEY is not set")
    url = settings.GEOAPIFY_REVERSE_URL
    params = {
        "lat": f"{lat_f:.6f}",
        "lon": f"{lon_f:.6f}",
        "format": "json",
        "apiKey": settings.GEOAPIFY_API_KEY,
    }
    data = _request_json(url, params)
    # Geoapify typically returns FeatureCollection
    features = data.get("features") if isinstance(data, dict) else None
    if features:
        props = features[0].get("properties", {})
        code = props.get("state_code") or props.get("state_code_iso")
        if code and len(code) == 2:
            return code.upper()
        state_name = props.get("state")
        if state_name and state_name in US_STATE_ABBR:
            return US_STATE_ABBR[state_name]
    return None


def route_directions(points: List[List[float]]) -> Dict:
    """Return a route dict using Geoapify Routing exclusively."""
    if len(points) < 2:
        raise ValueError("At least two points required for routing")
    if not getattr(settings, "GEOAPIFY_API_KEY", ""):
        raise RuntimeError("GEOAPIFY_API_KEY is not set")
    # Geoapify Routing API: try lon,lat first, then lat,lon as fallback
    url = settings.GEOAPIFY_ROUTING_URL
    attempts = []
    # attempt 1: lon,lat
    attempts.append("|".join([f"{lon:.6f},{lat:.6f}" for lon, lat in points]))
    # attempt 2: lat,lon
    attempts.append("|".join([f"{lat:.6f},{lon:.6f}" for lon, lat in points]))

    last_detail = None
    for waypoints in attempts:
        logger.debug("Geoapify routing: trying waypoints='%s'", waypoints)
        params = {
            "waypoints": waypoints,
            "mode": "drive",
            "apiKey": settings.GEOAPIFY_API_KEY,
        }
        data = _request_json(url, params)
        features = data.get("features") if isinstance(data, dict) else None
        if features:
            feat = features[0]
            break
        # collect detail for error reporting
        if isinstance(data, dict):
            last_detail = data.get("message") or data.get("error") or data.get("status")
            if not last_detail:
                try:
                    last_detail = json.dumps({k: data[k] for k in list(data.keys())[:3]})
                except Exception:
                    last_detail = str(data)[:200]
    else:
        raise RuntimeError(f"Geoapify routing returned no features: {last_detail}")

    props = feat.get("properties", {})
    geom = feat.get("geometry", {})
    distance_m = float(props.get("distance", props.get("distanceMeters", 0)) or 0)
    duration_s = float(props.get("time", props.get("duration", 0)) or 0)
    geometry = geom if geom else {"type": "LineString", "coordinates": []}
    # Flatten MultiLineString into a single LineString for downstream processing
    if isinstance(geometry, dict) and geometry.get("type") == "MultiLineString":
        lines = geometry.get("coordinates") or []
        flat: List[List[float]] = []
        for line in lines:
            # each line is a list of [lon, lat]
            if isinstance(line, list):
                flat.extend([pt for pt in line if isinstance(pt, (list, tuple)) and len(pt) >= 2])
        geometry = {"type": "LineString", "coordinates": flat}
    return {"distance": distance_m, "duration": duration_s, "geometry": geometry}


def _load_cache(path: str) -> Dict[str, Dict]:
    if not os.path.exists(path):
        return {}
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return {}


def _save_cache(path: str, cache: Dict[str, Dict]) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(cache, f)
    os.replace(tmp, path)


_GEOCODE_CACHE_PATH = os.path.join(settings.BASE_DIR, "data", "geocode_cache.json")
_GEOCODE_CACHE: Optional[Dict[str, Dict]] = None
_CITY_CACHE_PATH = os.path.join(settings.BASE_DIR, "data", "city_geocode_cache.json")
_CITY_CACHE: Optional[Dict[str, Dict]] = None


def geocode_station_with_cache(station: Dict) -> Optional[Tuple[float, float]]:
    """Geocode a station with a persistent JSON cache.

    Station dict should include keys: id, name, address, city, state
    Returns (lon, lat) or None
    """
    global _GEOCODE_CACHE
    if _GEOCODE_CACHE is None:
        _GEOCODE_CACHE = _load_cache(_GEOCODE_CACHE_PATH)
    cache = _GEOCODE_CACHE

    key = str(station.get("id"))
    if key in cache and "lon" in cache[key] and "lat" in cache[key]:
        return float(cache[key]["lon"]), float(cache[key]["lat"])

    # Build query strings (try a couple of variants)
    parts = [
        station.get("name") or "",
        station.get("address") or "",
        station.get("city") or "",
        station.get("state") or "",
        "USA",
    ]
    q1 = ", ".join([p for p in parts if p])
    try:
        lon, lat = geocode_text(q1)
    except Exception:
        # fallback to address, city, state
        parts2 = [station.get("address") or "", station.get("city") or "", station.get("state") or "", "USA"]
        q2 = ", ".join([p for p in parts2 if p])
        lon, lat = geocode_text(q2)

    cache[key] = {"lon": lon, "lat": lat}
    # Persist updated cache to disk
    try:
        _save_cache(_GEOCODE_CACHE_PATH, cache)
    except Exception:
        # Non-fatal if cache persistence fails
        logger.debug("Failed to persist geocode cache for station %s", key)
    # be polite; but keep short to avoid high latency. Adjust if needed.
    return lon, lat


def _geocode_first(query: str) -> Tuple[float, float]:
    # Legacy helper, use geocode_text for actual provider selection
    lon, lat = geocode_text(query)
    return lon, lat


def geocode_city_with_cache(city: str, state: str) -> Optional[Tuple[float, float]]:
    global _CITY_CACHE
    if not city or not state:
        return None
    if _CITY_CACHE is None:
        _CITY_CACHE = _load_cache(_CITY_CACHE_PATH)
    key = f"{city.strip().lower()},{state.strip().upper()}"
    if key in _CITY_CACHE and "lon" in _CITY_CACHE[key] and "lat" in _CITY_CACHE[key]:
        return float(_CITY_CACHE[key]["lon"]), float(_CITY_CACHE[key]["lat"])
    q = ", ".join([city, state, "USA"])
    try:
        lon, lat = geocode_text(q)
    except Exception:
        return None
    _CITY_CACHE[key] = {"lon": lon, "lat": lat}
    try:
        _save_cache(_CITY_CACHE_PATH, _CITY_CACHE)
    except Exception:
        logger.debug("Failed to persist city geocode cache for %s", key)
    return lon, lat
