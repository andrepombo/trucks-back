import math
from typing import List, Tuple

# Coordinates are [lon, lat]


def haversine_miles(a: Tuple[float, float], b: Tuple[float, float]) -> float:
    lon1, lat1 = a
    lon2, lat2 = b
    R = 3958.7613  # Earth radius in miles
    phi1 = math.radians(lat1)
    phi2 = math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlambda = math.radians(lon2 - lon1)
    s = (
        math.sin(dphi / 2) ** 2
        + math.cos(phi1) * math.cos(phi2) * math.sin(dlambda / 2) ** 2
    )
    return 2 * R * math.asin(math.sqrt(s))


def cumulative_distances_miles(coords: List[List[float]]) -> List[float]:
    dists = [0.0]
    for i in range(1, len(coords)):
        d = haversine_miles(tuple(coords[i - 1]), tuple(coords[i]))
        dists.append(dists[-1] + d)
    return dists


def interpolate_point_at_distance(coords: List[List[float]], cum: List[float], d: float) -> List[float]:
    if d <= 0:
        return coords[0]
    if d >= cum[-1]:
        return coords[-1]
    # find segment where cum[i] <= d <= cum[i+1]
    lo, hi = 0, len(cum) - 1
    while lo < hi - 1:
        mid = (lo + hi) // 2
        if cum[mid] == d:
            return coords[mid]
        if cum[mid] < d:
            lo = mid
        else:
            hi = mid
    seg_len = cum[hi] - cum[lo]
    if seg_len <= 0:
        return coords[lo]
    t = (d - cum[lo]) / seg_len
    lon = coords[lo][0] + (coords[hi][0] - coords[lo][0]) * t
    lat = coords[lo][1] + (coords[hi][1] - coords[lo][1]) * t
    return [lon, lat]


def _deg_to_miles_scale(lat_ref: float) -> Tuple[float, float]:
    # Approximate miles per degree at given latitude
    miles_per_deg_lat = 69.0
    miles_per_deg_lon = 69.172 * math.cos(math.radians(lat_ref))
    return miles_per_deg_lon, miles_per_deg_lat


def project_point_onto_polyline_miles(
    coords: List[List[float]],
    cum: List[float],
    point: List[float],
) -> Tuple[float, float]:
    """Project a lon/lat point onto a polyline (lon/lat) and return:
    (mile_marker_along_route, roundtrip_detour_miles)

    The detour reported is an approximate out-and-back distance (2x the shortest
    lateral offset) to account for leaving and returning to the route.
    """
    px, py = point[0], point[1]
    best_mile = 0.0
    best_detour = float("inf")

    for i in range(len(coords) - 1):
        ax, ay = coords[i]
        bx, by = coords[i + 1]

        lat_ref = (ay + by) / 2.0
        m_lon, m_lat = _deg_to_miles_scale(lat_ref)

        # Convert to local XY in miles using A as origin
        ABx = (bx - ax) * m_lon
        ABy = (by - ay) * m_lat
        APx = (px - ax) * m_lon
        APy = (py - ay) * m_lat

        seg_len2 = ABx * ABx + ABy * ABy
        if seg_len2 <= 1e-12:
            continue
        t = (APx * ABx + APy * ABy) / seg_len2
        if t < 0.0:
            t_clamped = 0.0
        elif t > 1.0:
            t_clamped = 1.0
        else:
            t_clamped = t

        # Q = A + t*AB in local XY miles
        Qx = ABx * t_clamped
        Qy = ABy * t_clamped

        # Detour distance from P to Q in miles (local XY approx)
        dx = APx - Qx
        dy = APy - Qy
        detour = math.hypot(dx, dy)

        if detour < best_detour:
            seg_len_miles = max(0.0, cum[i + 1] - cum[i])
            mile_here = cum[i] + max(0.0, min(1.0, t_clamped)) * seg_len_miles
            best_detour = detour
            best_mile = mile_here

    if math.isinf(best_detour):
        # Fallback to start
        return 0.0, haversine_miles(tuple(point), tuple(coords[0])) * 2.0
    return best_mile, best_detour * 2.0
