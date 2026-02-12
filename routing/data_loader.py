import csv
from typing import Dict, List
from django.conf import settings


def load_fuel_prices() -> List[Dict]:
    """Load and deduplicate stations from the configured CSV.

    Returns list of dicts with keys:
    - id: int
    - name: str
    - address: str
    - city: str
    - state: str (2-letter)
    - price: float (per gallon)
    """
    # Try DB first for performance, then fall back to CSV
    try:
        from .models import Station  # type: ignore

        qs = Station.objects.all().values(
            "id",
            "name",
            "address",
            "city",
            "state",
            "price",
            "lon",
            "lat",
        )
        if qs.exists():
            stations: List[Dict] = []
            for r in qs:
                s: Dict = {
                    "id": int(r["id"]),
                    "name": r.get("name", "") or "",
                    "address": r.get("address", "") or "",
                    "city": r.get("city", "") or "",
                    "state": (r.get("state", "") or "").upper(),
                    "price": float(r.get("price", 0.0) or 0.0),
                }
                if r.get("lon") is not None and r.get("lat") is not None:
                    try:
                        s["lon"] = float(r["lon"])  # type: ignore
                        s["lat"] = float(r["lat"])  # type: ignore
                    except Exception:
                        pass
                stations.append(s)
            return stations
    except Exception:
        # DB not ready or other error; continue to CSV
        pass

    path = settings.FUEL_PRICE_CSV
    best_by_id: Dict[str, Dict] = {}
    with open(path, "r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        # Normalize expected headers
        id_key = "OPIS Truckstop ID"
        name_key = "Truckstop Name"
        addr_key = "Address"
        city_key = "City"
        state_key = "State"
        price_key = "Retail Price"
        for row in reader:
            try:
                sid = str(row[id_key]).strip()
                if not sid:
                    continue
                price = float(str(row[price_key]).strip())
                current = best_by_id.get(sid)
                if current is None or price < current["price"]:
                    best_by_id[sid] = {
                        "id": int(sid),
                        "name": str(row.get(name_key, "")).strip(),
                        "address": str(row.get(addr_key, "")).strip(),
                        "city": str(row.get(city_key, "")).strip(),
                        "state": str(row.get(state_key, "")).strip(),
                        "price": price,
                    }
            except Exception:
                # skip malformed rows
                continue
    # Filter to USA-like states (2-letter upper)
    stations: List[Dict] = []
    for s in best_by_id.values():
        st = s.get("state", "").strip()
        if len(st) == 2 and st.isalpha():
            s["state"] = st.upper()
            stations.append(s)
    return stations
