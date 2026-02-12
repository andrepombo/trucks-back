import time
from typing import Dict

from django.core.management.base import BaseCommand

from routing.models import Station
from routing.services import geocode_station_with_cache


class Command(BaseCommand):
    help = "Geocode stations without lon/lat and store results in DB (uses on-disk cache)."

    def add_arguments(self, parser):
        parser.add_argument("--state", dest="state", help="Filter by 2-letter state code")
        parser.add_argument(
            "--limit", dest="limit", type=int, default=0, help="Max stations to process (0 = all)"
        )
        parser.add_argument(
            "--sleep", dest="sleep", type=float, default=0.05, help="Sleep between requests (seconds)"
        )

    def handle(self, *args, **options):
        state = options.get("state")
        limit = int(options.get("limit") or 0)
        sleep = float(options.get("sleep") or 0.0)

        qs = Station.objects.filter(lon__isnull=True, lat__isnull=True)
        if state:
            qs = qs.filter(state=state.upper())
        total = qs.count()
        if limit and limit > 0:
            qs = qs[:limit]

        processed = 0
        updated = 0
        for st in qs.iterator():
            station_dict: Dict = {
                "id": st.id,
                "name": st.name,
                "address": st.address,
                "city": st.city,
                "state": st.state,
            }
            try:
                lonlat = geocode_station_with_cache(station_dict)
                if lonlat:
                    st.lon, st.lat = float(lonlat[0]), float(lonlat[1])
                    st.save(update_fields=["lon", "lat", "updated_at"])
                    updated += 1
            except Exception:
                pass
            processed += 1
            if sleep > 0:
                time.sleep(sleep)
        self.stdout.write(
            self.style.SUCCESS(
                f"Geocode complete: processed={processed}/{total} updated={updated}"
            )
        )
