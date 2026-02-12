import csv
from typing import Dict

from django.core.management.base import BaseCommand, CommandError
from django.conf import settings

from routing.models import Station


class Command(BaseCommand):
    help = "Import fuel stations from CSV into the database (dedup by id, keep lowest price)."

    def add_arguments(self, parser):
        parser.add_argument(
            "--csv",
            dest="csv_path",
            default=str(settings.FUEL_PRICE_CSV),
            help="Path to the fuel prices CSV (defaults to settings.FUEL_PRICE_CSV)",
        )

    def handle(self, *args, **options):
        path = options["csv_path"]
        try:
            best_by_id: Dict[str, Dict] = {}
            with open(path, "r", encoding="utf-8-sig", newline="") as f:
                reader = csv.DictReader(f)
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
                                "state": str(row.get(state_key, "")).strip().upper(),
                                "price": price,
                            }
                    except Exception:
                        continue
            if not best_by_id:
                raise CommandError("No rows parsed from CSV; verify columns and path")

            ids = [int(k) for k in best_by_id.keys()]
            existing = set(
                Station.objects.filter(id__in=ids).values_list("id", flat=True)
            )
            to_create = []
            to_update = []
            for s in best_by_id.values():
                if len(s.get("state", "")) != 2:
                    continue
                if s["id"] in existing:
                    to_update.append(
                        Station(
                            id=s["id"],
                            name=s.get("name", ""),
                            address=s.get("address", ""),
                            city=s.get("city", ""),
                            state=s.get("state", "").upper(),
                            price=s.get("price", 0.0),
                        )
                    )
                else:
                    to_create.append(
                        Station(
                            id=s["id"],
                            name=s.get("name", ""),
                            address=s.get("address", ""),
                            city=s.get("city", ""),
                            state=s.get("state", "").upper(),
                            price=s.get("price", 0.0),
                        )
                    )

            created = 0
            updated = 0
            if to_create:
                Station.objects.bulk_create(to_create, ignore_conflicts=True)
                created = len(to_create)
            if to_update:
                Station.objects.bulk_update(
                    to_update, ["name", "address", "city", "state", "price"]
                )
                updated = len(to_update)

            self.stdout.write(
                self.style.SUCCESS(
                    f"Import complete: created={created} updated={updated} (total unique={len(best_by_id)})"
                )
            )
        except FileNotFoundError:
            raise CommandError(f"CSV file not found: {path}")
        except Exception as e:
            raise CommandError(str(e))
