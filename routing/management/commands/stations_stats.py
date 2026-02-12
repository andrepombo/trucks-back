from django.core.management.base import BaseCommand
from django.db.models import Count, Q

from routing.models import Station


class Command(BaseCommand):
    help = "Print per-state counts of stations with and without coordinates."

    def add_arguments(self, parser):
        parser.add_argument(
            "--state",
            dest="state",
            help="Optional 2-letter state code to limit the report",
        )

    def handle(self, *args, **options):
        state = options.get("state")
        qs = Station.objects.all()
        if state:
            qs = qs.filter(state=state.upper())

        agg = (
            qs.values("state")
            .annotate(
                total=Count("id"),
                with_coords=Count("id", filter=Q(lon__isnull=False) & Q(lat__isnull=False)),
                missing=Count("id", filter=Q(lon__isnull=True) | Q(lat__isnull=True)),
            )
            .order_by("state")
        )

        total_total = 0
        total_with = 0
        total_missing = 0

        header = f"{'State':<6} {'Total':>6} {'With':>6} {'Missing':>8} {'%With':>7}"
        self.stdout.write(header)
        self.stdout.write("-" * len(header))

        for row in agg:
            st = (row.get("state") or "").upper()
            total = int(row.get("total") or 0)
            with_c = int(row.get("with_coords") or 0)
            missing = int(row.get("missing") or 0)
            pct = (with_c / total * 100.0) if total else 0.0

            total_total += total
            total_with += with_c
            total_missing += missing

            self.stdout.write(f"{st:<6} {total:>6} {with_c:>6} {missing:>8} {pct:>6.1f}%")

        if not state:
            self.stdout.write("-" * len(header))
            pct_all = (total_with / total_total * 100.0) if total_total else 0.0
            self.stdout.write(
                self.style.SUCCESS(
                    f"TOTAL  {total_total:>6} {total_with:>6} {total_missing:>8} {pct_all:>6.1f}%"
                )
            )
