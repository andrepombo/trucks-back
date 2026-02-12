from django.db import models


class Station(models.Model):
    id = models.IntegerField(primary_key=True)
    name = models.CharField(max_length=200, blank=True)
    address = models.CharField(max_length=200, blank=True)
    city = models.CharField(max_length=120, blank=True)
    state = models.CharField(max_length=2, db_index=True)
    price = models.FloatField(db_index=True)
    lon = models.FloatField(null=True, blank=True, db_index=True)
    lat = models.FloatField(null=True, blank=True, db_index=True)
    updated_at = models.DateTimeField(auto_now=True)

    class Meta:
        indexes = [
            models.Index(fields=["state", "price"]),
        ]

    def __str__(self) -> str:
        return f"{self.id} {self.name} ({self.city}, {self.state})"
