from django.urls import path
from .views import health, plan_route, route_through_stops

urlpatterns = [
    path("health/", health, name="health"),
    path("plan-route/", plan_route, name="plan-route"),
    path("route-through-stops/", route_through_stops, name="route-through-stops"),
]
