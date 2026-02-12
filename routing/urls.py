from django.urls import path
from .views import health, plan_route

urlpatterns = [
    path("health/", health, name="health"),
    path("plan-route/", plan_route, name="plan-route"),
]
