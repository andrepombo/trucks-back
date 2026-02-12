#!/bin/sh
set -e

export DJANGO_SETTINGS_MODULE=trucks.settings

echo "Running database migrations..."
python manage.py migrate --noinput

echo "Collecting static files..."
python manage.py collectstatic --noinput

echo "Starting Django server..."
exec "$@"
