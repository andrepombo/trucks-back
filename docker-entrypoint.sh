#!/bin/sh
set -e

export DJANGO_SETTINGS_MODULE=trucks.settings
export PATH="/usr/local/bin:$PATH"

echo "Running database migrations..."
python manage.py migrate --noinput

echo "Collecting static files..."
python manage.py collectstatic --noinput

echo "Starting Django server..."
exec "$@"
