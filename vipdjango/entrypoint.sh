#!/bin/sh
set -e

cd /app/vipdjango

python manage.py migrate --noinput
python manage.py collectstatic --noinput

PORT="${PORT:-8080}"
WORKERS="${GUNICORN_WORKERS:-3}"
TIMEOUT="${GUNICORN_TIMEOUT:-120}"

exec gunicorn vipson_manager.wsgi:application --bind "0.0.0.0:${PORT}" --workers "${WORKERS}" --timeout "${TIMEOUT}"
