#!/bin/sh

python manage.py migrate --noinput

exec gunicorn AttendEase.wsgi:application --bind 0.0.0.0:8000 --workers 2
