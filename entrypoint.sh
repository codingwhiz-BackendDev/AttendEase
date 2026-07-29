#!/bin/sh

python manage.py migrate --noinput

python manage.py shell << EOF
from django.contrib.auth import get_user_model
User = get_user_model()
if not User.objects.filter(username='Admin').exists():
    User.objects.create_superuser(username='Admin', email='', password='admin')
    print("Superuser created: username=Admin, password=admin")
EOF

exec gunicorn AttendEase.wsgi:application --bind 0.0.0.0:8000 --workers 2
