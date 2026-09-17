"""Isolated offline test configuration; never use this module to serve users."""
import os

os.environ.setdefault("DJANGO_SECRET_KEY", "local-tests-only-not-a-deployment-secret")
from .settings import *  # noqa: E402,F403

# Local settings may load the ignored env.txt file. Tests must remain offline;
# individual Speech Engine tests supply explicit placeholder credentials.
os.environ.pop("ELEVENLABS_API_KEY", None)
os.environ.pop("ELEVENLABS_SPEECH_ENGINE_ID", None)

DATABASES = {"default": {"ENGINE": "django.db.backends.sqlite3", "NAME": ":memory:"}}
SECURE_SSL_REDIRECT = False
SESSION_COOKIE_SECURE = False
CSRF_COOKIE_SECURE = False
STORAGES = {"staticfiles": {"BACKEND": "django.contrib.staticfiles.storage.StaticFilesStorage"}}
PASSWORD_HASHERS = ["django.contrib.auth.hashers.MD5PasswordHasher"]
