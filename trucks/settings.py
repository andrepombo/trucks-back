import os
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent

SECRET_KEY = os.getenv("DJANGO_SECRET_KEY", "dev-secret-key")
DEBUG = os.getenv("DEBUG", "1") == "1"
ALLOWED_HOSTS = ["0.0.0.0", "localhost", "127.0.0.1", "backend", "app3.andrepombo.info"]
CORS_ALLOWED_ORIGINS = os.getenv('CORS_ALLOWED_ORIGINS', 'http://localhost:3000,http://127.0.0.1:3000,http://localhost:5173,http://127.0.0.1:5173').split(',')


INSTALLED_APPS = [
    "django.contrib.admin",
    "django.contrib.auth",
    "django.contrib.contenttypes",
    "django.contrib.sessions",
    "django.contrib.messages",
    "django.contrib.staticfiles",
    "corsheaders",
    "routing",
]

MIDDLEWARE = [
    "corsheaders.middleware.CorsMiddleware",
    "django.middleware.security.SecurityMiddleware",
    "django.contrib.sessions.middleware.SessionMiddleware",
    "django.middleware.common.CommonMiddleware",
    "django.middleware.csrf.CsrfViewMiddleware",
    "django.contrib.auth.middleware.AuthenticationMiddleware",
    "django.contrib.messages.middleware.MessageMiddleware",
    "django.middleware.clickjacking.XFrameOptionsMiddleware",
]

ROOT_URLCONF = "trucks.urls"

TEMPLATES = [
    {
        "BACKEND": "django.template.backends.django.DjangoTemplates",
        "DIRS": [],
        "APP_DIRS": True,
        "OPTIONS": {
            "context_processors": [
                "django.template.context_processors.debug",
                "django.template.context_processors.request",
                "django.contrib.auth.context_processors.auth",
                "django.contrib.messages.context_processors.messages",
            ],
        },
    },
]

WSGI_APPLICATION = "trucks.wsgi.application"
ASGI_APPLICATION = "trucks.asgi.application"

# Use PostgreSQL if DATABASE_URL is set (Docker), otherwise use SQLite
if os.environ.get('DATABASE_URL'):
    DATABASES = {
        'default': dj_database_url.parse(os.environ.get('DATABASE_URL'))
    }
else:
    DATABASES = {
        'default': {
            'ENGINE': 'django.db.backends.sqlite3',
            'NAME': BASE_DIR / 'db.sqlite3',
        }
    }

AUTH_PASSWORD_VALIDATORS = []

LANGUAGE_CODE = "en-us"
TIME_ZONE = "UTC"
USE_I18N = True
USE_TZ = True

STATIC_URL = "static/"

DEFAULT_AUTO_FIELD = "django.db.models.BigAutoField"

# CORS
CORS_ALLOW_ALL_ORIGINS = True

# Fuel price data file
FUEL_PRICE_CSV = os.getenv(
    "FUEL_PRICE_CSV",
    str(BASE_DIR / "fuel-prices-for-be-assessment.csv"),
)

# External services
MPG = float(os.getenv("MPG", "10"))
MAX_RANGE_MILES = float(os.getenv("MAX_RANGE_MILES", "500"))
LEG_TARGET_MILES = float(os.getenv("LEG_TARGET_MILES", "450"))
RESERVE_GALLONS = float(os.getenv("RESERVE_GALLONS", "2"))

MIN_GALLONS_PER_STOP = float(os.getenv("MIN_GALLONS_PER_STOP", "8"))
DETOUR_LIMIT_MILES = float(os.getenv("DETOUR_LIMIT_MILES", "12"))
DETOUR_LIMIT_WIDEN_MILES = float(os.getenv("DETOUR_LIMIT_WIDEN_MILES", "20"))
DETOUR_LIMIT_FALLBACK_MILES = float(os.getenv("DETOUR_LIMIT_FALLBACK_MILES", "50"))

# Logging: show routing phase progress in terminal
LOGGING = {
    "version": 1,
    "disable_existing_loggers": False,
    "handlers": {
        "console": {
            "class": "logging.StreamHandler",
        }
    },
    "loggers": {
        "routing": {
            "handlers": ["console"],
            "level": "INFO",
            "propagate": False,
        },
        "django.request": {
            "handlers": ["console"],
            "level": "WARNING",
            "propagate": False,
        },
    },
}

# Geoapify configuration (preferred provider for geocoding+routing if API key set)
GEOAPIFY_API_KEY = os.getenv("GEOAPIFY_API_KEY", "2630f15e279e4a94a6c578574ecb94a9")
GEOAPIFY_GEOCODE_URL = os.getenv("GEOAPIFY_GEOCODE_URL", "https://api.geoapify.com/v1/geocode/search")
GEOAPIFY_REVERSE_URL = os.getenv("GEOAPIFY_REVERSE_URL", "https://api.geoapify.com/v1/geocode/reverse")
GEOAPIFY_ROUTING_URL = os.getenv("GEOAPIFY_ROUTING_URL", "https://api.geoapify.com/v1/routing")
