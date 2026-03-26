import os

# Base Directories
# This file is located at backend/core/config.py
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
# REPO_ROOT should point to the root folder of the project
REPO_ROOT = os.path.abspath(os.path.join(BASE_DIR, "..", ".."))

DATA_DIR = os.path.join(REPO_ROOT, "data")
UPLOAD_FOLDER = os.path.join(DATA_DIR, "raw_uploads")
PROCESSED_FOLDER = os.path.join(DATA_DIR, "processed_galleries")
FRONTEND_FOLDER = os.path.join(REPO_ROOT, "frontend")
IMG_FOLDER = os.path.join(REPO_ROOT, "img")
DB_PATH = os.path.join(DATA_DIR, "app.db")

# File Config
ALLOWED_EXTENSIONS = {"png", "jpg", "jpeg", "gif", "webp"}
MAX_CONTENT_LENGTH = 1024 * 1024 * 1024 

# Filenames
PREVIEW_FILENAME = "preview.jpg"
WEB_PANO_FILENAME = "web.jpg"
STUDIO_WEB_PANO_FILENAME = "studio_web.jpg"
COVER_FILENAME = "cover.jpg"
COVER_META_FILENAME = "cover.meta.json"

# Auth Config
GALLERY_TEMPLATE_VERSION = 39
VISITOR_COOKIE_NAME = "lo_vid"
ADMIN_ANALYTICS_COOKIE_NAME = "lo_admin_analytics"
EMAIL_VERIFICATION_TTL_SEC = 24 * 60 * 60
PASSWORD_RESET_TTL_SEC = 60 * 60

# Plans
PLAN_FREE = "free"
PLAN_PRO = "pro"
PLAN_BUSINESS = "business"
PLAN_ORDER = {PLAN_FREE: 0, PLAN_PRO: 1, PLAN_BUSINESS: 2}
DEFAULT_PLAN_DEFS = {
    PLAN_FREE: {"name": "Free", "max_tours": 2, "watermark_enabled": 1, "price_monthly_cents": 0},
    PLAN_PRO: {"name": "Pro", "max_tours": 50, "watermark_enabled": 0, "price_monthly_cents": 4900},
    PLAN_BUSINESS: {"name": "Business", "max_tours": 500, "watermark_enabled": 0, "price_monthly_cents": 19900},
}

# External Services
RESEND_API_KEY = (os.getenv("RESEND_API_KEY") or "").strip()
MAIL_FROM = (os.getenv("MAIL_FROM") or "Lokalny Obiektyw <noreply@lokalnyobiektyw.pl>").strip()

STRIPE_SECRET_KEY = os.getenv("STRIPE_SECRET_KEY")
STRIPE_WEBHOOK_SECRET = os.getenv("STRIPE_WEBHOOK_SECRET")
BILLING_MODE = os.getenv("BILLING_MODE", "mock").lower()
ALLOW_MOCK_SUBSCRIBE = (os.getenv("ALLOW_MOCK_SUBSCRIBE") or "").strip() in {"1", "true", "yes", "on"}

STRIPE_PRICE_ID_PRO = os.getenv("STRIPE_PRICE_ID_PRO")
STRIPE_PRICE_ID_BUSINESS = os.getenv("STRIPE_PRICE_ID_BUSINESS")

# Redis / RQ
REDIS_URL = os.getenv("REDIS_URL", "redis://localhost:6379/0")
QUEUE_NAME = "default"

# Ensure core directories exist
os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(PROCESSED_FOLDER, exist_ok=True)
