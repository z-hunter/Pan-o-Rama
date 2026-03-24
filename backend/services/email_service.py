import os
import json
import urllib.request
import urllib.error
import datetime
import uuid
import secrets
from flask import request, current_app
from backend.core.config import (
    RESEND_API_KEY, MAIL_FROM, EMAIL_VERIFICATION_TTL_SEC, PASSWORD_RESET_TTL_SEC
)
from backend.core.database import get_db
from backend.core.auth import hash_token
from backend.core.models import now_iso

def app_base_url():
    # Attempt to get from env or request context
    base = os.getenv("APP_BASE_URL")
    if base:
        return base.strip().rstrip("/")
    try:
        return request.url_root.rstrip("/")
    except Exception:
        return "http://localhost:5000" # Fallback

def email_verification_enabled():
    raw = (os.getenv("EMAIL_VERIFICATION_ENABLED") or "1").strip().lower()
    return raw in {"1", "true", "yes", "on"}

def send_resend_email(recipient_email, subject, html):
    if not RESEND_API_KEY or not MAIL_FROM:
        current_app.logger.warning("Email send skipped: RESEND_API_KEY or MAIL_FROM is missing")
        return False, "mail_not_configured"
    payload = {
        "from": MAIL_FROM,
        "to": [recipient_email],
        "subject": subject,
        "html": html,
    }
    data = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(
        "https://api.resend.com/emails",
        data=data,
        method="POST",
        headers={
            "Authorization": f"Bearer {RESEND_API_KEY}",
            "Content-Type": "application/json",
            "Accept": "application/json",
            "User-Agent": "LokalnyObiektyw/1.0 (+https://lokalnyobiektyw.pl)",
        },
    )
    try:
        with urllib.request.urlopen(req, timeout=12) as resp:
            ok = 200 <= resp.status < 300
            if not ok:
                return False, f"resend_http_{resp.status}"
            return True, None
    except urllib.error.HTTPError as e:
        try:
            body = e.read().decode("utf-8", errors="ignore")
        except Exception:
            body = ""
        current_app.logger.warning("Resend HTTPError %s: %s", e.code, body[:500])
        return False, f"resend_http_{e.code}"
    except Exception as e:
        current_app.logger.warning("Resend send failed: %s", e)
        return False, "resend_send_failed"

def create_email_verification_token(user_id):
    token = secrets.token_urlsafe(48)
    token_hash = hash_token(token)
    ts = now_iso()
    expires = (datetime.datetime.utcnow() + datetime.timedelta(seconds=EMAIL_VERIFICATION_TTL_SEC)).replace(microsecond=0).isoformat() + "Z"
    db = get_db()
    db.execute(
        """
        INSERT INTO email_verification_tokens (id, user_id, token_hash, expires_at, used_at, created_at)
        VALUES (?, ?, ?, ?, NULL, ?)
        """,
        (str(uuid.uuid4()), user_id, token_hash, expires, ts),
    )
    db.commit()
    return token

def send_verification_email(recipient_email, token):
    verify_url = f"{app_base_url()}/auth/verify?token={token}"
    return send_resend_email(
        recipient_email,
        "Confirm your Lokalny Obiektyw account",
        (
            "<p>Welcome to Lokalny Obiektyw.</p>"
            "<p>Confirm your email to activate your account:</p>"
            f"<p><a href=\"{verify_url}\">{verify_url}</a></p>"
            "<p>This link expires in 24 hours.</p>"
        ),
    )

def create_password_reset_token(user_id):
    token = secrets.token_urlsafe(48)
    token_hash = hash_token(token)
    ts = now_iso()
    expires = (datetime.datetime.utcnow() + datetime.timedelta(seconds=PASSWORD_RESET_TTL_SEC)).replace(microsecond=0).isoformat() + "Z"
    db = get_db()
    db.execute(
        """
        INSERT INTO password_reset_tokens (id, user_id, token_hash, expires_at, used_at, created_at)
        VALUES (?, ?, ?, ?, NULL, ?)
        """,
        (str(uuid.uuid4()), user_id, token_hash, expires, ts),
    )
    db.commit()
    return token

def send_password_reset_email(recipient_email, token):
    reset_url = f"{app_base_url()}/reset-password?token={token}"
    return send_resend_email(
        recipient_email,
        "Reset your Lokalny Obiektyw password",
        (
            "<p>We received a request to reset your Lokalny Obiektyw password.</p>"
            f"<p><a href=\"{reset_url}\">{reset_url}</a></p>"
            "<p>This link expires in 1 hour. If you didn't request this, ignore this email.</p>"
        ),
    )

def queue_verification_email(user_id, email):
    token = create_email_verification_token(user_id)
    ok, err = send_verification_email(email, token)
    return ok, err
