import hashlib
import functools
import datetime
import uuid
import secrets
import os
from flask import request, g, jsonify
from backend.core.database import get_db
from backend.core.models import now_iso
from backend.core.config import VISITOR_COOKIE_NAME, ADMIN_ANALYTICS_COOKIE_NAME

def hash_token(token):
    return hashlib.sha256(token.encode("utf-8")).hexdigest()

def get_current_user():
    token = request.cookies.get("session_token")
    if not token:
        return None
    db = get_db()
    token_h = hash_token(token)
    row = db.execute(
        """
        SELECT u.*
        FROM sessions s
        JOIN users u ON u.id = s.user_id
        WHERE s.token_hash = ? AND s.expires_at > ? AND u.status = 'active'
        """,
        (token_h, now_iso()),
    ).fetchone()
    return row

def require_auth(view_fn):
    @functools.wraps(view_fn)
    def wrapper(*args, **kwargs):
        if g.current_user is None:
            return jsonify({"error": "Unauthorized"}), 401
        return view_fn(*args, **kwargs)
    return wrapper

def create_session_response(user_row):
    db = get_db()
    session_token = secrets.token_urlsafe(48)
    expires_dt = datetime.datetime.utcnow() + datetime.timedelta(days=7)
    expires = expires_dt.replace(microsecond=0).isoformat() + "Z"
    db.execute(
        """
        INSERT INTO sessions (id, user_id, token_hash, expires_at, created_at, user_agent, ip)
        VALUES (?, ?, ?, ?, ?, ?, ?)
        """,
        (
            str(uuid.uuid4()),
            user_row["id"],
            hash_token(session_token),
            expires,
            now_iso(),
            request.headers.get("User-Agent", ""),
            request.remote_addr or "",
        ),
    )
    db.commit()
    
    payload = {
        "id": user_row["id"], 
        "email": user_row["email"], 
        "display_name": user_row["display_name"]
    }
    resp = jsonify({"user": payload})
    max_age = 7 * 24 * 3600
    # In production you'd want secure=True if running on HTTPS.
    resp.set_cookie("session_token", session_token, httponly=True, samesite="Lax", secure=False, max_age=max_age)
    return resp

# Admin Analytics Auth Helpers
def get_admin_analytics_password():
    return (os.getenv("ADMIN_ANALYTICS_PASSWORD") or "").strip()

def admin_analytics_cookie_token():
    pwd = get_admin_analytics_password()
    if not pwd:
        return None
    return hashlib.sha256(f"lo_admin:{pwd}".encode("utf-8")).hexdigest()

def has_admin_analytics_access():
    expected = admin_analytics_cookie_token()
    if not expected:
        return False
    provided = request.cookies.get(ADMIN_ANALYTICS_COOKIE_NAME) or ""
    return secrets.compare_digest(provided, expected)

def require_admin_analytics(view_fn):
    @functools.wraps(view_fn)
    def wrapper(*args, **kwargs):
        if not get_admin_analytics_password():
            return jsonify({"error": "Admin analytics password is not configured"}), 503
        if not has_admin_analytics_access():
            return jsonify({"error": "Unauthorized"}), 401
        return view_fn(*args, **kwargs)
    return wrapper
