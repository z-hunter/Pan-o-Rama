import uuid
import datetime
from flask import Blueprint, request, jsonify, g, redirect, current_app
from werkzeug.security import generate_password_hash, check_password_hash

from backend.core.database import get_db
from backend.core.config import (
    PLAN_FREE, VISITOR_COOKIE_NAME, ADMIN_ANALYTICS_COOKIE_NAME
)
from backend.core.auth import (
    hash_token, create_session_response, require_auth, 
    require_admin_analytics, get_admin_analytics_password
)
from backend.core.models import now_iso
from backend.services.email_service import (
    email_verification_enabled, queue_verification_email,
    create_password_reset_token, send_password_reset_email
)

auth = Blueprint('auth', __name__)

# Note: We'll need a way to call set_user_plan which is still in app.py or move it to a service.
# For now, I'll assume it's moved to a service or I'll just do the DB insert here.

def set_user_plan_internal(user_id, plan_id, provider="mock"):
    db = get_db()
    ts = now_iso()
    db.execute(
        """
        INSERT OR REPLACE INTO subscriptions (id, user_id, plan_id, status, billing_provider, created_at, updated_at)
        VALUES (?, ?, ?, 'active', ?, ?, ?)
        """,
        (str(uuid.uuid4()), user_id, plan_id, provider, ts, ts),
    )
    db.commit()

@auth.route("/register", methods=["POST"])
def register():
    data = request.get_json(silent=True) or {}
    email = (data.get("email") or "").strip().lower()
    password = data.get("password") or ""
    display_name = (data.get("display_name") or email.split("@")[0] or "User").strip()
    if "@" not in email or len(password) < 8:
        return jsonify({"error": "Invalid email or password (min 8 chars)"}), 400
    db = get_db()
    exists = db.execute("SELECT id FROM users WHERE email = ?", (email,)).fetchone()
    if exists:
        return jsonify({"error": "Email already exists"}), 409
    uid = str(uuid.uuid4())
    ts = now_iso()
    verify_on = email_verification_enabled()
    email_verified = 0 if verify_on else 1
    email_verified_at = None if verify_on else ts
    db.execute(
        "INSERT INTO users (id, email, password_hash, display_name, email_verified, email_verified_at, created_at, updated_at) VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
        (uid, email, generate_password_hash(password), display_name or "User", email_verified, email_verified_at, ts, ts),
    )
    db.commit()
    
    set_user_plan_internal(uid, PLAN_FREE, provider="mock")
    
    if not verify_on:
        row = db.execute("SELECT * FROM users WHERE id = ?", (uid,)).fetchone()
        return create_session_response(row), 201
    
    ok, err = queue_verification_email(uid, email)
    return (
        jsonify(
            {
                "message": "Registration successful. Check your email to verify your account.",
                "email_sent": bool(ok),
                "email_error": err,
            }
        ),
        201,
    )

@auth.route("/login", methods=["POST"])
def login():
    data = request.get_json(silent=True) or {}
    email = (data.get("email") or "").strip().lower()
    password = data.get("password") or ""
    db = get_db()
    row = db.execute("SELECT * FROM users WHERE email = ?", (email,)).fetchone()
    if row is None or not check_password_hash(row["password_hash"], password):
        return jsonify({"error": "Invalid credentials"}), 401
    if email_verification_enabled() and int(row["email_verified"] or 0) != 1:
        return jsonify({"error": "Please verify your email before login", "code": "email_not_verified"}), 403
    return create_session_response(row), 200

@auth.route("/verification/resend", methods=["POST"])
def verification_resend():
    if not email_verification_enabled():
        return jsonify({"message": "Email verification is temporarily disabled"}), 200
    data = request.get_json(silent=True) or {}
    email = (data.get("email") or "").strip().lower()
    if "@" not in email:
        return jsonify({"error": "Invalid email"}), 400
    db = get_db()
    row = db.execute("SELECT id, email_verified FROM users WHERE email = ?", (email,)).fetchone()
    if row is None:
        return jsonify({"message": "If the account exists, verification email has been sent"}), 200
    if int(row["email_verified"] or 0) == 1:
        return jsonify({"message": "Email is already verified"}), 200
    ok, err = queue_verification_email(row["id"], email)
    return jsonify({"message": "Verification email sent", "email_sent": bool(ok), "email_error": err}), 200

@auth.route("/verify", methods=["GET"])
def verify():
    token = (request.args.get("token") or "").strip()
    if not token:
        return redirect("/login?verified=0&reason=missing_token", code=302)
    token_hash = hash_token(token)
    db = get_db()
    row = db.execute(
        """
        SELECT t.id AS token_id, t.user_id, t.expires_at, t.used_at, u.email_verified
        FROM email_verification_tokens t
        JOIN users u ON u.id = t.user_id
        WHERE t.token_hash = ?
        """,
        (token_hash,),
    ).fetchone()
    if row is None:
        return redirect("/login?verified=0&reason=invalid_token", code=302)
    if row["used_at"] is not None:
        return redirect("/login?verified=1", code=302)
    if row["expires_at"] <= now_iso():
        return redirect("/login?verified=0&reason=expired", code=302)
    ts = now_iso()
    db.execute(
        "UPDATE users SET email_verified = 1, email_verified_at = ?, updated_at = ? WHERE id = ?",
        (ts, ts, row["user_id"]),
    )
    db.execute("UPDATE email_verification_tokens SET used_at = ? WHERE id = ?", (ts, row["token_id"]))
    db.commit()
    return redirect("/login?verified=1", code=302)

@auth.route("/password/forgot", methods=["POST"])
def password_forgot():
    data = request.get_json(silent=True) or {}
    email = (data.get("email") or "").strip().lower()
    if "@" not in email:
        return jsonify({"error": "Invalid email"}), 400
    db = get_db()
    row = db.execute("SELECT id FROM users WHERE email = ? AND status = 'active'", (email,)).fetchone()
    if row is not None:
        try:
            token = create_password_reset_token(row["id"])
            send_password_reset_email(email, token)
        except Exception as e:
            current_app.logger.warning("Password reset email send failed: %s", e)
    return jsonify({"message": "If the account exists, a reset link has been sent"}), 200

@auth.route("/password/reset", methods=["POST"])
def password_reset():
    data = request.get_json(silent=True) or {}
    token = (data.get("token") or "").strip()
    new_password = data.get("new_password") or ""
    if not token:
        return jsonify({"error": "Missing token"}), 400
    if len(new_password) < 8:
        return jsonify({"error": "new_password must be at least 8 characters"}), 400
    token_hash = hash_token(token)
    db = get_db()
    row = db.execute(
        """
        SELECT id, user_id, expires_at, used_at
        FROM password_reset_tokens
        WHERE token_hash = ?
        """,
        (token_hash,),
    ).fetchone()
    if row is None:
        return jsonify({"error": "Invalid token"}), 400
    if row["used_at"] is not None:
        return jsonify({"error": "Token already used"}), 400
    if row["expires_at"] <= now_iso():
        return jsonify({"error": "Token expired"}), 400
    ts = now_iso()
    db.execute("UPDATE users SET password_hash = ?, updated_at = ? WHERE id = ?", (generate_password_hash(new_password), ts, row["user_id"]))
    db.execute("UPDATE password_reset_tokens SET used_at = ? WHERE id = ?", (ts, row["id"]))
    db.execute("DELETE FROM sessions WHERE user_id = ?", (row["user_id"],))
    db.commit()
    return jsonify({"message": "Password updated"}), 200

@auth.route("/logout", methods=["POST"])
def logout():
    token = request.cookies.get("session_token")
    if token:
        db = get_db()
        db.execute("DELETE FROM sessions WHERE token_hash = ?", (hash_token(token),))
        db.commit()
    resp = jsonify({"message": "Logged out"})
    resp.delete_cookie("session_token")
    return resp, 200
