from flask import Blueprint, request, jsonify, g
from werkzeug.security import generate_password_hash, check_password_hash

from backend.core.database import get_db
from backend.core.auth import require_auth
from backend.core.models import now_iso
from backend.services.billing_service import get_user_entitlements

users = Blueprint('users', __name__)

@users.route("/me", methods=["GET"])
@require_auth
def me_get():
    u = g.current_user
    ent = get_user_entitlements(u["id"])
    return jsonify(
        {
            "id": u["id"],
            "email": u["email"],
            "email_verified": bool(int(u["email_verified"] or 0)),
            "display_name": u["display_name"],
            "plan": {
                "id": ent["plan_id"],
                "name": ent["plan_name"],
            },
            "entitlements": {
                "max_tours": ent["max_tours"],
                "watermark_enabled": ent["watermark_enabled"],
                "remaining_tours": ent["remaining_tours"],
            },
            "usage": ent["usage"],
            "subscription": ent["subscription"],
        }
    ), 200

@users.route("/me", methods=["PATCH"])
@require_auth
def me_patch():
    data = request.get_json(silent=True) or {}
    display_name = (data.get("display_name") or "").strip()
    if not display_name:
        return jsonify({"error": "display_name is required"}), 400
    db = get_db()
    db.execute("UPDATE users SET display_name = ?, updated_at = ? WHERE id = ?", (display_name, now_iso(), g.current_user["id"]))
    db.commit()
    return jsonify({"message": "Profile updated"}), 200

@users.route("/me/password", methods=["PATCH"])
@require_auth
def me_password():
    data = request.get_json(silent=True) or {}
    current_password = data.get("current_password") or ""
    new_password = data.get("new_password") or ""
    if len(new_password) < 8:
        return jsonify({"error": "new_password must be at least 8 characters"}), 400
    db = get_db()
    user = db.execute("SELECT * FROM users WHERE id = ?", (g.current_user["id"],)).fetchone()
    if not check_password_hash(user["password_hash"], current_password):
        return jsonify({"error": "Current password is incorrect"}), 401
    db.execute("UPDATE users SET password_hash = ?, updated_at = ? WHERE id = ?", (generate_password_hash(new_password), now_iso(), user["id"]))
    db.commit()
    return jsonify({"message": "Password updated"}), 200
