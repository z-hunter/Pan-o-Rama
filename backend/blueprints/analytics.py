import datetime
from flask import Blueprint, request, jsonify, g
from backend.core.database import get_db
from backend.core.auth import require_admin_analytics
from backend.core.models import now_iso

analytics = Blueprint('analytics', __name__)

def build_analytics_summary_payload(period):
    db = get_db()
    now = datetime.datetime.utcnow()
    since_iso = None
    p = (period or "week").lower()
    if p == "month":
        since_iso = (now - datetime.timedelta(days=30)).isoformat() + "Z"
    elif p == "week":
        since_iso = (now - datetime.timedelta(days=7)).isoformat() + "Z"

    params = (since_iso,) if since_iso else ()
    
    tour_unique = db.execute(
        "SELECT COUNT(DISTINCT visitor_id) AS n FROM analytics_events WHERE event_type = 'tour_view'" 
        + (" AND created_at >= ?" if since_iso else ""),
        params
    ).fetchone()["n"] or 0
    
    tour_dur = db.execute(
        "SELECT SUM(duration_sec) AS s FROM analytics_events WHERE event_type = 'tour_view'"
        + (" AND created_at >= ?" if since_iso else ""),
        params
    ).fetchone()["s"] or 0
    
    home_unique = db.execute(
        "SELECT COUNT(DISTINCT visitor_id) AS n FROM analytics_events WHERE event_type = 'home_view'"
        + (" AND created_at >= ?" if since_iso else ""),
        params
    ).fetchone()["n"] or 0
    
    users_reg = db.execute(
        "SELECT COUNT(*) AS n FROM users" + (" WHERE created_at >= ?" if since_iso else ""),
        params
    ).fetchone()["n"] or 0
    
    tours_created = db.execute(
        "SELECT COUNT(*) AS n FROM tours" + (" WHERE created_at >= ?" if since_iso else ""),
        params
    ).fetchone()["n"] or 0
    
    checkouts = db.execute(
        "SELECT COUNT(*) AS n, COALESCE(SUM(amount_cents), 0) AS s FROM analytics_events WHERE event_type = 'checkout_completed'"
        + (" AND created_at >= ?" if since_iso else ""),
        params
    ).fetchone()

    return {
        "period": p,
        "tour_visitors_unique": int(tour_unique),
        "tour_time_total_sec": int(tour_dur),
        "home_visitors_unique": int(home_unique),
        "tours_created": int(tours_created),
        "users_registered": int(users_reg),
        "checkouts_count": int(checkouts["n"] or 0),
        "checkouts_amount_total_cents": int(checkouts["s"] or 0)
    }

@analytics.route("/summary", methods=["GET"])
@require_admin_analytics
def summary():
    return jsonify(build_analytics_summary_payload(request.args.get("period"))), 200

@analytics.route("/home", methods=["POST"])
def track_home():
    # Helper for home tracking via Beacon
    from backend.app import analytics_track
    analytics_track("home_view", visitor_id=g.visitor_id, user_id=g.current_user["id"] if g.current_user else None)
    return jsonify({"ok": True}), 200
