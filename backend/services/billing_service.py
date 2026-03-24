import uuid
import datetime
from flask import current_app, g
from backend.core.database import get_db
from backend.core.config import (
    PLAN_FREE, PLAN_ORDER, DEFAULT_PLAN_DEFS, PLAN_BUSINESS
)
from backend.core.models import now_iso

def get_plan_row(plan_id):
    db = get_db()
    return db.execute("SELECT * FROM plans WHERE id = ? AND is_active = 1", (plan_id,)).fetchone()

def get_active_subscription(user_id):
    db = get_db()
    return db.execute(
        """
        SELECT * FROM subscriptions
        WHERE user_id = ? AND status = 'active'
        ORDER BY created_at DESC
        LIMIT 1
        """,
        (user_id,),
    ).fetchone()

def set_user_plan(user_id, plan_id, provider="mock", status="active", provider_customer_id=None, provider_subscription_id=None, period_end=None):
    if plan_id not in PLAN_ORDER:
        plan_id = PLAN_FREE
    db = get_db()
    ts = now_iso()
    db.execute("UPDATE subscriptions SET status = 'canceled', updated_at = ? WHERE user_id = ? AND status = 'active'", (ts, user_id))
    sub_id = str(uuid.uuid4())
    db.execute(
        """
        INSERT INTO subscriptions (
            id, user_id, plan_id, status, billing_provider, provider_customer_id, provider_subscription_id, current_period_end, created_at, updated_at
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (sub_id, user_id, plan_id, status, provider, provider_customer_id, provider_subscription_id, period_end, ts, ts),
    )
    db.commit()
    return db.execute("SELECT * FROM subscriptions WHERE id = ?", (sub_id,)).fetchone()

def ensure_user_subscription(user_id):
    sub = get_active_subscription(user_id)
    if sub is not None:
        return sub
    return set_user_plan(user_id, PLAN_FREE, provider="mock")

def get_user_plan(user_id):
    sub = ensure_user_subscription(user_id)
    plan = get_plan_row(sub["plan_id"]) if sub is not None else None
    if plan is None:
        # Fallback safety for broken references.
        plan = get_plan_row(PLAN_FREE)
    return plan, sub

def compute_usage(user_id):
    db = get_db()
    row = db.execute(
        """
        SELECT COUNT(*) AS tours_count
        FROM tours
        WHERE owner_id = ? AND deleted_at IS NULL
        """,
        (user_id,),
    ).fetchone()
    count = int(row["tours_count"] if row else 0)
    db.execute(
        """
        INSERT INTO usage_counters (user_id, tours_count, updated_at)
        VALUES (?, ?, ?)
        ON CONFLICT(user_id) DO UPDATE SET tours_count = excluded.tours_count, updated_at = excluded.updated_at
        """,
        (user_id, count, now_iso()),
    )
    db.commit()
    return {"tours_count": count}

def get_user_entitlements(user_id):
    plan, sub = get_user_plan(user_id)
    usage = compute_usage(user_id)
    max_tours = int(plan["max_tours"]) if plan is not None else DEFAULT_PLAN_DEFS[PLAN_FREE]["max_tours"]
    watermark = bool(plan["watermark_enabled"]) if plan is not None else True
    remaining = max(0, max_tours - usage["tours_count"])
    return {
        "plan_id": plan["id"] if plan is not None else PLAN_FREE,
        "plan_name": plan["name"] if plan is not None else DEFAULT_PLAN_DEFS[PLAN_FREE]["name"],
        "max_tours": max_tours,
        "watermark_enabled": watermark,
        "usage": usage,
        "remaining_tours": remaining,
        "subscription": {
            "status": sub["status"] if sub is not None else "active",
            "billing_provider": sub["billing_provider"] if sub is not None else "mock",
            "current_period_end": sub["current_period_end"] if sub is not None else None,
        },
    }

def current_user_is_business():
    user = getattr(g, "current_user", None)
    if user is None:
        return False
    ent = get_user_entitlements(user["id"])
    return ent.get("plan_id") == PLAN_BUSINESS
