import sqlite3
import os
from flask import g
from backend.core.config import DB_PATH, DATA_DIR, DEFAULT_PLAN_DEFS

def get_db():
    if "db" not in g:
        g.db = sqlite3.connect(DB_PATH)
        g.db.row_factory = sqlite3.Row
    return g.db

def close_db(_error=None):
    db = g.pop("db", None)
    if db is not None:
        db.close()

def init_db():
    # Ensure now_iso is imported here to avoid circular dependencies if any
    from backend.core.models import now_iso
    
    os.makedirs(DATA_DIR, exist_ok=True)
    db = sqlite3.connect(DB_PATH)
    try:
        db.executescript(
            """
            PRAGMA foreign_keys = ON;
            CREATE TABLE IF NOT EXISTS users (
                id TEXT PRIMARY KEY,
                email TEXT NOT NULL UNIQUE,
                password_hash TEXT NOT NULL,
                display_name TEXT NOT NULL,
                status TEXT NOT NULL DEFAULT 'active',
                is_admin INTEGER NOT NULL DEFAULT 0,
                email_verified INTEGER NOT NULL DEFAULT 1,
                email_verified_at TEXT,
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL
            );
            CREATE TABLE IF NOT EXISTS sessions (
                id TEXT PRIMARY KEY,
                user_id TEXT NOT NULL,
                token_hash TEXT NOT NULL,
                expires_at TEXT NOT NULL,
                created_at TEXT NOT NULL,
                user_agent TEXT,
                ip TEXT,
                FOREIGN KEY (user_id) REFERENCES users(id) ON DELETE CASCADE
            );
            CREATE TABLE IF NOT EXISTS tours (
                id TEXT PRIMARY KEY,
                owner_id TEXT NOT NULL,
                title TEXT NOT NULL,
                description TEXT NOT NULL DEFAULT '',
                slug TEXT NOT NULL UNIQUE,
                visibility TEXT NOT NULL CHECK(visibility IN ('public','private')),
                start_scene_id TEXT,
                start_pitch REAL,
                start_yaw REAL,
                default_hfov REAL,
                status TEXT NOT NULL DEFAULT 'draft',
                deleted_at TEXT,
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL,
                FOREIGN KEY (owner_id) REFERENCES users(id) ON DELETE CASCADE
            );
            CREATE TABLE IF NOT EXISTS scenes (
                id TEXT PRIMARY KEY,
                tour_id TEXT NOT NULL,
                title TEXT NOT NULL,
                panorama_path TEXT,
                audio_path TEXT,
                preview_path TEXT,
                images_json TEXT NOT NULL DEFAULT '[]',
                order_index INTEGER NOT NULL,
                haov REAL NOT NULL DEFAULT 360,
                vaov REAL NOT NULL DEFAULT 180,
                pos_x REAL,
                pos_y REAL,
                pos_z REAL,
                world_yaw REAL,
                scene_type TEXT NOT NULL DEFAULT 'equirectangular',
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL,
                FOREIGN KEY (tour_id) REFERENCES tours(id) ON DELETE CASCADE
            );
            CREATE TABLE IF NOT EXISTS hotspots (
                id TEXT PRIMARY KEY,
                tour_id TEXT NOT NULL,
                from_scene_id TEXT NOT NULL,
                to_scene_id TEXT NOT NULL,
                yaw REAL NOT NULL,
                pitch REAL NOT NULL,
                entry_yaw REAL,
                entry_pitch REAL,
                distance_m REAL,
                label TEXT,
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL,
                FOREIGN KEY (tour_id) REFERENCES tours(id) ON DELETE CASCADE,
                FOREIGN KEY (from_scene_id) REFERENCES scenes(id) ON DELETE CASCADE,
                FOREIGN KEY (to_scene_id) REFERENCES scenes(id) ON DELETE CASCADE
            );
            CREATE TABLE IF NOT EXISTS plans (
                id TEXT PRIMARY KEY,
                name TEXT NOT NULL,
                max_tours INTEGER NOT NULL,
                watermark_enabled INTEGER NOT NULL DEFAULT 1,
                price_monthly_cents INTEGER NOT NULL DEFAULT 0,
                is_active INTEGER NOT NULL DEFAULT 1,
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL
            );
            CREATE TABLE IF NOT EXISTS subscriptions (
                id TEXT PRIMARY KEY,
                user_id TEXT NOT NULL,
                plan_id TEXT NOT NULL,
                status TEXT NOT NULL DEFAULT 'active',
                billing_provider TEXT NOT NULL DEFAULT 'mock',
                provider_customer_id TEXT,
                provider_subscription_id TEXT,
                current_period_end TEXT,
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL,
                FOREIGN KEY (user_id) REFERENCES users(id) ON DELETE CASCADE,
                FOREIGN KEY (plan_id) REFERENCES plans(id)
            );
            CREATE TABLE IF NOT EXISTS usage_counters (
                user_id TEXT PRIMARY KEY,
                tours_count INTEGER NOT NULL DEFAULT 0,
                updated_at TEXT NOT NULL,
                FOREIGN KEY (user_id) REFERENCES users(id) ON DELETE CASCADE
            );
            CREATE TABLE IF NOT EXISTS billing_kv (
                key TEXT PRIMARY KEY,
                value TEXT NOT NULL,
                updated_at TEXT NOT NULL
            );
            CREATE TABLE IF NOT EXISTS jobs (
                id TEXT PRIMARY KEY,
                kind TEXT NOT NULL,
                owner_id TEXT NOT NULL,
                tour_id TEXT,
                scene_id TEXT,
                status TEXT NOT NULL,
                stage TEXT,
                progress_pct INTEGER NOT NULL DEFAULT 0,
                message TEXT,
                payload_json TEXT NOT NULL DEFAULT '{}',
                result_json TEXT,
                error TEXT,
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL,
                FOREIGN KEY (owner_id) REFERENCES users(id) ON DELETE CASCADE
            );
            CREATE TABLE IF NOT EXISTS analytics_events (
                id TEXT PRIMARY KEY,
                event_type TEXT NOT NULL,
                visitor_id TEXT NOT NULL,
                user_id TEXT,
                tour_id TEXT,
                path TEXT,
                duration_sec INTEGER,
                amount_cents INTEGER,
                currency TEXT,
                dedupe_key TEXT,
                meta_json TEXT NOT NULL DEFAULT '{}',
                created_at TEXT NOT NULL
            );
            CREATE TABLE IF NOT EXISTS tour_access_grants (
                id TEXT PRIMARY KEY,
                tour_id TEXT NOT NULL,
                user_id TEXT NOT NULL,
                created_at TEXT NOT NULL,
                FOREIGN KEY (tour_id) REFERENCES tours(id) ON DELETE CASCADE,
                FOREIGN KEY (user_id) REFERENCES users(id) ON DELETE CASCADE
            );
            CREATE TABLE IF NOT EXISTS email_verification_tokens (
                id TEXT PRIMARY KEY,
                user_id TEXT NOT NULL,
                token_hash TEXT NOT NULL,
                expires_at TEXT NOT NULL,
                used_at TEXT,
                created_at TEXT NOT NULL,
                FOREIGN KEY (user_id) REFERENCES users(id) ON DELETE CASCADE
            );
            CREATE TABLE IF NOT EXISTS password_reset_tokens (
                id TEXT PRIMARY KEY,
                user_id TEXT NOT NULL,
                token_hash TEXT NOT NULL,
                expires_at TEXT NOT NULL,
                used_at TEXT,
                created_at TEXT NOT NULL,
                FOREIGN KEY (user_id) REFERENCES users(id) ON DELETE CASCADE
            );
            CREATE INDEX IF NOT EXISTS idx_tours_owner ON tours(owner_id);
            CREATE INDEX IF NOT EXISTS idx_tours_slug ON tours(slug);
            CREATE INDEX IF NOT EXISTS idx_scenes_tour ON scenes(tour_id);
            CREATE INDEX IF NOT EXISTS idx_hotspots_scene ON hotspots(from_scene_id);
            CREATE INDEX IF NOT EXISTS idx_subscriptions_user ON subscriptions(user_id, created_at);
            CREATE INDEX IF NOT EXISTS idx_subscriptions_provider_sub ON subscriptions(provider_subscription_id);
            CREATE INDEX IF NOT EXISTS idx_jobs_owner ON jobs(owner_id, created_at);
            CREATE INDEX IF NOT EXISTS idx_jobs_status ON jobs(status, created_at);
            CREATE INDEX IF NOT EXISTS idx_analytics_type_time ON analytics_events(event_type, created_at);
            CREATE INDEX IF NOT EXISTS idx_analytics_visitor_time ON analytics_events(visitor_id, created_at);
            CREATE UNIQUE INDEX IF NOT EXISTS idx_analytics_dedupe ON analytics_events(dedupe_key);
            CREATE INDEX IF NOT EXISTS idx_tour_access_tour ON tour_access_grants(tour_id, created_at);
            CREATE INDEX IF NOT EXISTS idx_tour_access_user ON tour_access_grants(user_id, created_at);
            CREATE UNIQUE INDEX IF NOT EXISTS idx_tour_access_unique ON tour_access_grants(tour_id, user_id);
            CREATE INDEX IF NOT EXISTS idx_email_verify_user ON email_verification_tokens(user_id, created_at);
            CREATE UNIQUE INDEX IF NOT EXISTS idx_email_verify_token_hash ON email_verification_tokens(token_hash);
            CREATE INDEX IF NOT EXISTS idx_pwd_reset_user ON password_reset_tokens(user_id, created_at);
            CREATE UNIQUE INDEX IF NOT EXISTS idx_pwd_reset_token_hash ON password_reset_tokens(token_hash);
            """
        )
        
        # Migrations / Column Checks
        cols = [r[1] for r in db.execute("PRAGMA table_info(hotspots)").fetchall()]
        if "entry_yaw" not in cols:
            db.execute("ALTER TABLE hotspots ADD COLUMN entry_yaw REAL")
        if "entry_pitch" not in cols:
            db.execute("ALTER TABLE hotspots ADD COLUMN entry_pitch REAL")
        if "distance_m" not in cols:
            db.execute("ALTER TABLE hotspots ADD COLUMN distance_m REAL")
            
        scene_cols = [r[1] for r in db.execute("PRAGMA table_info(scenes)").fetchall()]
        if "preview_path" not in scene_cols:
            db.execute("ALTER TABLE scenes ADD COLUMN preview_path TEXT")
        if "audio_path" not in scene_cols:
            db.execute("ALTER TABLE scenes ADD COLUMN audio_path TEXT")
        if "pos_x" not in scene_cols:
            db.execute("ALTER TABLE scenes ADD COLUMN pos_x REAL")
        if "pos_y" not in scene_cols:
            db.execute("ALTER TABLE scenes ADD COLUMN pos_y REAL")
        if "pos_z" not in scene_cols:
            db.execute("ALTER TABLE scenes ADD COLUMN pos_z REAL")
        if "world_yaw" not in scene_cols:
            db.execute("ALTER TABLE scenes ADD COLUMN world_yaw REAL")
        if "processing_status" not in scene_cols:
            db.execute("ALTER TABLE scenes ADD COLUMN processing_status TEXT NOT NULL DEFAULT 'ready'")
        if "processing_error" not in scene_cols:
            db.execute("ALTER TABLE scenes ADD COLUMN processing_error TEXT")
        if "job_id" not in scene_cols:
            db.execute("ALTER TABLE scenes ADD COLUMN job_id TEXT")
            
        tour_cols = [r[1] for r in db.execute("PRAGMA table_info(tours)").fetchall()]
        if "start_scene_id" not in tour_cols:
            db.execute("ALTER TABLE tours ADD COLUMN start_scene_id TEXT")
        if "start_pitch" not in tour_cols:
            db.execute("ALTER TABLE tours ADD COLUMN start_pitch REAL")
        if "start_yaw" not in tour_cols:
            db.execute("ALTER TABLE tours ADD COLUMN start_yaw REAL")
        if "default_hfov" not in tour_cols:
            db.execute("ALTER TABLE tours ADD COLUMN default_hfov REAL")
            
        user_cols = [r[1] for r in db.execute("PRAGMA table_info(users)").fetchall()]
        if "email_verified" not in user_cols:
            db.execute("ALTER TABLE users ADD COLUMN email_verified INTEGER NOT NULL DEFAULT 1")
        if "email_verified_at" not in user_cols:
            db.execute("ALTER TABLE users ADD COLUMN email_verified_at TEXT")
            
        ts = now_iso()
        for plan_id, d in DEFAULT_PLAN_DEFS.items():
            db.execute(
                """
                INSERT OR IGNORE INTO plans (id, name, max_tours, watermark_enabled, price_monthly_cents, is_active, created_at, updated_at)
                VALUES (?, ?, ?, ?, ?, 1, ?, ?)
                """,
                (plan_id, d["name"], d["max_tours"], d["watermark_enabled"], d["price_monthly_cents"], ts, ts),
            )
        db.commit()
    finally:
        db.close()
