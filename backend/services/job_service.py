import uuid
import json
from backend.core.database import get_db
from backend.core.models import now_iso
from backend.core.config import REDIS_URL, QUEUE_NAME

_queue = None
_redis_conn = None

def get_queue():
    global _queue, _redis_conn
    if _queue is not None: return _queue
    try:
        from redis import Redis
        from rq import Queue
        _redis_conn = Redis.from_url(REDIS_URL)
        _queue = Queue(QUEUE_NAME, connection=_redis_conn)
        return _queue
    except Exception as e:
        print(f"Skipping RQ initialization: {e}")
        return None

def enqueue_job(kind, owner_id, tour_id=None, scene_id=None, payload=None):
    """
    Creates a job record in DB and enqueues it in Redis or synchronously if unavailable.
    """
    db = get_db()
    jid = str(uuid.uuid4())
    ts = now_iso()
    
    # 1. Create DB record
    db.execute(
        """
        INSERT INTO jobs (id, kind, owner_id, tour_id, scene_id, status, stage, progress_pct, payload_json, created_at, updated_at)
        VALUES (?, ?, ?, ?, ?, 'queued', 'initializing', 0, ?, ?, ?)
        """,
        (jid, kind, owner_id, tour_id, scene_id, json.dumps(payload or {}), ts, ts)
    )
    db.commit()
    
    # 2. Try RQ Enqueue
    q = get_queue()
    enqueued = False
    if q:
        try:
            from backend.worker import process_job_task
            q.enqueue(process_job_task, jid)
            enqueued = True
        except Exception as e:
            print(f"Failed to enqueue to RQ: {e}")
    
    if not enqueued:
        print("Running job asynchronously in a fallback thread...")
        import threading
        try:
            from flask import current_app
            app_obj = current_app._get_current_object()
            from backend.services.scene_service import process_scene_from_raw_paths
            # Explicitly import the exact function and manually wrap it to avoid hitting worker.py
            def sync_fallback(app, jid):
                # Ensure we run within application context since we're in a new thread
                with app.app_context():
                    try:
                        job = get_job(jid)
                        if not job: return
                        update_job_status(jid, status="running", stage="started", message="Job started safely")
                        payload = json.loads(job["payload_json"] or "{}")
                        if job["kind"] == "scene_process":
                            result = process_scene_from_raw_paths(
                                payload.get("tour_id"), payload.get("scene_id"), payload.get("name"),
                                bool(payload.get("is_pano")), payload.get("raw_paths") or [], int(payload.get("order_index") or 0), job_id=jid
                            )
                            update_job_status(jid, status="done", stage="complete", progress_pct=100, result={"scene": result})
                    except Exception as fallback_e:
                        print(f"Fallback sync processing failed: {fallback_e}")
                        import traceback
                        update_job_status(jid, status="failed", stage="error", error=traceback.format_exc())
            
            # Start the background thread
            t = threading.Thread(target=sync_fallback, args=(app_obj, jid,))
            t.daemon = True
            t.start()
        except Exception as kickoff_e:
            print(f"Failed to start fallback thread: {kickoff_e}")
            import traceback
            update_job_status(jid, status="failed", stage="error", error=traceback.format_exc())
    
    return jid

def update_job_status(jid, status=None, stage=None, progress_pct=None, message=None, result=None, error=None):
    """
    Updates a job record in DB.
    """
    db = get_db()
    updates = []
    params = []
    
    if status:
        updates.append("status = ?")
        params.append(status)
    if stage:
        updates.append("stage = ?")
        params.append(stage)
    if progress_pct is not None:
        updates.append("progress_pct = ?")
        params.append(int(progress_pct))
    if message:
        updates.append("message = ?")
        params.append(message)
    if result is not None:
        updates.append("result_json = ?")
        params.append(json.dumps(result))
    if error:
        updates.append("error = ?")
        params.append(str(error))
        
    if not updates:
        return
        
    updates.append("updated_at = ?")
    params.append(now_iso())
    params.append(jid)
    
    db.execute(f"UPDATE jobs SET {', '.join(updates)} WHERE id = ?", params)
    db.commit()

def get_job(jid):
    db = get_db()
    return db.execute("SELECT * FROM jobs WHERE id = ?", (jid,)).fetchone()
