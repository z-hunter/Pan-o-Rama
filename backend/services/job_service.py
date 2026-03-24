import uuid
import json
from redis import Redis
from rq import Queue
from backend.core.config import REDIS_URL, QUEUE_NAME
from backend.core.database import get_db
from backend.core.models import now_iso

# Initialize Redis and RQ
_redis_conn = Redis.from_url(REDIS_URL)
_queue = Queue(QUEUE_NAME, connection=_redis_conn)

def enqueue_job(kind, owner_id, tour_id=None, scene_id=None, payload=None):
    """
    Creates a job record in DB and enqueues it in Redis.
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
    
    # 2. Enqueue in Redis
    # We pass the jid so the worker knows which DB record to update
    from backend.worker import process_job_task
    _queue.enqueue(process_job_task, jid)
    
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
