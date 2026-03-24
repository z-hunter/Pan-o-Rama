import os
import sys
import traceback
import json
from redis import Redis
from rq import Worker, Queue, Connection

# Add current dir to path to allow absolute imports
sys.path.append(os.getcwd())

from backend.core.config import REDIS_URL, QUEUE_NAME
from backend.core.database import get_db, close_db
from backend.core.models import now_iso
from backend.services.job_service import update_job_status, get_job
from backend.services.scene_service import process_scene_from_raw_paths

def process_job_task(jid):
    """
    Entrypoint for RQ worker.
    """
    # Create a minimal app-like context for DB access if needed
    # (Since we use get_db() which relies on Flask 'g', we might need to mock 'g' or use a different approach)
    # For background tasks, it's better to avoid 'g' and use direct DB connections.
    
    from flask import Flask, g
    app = Flask(__name__)
    with app.app_context():
        job = get_job(jid)
        if not job:
            print(f"Job {jid} not found in DB")
            return
            
        update_job_status(jid, status="running", stage="started", message="Job started by worker")
        payload = json.loads(job["payload_json"] or "{}")
        
        try:
            if job["kind"] == "scene_process":
                tour_id = payload.get("tour_id")
                scene_id = payload.get("scene_id")
                name = payload.get("name")
                is_pano = bool(payload.get("is_pano"))
                raw_paths = payload.get("raw_paths") or []
                order_index = int(payload.get("order_index") or 0)
                
                result = process_scene_from_raw_paths(
                    tour_id, scene_id, name, is_pano, raw_paths, order_index, job_id=jid
                )
                
                update_job_status(
                    jid, 
                    status="done", 
                    stage="complete", 
                    progress_pct=100, 
                    result={"scene": result}
                )
            else:
                raise ValueError(f"Unknown job kind: {job['kind']}")
                
        except Exception as e:
            err = f"{type(e).__name__}: {e}"
            tb = traceback.format_exc()
            update_job_status(
                jid, 
                status="failed", 
                stage="error", 
                error=err + "\n" + tb,
                message="Processing failed"
            )
            # Mark scene as failed too
            if job["kind"] == "scene_process" and payload.get("scene_id"):
                db = get_db()
                db.execute(
                    "UPDATE scenes SET processing_status = 'failed', processing_error = ?, updated_at = ? WHERE id = ?",
                    (err, now_iso(), payload.get("scene_id"))
                )
                db.commit()
            raise e

def main():
    print(f"Connecting to Redis at {REDIS_URL}...")
    redis_conn = Redis.from_url(REDIS_URL)
    with Connection(redis_conn):
        worker = Worker([QUEUE_NAME])
        print(f"Worker listening on queue: {QUEUE_NAME}")
        worker.work()

if __name__ == "__main__":
    main()
