import unittest
from unittest.mock import patch
import uuid
import time
import json
import os
from backend.app import app, create_app
from backend.services.job_service import enqueue_job, get_job
from backend.core.models import now_iso
from backend.core.database import get_db

class JobSystemTests(unittest.TestCase):
    def setUp(self):
        self.app = create_app()
        self.client = self.app.test_client()

        # Ensure we have a clean state for the test user
        self.user_id = str(uuid.uuid4())
        self.tour_id = str(uuid.uuid4())

        with self.app.app_context():
            db = get_db()
            ts = now_iso()
            db.execute("INSERT INTO users (id, email, password_hash, status, display_name, created_at, updated_at) VALUES (?, ?, ?, ?, ?, ?, ?)", 
                       (self.user_id, f"job_test_{uuid.uuid4().hex}@example.com", "hash", "active", "Tester", ts, ts))
            db.execute("INSERT INTO tours (id, owner_id, title, slug, visibility, created_at, updated_at) VALUES (?, ?, ?, ?, ?, ?, ?)",
                       (self.tour_id, self.user_id, "Job Test Tour", f"job-test-{uuid.uuid4().hex[:6]}", "public", ts, ts))
            db.commit()

        # Mock g.current_user for all requests
        self.patcher = unittest.mock.patch('backend.app.get_current_user')
        self.mock_get_user = self.patcher.start()
        self.mock_get_user.return_value = {"id": self.user_id, "email": "test@example.com"}

    def tearDown(self):
        self.patcher.stop()

    def test_job_enqueue_and_polling(self):
        with self.app.app_context():
            # 1. Enqueue a mock job (using fallback thread since Redis might not be running in CI/test)
            jid = enqueue_job(
                kind="scene_process",
                owner_id=self.user_id,
                tour_id=self.tour_id,
                payload={"name": "Test Scene", "raw_paths": []}
            )
            self.assertIsNotNone(jid)
            
            # 2. Check DB record immediately
            job = get_job(jid)
            self.assertEqual(job["status"], "queued")
            
            # 3. Test API endpoint for polling
            res = self.client.get(f"/jobs/{jid}")
            self.assertEqual(res.status_code, 200)
            data = res.get_json()
            self.assertEqual(data["job"]["id"], jid)
            self.assertEqual(data["job"]["tour_id"], self.tour_id)
            self.assertIn("scene_id", data["job"])
            self.assertIn("kind", data["job"])
        
    def test_fallback_thread_execution(self):
        with self.app.app_context():
            # This test verifies that the job eventually reaches a terminal state via the fallback thread
            # Note: scene_process with empty paths will likely fail or do nothing, 
            # but it should still update the status to something other than 'queued'.
            
            jid = enqueue_job(
                kind="scene_process",
                owner_id=self.user_id,
                tour_id=self.tour_id,
                payload={"name": "Async Test", "raw_paths": []}
            )
            
            # Wait for fallback thread to pick it up and process (timeout after 5s)
            max_wait = 5
            start_time = time.time()
            final_status = "queued"
            
            while time.time() - start_time < max_wait:
                res = self.client.get(f"/jobs/{jid}")
                data = res.get_json()
                if not data or "job" not in data:
                    time.sleep(0.5)
                    continue
                final_status = data["job"]["status"]
                if final_status in ["done", "failed"]:
                    break
                time.sleep(0.5)
                
            self.assertIn(final_status, ["done", "failed"], "Job should have finished in fallback thread")

if __name__ == "__main__":
    unittest.main()
