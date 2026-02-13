import unittest
import uuid
import os
import sqlite3

from backend.app import app, DB_PATH, now_iso


class MvpApiTests(unittest.TestCase):
    def setUp(self):
        self.client = app.test_client()
        self.email = f"test_{uuid.uuid4().hex[:8]}@example.com"
        self.password = "Password123"

    def register_and_login(self):
        res = self.client.post(
            "/auth/register",
            json={"email": self.email, "password": self.password, "display_name": "Tester"},
        )
        self.assertEqual(res.status_code, 201)

    def test_auth_and_tour_privacy(self):
        self.register_and_login()

        create_res = self.client.post(
            "/tours",
            json={"title": "Private QA Tour", "description": "qa", "visibility": "private"},
        )
        self.assertEqual(create_res.status_code, 201)
        tour = create_res.get_json()["tour"]

        finalize_res = self.client.post(f"/tours/{tour['id']}/finalize")
        self.assertEqual(finalize_res.status_code, 400)  # no scenes yet

        # Logout and verify private share is blocked
        self.client.post("/auth/logout")
        share_res = self.client.get(f"/t/{tour['slug']}")
        self.assertEqual(share_res.status_code, 403)

    def test_public_gallery_listing(self):
        self.register_and_login()
        create_res = self.client.post(
            "/tours",
            json={"title": "Public QA Tour", "description": "qa", "visibility": "public"},
        )
        self.assertEqual(create_res.status_code, 201)
        tour = create_res.get_json()["tour"]

        # Draft tours are not listed yet
        gallery_res = self.client.get("/gallery")
        self.assertEqual(gallery_res.status_code, 200)
        slugs = [x["slug"] for x in gallery_res.get_json().get("items", [])]
        self.assertNotIn(tour["slug"], slugs)

    def test_default_free_plan_and_limit(self):
        self.register_and_login()
        me = self.client.get("/me")
        self.assertEqual(me.status_code, 200)
        me_data = me.get_json()
        self.assertEqual(me_data["plan"]["id"], "free")
        self.assertEqual(me_data["entitlements"]["max_tours"], 2)
        self.assertTrue(me_data["entitlements"]["watermark_enabled"])

        for i in range(2):
            res = self.client.post("/tours", json={"title": f"T{i+1}", "visibility": "private"})
            self.assertEqual(res.status_code, 201)
        blocked = self.client.post("/tours", json={"title": "T3", "visibility": "private"})
        self.assertEqual(blocked.status_code, 403)
        blocked_data = blocked.get_json()
        self.assertEqual(blocked_data["code"], "plan_limit_exceeded")

    def test_mock_upgrade_removes_limit_block(self):
        self.register_and_login()
        for i in range(2):
            self.client.post("/tours", json={"title": f"F{i+1}"})
        blocked = self.client.post("/tours", json={"title": "Blocked"})
        self.assertEqual(blocked.status_code, 403)

        up = self.client.post("/billing/mock/subscribe", json={"plan_id": "pro"})
        self.assertEqual(up.status_code, 200)
        self.assertEqual(up.get_json()["plan_id"], "pro")

        allowed = self.client.post("/tours", json={"title": "After upgrade"})
        self.assertEqual(allowed.status_code, 201)

    def _insert_min_scene(self, tour_id, scene_id=None):
        if scene_id is None:
            scene_id = f"scene_{uuid.uuid4().hex[:8]}"
        db = sqlite3.connect(DB_PATH)
        try:
            db.execute(
                """
                INSERT INTO scenes (
                    id, tour_id, title, panorama_path, images_json, order_index,
                    haov, vaov, scene_type, created_at, updated_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (scene_id, tour_id, "Room", None, '["dummy.jpg"]', 0, 360, 180, "equirectangular", now_iso(), now_iso()),
            )
            db.commit()
        finally:
            db.close()

    def test_finalize_includes_watermark_for_free(self):
        self.register_and_login()
        create = self.client.post("/tours", json={"title": "WM Free", "visibility": "private"})
        self.assertEqual(create.status_code, 201)
        tour = create.get_json()["tour"]
        self._insert_min_scene(tour["id"])

        fin = self.client.post(f"/tours/{tour['id']}/finalize")
        self.assertEqual(fin.status_code, 200)
        html_path = os.path.join(app.config["PROCESSED_FOLDER"], tour["id"], "index.html")
        with open(html_path, "r", encoding="utf-8") as f:
            html = f.read()
        self.assertIn("Created with Pan-o-Rama Free", html)

    def test_finalize_without_watermark_for_pro(self):
        self.register_and_login()
        up = self.client.post("/billing/mock/subscribe", json={"plan_id": "pro"})
        self.assertEqual(up.status_code, 200)

        create = self.client.post("/tours", json={"title": "WM Pro", "visibility": "private"})
        self.assertEqual(create.status_code, 201)
        tour = create.get_json()["tour"]
        self._insert_min_scene(tour["id"])

        fin = self.client.post(f"/tours/{tour['id']}/finalize")
        self.assertEqual(fin.status_code, 200)
        html_path = os.path.join(app.config["PROCESSED_FOLDER"], tour["id"], "index.html")
        with open(html_path, "r", encoding="utf-8") as f:
            html = f.read()
        self.assertNotIn("Created with Pan-o-Rama Free", html)


if __name__ == "__main__":
    unittest.main()
