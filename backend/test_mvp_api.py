import unittest
import uuid
import os
import sqlite3

from backend.app import app, DB_PATH, now_iso


class MvpApiTests(unittest.TestCase):
    def setUp(self):
        self._old_verify = os.environ.get("EMAIL_VERIFICATION_ENABLED")
        self._old_allow_mock = os.environ.get("ALLOW_MOCK_SUBSCRIBE")
        os.environ["EMAIL_VERIFICATION_ENABLED"] = "0"
        os.environ["ALLOW_MOCK_SUBSCRIBE"] = "1"
        self.client = app.test_client()
        self.email = f"test_{uuid.uuid4().hex[:8]}@example.com"
        self.password = "Password123"

    def tearDown(self):
        if self._old_verify is None:
            os.environ.pop("EMAIL_VERIFICATION_ENABLED", None)
        else:
            os.environ["EMAIL_VERIFICATION_ENABLED"] = self._old_verify
        if self._old_allow_mock is None:
            os.environ.pop("ALLOW_MOCK_SUBSCRIBE", None)
        else:
            os.environ["ALLOW_MOCK_SUBSCRIBE"] = self._old_allow_mock

    def register_and_login(self):
        res = self.client.post(
            "/auth/register",
            json={"email": self.email, "password": self.password, "display_name": "Tester"},
        )
        self.assertEqual(res.status_code, 201)

    def test_auth_and_tour_privacy(self):
        self.register_and_login()
        up = self.client.post("/billing/mock/subscribe", json={"plan_id": "pro"})
        self.assertEqual(up.status_code, 200)

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
            res = self.client.post("/tours", json={"title": f"T{i+1}", "visibility": "public"})
            self.assertEqual(res.status_code, 201)
        blocked = self.client.post("/tours", json={"title": "T3", "visibility": "public"})
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
        return scene_id

    def test_free_cannot_create_private_tour(self):
        self.register_and_login()
        res = self.client.post("/tours", json={"title": "Private blocked", "visibility": "private"})
        self.assertEqual(res.status_code, 403)
        self.assertEqual(res.get_json().get("code"), "private_requires_paid_plan")

    def test_private_tour_access_list_allows_specific_user(self):
        # owner
        self.register_and_login()
        owner_email = self.email
        up = self.client.post("/billing/mock/subscribe", json={"plan_id": "pro"})
        self.assertEqual(up.status_code, 200)
        create = self.client.post("/tours", json={"title": "Private ACL", "visibility": "private"})
        self.assertEqual(create.status_code, 201)
        tour = create.get_json()["tour"]
        slug = tour["slug"]
        self.client.post("/auth/logout")

        # guest user (no access yet)
        guest_email = f"guest_{uuid.uuid4().hex[:8]}@example.com"
        guest_pw = "Password123"
        r2 = self.client.post("/auth/register", json={"email": guest_email, "password": guest_pw, "display_name": "Guest"})
        self.assertEqual(r2.status_code, 201)
        denied = self.client.get(f"/t/{slug}")
        self.assertEqual(denied.status_code, 403)
        self.client.post("/auth/logout")

        # owner grants access
        login_owner = self.client.post("/auth/login", json={"email": owner_email, "password": self.password})
        self.assertEqual(login_owner.status_code, 200)
        grant = self.client.post(f"/tours/{tour['id']}/access", json={"email": guest_email})
        self.assertEqual(grant.status_code, 200)
        self.client.post("/auth/logout")

        # guest can open now
        login_guest = self.client.post("/auth/login", json={"email": guest_email, "password": guest_pw})
        self.assertEqual(login_guest.status_code, 200)
        allowed = self.client.get(f"/t/{slug}")
        self.assertEqual(allowed.status_code, 302)

    def test_set_start_scene_and_default_view(self):
        self.register_and_login()
        create = self.client.post("/tours", json={"title": "Start scene settings", "visibility": "public"})
        self.assertEqual(create.status_code, 201)
        tour = create.get_json()["tour"]
        s1 = self._insert_min_scene(tour["id"])
        _s2 = self._insert_min_scene(tour["id"])
        patch = self.client.patch(
            f"/tours/{tour['id']}",
            json={"start_scene_id": s1, "start_pitch": 11.5, "start_yaw": -23.0, "default_hfov": 64.0},
        )
        self.assertEqual(patch.status_code, 200)
        got = self.client.get(f"/tours/{tour['id']}")
        self.assertEqual(got.status_code, 200)
        payload = got.get_json()["tour"]
        self.assertEqual(payload["start_scene_id"], s1)
        self.assertAlmostEqual(float(payload["start_pitch"]), 11.5, places=2)
        self.assertAlmostEqual(float(payload["start_yaw"]), -23.0, places=2)
        self.assertAlmostEqual(float(payload["default_hfov"]), 64.0, places=2)

    def test_finalize_includes_watermark_for_free(self):
        self.register_and_login()
        create = self.client.post("/tours", json={"title": "WM Free", "visibility": "public"})
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

        create = self.client.post("/tours", json={"title": "WM Pro", "visibility": "public"})
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
