import unittest
import uuid

from backend.app import app


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


if __name__ == "__main__":
    unittest.main()
