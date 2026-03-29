import argparse
import os
import sys


REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from backend.app import create_app
from backend.core.database import get_db
from backend.services.billing_service import get_user_entitlements
from backend.services.tour_service import generate_tour, load_tour_scenes_and_hotspots


def _fetch_tours(db, tour_id=None, all_published=False):
    if tour_id:
        row = db.execute(
            "SELECT * FROM tours WHERE id = ? AND deleted_at IS NULL",
            (tour_id,),
        ).fetchone()
        return [row] if row else []

    if all_published:
        return db.execute(
            "SELECT * FROM tours WHERE status = 'published' AND deleted_at IS NULL ORDER BY created_at ASC"
        ).fetchall()

    raise ValueError("Choose either --tour-id or --all-published")


def republish_tours(tours):
    total = 0
    failures = 0

    for tour in tours:
        if tour is None:
            continue

        scenes = load_tour_scenes_and_hotspots(tour["id"])
        if not scenes:
            print(f"skip {tour['id']} ({tour['slug']}): no scenes")
            continue

        try:
            entitlements = get_user_entitlements(tour["owner_id"])
            generate_tour(
                tour["id"],
                scenes,
                watermark_enabled=entitlements["watermark_enabled"],
                tour_settings=tour,
            )
            total += 1
            print(f"republished {tour['slug']} ({tour['id']})")
        except Exception as exc:
            failures += 1
            print(f"failed {tour['slug']} ({tour['id']}): {exc}")

    print(f"done: republished={total} failures={failures}")
    return 1 if failures else 0


def main():
    parser = argparse.ArgumentParser(
        description="Regenerate static published player HTML for tours."
    )
    parser.add_argument("--tour-id", help="Republish a single tour by UUID")
    parser.add_argument(
        "--all-published",
        action="store_true",
        help="Republish all published tours",
    )
    args = parser.parse_args()

    app = create_app()
    with app.app_context():
        db = get_db()
        tours = _fetch_tours(db, tour_id=args.tour_id, all_published=args.all_published)
        if not tours:
            print("No matching tours found.")
            return 1
        return republish_tours(tours)


if __name__ == "__main__":
    raise SystemExit(main())
