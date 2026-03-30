import uuid
import os
import json
from io import BytesIO
from flask import Blueprint, request, jsonify, g, send_file, current_app, after_this_request
import tempfile
import zipfile
import re
from PIL import Image, ImageOps

from backend.core.database import get_db
from backend.core.config import (
    PLAN_FREE, PLAN_PRO, PLAN_BUSINESS, ALLOWED_EXTENSIONS, 
    UPLOAD_FOLDER, PROCESSED_FOLDER, COVER_FILENAME
)
from backend.core.auth import require_auth, require_admin, current_user_is_admin
from backend.core.models import (
    now_iso, serialize_tour, slugify, normalize_visibility, 
    parse_optional_float, natural_sort_key
)
from backend.services.tour_service import (
    fetch_tour_with_access, load_tour_scenes_and_hotspots, 
    user_can_use_private_tours, list_tour_access_entries,
    generate_tour, build_gpano_xmp, inject_xmp_into_jpeg,
    build_self_hosted_readme, build_self_hosted_eula, add_directory_to_zip
)
from backend.services.billing_service import get_user_entitlements

tours = Blueprint('tours', __name__)

@tours.route("", methods=["POST"])
@require_auth
def tours_create():
    ent = get_user_entitlements(g.current_user["id"])
    if ent["usage"]["tours_count"] >= ent["max_tours"]:
        return jsonify({"error": "Plan limit reached", "code": "plan_limit_exceeded", "entitlements": ent}), 403

    data = request.get_json(silent=True) or {}
    title = (data.get("title") or "").strip() or "Untitled Tour"
    description = (data.get("description") or "").strip()
    visibility = normalize_visibility(data.get("visibility") or "public")
    
    if visibility == "private" and not user_can_use_private_tours(g.current_user["id"]):
        return jsonify({"error": "Private tours are available on paid plans", "code": "private_requires_paid_plan"}), 403
    
    tid = str(uuid.uuid4())
    ts = now_iso()
    slug = slugify(title)
    db = get_db()
    db.execute(
        """
        INSERT INTO tours (id, owner_id, title, description, slug, visibility, status, created_at, updated_at, start_scene_id, start_pitch, start_yaw, default_hfov)
        VALUES (?, ?, ?, ?, ?, ?, 'draft', ?, ?, NULL, NULL, NULL, 70)
        """,
        (tid, g.current_user["id"], title, description, slug, visibility, ts, ts),
    )
    db.commit()
    os.makedirs(os.path.join(current_app.config["PROCESSED_FOLDER"], tid), exist_ok=True)
    row = db.execute("SELECT * FROM tours WHERE id = ?", (tid,)).fetchone()
    return jsonify({"tour": serialize_tour(row)}), 201

@tours.route("/my", methods=["GET"])
@require_auth
def tours_my():
    db = get_db()
    rows = db.execute(
        "SELECT * FROM tours WHERE owner_id = ? AND deleted_at IS NULL ORDER BY created_at DESC",
        (g.current_user["id"],),
    ).fetchall()
    return jsonify({"tours": [serialize_tour(r) for r in rows]}), 200

@tours.route("/admin/all", methods=["GET"])
@require_admin
def tours_admin_all():
    db = get_db()
    rows = db.execute(
        """
        SELECT t.*, u.email AS owner_email, u.display_name AS owner_display_name
        FROM tours t
        JOIN users u ON u.id = t.owner_id
        ORDER BY
            CASE
                WHEN t.deleted_at IS NULL THEN 0
                ELSE 1
            END,
            t.updated_at DESC
        """
    ).fetchall()
    tours_payload = []
    for row in rows:
        item = serialize_tour(row)
        item["owner_email"] = row["owner_email"]
        item["owner_display_name"] = row["owner_display_name"]
        item["deleted_at"] = row["deleted_at"] if "deleted_at" in row.keys() else None
        tours_payload.append(item)
    return jsonify({"tours": tours_payload}), 200

@tours.route("/<tour_id>", methods=["GET"])
def tours_get(tour_id):
    tour, err = fetch_tour_with_access(tour_id, require_owner=False)
    if err: return err
    scenes = load_tour_scenes_and_hotspots(tour["id"])
    payload = serialize_tour(tour)
    payload["scenes"] = scenes
    if g.current_user is not None and (g.current_user["id"] == tour["owner_id"] or current_user_is_admin()):
        payload["access_list"] = list_tour_access_entries(tour["id"])
    return jsonify({"tour": payload}), 200


@tours.route("/<tour_id>/scenes", methods=["POST"])
@require_auth
def tours_add_scene_proxy(tour_id):
    from backend.blueprints.scenes import tours_add_scene

    return tours_add_scene(tour_id)

@tours.route("/<tour_id>", methods=["PATCH"])
@require_auth
def tours_patch(tour_id):
    tour, err = fetch_tour_with_access(tour_id, require_owner=True)
    if err: return err
    data = request.get_json(silent=True) or {}
    title = (data.get("title") or tour["title"]).strip() or tour["title"]
    description = (data.get("description") or tour["description"]).strip()
    visibility = normalize_visibility(data.get("visibility") or tour["visibility"])
    
    if visibility == "private" and not user_can_use_private_tours(g.current_user["id"]):
        return jsonify({"error": "Private tours are available on paid plans", "code": "private_requires_paid_plan"}), 403
    
    start_scene_id = data.get("start_scene_id", tour["start_scene_id"])
    start_pitch = data.get("start_pitch", tour["start_pitch"])
    start_yaw = data.get("start_yaw", tour["start_yaw"])
    default_hfov = data.get("default_hfov", tour["default_hfov"])
    
    if start_scene_id == "": start_scene_id = None
    try:
        start_pitch = parse_optional_float(start_pitch, "start_pitch", -90, 90)
        start_yaw = parse_optional_float(start_yaw, "start_yaw", -360, 360)
        default_hfov = parse_optional_float(default_hfov, "default_hfov", 30, 120)
    except ValueError as e:
        return jsonify({"error": str(e)}), 400
        
    db = get_db()
    if start_scene_id is not None:
        sc = db.execute("SELECT id FROM scenes WHERE id = ? AND tour_id = ?", (start_scene_id, tour["id"])).fetchone()
        if sc is None: return jsonify({"error": "start_scene_id must belong to this tour"}), 400
        
    db.execute(
        """
        UPDATE tours
        SET title = ?, description = ?, visibility = ?, start_scene_id = ?, start_pitch = ?, start_yaw = ?, default_hfov = ?, updated_at = ?
        WHERE id = ?
        """,
        (title, description, visibility, start_scene_id, start_pitch, start_yaw, default_hfov, now_iso(), tour["id"]),
    )
    db.commit()
    row = db.execute("SELECT * FROM tours WHERE id = ?", (tour["id"],)).fetchone()

    # Keep the published gallery player in sync with Studio start-view changes.
    gallery_index = os.path.join(PROCESSED_FOLDER, tour["id"], "index.html")
    if row is not None and (row["status"] == "published" or os.path.exists(gallery_index)):
        scenes_data = load_tour_scenes_and_hotspots(tour["id"])
        if scenes_data:
            ent = get_user_entitlements(g.current_user["id"])
            generate_tour(
                tour["id"],
                scenes_data,
                watermark_enabled=ent["watermark_enabled"],
                tour_settings=row,
            )

    return jsonify({"tour": serialize_tour(row)}), 200

@tours.route("/<tour_id>", methods=["DELETE"])
@require_auth
def tours_delete(tour_id):
    tour, err = fetch_tour_with_access(tour_id, require_owner=True)
    if err: return err
    db = get_db()
    db.execute("UPDATE tours SET deleted_at = ?, updated_at = ? WHERE id = ?", (now_iso(), now_iso(), tour["id"]))
    db.commit()
    return jsonify({"message": "Tour deleted"}), 200

@tours.route("/<tour_id>/finalize", methods=["POST"])
@require_auth
def tours_finalize(tour_id):
    tour, err = fetch_tour_with_access(tour_id, require_owner=True)
    if err: return err
    db = get_db()
    pending_row = db.execute(
        "SELECT COUNT(*) AS n FROM scenes WHERE tour_id = ? AND processing_status IS NOT NULL AND processing_status != 'ready'",
        (tour["id"],),
    ).fetchone()
    if int(pending_row["n"] or 0) > 0:
        return jsonify({"error": "Scenes are still processing", "code": "scenes_processing"}), 409
        
    scenes_data = load_tour_scenes_and_hotspots(tour["id"])
    if not scenes_data: return jsonify({"error": "Tour must have at least one scene"}), 400
    
    ent = get_user_entitlements(g.current_user["id"])
    gallery_url = generate_tour(tour["id"], scenes_data, watermark_enabled=ent["watermark_enabled"], tour_settings=tour)
    db.execute("UPDATE tours SET status = 'published', updated_at = ? WHERE id = ?", (now_iso(), tour["id"]))
    db.commit()
    return jsonify({"gallery_url": gallery_url, "share_url": f"/t/{tour['slug']}", "visibility": tour["visibility"]}), 200


@tours.route("/<tour_id>/hotspots/bulk", methods=["POST"])
@require_auth
def tours_hotspots_bulk(tour_id):
    from backend.blueprints.scenes import hotspots_bulk_save
    return hotspots_bulk_save(tour_id)

@tours.route("/<tour_id>/export/facebook360", methods=["GET"])
@require_auth
def tours_export_facebook360(tour_id):
    tour, err = fetch_tour_with_access(tour_id, require_owner=True)
    if err: return err
    db = get_db()
    scene = db.execute("SELECT * FROM scenes WHERE tour_id = ? ORDER BY order_index ASC LIMIT 1", (tour["id"],)).fetchone()
    if scene is None: return jsonify({"error": "Tour has no scenes"}), 400

    proc_dir = os.path.join(current_app.config["PROCESSED_FOLDER"], tour["id"], scene["id"])
    pano = (scene["panorama_path"] or "").strip()
    if not pano: return jsonify({"error": "First scene has no panorama"}), 400
    img_path = os.path.join(proc_dir, pano)
    if not os.path.exists(img_path): return jsonify({"error": "File not found"}), 404

    try:
        with Image.open(img_path) as im:
            im = ImageOps.exif_transpose(im)
            w, h = im.size
            buf = BytesIO()
            im.save(buf, format="JPEG", quality=92)
            raw = buf.getvalue()
        xmp = build_gpano_xmp(w, h)
        out = inject_xmp_into_jpeg(raw, xmp)
        return send_file(BytesIO(out), mimetype="image/jpeg", as_attachment=True, download_name=f"{tour['slug']}-facebook360.jpg")
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@tours.route("/<tour_id>/export/self-hosted", methods=["GET"])
@require_auth
def tours_export_self_hosted(tour_id):
    tour, err = fetch_tour_with_access(tour_id, require_owner=True)
    if err: return err
    ent = get_user_entitlements(g.current_user["id"])
    if ent.get("plan_id") != PLAN_BUSINESS:
        return jsonify({"error": "Business plan required"}), 403

    gallery_dir = os.path.join(current_app.config["PROCESSED_FOLDER"], tour["id"])
    fd, tmp_zip = tempfile.mkstemp(suffix=".zip")
    os.close(fd)
    with zipfile.ZipFile(tmp_zip, "w") as zf:
        add_directory_to_zip(zf, gallery_dir)
        zf.writestr("README.txt", build_self_hosted_readme(tour))
    
    @after_this_request
    def cleanup(resp):
        if os.path.exists(tmp_zip): os.remove(tmp_zip)
        return resp
    return send_file(tmp_zip, mimetype="application/zip", as_attachment=True, download_name=f"{tour['slug']}-self-hosted.zip")
