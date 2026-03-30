import uuid
import os
import json
from flask import Blueprint, request, jsonify, g, current_app
from werkzeug.utils import secure_filename

from backend.core.database import get_db
from backend.core.config import ALLOWED_EXTENSIONS, ALLOWED_AUDIO_EXTENSIONS
from backend.core.auth import require_auth, can_manage_owner_resource
from backend.core.models import (
    now_iso, serialize_scene, natural_sort_key, parse_optional_float
)
from backend.services.tour_service import (
    fetch_tour_with_access,
    load_tour_scenes_and_hotspots,
    generate_tour,
)

def _regenerate_published_tour_if_needed(tour_id, owner_id):
    db = get_db()
    tour_row = db.execute("SELECT * FROM tours WHERE id = ?", (tour_id,)).fetchone()
    if not tour_row:
        return
    gallery_index = os.path.join(current_app.config["PROCESSED_FOLDER"], tour_id, "index.html")
    if tour_row["status"] != "published" and not os.path.exists(gallery_index):
        return
    scenes_data = load_tour_scenes_and_hotspots(tour_id)
    if not scenes_data:
        return
    from backend.services.billing_service import get_user_entitlements
    ent = get_user_entitlements(owner_id)
    generate_tour(
        tour_id,
        scenes_data,
        watermark_enabled=ent["watermark_enabled"],
        tour_settings=tour_row,
    )

scenes = Blueprint('scenes', __name__)


def _parse_nullable_scene_float(raw_value, field_name, min_value, max_value):
    if raw_value is None or raw_value == "":
        return None
    return parse_optional_float(raw_value, field_name, min_value, max_value)

@scenes.route("/tours/<tour_id>/scenes", methods=["POST"])
@require_auth
def tours_add_scene(tour_id):
    sid = None
    try:
        tour, err = fetch_tour_with_access(tour_id, require_owner=True)
        if err: return err
        
        name = request.form.get("scene_name", "Unnamed")
        is_pano = request.form.get("is_panorama") == "true"
        want_async = request.args.get("async") in ("1", "true")
        files = request.files.getlist("files[]")
        
        if not files: return jsonify({"error": "No files uploaded"}), 400
        
        db = get_db()
        count_row = db.execute("SELECT COUNT(*) AS c FROM scenes WHERE tour_id = ?", (tour["id"],)).fetchone()
        order_index = int(count_row["c"])
        sid = str(uuid.uuid4())
        
        raw_dir = os.path.join(current_app.config["UPLOAD_FOLDER"], tour["id"], sid)
        proc_dir = os.path.join(current_app.config["PROCESSED_FOLDER"], tour["id"], sid)
        os.makedirs(raw_dir, exist_ok=True)
        os.makedirs(proc_dir, exist_ok=True)
        
        files.sort(key=lambda x: natural_sort_key(x.filename or ""))
        saved_raw = []
        for file in files:
            if file and ('.' in file.filename and file.filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS):
                fn = secure_filename(file.filename)
                rp = os.path.join(raw_dir, fn)
                file.save(rp)
                saved_raw.append(rp)
                
        if not saved_raw: return jsonify({"error": "No valid images"}), 400

        ts = now_iso()
        db.execute(
            """
            INSERT INTO scenes (id, tour_id, title, panorama_path, preview_path, images_json, order_index, haov, vaov, scene_type, processing_status, created_at, updated_at)
            VALUES (?, ?, ?, NULL, NULL, '[]', ?, 360, 180, 'equirectangular', 'queued', ?, ?)
            """,
            (sid, tour["id"], name, order_index, ts, ts),
        )
        db.execute("UPDATE tours SET updated_at = ? WHERE id = ?", (ts, tour["id"]))
        db.commit()
        
        # Enqueue background processing
        from backend.services.job_service import enqueue_job
        jid = enqueue_job(
            kind="scene_process",
            owner_id=g.current_user["id"],
            tour_id=tour["id"],
            scene_id=sid,
            payload={
                "tour_id": tour["id"],
                "scene_id": sid,
                "name": name,
                "is_pano": is_pano,
                "raw_paths": saved_raw,
                "order_index": order_index
            }
        )
        
        # Update scene with job_id
        db.execute("UPDATE scenes SET job_id = ? WHERE id = ?", (jid, sid))
        db.commit()
        
        row = db.execute("SELECT * FROM scenes WHERE id = ?", (sid,)).fetchone()
        return jsonify({"scene": serialize_scene(row), "job_id": jid}), 202
    except Exception as e:
        import traceback
        if sid:
            try:
                db = get_db()
                db.execute(
                    "UPDATE scenes SET processing_status = 'failed', processing_error = ?, updated_at = ? WHERE id = ?",
                    (str(e), now_iso(), sid),
                )
                db.commit()
            except Exception:
                pass
        current_app.logger.error(f"Error in tours_add_scene: {traceback.format_exc()}")
        return jsonify({"error": f"Server crash: {str(e)}"}), 500

@scenes.route("/<scene_id>", methods=["PATCH"])
@require_auth
def scenes_patch(scene_id):
    db = get_db()
    row = db.execute(
        "SELECT s.*, t.owner_id FROM scenes s JOIN tours t ON t.id = s.tour_id WHERE s.id = ?", 
        (scene_id,)
    ).fetchone()
    if not row: return jsonify({"error": "Not found"}), 404
    if not can_manage_owner_resource(row["owner_id"]): return jsonify({"error": "Forbidden"}), 403
    
    data = request.get_json(silent=True) or {}
    title = (data.get("title") or row["title"]).strip()
    haov = parse_optional_float(data.get("haov", row["haov"]), "haov", 10, 360)
    vaov = parse_optional_float(data.get("vaov", row["vaov"]), "vaov", 10, 180)
    pos_payload = data.get("pos") or {}
    try:
        pos_x = _parse_nullable_scene_float(
            pos_payload.get("x", row["pos_x"] if "pos_x" in row.keys() else None),
            "pos.x",
            -100000,
            100000,
        )
        pos_y = _parse_nullable_scene_float(
            pos_payload.get("y", row["pos_y"] if "pos_y" in row.keys() else None),
            "pos.y",
            -100000,
            100000,
        )
        pos_z = _parse_nullable_scene_float(
            pos_payload.get("z", row["pos_z"] if "pos_z" in row.keys() else None),
            "pos.z",
            -100000,
            100000,
        )
        world_yaw = _parse_nullable_scene_float(
            data.get("world_yaw", row["world_yaw"] if "world_yaw" in row.keys() else None),
            "world_yaw",
            -360,
            360,
        )
    except ValueError as exc:
        return jsonify({"error": str(exc)}), 400
    
    db.execute(
        """
        UPDATE scenes
        SET title = ?, haov = ?, vaov = ?, pos_x = ?, pos_y = ?, pos_z = ?, world_yaw = ?, updated_at = ?
        WHERE id = ?
        """,
        (title, haov, vaov, pos_x, pos_y, pos_z, world_yaw, now_iso(), scene_id)
    )
    db.commit()
    updated = db.execute("SELECT * FROM scenes WHERE id = ?", (scene_id,)).fetchone()
    return jsonify({"message": "Updated", "scene": serialize_scene(updated)}), 200

@scenes.route("/<scene_id>", methods=["DELETE"])
@require_auth
def scenes_delete(scene_id):
    db = get_db()
    row = db.execute(
        "SELECT s.id, s.tour_id, t.owner_id FROM scenes s JOIN tours t ON t.id = s.tour_id WHERE s.id = ?", 
        (scene_id,)
    ).fetchone()
    if not row: return jsonify({"error": "Not found"}), 404
    if not can_manage_owner_resource(row["owner_id"]): return jsonify({"error": "Forbidden"}), 403
    
    db.execute("DELETE FROM scenes WHERE id = ?", (scene_id,))
    db.commit()
    return jsonify({"message": "Deleted"}), 200

@scenes.route("/<scene_id>/audio", methods=["POST"])
@require_auth
def scene_audio_upload(scene_id):
    db = get_db()
    row = db.execute(
        "SELECT s.*, t.owner_id FROM scenes s JOIN tours t ON t.id = s.tour_id WHERE s.id = ?",
        (scene_id,)
    ).fetchone()
    if not row:
        return jsonify({"error": "Not found"}), 404
    if not can_manage_owner_resource(row["owner_id"]):
        return jsonify({"error": "Forbidden"}), 403

    audio_file = request.files.get("audio")
    if not audio_file or not audio_file.filename:
        return jsonify({"error": "No audio file uploaded"}), 400

    ext = audio_file.filename.rsplit(".", 1)[-1].lower() if "." in audio_file.filename else ""
    if ext not in ALLOWED_AUDIO_EXTENSIONS:
        return jsonify({"error": "Unsupported audio format"}), 400

    proc_dir = os.path.join(current_app.config["PROCESSED_FOLDER"], row["tour_id"], row["id"])
    os.makedirs(proc_dir, exist_ok=True)

    existing_audio = (row["audio_path"] or "").strip()
    if existing_audio:
        existing_path = os.path.join(proc_dir, existing_audio)
        if os.path.exists(existing_path):
            try:
                os.remove(existing_path)
            except OSError:
                current_app.logger.warning("Failed to remove existing audio for scene %s", scene_id)

    filename = secure_filename(audio_file.filename)
    stored_name = f"audio_{filename}"
    audio_file.save(os.path.join(proc_dir, stored_name))

    ts = now_iso()
    db.execute(
        "UPDATE scenes SET audio_path = ?, updated_at = ? WHERE id = ?",
        (stored_name, ts, scene_id)
    )
    db.execute("UPDATE tours SET updated_at = ? WHERE id = ?", (ts, row["tour_id"]))
    db.commit()
    _regenerate_published_tour_if_needed(row["tour_id"], row["owner_id"])

    updated = db.execute("SELECT * FROM scenes WHERE id = ?", (scene_id,)).fetchone()
    return jsonify({"scene": serialize_scene(updated)}), 200

@scenes.route("/<scene_id>/audio", methods=["DELETE"])
@require_auth
def scene_audio_delete(scene_id):
    db = get_db()
    row = db.execute(
        "SELECT s.*, t.owner_id FROM scenes s JOIN tours t ON t.id = s.tour_id WHERE s.id = ?",
        (scene_id,)
    ).fetchone()
    if not row:
        return jsonify({"error": "Not found"}), 404
    if not can_manage_owner_resource(row["owner_id"]):
        return jsonify({"error": "Forbidden"}), 403

    audio_path = (row["audio_path"] or "").strip()
    if audio_path:
        file_path = os.path.join(current_app.config["PROCESSED_FOLDER"], row["tour_id"], row["id"], audio_path)
        if os.path.exists(file_path):
            try:
                os.remove(file_path)
            except OSError:
                current_app.logger.warning("Failed to remove audio file for scene %s", scene_id)

    ts = now_iso()
    db.execute(
        "UPDATE scenes SET audio_path = NULL, updated_at = ? WHERE id = ?",
        (ts, scene_id)
    )
    db.execute("UPDATE tours SET updated_at = ? WHERE id = ?", (ts, row["tour_id"]))
    db.commit()
    _regenerate_published_tour_if_needed(row["tour_id"], row["owner_id"])

    updated = db.execute("SELECT * FROM scenes WHERE id = ?", (scene_id,)).fetchone()
    return jsonify({"scene": serialize_scene(updated)}), 200

@scenes.route("/hotspots/<hotspot_id>", methods=["PATCH"])
@require_auth
def hotspots_patch(hotspot_id):
    db = get_db()
    row = db.execute(
        "SELECT h.*, t.owner_id FROM hotspots h JOIN tours t ON t.id = h.tour_id WHERE h.id = ?",
        (hotspot_id,)
    ).fetchone()
    if not row: return jsonify({"error": "Not found"}), 404
    if not can_manage_owner_resource(row["owner_id"]): return jsonify({"error": "Forbidden"}), 403
    
    data = request.get_json(silent=True) or {}
    try:
        distance_m = _parse_nullable_scene_float(data.get("distance_m", row["distance_m"] if "distance_m" in row.keys() else None), "distance_m", 0.1, 1000)
    except ValueError as exc:
        return jsonify({"error": str(exc)}), 400
    db.execute(
        "UPDATE hotspots SET yaw = ?, pitch = ?, label = ?, distance_m = ?, updated_at = ? WHERE id = ?",
        (
            float(data.get("yaw", row["yaw"])),
            float(data.get("pitch", row["pitch"])),
            data.get("label", row["label"]),
            distance_m,
            now_iso(),
            hotspot_id,
        )
    )
    db.commit()
    return jsonify({"message": "Updated"}), 200

@scenes.route("/hotspots/<hotspot_id>", methods=["DELETE"])
@require_auth
def hotspots_delete(hotspot_id):
    db = get_db()
    row = db.execute(
        "SELECT h.id, t.owner_id FROM hotspots h JOIN tours t ON t.id = h.tour_id WHERE h.id = ?",
        (hotspot_id,)
    ).fetchone()
    if not row: return jsonify({"error": "Not found"}), 404
    if not can_manage_owner_resource(row["owner_id"]): return jsonify({"error": "Forbidden"}), 403
    
    db.execute("DELETE FROM hotspots WHERE id = ?", (hotspot_id,))
    db.commit()
    return jsonify({"message": "Deleted"}), 200


@scenes.route("/tours/<tour_id>/hotspots/bulk", methods=["POST"])
@require_auth
def hotspots_bulk_save(tour_id):
    tour, err = fetch_tour_with_access(tour_id, require_owner=True)
    if err:
        return err

    payload = request.get_json(silent=True) or {}
    if not isinstance(payload, dict):
        return jsonify({"error": "Expected hotspot payload grouped by source scene id"}), 400

    db = get_db()
    scene_rows = db.execute("SELECT id FROM scenes WHERE tour_id = ?", (tour["id"],)).fetchall()
    valid_scene_ids = {row["id"] for row in scene_rows}
    ts = now_iso()

    try:
        db.execute("DELETE FROM hotspots WHERE tour_id = ?", (tour["id"],))
        for from_scene_id, hotspot_list in payload.items():
            if from_scene_id not in valid_scene_ids:
                return jsonify({"error": f"Unknown source scene: {from_scene_id}"}), 400
            if not isinstance(hotspot_list, list):
                return jsonify({"error": f"Hotspots for {from_scene_id} must be a list"}), 400

            for hotspot in hotspot_list:
                if not isinstance(hotspot, dict):
                    return jsonify({"error": "Each hotspot must be an object"}), 400
                to_scene_id = hotspot.get("target_id")
                if to_scene_id not in valid_scene_ids:
                    return jsonify({"error": f"Unknown target scene: {to_scene_id}"}), 400
                yaw = parse_optional_float(hotspot.get("yaw"), "yaw", -360, 360)
                pitch = parse_optional_float(hotspot.get("pitch"), "pitch", -90, 90)
                entry_yaw = _parse_nullable_scene_float(hotspot.get("entry_yaw"), "entry_yaw", -360, 360)
                entry_pitch = _parse_nullable_scene_float(hotspot.get("entry_pitch"), "entry_pitch", -90, 90)
                distance_m = _parse_nullable_scene_float(hotspot.get("distance_m"), "distance_m", 0.1, 1000)
                label = (hotspot.get("label") or "").strip() or None
                hotspot_id = hotspot.get("id") or str(uuid.uuid4())
                db.execute(
                    """
                    INSERT INTO hotspots (
                        id, tour_id, from_scene_id, to_scene_id, yaw, pitch,
                        entry_yaw, entry_pitch, distance_m, label, created_at, updated_at
                    )
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        hotspot_id,
                        tour["id"],
                        from_scene_id,
                        to_scene_id,
                        yaw,
                        pitch,
                        entry_yaw,
                        entry_pitch,
                        distance_m,
                        label,
                        ts,
                        ts,
                    ),
                )

        db.execute("UPDATE tours SET updated_at = ? WHERE id = ?", (ts, tour["id"]))
        db.commit()
    except ValueError as exc:
        db.rollback()
        return jsonify({"error": str(exc)}), 400

    scenes_data = load_tour_scenes_and_hotspots(tour["id"])
    _regenerate_published_tour_if_needed(tour["id"], tour["owner_id"])
    return jsonify({"scenes": scenes_data, "message": "Hotspots saved"}), 200
