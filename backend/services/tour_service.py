import os
import json
import subprocess
import shutil
import zipfile
import re
import datetime
from flask import g, jsonify, current_app
from backend.core.config import (
    PROCESSED_FOLDER, COVER_FILENAME, COVER_META_FILENAME, 
    WEB_PANO_FILENAME, GALLERY_TEMPLATE_VERSION, FRONTEND_FOLDER
)
from backend.core.database import get_db
from backend.core.models import serialize_scene, now_iso

def fetch_tour_with_access(tour_id, require_owner=False):
    db = get_db()
    tour = db.execute("SELECT * FROM tours WHERE id = ? AND deleted_at IS NULL", (tour_id,)).fetchone()
    if tour is None:
        return None, (jsonify({"error": "Tour not found"}), 404)
    
    if g.current_user is None:
        if tour["visibility"] != "public":
            return None, (jsonify({"error": "Unauthorized"}), 401)
        return tour, None

    if tour["owner_id"] == g.current_user["id"]:
        return tour, None

    if require_owner:
        return None, (jsonify({"error": "Forbidden"}), 403)

    if tour["visibility"] == "public":
        return tour, None

    # Check explicit access grants
    grant = db.execute(
        "SELECT id FROM tour_access_grants WHERE tour_id = ? AND user_id = ?",
        (tour["id"], g.current_user["id"]),
    ).fetchone()
    if grant:
        return tour, None

    return None, (jsonify({"error": "Forbidden"}), 403)

def load_tour_scenes_and_hotspots(tour_id):
    db = get_db()
    scene_rows = db.execute(
        "SELECT * FROM scenes WHERE tour_id = ? ORDER BY order_index ASC",
        (tour_id,),
    ).fetchall()
    hotspot_rows = db.execute(
        "SELECT * FROM hotspots WHERE tour_id = ?",
        (tour_id,),
    ).fetchall()
    
    # Group hotspots by scene
    hs_map = {}
    for h in hotspot_rows:
        target_scene = next((s for s in scene_rows if s["id"] == h["to_scene_id"]), None)
        hs_map.setdefault(h["from_scene_id"], []).append({
            "id": h["id"],
            "pitch": h["pitch"],
            "yaw": h["yaw"],
            "target_id": h["to_scene_id"],
            "target_name": target_scene["title"] if target_scene else "Unknown",
            "entry_pitch": h["entry_pitch"],
            "entry_yaw": h["entry_yaw"],
            "label": h["label"]
        })
    
    scenes = []
    for s in scene_rows:
        s_data = serialize_scene(s)
        s_data["hotspots"] = hs_map.get(s["id"], [])
        scenes.append(s_data)
    return scenes

def user_can_use_private_tours(user_id):
    from backend.services.billing_service import get_user_plan
    plan, _ = get_user_plan(user_id)
    # Free plan cannot have private tours
    return plan["id"] != "free"

def list_tour_access_entries(tour_id):
    db = get_db()
    rows = db.execute(
        """
        SELECT g.id, g.user_id, u.email, u.display_name, g.created_at
        FROM tour_access_grants g
        JOIN users u ON u.id = g.user_id
        WHERE g.tour_id = ?
        ORDER BY g.created_at DESC
        """,
        (tour_id,),
    ).fetchall()
    return [dict(r) for r in rows]

# Image & Export Utils
def build_gpano_xmp(w, h):
    return (
        f'<?xpacket begin="" id="W5M0MpCehiHzreSzNTczkc9d"?>'
        f'<x:xmpmeta xmlns:x="adobe:ns:meta/">'
        f'<rdf:RDF xmlns:rdf="http://www.w3.org/1999/02/22-rdf-syntax-ns#">'
        f'<rdf:Description rdf:about="" xmlns:GPano="http://ns.google.com/photos/1.0/panorama/">'
        f'<GPano:UsePanoramaViewer>True</GPano:UsePanoramaViewer>'
        f'<GPano:ProjectionType>equirectangular</GPano:ProjectionType>'
        f'<GPano:FullPanoWidthPixels>{w}</GPano:FullPanoWidthPixels>'
        f'<GPano:FullPanoHeightPixels>{h}</GPano:FullPanoHeightPixels>'
        f'<GPano:CroppedAreaImageWidthPixels>{w}</GPano:CroppedAreaImageWidthPixels>'
        f'<GPano:CroppedAreaImageHeightPixels>{h}</GPano:CroppedAreaImageHeightPixels>'
        f'<GPano:CroppedAreaLeftPixels>0</GPano:CroppedAreaLeftPixels>'
        f'<GPano:CroppedAreaTopPixels>0</GPano:CroppedAreaTopPixels>'
        f'</rdf:Description></rdf:RDF></x:xmpmeta>'
        f'<?xpacket end="r"?>'
    ).encode("utf-8")

def inject_xmp_into_jpeg(image_bytes, xmp_xml_bytes):
    xmp_header = b"http://ns.adobe.com/xap/1.0/\x00"
    xmp_payload = xmp_header + (xmp_xml_bytes or b"")
    seg_len = len(xmp_payload) + 2
    if seg_len > 0xFFFF:
        raise ValueError("XMP payload too large for APP1")
    app1_marker = b"\xFF\xE1" + seg_len.to_bytes(2, "big") + xmp_payload
    if image_bytes[:2] != b"\xFF\xD8":
        return image_bytes
    return image_bytes[:2] + app1_marker + image_bytes[2:]

def add_directory_to_zip(zip_handle, folder_path, zip_path_prefix=""):
    for root, dirs, files in os.walk(folder_path):
        for file in files:
            file_path = os.path.join(root, file)
            rel_path = os.path.relpath(file_path, folder_path)
            arcname = os.path.join(zip_path_prefix, rel_path)
            zip_handle.write(file_path, arcname)

def build_self_hosted_readme(tour):
    return f"TOUR: {tour['title']}\nID: {tour['id']}\nEXPORTED: {now_iso()}\n\nTo view locally:\n1. Unzip all files.\n2. Run a local web server (e.g., 'python -m http.server 8080').\n3. Open http://localhost:8080 in your browser."

def build_self_hosted_eula(tour):
    return "This export is provided for self-hosting. Redistribution or resale of the player code is prohibited."

PUBLIC_PLAYER_TEMPLATE = "player_template_legacy.html"
# Keep the experimental Three.js player in the repo, but route published tours
# through the stable Pannellum template until the new viewer is production-ready.

def generate_tour(tour_id, scenes, watermark_enabled=True, force_previews=False, tour_settings=None):
    """
    Generates the static index.html for a tour.
    """
    from backend.core.models import ensure_scene_web_pano
    out_dir = os.path.join(PROCESSED_FOLDER, tour_id)
    os.makedirs(out_dir, exist_ok=True)
    
    # Simplified version of the generation logic
    # In a real app, this would use a template engine (Jinja2)
    template_path = os.path.join(FRONTEND_FOLDER, PUBLIC_PLAYER_TEMPLATE)
    if not os.path.exists(template_path):
        # Fallback to a very simple internal template if file missing
        html = "<html><body><h1>Tour Player Placeholder</h1></body></html>"
    else:
        with open(template_path, "r", encoding="utf-8") as f:
            html = f.read()
    
    settings = dict(tour_settings) if tour_settings is not None else {}

    # Inject data
    tour_data = {
        "id": tour_id,
        "scenes": scenes,
        "settings": {
            "start_scene_id": settings.get("start_scene_id"),
            "start_pitch": settings.get("start_pitch", 0),
            "start_yaw": settings.get("start_yaw", 0),
            "default_hfov": settings.get("default_hfov", 75),
            "title": settings.get("title", "Tour")
        },
        "watermark": watermark_enabled,
        "v": GALLERY_TEMPLATE_VERSION
    }
    html = html.replace("/*{{TOUR_DATA}}*/", f"window.TOUR_DATA = {json.dumps(tour_data, ensure_ascii=False)};")
    
    with open(os.path.join(out_dir, "index.html"), "w", encoding="utf-8") as f:
        f.write(html)
    
    return f"/t/{settings.get('slug', tour_id)}"

def ensure_tour_cover_image(tour_row):
    if tour_row is None:
        return None
    tour_id = tour_row["id"]
    db = get_db()
    
    keys = tour_row.keys()
    start_scene_id = tour_row["start_scene_id"] if "start_scene_id" in keys else None
    if start_scene_id:
        scene = db.execute("SELECT * FROM scenes WHERE id = ? AND tour_id = ?", (start_scene_id, tour_id)).fetchone()
    else:
        scene = db.execute("SELECT * FROM scenes WHERE tour_id = ? ORDER BY order_index ASC LIMIT 1", (tour_id,)).fetchone()
        
    if scene is None: return None
    pano_name = (scene["panorama_path"] or "").strip()
    if not pano_name:
        try: imgs = json.loads(scene["images_json"] or "[]")
        except: imgs = []
        pano_name = (imgs[0] if imgs else "") or ""
    if not pano_name: return None

    scene_dir = os.path.join(PROCESSED_FOLDER, tour_id, scene["id"])
    src_path = os.path.join(scene_dir, pano_name)
    if not os.path.exists(src_path): return None

    out_dir = os.path.join(PROCESSED_FOLDER, tour_id)
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, COVER_FILENAME)
    meta_path = os.path.join(out_dir, COVER_META_FILENAME)

    try:
        yaw = float(tour_row["start_yaw"] if "start_yaw" in keys else 0.0)
        pitch = float(tour_row["start_pitch"] if "start_pitch" in keys else 0.0)
    except: yaw, pitch = 0.0, 0.0

    sig = {"scene_id": scene["id"], "source_file": pano_name, "yaw": round(yaw, 2), "pitch": round(pitch, 2)}
    if os.path.exists(out_path) and os.path.exists(meta_path):
        try:
            with open(meta_path, "r") as f:
                if json.load(f) == sig: return f"/galleries/{tour_id}/{COVER_FILENAME}"
        except: pass

    try:
        from PIL import Image
        with Image.open(src_path) as im:
            w, h = im.size
            target_aspect = 16/9
            if w/h > target_aspect:
                new_w = h * target_aspect
                left = (w - new_w) / 2
                im = im.crop((left, 0, left + new_w, h))
            else:
                new_h = w / target_aspect
                top = (h - new_h) / 2
                im = im.crop((0, top, w, top + new_h))
            im.thumbnail((800, 450), Image.Resampling.LANCZOS)
            im.save(out_path, "JPEG", quality=85)
        with open(meta_path, "w") as f: json.dump(sig, f)
        return f"/galleries/{tour_id}/{COVER_FILENAME}"
    except: return None
