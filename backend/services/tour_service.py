import os
import json
import subprocess
from backend.core.config import (
    PROCESSED_FOLDER, COVER_FILENAME, COVER_META_FILENAME
)
from backend.core.database import get_db

def ensure_tour_cover_image(tour_row):
    """
    Create/update a cached tour cover image from the configured start scene.
    Returns a relative URL (galleries/<tour_id>/cover.jpg) or None.
    """
    if tour_row is None:
        return None
    tour_id = tour_row["id"]
    db = get_db()
    
    # 1. Determine the scene to use for the cover
    keys = tour_row.keys()
    start_scene_id = tour_row["start_scene_id"] if "start_scene_id" in keys else None
    if start_scene_id:
        scene = db.execute(
            "SELECT * FROM scenes WHERE id = ? AND tour_id = ?",
            (start_scene_id, tour_id),
        ).fetchone()
    else:
        scene = None
        
    if scene is None:
        scene = db.execute(
            "SELECT * FROM scenes WHERE tour_id = ? ORDER BY order_index ASC LIMIT 1",
            (tour_id,),
        ).fetchone()
        
    if scene is None:
        return None

    # 2. Identify the panorama file
    pano_name = (scene["panorama_path"] or "").strip()
    if not pano_name:
        try:
            imgs = json.loads(scene["images_json"] or "[]")
        except Exception:
            imgs = []
        pano_name = (imgs[0] if imgs else "") or ""
    
    if not pano_name:
        return None

    scene_dir = os.path.join(PROCESSED_FOLDER, tour_id, scene["id"])
    src_path = os.path.join(scene_dir, pano_name)
    if not os.path.exists(src_path):
        return None

    out_dir = os.path.join(PROCESSED_FOLDER, tour_id)
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, COVER_FILENAME)
    meta_path = os.path.join(out_dir, COVER_META_FILENAME)

    # 3. Get view orientation
    try:
        yaw = float(tour_row["start_yaw"] if "start_yaw" in keys else 0.0)
        pitch = float(tour_row["start_pitch"] if "start_pitch" in keys else 0.0)
    except Exception:
        yaw, pitch = 0.0, 0.0

    sig = {
        "scene_id": scene["id"],
        "source_file": pano_name,
        "yaw": round(yaw, 2),
        "pitch": round(pitch, 2),
    }

    # 4. Check cache validity
    cache_valid = False
    if os.path.exists(out_path) and os.path.exists(meta_path):
        try:
            with open(meta_path, "r") as f:
                old_sig = json.load(f)
            if old_sig == sig:
                cache_valid = True
        except Exception:
            pass

    if cache_valid:
        return f"/galleries/{tour_id}/{COVER_FILENAME}"

    # 5. Generate cover using ffmpeg/vtools logic (simplified placeholder for now, or use existing logic)
    # For now, if we don't have the sophisticated 'equirect_to_rect' tool available as a python lib,
    # we might just copy the preview or a cropped version.
    # Re-using the logic from app.py usually involves calling a subprocess or PIL.
    
    # In the original app.py, there was a specific complex logic for cover generation.
    # I will try to implement a basic version using PIL for now to ensure it works.
    try:
        from PIL import Image
        with Image.open(src_path) as im:
            # Simple center crop for cover if view is 0,0
            # A real version should handle equirectangular projection math
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
            
        with open(meta_path, "w") as f:
            json.dump(sig, f)
            
        return f"/galleries/{tour_id}/{COVER_FILENAME}"
    except Exception:
        return None
