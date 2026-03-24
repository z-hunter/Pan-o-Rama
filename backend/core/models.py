import datetime
import json
import re
import os
from PIL import Image, ImageOps

# Helper to avoid circular imports - config can be imported directly
from backend.core.config import (
    PROCESSED_FOLDER, 
    STUDIO_WEB_PANO_FILENAME
)

def now_iso():
    return datetime.datetime.utcnow().isoformat() + "Z"

def safe_float(val, default=0.0):
    try:
        if isinstance(val, (list, tuple)):
            if len(val) >= 2 and val[1] != 0: return float(val[0]) / float(val[1])
            return float(val[0]) if len(val) > 0 else default
        return float(val)
    except: return default

def natural_sort_key(s):
    return [int(text) if text.isdigit() else text.lower() for text in re.split('([0-9]+)', s)]

def _target_thumbnail_size(src_w, src_h, max_size):
    if src_w <= 0 or src_h <= 0:
        return (0, 0)
    ratio = min(max_size[0] / float(src_w), max_size[1] / float(src_h), 1.0)
    return (max(1, int(round(src_w * ratio))), max(1, int(round(src_h * ratio))))

def ensure_scene_web_pano(tour_id, scene_id, source_filename, web_filename, max_size=(6144, 6144), quality=90):
    if not tour_id or not scene_id or not source_filename:
        return None
    proc_dir = os.path.join(PROCESSED_FOLDER, tour_id, scene_id)
    src_path = os.path.join(proc_dir, source_filename)
    out_path = os.path.join(proc_dir, web_filename)
    try:
        if not os.path.exists(src_path):
            return None
        old_max = getattr(Image, "MAX_IMAGE_PIXELS", None)
        try:
            Image.MAX_IMAGE_PIXELS = None
        except Exception:
            pass
        with Image.open(src_path) as im:
            im = ImageOps.exif_transpose(im)
            if im.mode in ("RGBA", "P"):
                im = im.convert("RGB")
            target_w, target_h = _target_thumbnail_size(im.width, im.height, max_size)
            if os.path.exists(out_path):
                try:
                    with Image.open(out_path) as existing:
                        if existing.width == target_w and existing.height == target_h:
                            return web_filename
                except Exception:
                    pass
            if im.width <= max_size[0] and im.height <= max_size[1]:
                return None
            try:
                resample = Image.Resampling.LANCZOS
            except Exception:
                resample = Image.LANCZOS
            if im.width > max_size[0] or im.height > max_size[1]:
                im.thumbnail(max_size, resample)
            im.save(out_path, "JPEG", quality=int(quality), optimize=True, progressive=True, subsampling=0)
        try:
            Image.MAX_IMAGE_PIXELS = old_max
        except Exception:
            pass
        return web_filename
    except Exception:
        return None

def ensure_scene_studio_web_pano(tour_id, scene_id, source_filename):
    return ensure_scene_web_pano(
        tour_id,
        scene_id,
        source_filename,
        web_filename=STUDIO_WEB_PANO_FILENAME,
        max_size=(4096, 4096),
        quality=88,
    )

def serialize_tour(row):
    return {
        "id": row["id"],
        "owner_id": row["owner_id"],
        "title": row["title"],
        "description": row["description"],
        "slug": row["slug"],
        "visibility": row["visibility"],
        "start_scene_id": row["start_scene_id"] if "start_scene_id" in row.keys() else None,
        "start_pitch": row["start_pitch"] if "start_pitch" in row.keys() else None,
        "start_yaw": row["start_yaw"] if "start_yaw" in row.keys() else None,
        "default_hfov": row["default_hfov"] if "default_hfov" in row.keys() else None,
        "status": row["status"],
        "created_at": row["created_at"],
        "updated_at": row["updated_at"],
    }

def serialize_scene(row):
    studio_web_pano = None
    try:
        pano_name = row["panorama_path"]
        if pano_name:
            studio_web_pano = ensure_scene_studio_web_pano(row["tour_id"], row["id"], pano_name)
    except Exception:
        studio_web_pano = None
    return {
        "id": row["id"],
        "tour_id": row["tour_id"],
        "name": row["title"],
        "panorama": row["panorama_path"],
        "web": studio_web_pano,
        "preview": row["preview_path"],
        "images": json.loads(row["images_json"] or "[]"),
        "haov": row["haov"],
        "vaov": row["vaov"],
        "type": row["scene_type"],
        "order_index": row["order_index"],
        "processing_status": row["processing_status"],
        "processing_error": row["processing_error"],
        "job_id": row["job_id"],
    }

def serialize_job(row):
    return {
        "id": row["id"],
        "kind": row["kind"],
        "owner_id": row["owner_id"],
        "tour_id": row["tour_id"],
        "scene_id": row["scene_id"],
        "status": row["status"],
        "stage": row["stage"],
        "progress_pct": int(row["progress_pct"] or 0),
        "message": row["message"],
        "result": json.loads(row["result_json"] or "null"),
        "error": row["error"],
        "created_at": row["created_at"],
        "updated_at": row["updated_at"],
    }
