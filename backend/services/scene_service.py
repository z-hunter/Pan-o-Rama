import os
import json
import math
import cv2
import numpy as np
import traceback
from PIL import Image, ImageOps
from flask import current_app

from backend.core.config import (
    UPLOAD_FOLDER, PROCESSED_FOLDER, PREVIEW_FILENAME, 
    WEB_PANO_FILENAME, STUDIO_WEB_PANO_FILENAME
)
from backend.core.database import get_db
from backend.core.models import (
    now_iso, safe_float, serialize_scene, 
    ensure_scene_web_pano, ensure_scene_studio_web_pano
)

def looks_like_equirectangular_aspect(aspect):
    # Equirectangular is 2:1
    return 1.8 < aspect < 2.2

def process_image(input_path, output_path, max_size=(16384, 16384), quality=98):
    try:
        with Image.open(input_path) as img:
            img = ImageOps.exif_transpose(img)
            if img.mode in ("RGBA", "P"):
                img = img.convert("RGB")
            if img.width > max_size[0] or img.height > max_size[1]:
                img.thumbnail(max_size, Image.Resampling.LANCZOS)
            img.save(output_path, "JPEG", quality=quality, optimize=True)
        return True
    except Exception:
        return False

def ensure_scene_preview(tour_id, scene_id, source_filename, preview_filename=PREVIEW_FILENAME, max_size=(2048, 1024), quality=82, force=False):
    proc_dir = os.path.join(PROCESSED_FOLDER, tour_id, scene_id)
    src_path = os.path.join(proc_dir, source_filename)
    out_path = os.path.join(proc_dir, preview_filename)
    if os.path.exists(out_path) and not force:
        return preview_filename
    try:
        with Image.open(src_path) as im:
            im.thumbnail(max_size, Image.Resampling.LANCZOS)
            im.save(out_path, "JPEG", quality=quality, optimize=True)
        return preview_filename
    except Exception:
        return None

def stitch_panorama_basic(image_paths, output_path, is_360=True):
    # Simplified fallback stitcher using OpenCV
    stitcher = cv2.Stitcher_create(cv2.Stitcher_PANORAMA)
    imgs = []
    for p in image_paths:
        img = cv2.imread(p)
        if img is not None:
            imgs.append(img)
    if len(imgs) < 2: return False
    status, pano = stitcher.stitch(imgs)
    if status == cv2.Stitcher_OK:
        cv2.imwrite(output_path, pano)
        return True
    return False

def stitch_panorama(image_paths, output_path, is_360=True):
    # Try basic OpenCV stitcher
    return stitch_panorama_basic(image_paths, output_path, is_360)

def postprocess_panorama(path):
    # Placeholder for panorama cleanup logic
    try:
        with Image.open(path) as img:
            return img.width, img.height, False
    except:
        return 0, 0, False

def process_scene_from_raw_paths(tour_id, scene_id, name, is_pano, raw_paths, order_index, job_id=None):
    """
    Core processing logic.
    """
    from backend.services.job_service import update_job_status
    
    if job_id: update_job_status(job_id, stage="processing", progress_pct=10, message="Initializing asset processing...")

    raw_dir = os.path.join(UPLOAD_FOLDER, tour_id, scene_id)
    proc_dir = os.path.join(PROCESSED_FOLDER, tour_id, scene_id)
    os.makedirs(proc_dir, exist_ok=True)

    # 1. Process Raw Images
    processed = []
    for i, rp in enumerate(raw_paths):
        fn = os.path.basename(rp)
        out_fn = f"proc_{fn}"
        out_path = os.path.join(proc_dir, out_fn)
        if process_image(rp, out_path):
            processed.append(out_fn)
        if job_id: update_job_status(job_id, progress_pct=10 + (i/len(raw_paths)*30))

    if not processed:
        raise RuntimeError("No valid images after processing")

    # 2. Stitching / Identification
    focal_35 = 26.0 # Default
    pano_file = None
    haov, vaov = 100, 60
    
    if len(processed) >= 2:
        if job_id: update_job_status(job_id, stage="stitching", progress_pct=40, message="Stitching panorama...")
        ppano = os.path.join(proc_dir, "panorama.jpg")
        if stitch_panorama([os.path.join(proc_dir, fn) for fn in processed], ppano, is_360=is_pano):
            pano_file = "panorama.jpg"
            haov, vaov = 360, 180 # Assumed for stitched
        else:
            pano_file = processed[0]
    else:
        pano_file = processed[0]
        with Image.open(os.path.join(proc_dir, pano_file)) as img:
            aspect = img.width / img.height
            if is_pano or looks_like_equirectangular_aspect(aspect):
                haov, vaov = 360, 180
            else:
                haov, vaov = 100, 60

    # 3. Generating Derivatives
    if job_id: update_job_status(job_id, stage="derivatives", progress_pct=70, message="Generating web assets...")
    
    src_for_preview = pano_file
    preview_path = ensure_scene_preview(tour_id, scene_id, src_for_preview)
    ensure_scene_web_pano(tour_id, scene_id, src_for_preview, WEB_PANO_FILENAME)
    ensure_scene_studio_web_pano(tour_id, scene_id, src_for_preview)

    # 4. Update Database
    ts = now_iso()
    db = get_db()
    db.execute(
        """
        UPDATE scenes
        SET title = ?, panorama_path = ?, preview_path = ?, images_json = ?, order_index = ?,
            haov = ?, vaov = ?, scene_type = 'equirectangular',
            processing_status = 'ready', processing_error = NULL,
            updated_at = ?
        WHERE id = ? AND tour_id = ?
        """,
        (name, pano_file, preview_path, json.dumps(processed), int(order_index),
         round(haov, 2), round(vaov, 2), ts, scene_id, tour_id),
    )
    db.execute("UPDATE tours SET updated_at = ? WHERE id = ?", (ts, tour_id))
    db.commit()
    
    if job_id: update_job_status(job_id, stage="done", progress_pct=100, status="done", message="Asset processing complete.")
    
    row = db.execute("SELECT * FROM scenes WHERE id = ? AND tour_id = ?", (scene_id, tour_id)).fetchone()
    return serialize_scene(row)
