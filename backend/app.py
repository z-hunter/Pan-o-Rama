from flask import Flask, request, jsonify, send_from_directory, send_file, g, redirect
import os
import platform
import time
from werkzeug.utils import secure_filename
from werkzeug.security import generate_password_hash, check_password_hash
from PIL import Image, ExifTags, ImageOps
from io import BytesIO
import uuid
import datetime
import cv2
import logging
from flask_cors import CORS
import json
import math
import numpy as np
import re
import sqlite3
import hashlib
import secrets
import functools
try:
    import stripe
except Exception:
    stripe = None

app = Flask(__name__)
CORS(app)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, '..', 'data')
UPLOAD_FOLDER = os.path.join(DATA_DIR, 'raw_uploads')
PROCESSED_FOLDER = os.path.join(DATA_DIR, 'processed_galleries')
FRONTEND_FOLDER = os.path.join(BASE_DIR, '..', 'frontend')
IMG_FOLDER = os.path.join(BASE_DIR, '..', 'img')
DB_PATH = os.path.join(DATA_DIR, 'app.db')

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['PROCESSED_FOLDER'] = PROCESSED_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 1024 * 1024 * 1024 

# Setup Logging
log_file = os.path.join(BASE_DIR, 'flask_app.log')
handler = logging.FileHandler(log_file)
handler.setFormatter(logging.Formatter('%(asctime)s %(levelname)s: %(message)s'))
app.logger.addHandler(handler)
app.logger.setLevel(logging.DEBUG)

ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif', 'webp'}

PREVIEW_FILENAME = "preview.jpg"
WEB_PANO_FILENAME = "web.jpg"
GALLERY_TEMPLATE_VERSION = 34

@app.route("/__debug/version")
def debug_version():
    # Local debugging helper. Do not return environment variables or secrets.
    return (
        jsonify(
            {
                "now_unix": int(time.time()),
                "pid": os.getpid(),
                "platform": platform.platform(),
                "cwd": os.getcwd(),
                "app_py": os.path.abspath(__file__),
                "gallery_template_version": GALLERY_TEMPLATE_VERSION,
            }
        ),
        200,
    )

PLAN_FREE = "free"
PLAN_PRO = "pro"
PLAN_BUSINESS = "business"
PLAN_ORDER = {PLAN_FREE: 0, PLAN_PRO: 1, PLAN_BUSINESS: 2}
DEFAULT_PLAN_DEFS = {
    PLAN_FREE: {"name": "Free", "max_tours": 2, "watermark_enabled": 1, "price_monthly_cents": 0},
    PLAN_PRO: {"name": "Pro", "max_tours": 50, "watermark_enabled": 0, "price_monthly_cents": 4900},
    PLAN_BUSINESS: {"name": "Business", "max_tours": 500, "watermark_enabled": 0, "price_monthly_cents": 19900},
}

os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(PROCESSED_FOLDER, exist_ok=True)

def natural_sort_key(s):
    return [int(text) if text.isdigit() else text.lower() for text in re.split('([0-9]+)', s)]

def safe_float(val, default=0.0):
    try:
        if isinstance(val, (list, tuple)):
            if len(val) >= 2 and val[1] != 0: return float(val[0]) / float(val[1])
            return float(val[0]) if len(val) > 0 else default
        return float(val)
    except: return default

def build_gpano_xmp(width, height):
    # Minimal GPano payload for Facebook/Google-style panorama viewers.
    # Keep ASCII-only to avoid encoding edge cases.
    return (
        "<x:xmpmeta xmlns:x='adobe:ns:meta/'>"
        "<rdf:RDF xmlns:rdf='http://www.w3.org/1999/02/22-rdf-syntax-ns#'>"
        "<rdf:Description xmlns:GPano='http://ns.google.com/photos/1.0/panorama/' "
        "GPano:UsePanoramaViewer='True' "
        "GPano:ProjectionType='equirectangular' "
        f"GPano:FullPanoWidthPixels='{int(width)}' "
        f"GPano:FullPanoHeightPixels='{int(height)}' "
        f"GPano:CroppedAreaImageWidthPixels='{int(width)}' "
        f"GPano:CroppedAreaImageHeightPixels='{int(height)}' "
        "GPano:CroppedAreaLeftPixels='0' "
        "GPano:CroppedAreaTopPixels='0'/>"
        "</rdf:RDF>"
        "</x:xmpmeta>"
    ).encode("utf-8")

def inject_xmp_into_jpeg(jpeg_bytes, xmp_xml_bytes):
    """
    Insert/replace XMP APP1 segment into a JPEG.
    - Removes existing XMP segments (APP1 with Adobe XMP namespace header).
    - Inserts new XMP APP1 right after SOI.
    """
    if not jpeg_bytes or len(jpeg_bytes) < 4 or jpeg_bytes[0:2] != b"\xff\xd8":
        raise ValueError("Not a JPEG (missing SOI)")

    xmp_header = b"http://ns.adobe.com/xap/1.0/\x00"
    xmp_payload = xmp_header + (xmp_xml_bytes or b"")
    seg_len = len(xmp_payload) + 2
    if seg_len > 0xFFFF:
        raise ValueError("XMP payload too large for APP1")
    app1 = b"\xff\xe1" + seg_len.to_bytes(2, "big") + xmp_payload

    # Rebuild segments: SOI + new APP1 + all non-XMP segments up to SOS + rest.
    i = 2
    out = bytearray()
    out += b"\xff\xd8"
    out += app1

    # Walk segments until SOS/EOI.
    while i + 4 <= len(jpeg_bytes):
        if jpeg_bytes[i] != 0xFF:
            # We are in entropy-coded data (should only happen after SOS), stop copying as segments.
            out += jpeg_bytes[i:]
            return bytes(out)
        # Skip fill bytes 0xFF..0xFF
        while i < len(jpeg_bytes) and jpeg_bytes[i] == 0xFF:
            i += 1
        if i >= len(jpeg_bytes):
            break
        marker = jpeg_bytes[i]
        i += 1

        # Markers without length
        if marker in (0xD8, 0xD9):  # SOI, EOI
            out += b"\xff" + bytes([marker])
            if marker == 0xD9:
                return bytes(out)
            continue
        if marker == 0xDA:  # SOS: copy header (with length) then rest of file as-is.
            if i + 2 > len(jpeg_bytes):
                break
            segl = int.from_bytes(jpeg_bytes[i:i+2], "big")
            seg_start = i - 2  # include length bytes
            seg_end = seg_start + segl
            if seg_end > len(jpeg_bytes):
                break
            out += b"\xff" + bytes([marker]) + jpeg_bytes[seg_start:seg_end]
            out += jpeg_bytes[seg_end:]
            return bytes(out)

        # Normal segment with length
        if i + 2 > len(jpeg_bytes):
            break
        segl = int.from_bytes(jpeg_bytes[i:i+2], "big")
        seg_start = i
        seg_end = i + segl
        if seg_end > len(jpeg_bytes):
            break
        seg_data = jpeg_bytes[seg_start:seg_end]

        if marker == 0xE1 and seg_data.startswith(xmp_header):
            # Drop existing XMP APP1.
            i = seg_end
            continue

        out += b"\xff" + bytes([marker]) + seg_data
        i = seg_end

    # Fallback: if parsing failed, at least return original bytes (without modifications).
    return jpeg_bytes

def detect_and_crop_overlap(image_path):
    """
    CV-based overlap detection for 360 looping.
    EXTREMELY strict to avoid mid-image cuts.
    """
    try:
        img = cv2.imread(image_path)
        if img is None: return None
        h, w, _ = img.shape
        # Only look at the very edges (3%) to ensure we only match start vs end
        strip_w = int(w * 0.03)
        if strip_w < 50: strip_w = 50
        
        left_strip = img[:, :strip_w]
        right_strip = img[:, (w - strip_w):]
        
        orb = cv2.ORB_create(nfeatures=5000)
        kp1, des1 = orb.detectAndCompute(left_strip, None)
        kp2, des2 = orb.detectAndCompute(right_strip, None)
        
        if des1 is None or des2 is None: return None
        
        bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
        matches = bf.match(des1, des2)
        
        if len(matches) < 10: return None
        
        offsets = []
        for m in matches:
            x_left, y_left = kp1[m.queryIdx].pt
            x_right, y_right = kp2[m.trainIdx].pt
            # Very tight vertical tolerance
            if abs(y_left - y_right) < h * 0.002:
                # Calculate real image width if we were to cut here
                offset = (w - strip_w) + x_right - x_left
                # Only accept if it's a 360 wrap-around (> 95% width)
                if offset > w * 0.95:
                    offsets.append(offset)
        
        if len(offsets) < 5: return None
        best_offset = int(np.median(offsets))
        
        # Only crop if it's a safe 360 loop overlap (max 10% overlap)
        if best_offset > w * 0.90:
            app.logger.info(f"Loop closure detected. Trimming {w - best_offset}px")
            cropped = img[:, :best_offset]
            cv2.imwrite(image_path, cropped, [cv2.IMWRITE_JPEG_QUALITY, 100])
            return best_offset
    except Exception as e: app.logger.error(f"Overlap error: {e}")
    return None

def detect_and_crop_overlap_wide(image_path, min_ratio=0.08, max_ratio=0.45):
    """
    Overlap detection for single wide panoramas.
    Uses coarse SAD matching between left and right ends to find best overlap width.
    """
    try:
        img = cv2.imread(image_path)
        if img is None:
            return None
        h, w = img.shape[:2]
        if w < 200:
            return None
        # Downscale for speed
        target_h = 400
        scale = target_h / h
        if scale < 1.0:
            img_s = cv2.resize(img, (int(w * scale), target_h), interpolation=cv2.INTER_AREA)
        else:
            img_s = img.copy()
        gray = cv2.cvtColor(img_s, cv2.COLOR_BGR2GRAY)
        hs, ws = gray.shape[:2]

        min_ov = int(ws * min_ratio)
        max_ov = int(ws * max_ratio)
        if max_ov <= min_ov + 10:
            return None

        best_ov = None
        best_score = None
        for ov in range(min_ov, max_ov):
            left = gray[:, :ov]
            right = gray[:, ws - ov:]
            # Use correlation (higher is better)
            l = left.astype(np.float32)
            r = right.astype(np.float32)
            l -= l.mean()
            r -= r.mean()
            denom = (np.linalg.norm(l) * np.linalg.norm(r)) + 1e-6
            score = float((l * r).sum() / denom)
            if best_score is None or score > best_score:
                best_score = score
                best_ov = ov

        # Heuristic threshold: if overlap is plausible (high correlation), trim
        if best_ov is not None and best_score is not None and best_score > 0.25:
            # Refine cut position to avoid slicing through strong edges
            cut_center = ws - best_ov
            # Precompute gradient magnitude
            gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
            gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
            mag = cv2.magnitude(gx, gy)
            window = int(ws * 0.02)  # search +/-2% width
            band = 6  # pixels around seam
            best_cut = cut_center
            best_edge = None
            for dx in range(-window, window + 1, 1):
                x = cut_center + dx
                if x - band < 0 or x + band >= ws:
                    continue
                seam_band = mag[:, x - band:x + band]
                edge = float(np.mean(seam_band))
                if best_edge is None or edge < best_edge:
                    best_edge = edge
                    best_cut = x

            # Convert cut position back to overlap width
            best_ov = ws - best_cut
            # Map overlap back to original scale
            ov_full = int(best_ov / (ws / w))
            if ov_full > 0 and ov_full < w:
                new_w = w - ov_full
                app.logger.info(f"Wide overlap detected. Trimming {ov_full}px (corr={best_score:.3f}, edge={best_edge:.2f})")
                cropped = img[:, :new_w]
                cv2.imwrite(image_path, cropped, [cv2.IMWRITE_JPEG_QUALITY, 100])
                return new_w
    except Exception as e:
        app.logger.error(f"Wide overlap error: {e}")
    return None

def stitch_panorama_detail(image_paths, output_path, is_360=True):
    # Detail pipeline disabled (quality/memory issues); use basic Stitcher.
    return False
    """
    Custom detail stitching pipeline (cv2.detail) to reduce frame dropping
    for handheld 360 captures. Returns True on success.
    """
    if not hasattr(cv2, "detail"):
        app.logger.warning("cv2.detail not available; skipping detail stitcher")
        return False
    try:
        app.logger.info(f"Detail stitching {len(image_paths)} images...")
        # Lower scales to reduce memory usage on VPS
        work_megapix = 0.3
        seam_megapix = 0.05
        compose_megapix = 0.3
        conf_thresh = 0.3

        full_images = []
        images = []
        full_sizes = []
        work_scale = None
        seam_scale = None
        seam_work_aspect = None

        finder = cv2.ORB_create(nfeatures=2000)
        features = []
        app.logger.info(f"Stitch input paths: {', '.join([os.path.basename(p) for p in image_paths])}")

        for path in image_paths:
            full_img = cv2.imread(path)
            if full_img is None:
                app.logger.warning(f"Failed to read image: {path}")
                continue
            full_images.append(full_img)
            full_sizes.append((full_img.shape[1], full_img.shape[0]))
            if work_scale is None:
                work_scale = min(1.0, math.sqrt((work_megapix * 1e6) / (full_img.shape[0] * full_img.shape[1])))
                seam_scale = min(1.0, math.sqrt((seam_megapix * 1e6) / (full_img.shape[0] * full_img.shape[1])))
                seam_work_aspect = seam_scale / work_scale

            img = cv2.resize(full_img, (int(full_img.shape[1] * work_scale), int(full_img.shape[0] * work_scale)), interpolation=cv2.INTER_LINEAR)
            images.append(img)
            feat = cv2.detail.computeImageFeatures2(finder, img)
            features.append(feat)

        if len(images) < 2:
            return False

        # Feature matching
        try:
            matcher = cv2.detail_BestOf2NearestMatcher_create(False, conf_thresh)
        except Exception:
            matcher = cv2.detail.BestOf2NearestMatcher(False, conf_thresh)
        matches = matcher.apply2(features)
        matcher.collectGarbage()

        # Keep only the largest connected component
        try:
            indices = cv2.detail.leaveBiggestComponent(features, matches, conf_thresh)
        except Exception:
            indices = list(range(len(images)))
        if indices is None or len(indices) < 2:
            app.logger.warning("Detail stitcher: not enough connected images")
            return False

        features = [features[i] for i in indices]
        images = [images[i] for i in indices]
        full_images = [full_images[i] for i in indices]
        full_sizes = [full_sizes[i] for i in indices]

        # Camera parameters
        estimator = cv2.detail_HomographyBasedEstimator()
        ok, cameras = estimator.apply(features, matches, None)
        if not ok:
            app.logger.warning("Detail stitcher: camera estimation failed")
            return False
        for cam in cameras:
            cam.R = cam.R.astype(np.float32)

        adjuster = cv2.detail_BundleAdjusterRay()
        adjuster.setConfThresh(1.0)
        refine_mask = np.zeros((3, 3), np.uint8)
        refine_mask[0, 0] = 1
        refine_mask[0, 1] = 1
        refine_mask[0, 2] = 1
        refine_mask[1, 1] = 1
        refine_mask[1, 2] = 1
        adjuster.setRefinementMask(refine_mask)
        ok, cameras = adjuster.apply(features, matches, cameras)
        if not ok:
            app.logger.warning("Detail stitcher: bundle adjustment failed")
            return False

        # Wave correction
        try:
            cam_Rs = [cam.R for cam in cameras]
            cv2.detail.waveCorrect(cam_Rs, cv2.detail.WAVE_CORRECT_HORIZ)
            for i, cam in enumerate(cameras):
                cam.R = cam_Rs[i]
        except Exception:
            pass

        # Prefer 35mm-equivalent focal from EXIF (more stable for handheld)
        exif_focal_35 = None
        try:
            with Image.open(image_paths[0]) as img_ex:
                exif = img_ex.getexif()
                if exif:
                    exif_focal_35 = safe_float(exif.get(41989))
        except Exception:
            exif_focal_35 = None

        if exif_focal_35 and full_sizes:
            full_w = full_sizes[0][0]
            # Focal length in pixels at full resolution, scaled to work size
            warped_image_scale = float((full_w * exif_focal_35 / 36.0) * work_scale)
            app.logger.info(f"Detail stitcher: using EXIF focal_35={exif_focal_35}, warped_scale={warped_image_scale:.2f}")
        else:
            focals = [cam.focal for cam in cameras]
            warped_image_scale = float(np.median(focals))

        # Warp images for seam estimation
        warper = cv2.PyRotationWarper('spherical', warped_image_scale * seam_work_aspect)
        corners = []
        masks_warped = []
        images_warped = []
        sizes = []

        for i, img in enumerate(images):
            K = cameras[i].K().astype(np.float32)
            K[0, 0] *= seam_work_aspect
            K[1, 1] *= seam_work_aspect
            K[0, 2] *= seam_work_aspect
            K[1, 2] *= seam_work_aspect

            corner, image_warped = warper.warp(img, K, cameras[i].R, cv2.INTER_LINEAR, cv2.BORDER_REFLECT)
            mask = 255 * np.ones((img.shape[0], img.shape[1]), np.uint8)
            _, mask_warped = warper.warp(mask, K, cameras[i].R, cv2.INTER_NEAREST, cv2.BORDER_CONSTANT)
            corners.append(corner)
            images_warped.append(image_warped)
            masks_warped.append(mask_warped)
            sizes.append((image_warped.shape[1], image_warped.shape[0]))

        try:
            seam_finder = cv2.detail_DpSeamFinder('COLOR_GRAD')
        except Exception:
            try:
                seam_finder = cv2.detail.SeamFinder_createDefault(cv2.detail.SeamFinder_VORONOI_SEAM)
            except Exception:
                seam_finder = cv2.detail_SeamFinder_createDefault(cv2.detail.SeamFinder_VORONOI_SEAM)
        seam_finder.find(images_warped, corners, masks_warped)

        # Exposure compensation (disabled to reduce memory)
        compensator = None

        # Normalize corners to avoid negative offsets in blender
        min_x = min([c[0] for c in corners]) if corners else 0
        min_y = min([c[1] for c in corners]) if corners else 0
        shift = (0, 0)
        if min_x < 0 or min_y < 0:
            shift = (-min_x if min_x < 0 else 0, -min_y if min_y < 0 else 0)
            corners = [(c[0] + shift[0], c[1] + shift[1]) for c in corners]
            app.logger.info(f"Detail stitcher: shifted corners by {shift}")

        # Simple composite fallback (no multiband blending) to avoid blender issues
        use_simple_blend = True
        if not use_simple_blend:
            blender = cv2.detail.Blender_createDefault(cv2.detail.Blender_MULTI_BAND, False)
            blender.prepare(corners, sizes)

        # Composition at full resolution (or compose_megapix)
        for i, full_img in enumerate(full_images):
            if compose_megapix > 0:
                compose_scale = min(1.0, math.sqrt((compose_megapix * 1e6) / (full_img.shape[0] * full_img.shape[1])))
            else:
                compose_scale = 1.0
            compose_work_aspect = compose_scale / work_scale

            img = cv2.resize(full_img, (int(full_img.shape[1] * compose_scale), int(full_img.shape[0] * compose_scale)), interpolation=cv2.INTER_LINEAR)
            K = cameras[i].K().astype(np.float32)
            K[0, 0] *= compose_work_aspect
            K[1, 1] *= compose_work_aspect
            K[0, 2] *= compose_work_aspect
            K[1, 2] *= compose_work_aspect

            warper = cv2.PyRotationWarper('spherical', warped_image_scale * compose_work_aspect)
            corner, image_warped = warper.warp(img, K, cameras[i].R, cv2.INTER_LINEAR, cv2.BORDER_REFLECT)
            mask = 255 * np.ones((img.shape[0], img.shape[1]), np.uint8)
            _, mask_warped = warper.warp(mask, K, cameras[i].R, cv2.INTER_NEAREST, cv2.BORDER_CONSTANT)

            # Apply same shift used for blender.prepare to avoid negative corners
            corner = (corner[0] + shift[0], corner[1] + shift[1])
            if compensator is not None:
                compensator.apply(i, corner, image_warped, mask_warped)
            if use_simple_blend:
                if i == 0:
                    simple_img = image_warped.copy()
                    simple_mask = mask_warped.copy()
                    simple_corner = corner
                else:
                    # Expand canvas if needed
                    min_x = min(simple_corner[0], corner[0])
                    min_y = min(simple_corner[1], corner[1])
                    max_x = max(simple_corner[0] + simple_img.shape[1], corner[0] + image_warped.shape[1])
                    max_y = max(simple_corner[1] + simple_img.shape[0], corner[1] + image_warped.shape[0])

                    new_w = max_x - min_x
                    new_h = max_y - min_y
                    new_img = np.zeros((new_h, new_w, 3), dtype=np.uint8)
                    new_mask = np.zeros((new_h, new_w), dtype=np.uint8)

                    # Paste existing
                    sx = simple_corner[0] - min_x
                    sy = simple_corner[1] - min_y
                    new_img[sy:sy + simple_img.shape[0], sx:sx + simple_img.shape[1]] = simple_img
                    new_mask[sy:sy + simple_mask.shape[0], sx:sx + simple_mask.shape[1]] = simple_mask

                    # Paste new where mask is set (with feather blending)
                    nx = corner[0] - min_x
                    ny = corner[1] - min_y
                    roi_img = new_img[ny:ny + image_warped.shape[0], nx:nx + image_warped.shape[1]]
                    roi_mask = new_mask[ny:ny + mask_warped.shape[0], nx:nx + mask_warped.shape[1]]
                    m_new = (mask_warped > 0).astype(np.uint8)
                    m_old = (roi_mask > 0).astype(np.uint8)
                    # Distance to mask edge for feather weights
                    dist_new = cv2.distanceTransform(m_new, cv2.DIST_L2, 3).astype(np.float32)
                    dist_old = cv2.distanceTransform(m_old, cv2.DIST_L2, 3).astype(np.float32)
                    w_new = dist_new
                    w_old = dist_old
                    w_sum = w_new + w_old
                    w_new = np.where(w_sum > 0, w_new / w_sum, 0)
                    w_old = np.where(w_sum > 0, w_old / w_sum, 0)

                    # Blend in overlap only on smooth areas; use seam-cut on detailed areas
                    if roi_img.dtype != np.float32:
                        roi_img_f = roi_img.astype(np.float32)
                    else:
                        roi_img_f = roi_img
                    new_img_f = image_warped.astype(np.float32)

                    overlap = (m_new > 0) & (m_old > 0)
                    only_new = (m_new > 0) & (m_old == 0)

                    if overlap.any():
                        # Edge mask based on gradient magnitude
                        roi_gray = cv2.cvtColor(roi_img, cv2.COLOR_BGR2GRAY)
                        new_gray = cv2.cvtColor(image_warped, cv2.COLOR_BGR2GRAY)
                        gx1 = cv2.Sobel(roi_gray, cv2.CV_32F, 1, 0, ksize=3)
                        gy1 = cv2.Sobel(roi_gray, cv2.CV_32F, 0, 1, ksize=3)
                        gx2 = cv2.Sobel(new_gray, cv2.CV_32F, 1, 0, ksize=3)
                        gy2 = cv2.Sobel(new_gray, cv2.CV_32F, 0, 1, ksize=3)
                        mag1 = cv2.magnitude(gx1, gy1)
                        mag2 = cv2.magnitude(gx2, gy2)
                        smooth = (mag1 < 12) & (mag2 < 12)

                        smooth_overlap = overlap & smooth
                        hard_overlap = overlap & (~smooth)

                        if smooth_overlap.any():
                            w_new_o = w_new[smooth_overlap][..., None]
                            w_old_o = w_old[smooth_overlap][..., None]
                            roi_img_f[smooth_overlap] = roi_img_f[smooth_overlap] * w_old_o + new_img_f[smooth_overlap] * w_new_o

                        if hard_overlap.any():
                            # Seam-cut: prefer pixels from new mask
                            take_new = hard_overlap & (m_new > 0)
                            roi_img_f[take_new] = new_img_f[take_new]
                    if only_new.any():
                        roi_img_f[only_new] = new_img_f[only_new]

                    roi_img[:] = np.clip(roi_img_f, 0, 255).astype(np.uint8)
                    roi_mask[m_new > 0] = 255

                    simple_img = new_img
                    simple_mask = new_mask
                    simple_corner = (min_x, min_y)
            else:
                blender.feed(image_warped, mask_warped, corner)

        if use_simple_blend:
            result = simple_img
            result_mask = simple_mask
        else:
            result, result_mask = blender.blend(None, None)
            if result is None:
                return False

        # Auto-crop by largest valid mask region to remove empty zones
        try:
            mask = (result_mask > 0).astype(np.uint8) * 255
            # Close small holes and connect nearby regions
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (21, 21))
            mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            if contours:
                # Take largest contour
                cnt = max(contours, key=cv2.contourArea)
                x, y, w, h = cv2.boundingRect(cnt)
                result = result[y:y + h, x:x + w]
                app.logger.info(f"Detail stitcher: mask-cropped to {w}x{h}")
        except Exception as e:
            app.logger.warning(f"Mask crop failed: {e}")
        if result.dtype != np.uint8:
            result = np.clip(result, 0, 255).astype(np.uint8)
        cv2.imwrite(output_path, result, [cv2.IMWRITE_JPEG_QUALITY, 100])
        return True
    except Exception as e:
        app.logger.error(f"Detail stitch error: {e}", exc_info=True)
        return False

def stitch_panorama_basic(image_paths, output_path, is_360=True):
    """Robust stitching with verification."""
    app.logger.info(f"Stitching {len(image_paths)} images...")
    images = []
    total_in_w = 0
    # Higher resolution for stitching (3000px height)
    target_h = 3000
    app.logger.info(f"Stitch input paths: {', '.join([os.path.basename(p) for p in image_paths])}")
    for path in image_paths:
        img = cv2.imread(path)
        if img is not None:
            h, w = img.shape[:2]
            scale = target_h / h
            img_r = cv2.resize(img, (int(w * scale), target_h), interpolation=cv2.INTER_LANCZOS4)
            images.append(img_r)
            total_in_w += img_r.shape[1]
        else:
            app.logger.warning(f"Failed to read image: {path}")
    
    if len(images) < 2: return False
    
    # Modes to try: PANORAMA (0), SCANS (1)
    # For 360, stick to PANORAMA to avoid SCANS "collage" artifacts.
    modes = [0] if is_360 else [1, 0]
    best_img = None
    best_w = 0
    
    def max_gap_ratio(img):
        try:
            h, w = img.shape[:2]
            if w < 2: return 0.0
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            col_black = (gray < 5).mean(axis=0)
            gap_cols = col_black > 0.9
            max_gap = 0
            run = 0
            for g in gap_cols:
                if g:
                    run += 1
                    if run > max_gap: max_gap = run
                else:
                    run = 0
            return max_gap / max(1, w)
        except Exception as e:
            app.logger.warning(f"Gap ratio error: {e}")
            return 0.0

    for mode in modes:
        try:
            # Try a strict pass first, then a more permissive pass if we likely dropped frames
            configs = [
                {"conf": 0.8, "wave": True},
                {"conf": 0.6, "wave": False},
            ] if is_360 and mode == 0 else [{"conf": 0.8, "wave": True}]

            for cfg in configs:
                app.logger.info(f"Attempting stitch mode {mode} conf {cfg['conf']} wave {cfg['wave']}...")
                stitcher = cv2.Stitcher.create(mode)
                stitcher.setPanoConfidenceThresh(cfg["conf"])
                try:
                    stitcher.setWaveCorrection(cfg["wave"])
                except Exception:
                    pass

                status, stitched = stitcher.stitch(images)
                if status == 0:
                    h, w = stitched.shape[:2]
                    gap_ratio = max_gap_ratio(stitched)
                    aspect = w / h if h else 0
                    app.logger.info(
                        f"Success mode {mode} conf {cfg['conf']}: {w}x{h}, aspect={aspect:.3f}, "
                        f"gap_ratio={gap_ratio:.4f}, total_in_w={total_in_w}"
                    )

                    # Prefer wider, but avoid big missing vertical gaps
                    if best_img is None:
                        best_img, best_w, best_gap = stitched, w, gap_ratio
                    else:
                        # If new is wider and not much worse in gap, take it
                        if w > best_w and gap_ratio <= best_gap + 0.01:
                            best_img, best_w, best_gap = stitched, w, gap_ratio
                        # If current best has large gaps and new is much cleaner, take it
                        elif best_gap > 0.02 and gap_ratio < best_gap and w >= best_w * 0.9:
                            best_img, best_w, best_gap = stitched, w, gap_ratio
                else:
                    app.logger.warning(f"Mode {mode} conf {cfg['conf']} failed with status {status}")
        except Exception as e: app.logger.error(f"Stitch error mode {mode}: {e}")

    if best_img is not None:
        cv2.imwrite(output_path, best_img, [cv2.IMWRITE_JPEG_QUALITY, 100])
        return True
    return False

def stitch_panorama(image_paths, output_path, is_360=True):
    # Use basic Stitcher for quality and stability
    return stitch_panorama_basic(image_paths, output_path, is_360=is_360)

def postprocess_panorama(image_path):
    """
    Remove large black gaps/borders from stitched panorama.
    Returns (width, height, changed).
    """
    try:
        img = cv2.imread(image_path)
        if img is None:
            return None, None, False
        h, w = img.shape[:2]
        changed = False

        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        col_black = (gray < 10).mean(axis=0)
        row_black = (gray < 10).mean(axis=1)

        # Trim black borders (top/bottom/left/right)
        top = 0
        while top < h and row_black[top] > 0.98: top += 1
        bottom = h - 1
        while bottom > 0 and row_black[bottom] > 0.98: bottom -= 1
        left = 0
        while left < w and col_black[left] > 0.98: left += 1
        right = w - 1
        while right > 0 and col_black[right] > 0.98: right -= 1

        if top > 0 or bottom < h - 1 or left > 0 or right < w - 1:
            img = img[top:bottom + 1, left:right + 1]
            changed = True
            h, w = img.shape[:2]
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            col_black = (gray < 10).mean(axis=0)

        # Remove largest internal black vertical gap
        gap_cols = col_black > 0.95
        if gap_cols.any():
            max_gap = 0
            max_start = 0
            run = 0
            start = 0
            for i, g in enumerate(gap_cols):
                if g and run == 0:
                    start = i
                if g:
                    run += 1
                    if run > max_gap:
                        max_gap = run
                        max_start = start
                else:
                    run = 0
            if max_gap > w * 0.02 and max_start > 0 and (max_start + max_gap) < w - 1:
                left_img = img[:, :max_start]
                right_img = img[:, max_start + max_gap:]
                img = np.concatenate([left_img, right_img], axis=1)
                changed = True
                h, w = img.shape[:2]

        if changed:
            cv2.imwrite(image_path, img, [cv2.IMWRITE_JPEG_QUALITY, 100])
        return w, h, changed
    except Exception as e:
        app.logger.warning(f"Postprocess panorama failed: {e}")
        return None, None, False

def process_image(input_path, output_path, max_size=(16384, 16384), quality=98):
    try:
        with Image.open(input_path) as img:
            img = ImageOps.exif_transpose(img)
            if img.mode in ("RGBA", "P"): img = img.convert("RGB")
            if img.width > max_size[0]: img.thumbnail(max_size, Image.Resampling.LANCZOS)
            img.save(output_path, "JPEG", quality=quality, optimize=True, subsampling=0)
            return True
    except: return False

def ensure_scene_preview(tour_id, scene_id, source_filename, preview_filename=PREVIEW_FILENAME, max_size=(2048, 1024), quality=82, force=False):
    """
    Create a low-res preview for quick transitions. Stored next to scene assets under processed_galleries.
    Returns preview_filename when successful, else None.
    """
    if not tour_id or not scene_id or not source_filename:
        return None
    proc_dir = os.path.join(PROCESSED_FOLDER, tour_id, scene_id)
    src_path = os.path.join(proc_dir, source_filename)
    out_path = os.path.join(proc_dir, preview_filename)
    try:
        if (not force) and os.path.exists(out_path):
            return preview_filename
        if not os.path.exists(src_path):
            return None
        # Panoramas can be very large; allow large images and try to hint to decoder that we only need a draft.
        old_max = getattr(Image, "MAX_IMAGE_PIXELS", None)
        try:
            Image.MAX_IMAGE_PIXELS = None
        except Exception:
            pass
        with Image.open(src_path) as im:
            try:
                im.draft("RGB", max_size)
            except Exception:
                pass
            im = ImageOps.exif_transpose(im)
            if im.mode in ("RGBA", "P"):
                im = im.convert("RGB")
            # Preview must preserve proportions. Do NOT crop or force a 2:1 canvas here; some tours use
            # non-2:1 sources and forcing 2:1 can look like distortion/zoom.
            try:
                resample = Image.Resampling.LANCZOS
            except Exception:
                resample = Image.LANCZOS
            try:
                im.thumbnail(max_size, resample)  # preserves aspect ratio
            except Exception:
                # If thumbnail fails, keep original image (better than distorting).
                pass
            im.save(out_path, "JPEG", quality=int(quality), optimize=True, progressive=True, subsampling=0)
        try:
            Image.MAX_IMAGE_PIXELS = old_max
        except Exception:
            pass
        return preview_filename
    except Exception as e:
        app.logger.warning(f"Preview generation failed: {e}")
        return None

def ensure_scene_web_pano(tour_id, scene_id, source_filename, web_filename=WEB_PANO_FILENAME, max_size=(8192, 8192), quality=90):
    """
    Create a "web-safe" panorama size to avoid extremely large textures that can freeze the browser.
    Stored next to scene assets under processed_galleries.
    Returns web_filename when successful, else None.
    """
    if not tour_id or not scene_id or not source_filename:
        return None
    proc_dir = os.path.join(PROCESSED_FOLDER, tour_id, scene_id)
    src_path = os.path.join(proc_dir, source_filename)
    out_path = os.path.join(proc_dir, web_filename)
    try:
        if os.path.exists(out_path):
            return web_filename
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
            # If the source is already reasonably sized, don't create a redundant web copy.
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
    except Exception as e:
        app.logger.warning(f"Web pano generation failed: {e}")
        return None

def now_iso():
    return datetime.datetime.utcnow().replace(microsecond=0).isoformat() + "Z"

def get_db():
    if "db" not in g:
        g.db = sqlite3.connect(DB_PATH)
        g.db.row_factory = sqlite3.Row
    return g.db

@app.teardown_appcontext
def close_db(_error):
    db = g.pop("db", None)
    if db is not None:
        db.close()

def init_db():
    os.makedirs(DATA_DIR, exist_ok=True)
    db = sqlite3.connect(DB_PATH)
    try:
        db.executescript(
            """
            PRAGMA foreign_keys = ON;
            CREATE TABLE IF NOT EXISTS users (
                id TEXT PRIMARY KEY,
                email TEXT NOT NULL UNIQUE,
                password_hash TEXT NOT NULL,
                display_name TEXT NOT NULL,
                status TEXT NOT NULL DEFAULT 'active',
                is_admin INTEGER NOT NULL DEFAULT 0,
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL
            );
            CREATE TABLE IF NOT EXISTS sessions (
                id TEXT PRIMARY KEY,
                user_id TEXT NOT NULL,
                token_hash TEXT NOT NULL,
                expires_at TEXT NOT NULL,
                created_at TEXT NOT NULL,
                user_agent TEXT,
                ip TEXT,
                FOREIGN KEY (user_id) REFERENCES users(id) ON DELETE CASCADE
            );
            CREATE TABLE IF NOT EXISTS tours (
                id TEXT PRIMARY KEY,
                owner_id TEXT NOT NULL,
                title TEXT NOT NULL,
                description TEXT NOT NULL DEFAULT '',
                slug TEXT NOT NULL UNIQUE,
                visibility TEXT NOT NULL CHECK(visibility IN ('public','private')),
                status TEXT NOT NULL DEFAULT 'draft',
                deleted_at TEXT,
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL,
                FOREIGN KEY (owner_id) REFERENCES users(id) ON DELETE CASCADE
            );
	            CREATE TABLE IF NOT EXISTS scenes (
	                id TEXT PRIMARY KEY,
	                tour_id TEXT NOT NULL,
	                title TEXT NOT NULL,
	                panorama_path TEXT,
	                preview_path TEXT,
	                images_json TEXT NOT NULL DEFAULT '[]',
	                order_index INTEGER NOT NULL,
	                haov REAL NOT NULL DEFAULT 360,
	                vaov REAL NOT NULL DEFAULT 180,
                scene_type TEXT NOT NULL DEFAULT 'equirectangular',
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL,
                FOREIGN KEY (tour_id) REFERENCES tours(id) ON DELETE CASCADE
            );
            CREATE TABLE IF NOT EXISTS hotspots (
                id TEXT PRIMARY KEY,
                tour_id TEXT NOT NULL,
                from_scene_id TEXT NOT NULL,
                to_scene_id TEXT NOT NULL,
                yaw REAL NOT NULL,
                pitch REAL NOT NULL,
                entry_yaw REAL,
                entry_pitch REAL,
                label TEXT,
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL,
                FOREIGN KEY (tour_id) REFERENCES tours(id) ON DELETE CASCADE,
                FOREIGN KEY (from_scene_id) REFERENCES scenes(id) ON DELETE CASCADE,
                FOREIGN KEY (to_scene_id) REFERENCES scenes(id) ON DELETE CASCADE
            );
            CREATE TABLE IF NOT EXISTS plans (
                id TEXT PRIMARY KEY,
                name TEXT NOT NULL,
                max_tours INTEGER NOT NULL,
                watermark_enabled INTEGER NOT NULL DEFAULT 1,
                price_monthly_cents INTEGER NOT NULL DEFAULT 0,
                is_active INTEGER NOT NULL DEFAULT 1,
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL
            );
            CREATE TABLE IF NOT EXISTS subscriptions (
                id TEXT PRIMARY KEY,
                user_id TEXT NOT NULL,
                plan_id TEXT NOT NULL,
                status TEXT NOT NULL DEFAULT 'active',
                billing_provider TEXT NOT NULL DEFAULT 'mock',
                provider_customer_id TEXT,
                provider_subscription_id TEXT,
                current_period_end TEXT,
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL,
                FOREIGN KEY (user_id) REFERENCES users(id) ON DELETE CASCADE,
                FOREIGN KEY (plan_id) REFERENCES plans(id)
            );
                CREATE TABLE IF NOT EXISTS usage_counters (
                    user_id TEXT PRIMARY KEY,
                    tours_count INTEGER NOT NULL DEFAULT 0,
                    updated_at TEXT NOT NULL,
                    FOREIGN KEY (user_id) REFERENCES users(id) ON DELETE CASCADE
                );
                -- Small KV cache for billing-related runtime ids (e.g., Stripe price ids created from $ amounts in dev).
                CREATE TABLE IF NOT EXISTS billing_kv (
                    key TEXT PRIMARY KEY,
                    value TEXT NOT NULL,
                    updated_at TEXT NOT NULL
                );
                CREATE INDEX IF NOT EXISTS idx_tours_owner ON tours(owner_id);
                CREATE INDEX IF NOT EXISTS idx_tours_slug ON tours(slug);
                CREATE INDEX IF NOT EXISTS idx_scenes_tour ON scenes(tour_id);
                CREATE INDEX IF NOT EXISTS idx_hotspots_scene ON hotspots(from_scene_id);
                CREATE INDEX IF NOT EXISTS idx_subscriptions_user ON subscriptions(user_id, created_at);
                CREATE INDEX IF NOT EXISTS idx_subscriptions_provider_sub ON subscriptions(provider_subscription_id);
            """
        )
        cols = [r[1] for r in db.execute("PRAGMA table_info(hotspots)").fetchall()]
        if "entry_yaw" not in cols:
            db.execute("ALTER TABLE hotspots ADD COLUMN entry_yaw REAL")
        if "entry_pitch" not in cols:
            db.execute("ALTER TABLE hotspots ADD COLUMN entry_pitch REAL")
        scene_cols = [r[1] for r in db.execute("PRAGMA table_info(scenes)").fetchall()]
        if "preview_path" not in scene_cols:
            db.execute("ALTER TABLE scenes ADD COLUMN preview_path TEXT")
        ts = now_iso()
        for plan_id, d in DEFAULT_PLAN_DEFS.items():
            db.execute(
                """
                INSERT OR IGNORE INTO plans (id, name, max_tours, watermark_enabled, price_monthly_cents, is_active, created_at, updated_at)
                VALUES (?, ?, ?, ?, ?, 1, ?, ?)
                """,
                (plan_id, d["name"], d["max_tours"], d["watermark_enabled"], d["price_monthly_cents"], ts, ts),
            )
        db.commit()
    finally:
        db.close()

def hash_token(token):
    return hashlib.sha256(token.encode("utf-8")).hexdigest()

def get_current_user():
    token = request.cookies.get("session_token")
    if not token:
        return None
    db = get_db()
    token_h = hash_token(token)
    row = db.execute(
        """
        SELECT u.*
        FROM sessions s
        JOIN users u ON u.id = s.user_id
        WHERE s.token_hash = ? AND s.expires_at > ? AND u.status = 'active'
        """,
        (token_h, now_iso()),
    ).fetchone()
    return row

@app.before_request
def attach_current_user():
    g.current_user = get_current_user()

def require_auth(view_fn):
    @functools.wraps(view_fn)
    def wrapper(*args, **kwargs):
        if g.current_user is None:
            return jsonify({"error": "Unauthorized"}), 401
        return view_fn(*args, **kwargs)
    return wrapper

def normalize_visibility(val):
    return "public" if val == "public" else "private"

def slugify(title):
    base = re.sub(r"[^a-zA-Z0-9]+", "-", (title or "").strip().lower()).strip("-")
    if not base:
        base = "tour"
    return f"{base}-{secrets.token_hex(3)}"

def serialize_tour(row):
    return {
        "id": row["id"],
        "owner_id": row["owner_id"],
        "title": row["title"],
        "description": row["description"],
        "slug": row["slug"],
        "visibility": row["visibility"],
        "status": row["status"],
        "created_at": row["created_at"],
        "updated_at": row["updated_at"],
    }

def serialize_scene(row):
    return {
        "id": row["id"],
        "tour_id": row["tour_id"],
        "name": row["title"],
        "panorama": row["panorama_path"],
        "preview": row["preview_path"],
        "images": json.loads(row["images_json"] or "[]"),
        "haov": row["haov"],
        "vaov": row["vaov"],
        "type": row["scene_type"],
        "order_index": row["order_index"],
    }

def get_billing_mode():
    mode = (os.getenv("BILLING_MODE") or "hybrid").strip().lower()
    if mode not in {"mock", "hybrid", "stripe"}:
        return "hybrid"
    return mode

def stripe_can_run():
    mode = get_billing_mode()
    return mode in {"hybrid", "stripe"} and stripe is not None and bool(os.getenv("STRIPE_SECRET_KEY"))

def billing_kv_get(key):
    db = get_db()
    row = db.execute("SELECT value FROM billing_kv WHERE key = ?", (key,)).fetchone()
    return row["value"] if row else None

def billing_kv_set(key, value):
    db = get_db()
    db.execute(
        """
        INSERT INTO billing_kv (key, value, updated_at)
        VALUES (?, ?, ?)
        ON CONFLICT(key) DO UPDATE SET value = excluded.value, updated_at = excluded.updated_at
        """,
        (key, value, now_iso()),
    )
    db.commit()

def parse_usd_dollars_to_cents(raw):
    """
    Accepts strings like '5', '50', '5.00' and returns integer cents (500, 5000, ...).
    Returns None for non-numeric inputs.
    """
    s = (raw or "").strip()
    if not s:
        return None
    try:
        v = float(s)
    except Exception:
        return None
    if not (v > 0):
        return None
    return int(round(v * 100.0))

def configured_stripe_price_id(plan_id, allow_create=False):
    """
    Supports two formats in env vars:
      - a real Stripe price id: 'price_...'
      - a USD amount in dollars: '5' (dev convenience; creates a recurring monthly price once and caches it)
    """
    if plan_id == PLAN_PRO:
        env_name = "STRIPE_PRICE_ID_PRO"
        plan_label = "Pro"
    elif plan_id == PLAN_BUSINESS:
        env_name = "STRIPE_PRICE_ID_BUSINESS"
        plan_label = "Business"
    else:
        return None

    raw = (os.getenv(env_name) or "").strip()
    if raw.startswith("price_"):
        return raw

    cents = parse_usd_dollars_to_cents(raw)
    if cents is None:
        return None

    # Prefer cached real price id to avoid creating duplicates.
    cache_key = f"stripe.price_id.{env_name}.{cents}"
    cached = billing_kv_get(cache_key)
    if cached and cached.startswith("price_"):
        return cached
    if not allow_create:
        return cached

    if not stripe_can_run():
        return None

    # Create (or reuse cached) product per plan, then a recurring monthly price.
    product_key = f"stripe.product_id.{plan_id}"
    product_id = billing_kv_get(product_key)
    try:
        if not product_id:
            product = stripe.Product.create(
                name=f"PAN-O-RAMA {plan_label}",
                metadata={"plan_id": plan_id},
            )
            product_id = product.get("id")
            if product_id:
                billing_kv_set(product_key, product_id)

        price = stripe.Price.create(
            product=product_id,
            currency="usd",
            unit_amount=cents,
            recurring={"interval": "month"},
            metadata={"plan_id": plan_id},
        )
        pid = price.get("id")
        if pid:
            billing_kv_set(cache_key, pid)
        return pid
    except Exception as e:
        app.logger.error(f"Stripe price auto-create failed: {e}")
        return None

def get_plan_row(plan_id):
    db = get_db()
    return db.execute("SELECT * FROM plans WHERE id = ? AND is_active = 1", (plan_id,)).fetchone()

def get_active_subscription(user_id):
    db = get_db()
    return db.execute(
        """
        SELECT * FROM subscriptions
        WHERE user_id = ? AND status = 'active'
        ORDER BY created_at DESC
        LIMIT 1
        """,
        (user_id,),
    ).fetchone()

def set_user_plan(user_id, plan_id, provider="mock", status="active", provider_customer_id=None, provider_subscription_id=None, period_end=None):
    if plan_id not in PLAN_ORDER:
        plan_id = PLAN_FREE
    db = get_db()
    ts = now_iso()
    db.execute("UPDATE subscriptions SET status = 'canceled', updated_at = ? WHERE user_id = ? AND status = 'active'", (ts, user_id))
    sub_id = str(uuid.uuid4())
    db.execute(
        """
        INSERT INTO subscriptions (
            id, user_id, plan_id, status, billing_provider, provider_customer_id, provider_subscription_id, current_period_end, created_at, updated_at
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (sub_id, user_id, plan_id, status, provider, provider_customer_id, provider_subscription_id, period_end, ts, ts),
    )
    db.commit()
    return db.execute("SELECT * FROM subscriptions WHERE id = ?", (sub_id,)).fetchone()

def ensure_user_subscription(user_id):
    sub = get_active_subscription(user_id)
    if sub is not None:
        return sub
    return set_user_plan(user_id, PLAN_FREE, provider="mock")

def get_user_plan(user_id):
    sub = ensure_user_subscription(user_id)
    plan = get_plan_row(sub["plan_id"]) if sub is not None else None
    if plan is None:
        # Fallback safety for broken references.
        plan = get_plan_row(PLAN_FREE)
    return plan, sub

def compute_usage(user_id):
    db = get_db()
    row = db.execute(
        """
        SELECT COUNT(*) AS tours_count
        FROM tours
        WHERE owner_id = ? AND deleted_at IS NULL
        """,
        (user_id,),
    ).fetchone()
    count = int(row["tours_count"] if row else 0)
    db.execute(
        """
        INSERT INTO usage_counters (user_id, tours_count, updated_at)
        VALUES (?, ?, ?)
        ON CONFLICT(user_id) DO UPDATE SET tours_count = excluded.tours_count, updated_at = excluded.updated_at
        """,
        (user_id, count, now_iso()),
    )
    db.commit()
    return {"tours_count": count}

def get_user_entitlements(user_id):
    plan, sub = get_user_plan(user_id)
    usage = compute_usage(user_id)
    max_tours = int(plan["max_tours"]) if plan is not None else DEFAULT_PLAN_DEFS[PLAN_FREE]["max_tours"]
    watermark = bool(plan["watermark_enabled"]) if plan is not None else True
    remaining = max(0, max_tours - usage["tours_count"])
    return {
        "plan_id": plan["id"] if plan is not None else PLAN_FREE,
        "plan_name": plan["name"] if plan is not None else DEFAULT_PLAN_DEFS[PLAN_FREE]["name"],
        "max_tours": max_tours,
        "watermark_enabled": watermark,
        "usage": usage,
        "remaining_tours": remaining,
        "subscription": {
            "status": sub["status"] if sub is not None else "active",
            "billing_provider": sub["billing_provider"] if sub is not None else "mock",
            "current_period_end": sub["current_period_end"] if sub is not None else None,
        },
    }

def fetch_tour_with_access(tour_id, require_owner=False):
    db = get_db()
    row = db.execute(
        "SELECT * FROM tours WHERE id = ? AND deleted_at IS NULL",
        (tour_id,),
    ).fetchone()
    if row is None:
        return None, (jsonify({"error": "Tour not found"}), 404)

    user = g.current_user
    is_owner = user is not None and user["id"] == row["owner_id"]
    if require_owner and not is_owner:
        return None, (jsonify({"error": "Forbidden"}), 403)
    if row["visibility"] == "private" and not is_owner:
        return None, (jsonify({"error": "Forbidden"}), 403)
    return row, None

def create_session_response(user_row):
    db = get_db()
    session_token = secrets.token_urlsafe(48)
    expires = (datetime.datetime.utcnow() + datetime.timedelta(days=7)).replace(microsecond=0).isoformat() + "Z"
    db.execute(
        """
        INSERT INTO sessions (id, user_id, token_hash, expires_at, created_at, user_agent, ip)
        VALUES (?, ?, ?, ?, ?, ?, ?)
        """,
        (
            str(uuid.uuid4()),
            user_row["id"],
            hash_token(session_token),
            expires,
            now_iso(),
            request.headers.get("User-Agent", ""),
            request.remote_addr or "",
        ),
    )
    db.commit()
    payload = {"id": user_row["id"], "email": user_row["email"], "display_name": user_row["display_name"]}
    resp = jsonify({"user": payload})
    max_age = 7 * 24 * 3600
    resp.set_cookie("session_token", session_token, httponly=True, samesite="Lax", secure=False, max_age=max_age)
    return resp

def load_tour_scenes_and_hotspots(tour_id):
    db = get_db()
    scene_rows = db.execute(
        "SELECT * FROM scenes WHERE tour_id = ? ORDER BY order_index ASC",
        (tour_id,),
    ).fetchall()
    scene_ids = [r["id"] for r in scene_rows]
    hotspot_rows = []
    if scene_ids:
        ph = ",".join("?" for _ in scene_ids)
        hotspot_rows = db.execute(
            f"SELECT * FROM hotspots WHERE from_scene_id IN ({ph}) ORDER BY created_at ASC",
            scene_ids,
        ).fetchall()
    scene_name = {r["id"]: r["title"] for r in scene_rows}
    grouped = {}
    for h in hotspot_rows:
        grouped.setdefault(h["from_scene_id"], []).append(
            {
                "pitch": h["pitch"],
                "yaw": h["yaw"],
                "entry_pitch": h["entry_pitch"] if h["entry_pitch"] is not None else 0.0,
                "entry_yaw": h["entry_yaw"] if h["entry_yaw"] is not None else 0.0,
                "target_id": h["to_scene_id"],
                "target_name": scene_name.get(h["to_scene_id"], "Scene"),
                "label": h["label"] or "",
            }
        )
    scenes = []
    for r in scene_rows:
        s = serialize_scene(r)
        s["hotspots"] = grouped.get(r["id"], [])
        scenes.append(s)
    return scenes

def load_disk_metadata_scenes(tour_id):
    """Fallback for older galleries that exist on disk but not in DB."""
    try:
        pdir = os.path.join(app.config["PROCESSED_FOLDER"], tour_id)
        mpath = os.path.join(pdir, "metadata.json")
        if not os.path.exists(mpath):
            return []
        with open(mpath, "r", encoding="utf-8") as f:
            meta = json.load(f)
        scenes = meta.get("scenes") or []
        return scenes if isinstance(scenes, list) else []
    except Exception:
        return []

def load_disk_scenes_from_index_html(tour_id):
    """Parse an existing generated player (index.html) to reconstruct a minimal scene list.

    This is a best-effort compatibility path for galleries that predate the DB-backed tour model.
    """
    try:
        pdir = os.path.join(app.config["PROCESSED_FOLDER"], tour_id)
        ipath = os.path.join(pdir, "index.html")
        if not os.path.exists(ipath):
            return []
        with open(ipath, "r", encoding="utf-8", errors="ignore") as f:
            txt = f.read()
        needle = "const tourConfig ="
        pos = txt.find(needle)
        if pos < 0:
            return []
        brace_start = txt.find("{", pos)
        if brace_start < 0:
            return []
        # Extract a balanced {...} JSON object. The config is JSON, so braces in strings are unlikely,
        # but handle strings/escapes anyway.
        depth = 0
        in_str = False
        esc = False
        end = -1
        for i in range(brace_start, len(txt)):
            c = txt[i]
            if in_str:
                if esc:
                    esc = False
                elif c == "\\":
                    esc = True
                elif c == '"':
                    in_str = False
            else:
                if c == '"':
                    in_str = True
                elif c == "{":
                    depth += 1
                elif c == "}":
                    depth -= 1
                    if depth == 0:
                        end = i + 1
                        break
        if end < 0:
            return []
        cfg = json.loads(txt[brace_start:end])
        scenes_cfg = (cfg.get("scenes") or {}) if isinstance(cfg, dict) else {}
        if not isinstance(scenes_cfg, dict) or not scenes_cfg:
            return []
        scene_title = {sid: (sc.get("title") if isinstance(sc, dict) else None) for sid, sc in scenes_cfg.items()}
        out = []
        for sid, sc in scenes_cfg.items():
            if not isinstance(sc, dict):
                continue
            pano = sc.get("panorama") or ""
            pano_file = os.path.basename(pano) if isinstance(pano, str) else ""
            if not pano_file:
                continue
            hs_out = []
            for hs in (sc.get("hotSpots") or []):
                if not isinstance(hs, dict):
                    continue
                args = hs.get("clickHandlerArgs") or {}
                tgt = args.get("targetSceneId") or args.get("target_id") or ""
                if not tgt:
                    continue
                hs_out.append(
                    {
                        "pitch": hs.get("pitch", 0.0),
                        "yaw": hs.get("yaw", 0.0),
                        "entry_pitch": args.get("entryPitch", args.get("entry_pitch", 0.0)) or 0.0,
                        "entry_yaw": args.get("entryYaw", args.get("entry_yaw", 0.0)) or 0.0,
                        "target_id": tgt,
                        "target_name": scene_title.get(tgt) or "Scene",
                        "label": hs.get("text") or "",
                    }
                )
            out.append(
                {
                    "id": sid,
                    "name": sc.get("title") or sid,
                    "panorama": pano_file,
                    "images": [pano_file],
                    "hotspots": hs_out,
                    "haov": sc.get("haov", 360),
                    "vaov": sc.get("vaov", 180),
                    "type": sc.get("type") or "equirectangular",
                }
            )
        return out
    except Exception:
        return []

@app.route("/auth/register", methods=["POST"])
def auth_register():
    data = request.get_json(silent=True) or {}
    email = (data.get("email") or "").strip().lower()
    password = data.get("password") or ""
    display_name = (data.get("display_name") or email.split("@")[0] or "User").strip()
    if "@" not in email or len(password) < 8:
        return jsonify({"error": "Invalid email or password (min 8 chars)"}), 400
    db = get_db()
    exists = db.execute("SELECT id FROM users WHERE email = ?", (email,)).fetchone()
    if exists:
        return jsonify({"error": "Email already exists"}), 409
    uid = str(uuid.uuid4())
    ts = now_iso()
    db.execute(
        "INSERT INTO users (id, email, password_hash, display_name, created_at, updated_at) VALUES (?, ?, ?, ?, ?, ?)",
        (uid, email, generate_password_hash(password), display_name or "User", ts, ts),
    )
    db.commit()
    set_user_plan(uid, PLAN_FREE, provider="mock")
    row = db.execute("SELECT * FROM users WHERE id = ?", (uid,)).fetchone()
    return create_session_response(row), 201

@app.route("/auth/login", methods=["POST"])
def auth_login():
    data = request.get_json(silent=True) or {}
    email = (data.get("email") or "").strip().lower()
    password = data.get("password") or ""
    db = get_db()
    row = db.execute("SELECT * FROM users WHERE email = ?", (email,)).fetchone()
    if row is None or not check_password_hash(row["password_hash"], password):
        return jsonify({"error": "Invalid credentials"}), 401
    return create_session_response(row), 200

@app.route("/auth/logout", methods=["POST"])
def auth_logout():
    token = request.cookies.get("session_token")
    if token:
        db = get_db()
        db.execute("DELETE FROM sessions WHERE token_hash = ?", (hash_token(token),))
        db.commit()
    resp = jsonify({"message": "Logged out"})
    resp.delete_cookie("session_token")
    return resp, 200

@app.route("/me", methods=["GET"])
@require_auth
def me_get():
    u = g.current_user
    ent = get_user_entitlements(u["id"])
    return jsonify(
        {
            "id": u["id"],
            "email": u["email"],
            "display_name": u["display_name"],
            "plan": {
                "id": ent["plan_id"],
                "name": ent["plan_name"],
            },
            "entitlements": {
                "max_tours": ent["max_tours"],
                "watermark_enabled": ent["watermark_enabled"],
                "remaining_tours": ent["remaining_tours"],
            },
            "usage": ent["usage"],
            "subscription": ent["subscription"],
        }
    ), 200

@app.route("/me", methods=["PATCH"])
@require_auth
def me_patch():
    data = request.get_json(silent=True) or {}
    display_name = (data.get("display_name") or "").strip()
    if not display_name:
        return jsonify({"error": "display_name is required"}), 400
    db = get_db()
    db.execute("UPDATE users SET display_name = ?, updated_at = ? WHERE id = ?", (display_name, now_iso(), g.current_user["id"]))
    db.commit()
    return jsonify({"message": "Profile updated"}), 200

@app.route("/me/password", methods=["PATCH"])
@require_auth
def me_password():
    data = request.get_json(silent=True) or {}
    current_password = data.get("current_password") or ""
    new_password = data.get("new_password") or ""
    if len(new_password) < 8:
        return jsonify({"error": "new_password must be at least 8 characters"}), 400
    db = get_db()
    user = db.execute("SELECT * FROM users WHERE id = ?", (g.current_user["id"],)).fetchone()
    if not check_password_hash(user["password_hash"], current_password):
        return jsonify({"error": "Current password is incorrect"}), 401
    db.execute("UPDATE users SET password_hash = ?, updated_at = ? WHERE id = ?", (generate_password_hash(new_password), now_iso(), user["id"]))
    db.commit()
    return jsonify({"message": "Password updated"}), 200

@app.route("/tours", methods=["POST"])
@require_auth
def tours_create():
    ent = get_user_entitlements(g.current_user["id"])
    if ent["usage"]["tours_count"] >= ent["max_tours"]:
        return jsonify({"error": "Plan limit reached", "code": "plan_limit_exceeded", "entitlements": ent}), 403

    data = request.get_json(silent=True) or {}
    title = (data.get("title") or "").strip() or "Untitled Tour"
    description = (data.get("description") or "").strip()
    visibility = normalize_visibility(data.get("visibility"))
    tid = str(uuid.uuid4())
    ts = now_iso()
    slug = slugify(title)
    db = get_db()
    db.execute(
        """
        INSERT INTO tours (id, owner_id, title, description, slug, visibility, status, created_at, updated_at)
        VALUES (?, ?, ?, ?, ?, ?, 'draft', ?, ?)
        """,
        (tid, g.current_user["id"], title, description, slug, visibility, ts, ts),
    )
    db.commit()
    os.makedirs(os.path.join(app.config["PROCESSED_FOLDER"], tid), exist_ok=True)
    row = db.execute("SELECT * FROM tours WHERE id = ?", (tid,)).fetchone()
    return jsonify({"tour": serialize_tour(row)}), 201

@app.route("/tours/my", methods=["GET"])
@require_auth
def tours_my():
    db = get_db()
    rows = db.execute(
        "SELECT * FROM tours WHERE owner_id = ? AND deleted_at IS NULL ORDER BY created_at DESC",
        (g.current_user["id"],),
    ).fetchall()
    return jsonify({"tours": [serialize_tour(r) for r in rows]}), 200

@app.route("/tours/<tour_id>", methods=["GET"])
def tours_get(tour_id):
    tour, err = fetch_tour_with_access(tour_id, require_owner=False)
    if err:
        return err
    scenes = load_tour_scenes_and_hotspots(tour["id"])
    payload = serialize_tour(tour)
    payload["scenes"] = scenes
    return jsonify({"tour": payload}), 200

@app.route("/tours/<tour_id>", methods=["PATCH"])
@require_auth
def tours_patch(tour_id):
    tour, err = fetch_tour_with_access(tour_id, require_owner=True)
    if err:
        return err
    data = request.get_json(silent=True) or {}
    title = (data.get("title") or tour["title"]).strip() or tour["title"]
    description = (data.get("description") or tour["description"]).strip()
    visibility = normalize_visibility(data.get("visibility") or tour["visibility"])
    db = get_db()
    db.execute(
        "UPDATE tours SET title = ?, description = ?, visibility = ?, updated_at = ? WHERE id = ?",
        (title, description, visibility, now_iso(), tour["id"]),
    )
    db.commit()
    row = db.execute("SELECT * FROM tours WHERE id = ?", (tour["id"],)).fetchone()
    return jsonify({"tour": serialize_tour(row)}), 200

@app.route("/tours/<tour_id>", methods=["DELETE"])
@require_auth
def tours_delete(tour_id):
    tour, err = fetch_tour_with_access(tour_id, require_owner=True)
    if err:
        return err
    db = get_db()
    db.execute("UPDATE tours SET deleted_at = ?, updated_at = ? WHERE id = ?", (now_iso(), now_iso(), tour["id"]))
    db.commit()
    return jsonify({"message": "Tour deleted"}), 200

@app.route("/tours/<tour_id>/scenes", methods=["POST"])
@require_auth
def tours_add_scene(tour_id):
    tour, err = fetch_tour_with_access(tour_id, require_owner=True)
    if err:
        return err
    name = request.form.get("scene_name", "Unnamed")
    is_pano = request.form.get("is_panorama") == "true"
    files = request.files.getlist("files[]")
    if not files:
        return jsonify({"error": "No files[] uploaded"}), 400
    db = get_db()
    count_row = db.execute("SELECT COUNT(*) AS c FROM scenes WHERE tour_id = ?", (tour["id"],)).fetchone()
    order_index = int(count_row["c"])
    # scenes.id is globally unique (not scoped by tour). Use UUID to avoid collisions across tours.
    sid = str(uuid.uuid4())
    raw_dir = os.path.join(app.config["UPLOAD_FOLDER"], tour["id"], sid)
    proc_dir = os.path.join(app.config["PROCESSED_FOLDER"], tour["id"], sid)
    os.makedirs(raw_dir, exist_ok=True)
    os.makedirs(proc_dir, exist_ok=True)
    files.sort(key=lambda x: natural_sort_key(x.filename))
    saved_raw, processed = [], []
    for file in files:
        if file and ('.' in file.filename and file.filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS):
            fn = secure_filename(file.filename)
            rp = os.path.join(raw_dir, fn)
            file.save(rp)
            saved_raw.append(rp)
    for rp in saved_raw:
        fn = f"proc_{os.path.basename(rp)}"
        pp = os.path.join(proc_dir, fn)
        if process_image(rp, pp):
            processed.append(fn)
    if not processed:
        return jsonify({"error": "No valid images"}), 400

    focal_35 = 26.0
    try:
        with Image.open(saved_raw[0]) as img:
            exif = img.getexif()
            if exif:
                f = safe_float(exif.get(41989)) or (safe_float(exif.get(37386)) * 6.0)
                if f > 0:
                    focal_35 = f
    except Exception:
        pass

    pano_file = None
    haov, vaov = 100, 60
    if len(processed) >= 2:
        ppano = os.path.join(proc_dir, "panorama.jpg")
        if stitch_panorama([os.path.join(proc_dir, fn) for fn in processed], ppano, is_360=is_pano):
            pano_file = "panorama.jpg"
            _pp_w, _pp_h, pp_changed = postprocess_panorama(ppano) if is_pano else (None, None, False)
            with Image.open(ppano) as img:
                aspect = img.width / img.height
                vaov_c = 2 * math.degrees(math.atan(18.0 / focal_35))
                if is_pano:
                    haov = 360 if pp_changed else vaov_c * aspect
                    vaov = (360 / aspect) if pp_changed else vaov_c
                else:
                    haov = vaov_c * aspect
                    vaov = vaov_c
        else:
            haov = 2 * math.degrees(math.atan(18.0 / focal_35))
            vaov = haov / (16 / 9)
    else:
        img_p = os.path.join(proc_dir, processed[0])
        with Image.open(img_p) as img:
            aspect = img.width / img.height
            vaov_c = 2 * math.degrees(math.atan(18.0 / focal_35))
            if is_pano:
                # If user uploads a single already-stitched equirectangular 360, do NOT crop overlap.
                # Overlap trimming is only for wide panoramas that accidentally contain duplicated seam content.
                pano_file = processed[0]
                haov = 360
                vaov = 360 / aspect
            else:
                haov = vaov_c * aspect
                vaov = vaov_c

    # Keep FOVs within Pannellum stable ranges for equirectangular panos.
    if is_pano:
        try:
            ref_path = os.path.join(proc_dir, pano_file) if pano_file else os.path.join(proc_dir, processed[0])
            with Image.open(ref_path) as img:
                aspect = img.width / img.height
            if vaov > 180.0 or haov > 360.0:
                vaov = 180.0
                haov = min(360.0, aspect * 180.0)
            vaov = min(180.0, max(30.0, vaov))
            haov = min(360.0, max(30.0, haov))
        except Exception:
            pass

    ts = now_iso()
    # Generate preview for faster transitions (use pano when present, else the first processed frame).
    src_for_preview = pano_file or (processed[0] if processed else None)
    preview_path = ensure_scene_preview(tour["id"], sid, src_for_preview) if src_for_preview else None
    db.execute(
        """
        INSERT INTO scenes (id, tour_id, title, panorama_path, preview_path, images_json, order_index, haov, vaov, scene_type, created_at, updated_at)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, 'equirectangular', ?, ?)
        """,
        (sid, tour["id"], name, pano_file, preview_path, json.dumps(processed), order_index, round(haov, 2), round(vaov, 2), ts, ts),
    )
    db.execute("UPDATE tours SET updated_at = ? WHERE id = ?", (ts, tour["id"]))
    db.commit()
    row = db.execute("SELECT * FROM scenes WHERE id = ? AND tour_id = ?", (sid, tour["id"])).fetchone()
    scene_payload = serialize_scene(row)
    scene_payload["hotspots"] = []
    return jsonify({"scene": scene_payload}), 201

@app.route("/scenes/<scene_id>", methods=["PATCH"])
@require_auth
def scenes_patch(scene_id):
    db = get_db()
    row = db.execute("SELECT s.*, t.owner_id FROM scenes s JOIN tours t ON t.id = s.tour_id WHERE s.id = ? AND t.deleted_at IS NULL", (scene_id,)).fetchone()
    if row is None:
        return jsonify({"error": "Scene not found"}), 404
    if row["owner_id"] != g.current_user["id"]:
        return jsonify({"error": "Forbidden"}), 403
    data = request.get_json(silent=True) or {}
    title = (data.get("name") or row["title"]).strip() or row["title"]
    db.execute("UPDATE scenes SET title = ?, updated_at = ? WHERE id = ?", (title, now_iso(), scene_id))
    db.commit()
    return jsonify({"message": "Scene updated"}), 200

@app.route("/scenes/<scene_id>", methods=["DELETE"])
@require_auth
def scenes_delete(scene_id):
    db = get_db()
    row = db.execute("SELECT s.*, t.owner_id FROM scenes s JOIN tours t ON t.id = s.tour_id WHERE s.id = ? AND t.deleted_at IS NULL", (scene_id,)).fetchone()
    if row is None:
        return jsonify({"error": "Scene not found"}), 404
    if row["owner_id"] != g.current_user["id"]:
        return jsonify({"error": "Forbidden"}), 403
    db.execute("DELETE FROM hotspots WHERE from_scene_id = ? OR to_scene_id = ?", (scene_id, scene_id))
    db.execute("DELETE FROM scenes WHERE id = ?", (scene_id,))
    db.execute("UPDATE tours SET updated_at = ? WHERE id = ?", (now_iso(), row["tour_id"]))
    db.commit()
    return jsonify({"message": "Scene deleted"}), 200

@app.route("/scenes/<scene_id>/hotspots", methods=["POST"])
@require_auth
def hotspots_create(scene_id):
    db = get_db()
    source = db.execute("SELECT s.*, t.owner_id FROM scenes s JOIN tours t ON t.id = s.tour_id WHERE s.id = ? AND t.deleted_at IS NULL", (scene_id,)).fetchone()
    if source is None:
        return jsonify({"error": "Scene not found"}), 404
    if source["owner_id"] != g.current_user["id"]:
        return jsonify({"error": "Forbidden"}), 403
    data = request.get_json(silent=True) or {}
    to_scene_id = data.get("to_scene_id")
    yaw = data.get("yaw")
    pitch = data.get("pitch")
    entry_yaw = data.get("entry_yaw", 0.0)
    entry_pitch = data.get("entry_pitch", 0.0)
    label = (data.get("label") or "").strip()
    target = db.execute("SELECT id FROM scenes WHERE id = ? AND tour_id = ?", (to_scene_id, source["tour_id"])).fetchone()
    if target is None:
        return jsonify({"error": "Target scene not found in same tour"}), 400
    try:
        yaw = float(yaw)
        pitch = float(pitch)
        entry_yaw = float(entry_yaw)
        entry_pitch = float(entry_pitch)
    except Exception:
        return jsonify({"error": "yaw, pitch, entry_yaw, entry_pitch must be numbers"}), 400
    hid = str(uuid.uuid4())
    ts = now_iso()
    db.execute(
        """
        INSERT INTO hotspots (id, tour_id, from_scene_id, to_scene_id, yaw, pitch, entry_yaw, entry_pitch, label, created_at, updated_at)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (hid, source["tour_id"], scene_id, to_scene_id, yaw, pitch, entry_yaw, entry_pitch, label, ts, ts),
    )
    db.execute("UPDATE tours SET updated_at = ? WHERE id = ?", (ts, source["tour_id"]))
    db.commit()
    return jsonify({"hotspot": {"id": hid, "from_scene_id": scene_id, "to_scene_id": to_scene_id, "yaw": yaw, "pitch": pitch, "entry_yaw": entry_yaw, "entry_pitch": entry_pitch, "label": label}}), 201

@app.route("/hotspots/<hotspot_id>", methods=["PATCH"])
@require_auth
def hotspots_patch(hotspot_id):
    db = get_db()
    row = db.execute(
        """
        SELECT h.*, t.owner_id
        FROM hotspots h
        JOIN tours t ON t.id = h.tour_id
        WHERE h.id = ? AND t.deleted_at IS NULL
        """,
        (hotspot_id,),
    ).fetchone()
    if row is None:
        return jsonify({"error": "Hotspot not found"}), 404
    if row["owner_id"] != g.current_user["id"]:
        return jsonify({"error": "Forbidden"}), 403
    data = request.get_json(silent=True) or {}
    yaw = float(data.get("yaw", row["yaw"]))
    pitch = float(data.get("pitch", row["pitch"]))
    entry_yaw = float(data.get("entry_yaw", row["entry_yaw"] if row["entry_yaw"] is not None else 0.0))
    entry_pitch = float(data.get("entry_pitch", row["entry_pitch"] if row["entry_pitch"] is not None else 0.0))
    label = (data.get("label") or row["label"] or "").strip()
    db.execute(
        "UPDATE hotspots SET yaw = ?, pitch = ?, entry_yaw = ?, entry_pitch = ?, label = ?, updated_at = ? WHERE id = ?",
        (yaw, pitch, entry_yaw, entry_pitch, label, now_iso(), hotspot_id),
    )
    db.commit()
    return jsonify({"message": "Hotspot updated"}), 200

@app.route("/hotspots/<hotspot_id>", methods=["DELETE"])
@require_auth
def hotspots_delete(hotspot_id):
    db = get_db()
    row = db.execute(
        """
        SELECT h.id, t.owner_id
        FROM hotspots h
        JOIN tours t ON t.id = h.tour_id
        WHERE h.id = ? AND t.deleted_at IS NULL
        """,
        (hotspot_id,),
    ).fetchone()
    if row is None:
        return jsonify({"error": "Hotspot not found"}), 404
    if row["owner_id"] != g.current_user["id"]:
        return jsonify({"error": "Forbidden"}), 403
    db.execute("DELETE FROM hotspots WHERE id = ?", (hotspot_id,))
    db.commit()
    return jsonify({"message": "Hotspot deleted"}), 200

@app.route("/tours/<tour_id>/finalize", methods=["POST"])
@require_auth
def tours_finalize(tour_id):
    tour, err = fetch_tour_with_access(tour_id, require_owner=True)
    if err:
        return err
    scenes = load_tour_scenes_and_hotspots(tour["id"])
    if not scenes:
        return jsonify({"error": "Tour must have at least one scene"}), 400
    ent = get_user_entitlements(g.current_user["id"])
    gallery_url = generate_tour(tour["id"], scenes, watermark_enabled=ent["watermark_enabled"])
    db = get_db()
    db.execute("UPDATE tours SET status = 'published', updated_at = ? WHERE id = ?", (now_iso(), tour["id"]))
    db.commit()
    share_url = f"/t/{tour['slug']}"
    return jsonify({"gallery_url": gallery_url, "share_url": share_url, "visibility": tour["visibility"]}), 200

@app.route("/tours/<tour_id>/export/facebook360", methods=["GET"])
@require_auth
def tours_export_facebook360(tour_id):
    tour, err = fetch_tour_with_access(tour_id, require_owner=True)
    if err:
        return err

    db = get_db()
    scene = db.execute(
        "SELECT * FROM scenes WHERE tour_id = ? ORDER BY order_index ASC LIMIT 1",
        (tour["id"],),
    ).fetchone()
    if scene is None:
        return jsonify({"error": "Tour has no scenes"}), 400

    try:
        haov = float(scene["haov"] or 0.0)
    except Exception:
        haov = 0.0
    if haov < 300.0:
        return jsonify({"error": "First scene is not a 360 panorama (haov < 300). Upload a 360 pano for scene 1."}), 400

    proc_dir = os.path.join(app.config["PROCESSED_FOLDER"], tour["id"], scene["id"])
    pano = (scene["panorama_path"] or "").strip()
    images = []
    try:
        images = json.loads(scene["images_json"] or "[]")
    except Exception:
        images = []
    candidate = pano or (images[0] if images else "")
    if not candidate:
        return jsonify({"error": "First scene has no image assets"}), 400

    img_path = os.path.join(proc_dir, candidate)
    if not os.path.exists(img_path):
        return jsonify({"error": "Scene image file not found on server"}), 404

    try:
        max_w = 6000
        max_h = 3000
        try:
            max_w = int(os.getenv("FB360_MAX_WIDTH") or max_w)
            max_h = int(os.getenv("FB360_MAX_HEIGHT") or max_h)
        except Exception:
            max_w, max_h = 6000, 3000

        with Image.open(img_path) as im:
            im = ImageOps.exif_transpose(im)
            if im.mode in ("RGBA", "P"):
                im = im.convert("RGB")
            w, h = im.size
            needs_resize = (w > max_w) or (h > max_h)
            if needs_resize:
                scale = min(max_w / float(w), max_h / float(h))
                nw = max(2, int(w * scale))
                nh = max(2, int(h * scale))
                # JPEG encoders and some viewers behave better with even dimensions.
                nw -= (nw % 2)
                nh -= (nh % 2)
                if nw < 2:
                    nw = 2
                if nh < 2:
                    nh = 2
                try:
                    resample = Image.Resampling.LANCZOS
                except Exception:
                    resample = Image.LANCZOS
                im = im.resize((nw, nh), resample=resample)
                buf = BytesIO()
                im.save(buf, format="JPEG", quality=92, optimize=True, progressive=True, subsampling=0)
                raw = buf.getvalue()
                w, h = nw, nh
            else:
                with open(img_path, "rb") as f:
                    raw = f.read()

        xmp = build_gpano_xmp(w, h)
        out = inject_xmp_into_jpeg(raw, xmp)
        slug = (tour["slug"] or "tour").strip() or "tour"
        filename = f"{slug}-facebook360.jpg"
        return send_file(
            BytesIO(out),
            mimetype="image/jpeg",
            as_attachment=True,
            download_name=filename,
            max_age=0,
        )
    except Exception as e:
        app.logger.error(f"FB360 export failed: {e}", exc_info=True)
        return jsonify({"error": "Failed to export Facebook 360 image"}), 500

@app.route("/tours/<tour_id>/hotspots/bulk", methods=["POST"])
@require_auth
def tours_hotspots_bulk(tour_id):
    tour, err = fetch_tour_with_access(tour_id, require_owner=True)
    if err:
        return err
    data = request.get_json(silent=True) or {}
    scenes_map = data if isinstance(data, dict) else {}
    db = get_db()
    scene_rows = db.execute("SELECT id, title FROM scenes WHERE tour_id = ?", (tour["id"],)).fetchall()
    valid_scene_ids = {r["id"] for r in scene_rows}
    scene_names = {r["id"]: r["title"] for r in scene_rows}

    db.execute("DELETE FROM hotspots WHERE tour_id = ?", (tour["id"],))
    ts = now_iso()
    for from_scene_id, items in scenes_map.items():
        if from_scene_id not in valid_scene_ids or not isinstance(items, list):
            continue
        for hs in items:
            to_scene_id = hs.get("target_id")
            if to_scene_id not in valid_scene_ids:
                continue
            try:
                yaw = float(hs.get("yaw"))
                pitch = float(hs.get("pitch"))
                entry_yaw = float(hs.get("entry_yaw", 0.0))
                entry_pitch = float(hs.get("entry_pitch", 0.0))
            except Exception:
                continue
            label = (hs.get("label") or f"Go to {scene_names.get(to_scene_id, 'Scene')}").strip()
            db.execute(
                """
                INSERT INTO hotspots (id, tour_id, from_scene_id, to_scene_id, yaw, pitch, entry_yaw, entry_pitch, label, created_at, updated_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (str(uuid.uuid4()), tour["id"], from_scene_id, to_scene_id, yaw, pitch, entry_yaw, entry_pitch, label, ts, ts),
            )
    db.execute("UPDATE tours SET updated_at = ? WHERE id = ?", (ts, tour["id"]))
    db.commit()
    return jsonify({"message": "Hotspots replaced"}), 200

def parse_plan_id(raw):
    p = (raw or "").strip().lower()
    if p in PLAN_ORDER:
        return p
    return None

def stripe_price_map():
    # Webhook inference: do not create new prices here; only map configured or cached ids.
    pro_pid = configured_stripe_price_id(PLAN_PRO, allow_create=False)
    biz_pid = configured_stripe_price_id(PLAN_BUSINESS, allow_create=False)
    out = {}
    if pro_pid:
        out[pro_pid] = PLAN_PRO
    if biz_pid:
        out[biz_pid] = PLAN_BUSINESS
    return out

def infer_plan_from_subscription(sub_id):
    """
    Best-effort plan inference from a Stripe subscription by looking at item price ids.
    Returns plan_id or None.
    """
    if not stripe_can_run() or not sub_id:
        return None
    try:
        sub = stripe.Subscription.retrieve(sub_id, expand=["items.data.price"])
        price_to_plan = stripe_price_map()
        best = None
        for item in (sub.get("items") or {}).get("data", []):
            price = item.get("price") or {}
            pid = price.get("id")
            plan = price_to_plan.get(pid)
            if plan and (best is None or PLAN_ORDER.get(plan, 0) > PLAN_ORDER.get(best, 0)):
                best = plan
        return best
    except Exception as e:
        app.logger.error(f"Stripe infer plan failed: {e}")
        return None

def find_user_id_for_checkout_object(obj):
    md = obj.get("metadata") or {}
    user_id = (md.get("user_id") or "").strip()
    if user_id:
        return user_id
    cr = (obj.get("client_reference_id") or "").strip()
    if cr:
        return cr
    email = None
    cd = obj.get("customer_details") or {}
    email = (cd.get("email") or obj.get("customer_email") or "").strip().lower()
    if not email:
        return None
    db = get_db()
    row = db.execute("SELECT id FROM users WHERE email = ?", (email,)).fetchone()
    return row["id"] if row else None

@app.route("/billing/mock/subscribe", methods=["POST"])
@require_auth
def billing_mock_subscribe():
    data = request.get_json(silent=True) or {}
    plan_id = parse_plan_id(data.get("plan_id"))
    if plan_id is None:
        return jsonify({"error": "Invalid plan_id"}), 400
    set_user_plan(g.current_user["id"], plan_id, provider="mock")
    ent = get_user_entitlements(g.current_user["id"])
    return jsonify({"message": "Plan updated", "plan_id": ent["plan_id"], "entitlements": ent}), 200

@app.route("/billing/checkout", methods=["POST"])
@require_auth
def billing_checkout():
    if get_billing_mode() not in {"hybrid", "stripe"}:
        return jsonify({"error": "Stripe checkout disabled by BILLING_MODE"}), 403
    if not stripe_can_run():
        return jsonify({"error": "Stripe is not configured"}), 503

    data = request.get_json(silent=True) or {}
    plan_id = parse_plan_id(data.get("plan_id"))
    if plan_id not in {PLAN_PRO, PLAN_BUSINESS}:
        return jsonify({"error": "Only paid plans supported in checkout"}), 400

    stripe.api_key = os.getenv("STRIPE_SECRET_KEY")
    customer_email = g.current_user["email"]
    origin = request.headers.get("Origin") or request.host_url.rstrip("/")
    success_url = f"{origin}/account?billing=success"
    cancel_url = f"{origin}/account?billing=cancel"
    price_id = configured_stripe_price_id(plan_id, allow_create=True)
    if not price_id:
        return jsonify({"error": "Stripe price id is not configured"}), 503

    try:
        session = stripe.checkout.Session.create(
            mode="subscription",
            line_items=[{"price": price_id, "quantity": 1}],
            success_url=success_url,
            cancel_url=cancel_url,
            customer_email=customer_email,
            client_reference_id=g.current_user["id"],
            metadata={"user_id": g.current_user["id"], "target_plan_id": plan_id},
            subscription_data={"metadata": {"user_id": g.current_user["id"], "target_plan_id": plan_id}},
        )
        return jsonify({"url": session.url, "session_id": session.id}), 200
    except Exception as e:
        app.logger.error(f"Stripe checkout error: {e}")
        return jsonify({"error": "Failed to create checkout session"}), 500

@app.route("/billing/webhook/stripe", methods=["POST"])
def billing_webhook_stripe():
    if not stripe_can_run():
        return jsonify({"error": "Stripe is not configured"}), 503
    stripe.api_key = os.getenv("STRIPE_SECRET_KEY")
    payload = request.data
    signature = request.headers.get("Stripe-Signature")
    endpoint_secret = os.getenv("STRIPE_WEBHOOK_SECRET")
    try:
        if endpoint_secret:
            event = stripe.Webhook.construct_event(payload, signature, endpoint_secret)
        else:
            event = request.get_json(force=True)
    except Exception:
        return jsonify({"error": "Invalid webhook payload"}), 400

    event_type = event.get("type")
    obj = (event.get("data") or {}).get("object") or {}
    db = get_db()

    if event_type == "checkout.session.completed":
        user_id = find_user_id_for_checkout_object(obj)
        md = obj.get("metadata") or {}
        target_plan_id = parse_plan_id(md.get("target_plan_id"))
        customer_id = obj.get("customer")
        sub_id = obj.get("subscription")
        if target_plan_id is None:
            target_plan_id = infer_plan_from_subscription(sub_id)
        if user_id and target_plan_id in {PLAN_PRO, PLAN_BUSINESS}:
            set_user_plan(
                user_id,
                target_plan_id,
                provider="stripe",
                provider_customer_id=customer_id,
                provider_subscription_id=sub_id,
            )
    elif event_type in {"customer.subscription.updated", "customer.subscription.deleted"}:
        sub_id = obj.get("id")
        if sub_id:
            row = db.execute(
                "SELECT * FROM subscriptions WHERE provider_subscription_id = ? ORDER BY created_at DESC LIMIT 1",
                (sub_id,),
            ).fetchone()
            if row is not None:
                status = obj.get("status") or "active"
                if status not in {"active", "canceled", "past_due"}:
                    status = "active"
                current_period_end = None
                cpe = obj.get("current_period_end")
                if isinstance(cpe, (int, float)):
                    current_period_end = datetime.datetime.utcfromtimestamp(cpe).replace(microsecond=0).isoformat() + "Z"
                db.execute(
                    "UPDATE subscriptions SET status = ?, current_period_end = ?, updated_at = ? WHERE id = ?",
                    (status, current_period_end, now_iso(), row["id"]),
                )
                db.commit()
                if event_type == "customer.subscription.deleted":
                    set_user_plan(row["user_id"], PLAN_FREE, provider="mock")
    elif event_type == "customer.subscription.created":
        # Some setups rely on subscription events rather than checkout.session metadata.
        sub_id = obj.get("id")
        customer_id = obj.get("customer")
        md = obj.get("metadata") or {}
        user_id = (md.get("user_id") or "").strip()
        plan_id = parse_plan_id(md.get("target_plan_id")) or infer_plan_from_subscription(sub_id)
        if user_id and plan_id in {PLAN_PRO, PLAN_BUSINESS}:
            set_user_plan(
                user_id,
                plan_id,
                provider="stripe",
                provider_customer_id=customer_id,
                provider_subscription_id=sub_id,
            )

    return jsonify({"received": True}), 200

@app.route("/gallery", methods=["GET"])
def public_gallery():
    db = get_db()
    rows = db.execute(
        """
        SELECT id, slug, title, description, created_at
        FROM tours
        WHERE deleted_at IS NULL AND visibility = 'public' AND status = 'published'
        ORDER BY created_at DESC
        """
    ).fetchall()
    return jsonify({"items": [dict(r) for r in rows]}), 200

@app.route("/t/<slug>", methods=["GET"])
def open_share_link(slug):
    db = get_db()
    tour = db.execute(
        "SELECT * FROM tours WHERE slug = ? AND deleted_at IS NULL",
        (slug,),
    ).fetchone()
    if tour is None:
        return jsonify({"error": "Tour not found"}), 404
    is_owner = g.current_user is not None and g.current_user["id"] == tour["owner_id"]
    if tour["visibility"] == "private" and not is_owner:
        return jsonify({"error": "Forbidden"}), 403
    return redirect(f"/galleries/{tour['id']}/index.html", code=302)

@app.route('/')
def index(): return send_from_directory(FRONTEND_FOLDER, 'index.html')

@app.route('/login')
def login_page():
    return send_from_directory(FRONTEND_FOLDER, 'login.html')

@app.route('/dashboard')
def dashboard_page():
    return send_from_directory(FRONTEND_FOLDER, 'dashboard.html')

@app.route('/account')
def account_page():
    return send_from_directory(FRONTEND_FOLDER, 'account.html')

@app.route('/img/<path:filename>')
def static_img(filename):
    return send_from_directory(IMG_FOLDER, filename)

@app.route('/all-projects')
def all_projects_page():
    return send_from_directory(FRONTEND_FOLDER, 'projects.html')

@app.route('/projects.html')
def projects_page():
    return send_from_directory(FRONTEND_FOLDER, 'projects.html')

@app.route('/browse')
def browse_page():
    return send_from_directory(FRONTEND_FOLDER, 'projects.html')

@app.route('/api/projects', methods=['GET'])
def list_projects():
    try:
        db = get_db()
        rows = db.execute(
            """
            SELECT id, title, description, slug, created_at
            FROM tours
            WHERE deleted_at IS NULL AND visibility = 'public' AND status = 'published'
            ORDER BY created_at DESC
            """
        ).fetchall()
        projects = []
        for r in rows:
            projects.append(
                {
                    "project_id": r["id"],
                    "title": r["title"],
                    "description": r["description"] or "",
                    "slug": r["slug"],
                    "created_at": r["created_at"],
                    "gallery_url": f"/t/{r['slug']}",
                    "preview_url": None,
                }
            )
        return jsonify(projects), 200
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/project/create', methods=['POST'])
def create_project():
    try:
        pid = str(uuid.uuid4())
        p_dir = os.path.join(app.config['PROCESSED_FOLDER'], pid)
        os.makedirs(p_dir, exist_ok=True)
        meta = {'project_id': pid, 'created_at': datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S'), 'scenes': []}
        with open(os.path.join(p_dir, 'metadata.json'), 'w') as f: json.dump(meta, f)
        return jsonify({'project_id': pid}), 200
    except Exception as e: return jsonify({'error': str(e)}), 500

@app.route('/api/project/<project_id>/scene/add', methods=['POST'])
def add_scene(project_id):
    try:
        name = request.form.get('scene_name', 'Unnamed')
        is_pano = request.form.get('is_panorama') == 'true'
        files = request.files.getlist('files[]')
        p_dir = os.path.join(app.config['PROCESSED_FOLDER'], project_id)
        m_path = os.path.join(p_dir, 'metadata.json')
        with open(m_path, 'r') as f: meta = json.load(f)
        sid = f"scene_{len(meta['scenes'])}"
        raw_dir = os.path.join(app.config['UPLOAD_FOLDER'], project_id, sid)
        proc_dir = os.path.join(p_dir, sid)
        os.makedirs(raw_dir, exist_ok=True); os.makedirs(proc_dir, exist_ok=True)
        files.sort(key=lambda x: natural_sort_key(x.filename))
        saved_raw, processed = [], []
        for file in files:
            if file and ('.' in file.filename and file.filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS):
                fn = secure_filename(file.filename); rp = os.path.join(raw_dir, fn); file.save(rp); saved_raw.append(rp)
        for rp in saved_raw:
            fn = f"proc_{os.path.basename(rp)}"; pp = os.path.join(proc_dir, fn)
            if process_image(rp, pp): processed.append(fn)
        if not processed: return jsonify({'error': 'No images'}), 400
        focal_35 = 26.0
        try:
            with Image.open(saved_raw[0]) as img:
                exif = img.getexif()
                if exif:
                    f = safe_float(exif.get(41989)) or (safe_float(exif.get(37386)) * 6.0)
                    if f > 0: focal_35 = f
        except: pass
        pano_file = None
        haov, vaov = 100, 60
        if len(processed) >= 2:
            ppano = os.path.join(proc_dir, "panorama.jpg")
            if stitch_panorama([os.path.join(proc_dir, fn) for fn in processed], ppano, is_360=is_pano):
                pano_file = "panorama.jpg"
                pp_w, pp_h, pp_changed = postprocess_panorama(ppano) if is_pano else (None, None, False)
                with Image.open(ppano) as img:
                    aspect = img.width / img.height
                    vaov_c = 2 * math.degrees(math.atan(18.0 / focal_35))
                    if is_pano:
                        # If we did not explicitly close the loop, avoid forcing 360 to prevent black gaps
                        crop_offset = None
                        if crop_offset is not None or pp_changed:
                            # If we altered the pano to remove gaps, assume full 360
                            with Image.open(ppano) as img_c:
                                haov = 360
                                vaov = 360 / (img_c.width / img_c.height)
                        else:
                            haov = vaov_c * aspect
                            vaov = vaov_c
                    else:
                        haov = vaov_c * aspect
                        vaov = vaov_c
            else:
                haov = 2 * math.degrees(math.atan(18.0 / focal_35)); vaov = haov / (16/9)
        else:
            img_p = os.path.join(proc_dir, processed[0])
            if is_pano: 
                with Image.open(img_p) as img:
                    aspect = img.width / img.height
                    vaov_c = 2 * math.degrees(math.atan(18.0 / focal_35))
                    # If this is already a very wide pano (likely phone/360 output), force full 360
                    if aspect >= 2.0:
                        # Try to detect and trim overlap to avoid double content
                        detect_and_crop_overlap_wide(img_p)
                        with Image.open(img_p) as img_c:
                            aspect = img_c.width / img_c.height
                            haov = 360
                            vaov = 360 / aspect
                    else:
                        haov = vaov_c * aspect
                        vaov = vaov_c
            else:
                with Image.open(img_p) as img:
                    vaov = 2 * math.degrees(math.atan(18.0 / focal_35))
                    haov = vaov * (img.width / img.height)
        s_data = {'id': sid, 'name': name, 'panorama': pano_file, 'images': processed, 'hotspots': [], 'haov': round(haov, 2), 'vaov': round(vaov, 2), 'type': 'equirectangular'}
        meta['scenes'].append(s_data)
        with open(m_path, 'w') as f: json.dump(meta, f)
        return jsonify({'scene': s_data}), 200
    except Exception as e: app.logger.error(f"Error: {e}", exc_info=True); return jsonify({'error': str(e)}), 500

def generate_tour(project_id, scenes, watermark_enabled=False, force_previews=False):
    # Pannellum's built-in `preview` works best when the scene switch is immediate; otherwise it can
    # look like the previous scene is "frozen" until full-res is ready.
    tour_config = {"default": {"firstScene": scenes[0]['id'], "sceneFadeDuration": 0, "autoLoad": False, "autoRotate": -2, "hfov": 70}, "scenes": {}}
    for scene in scenes:
        pano_name = scene.get('panorama') or ((scene.get('images') or [None])[0])
        if not pano_name:
            continue
        # Serve a web-safe pano by default to avoid browser freezes on very large textures.
        # Keep the original hi-res URL available for an optional "HD" user toggle in the player.
        orig_name = scene.get("hires") or pano_name
        web_name = scene.get("web") or ensure_scene_web_pano(project_id, scene["id"], orig_name)
        served_name = web_name or orig_name
        panorama_url = f"{scene['id']}/{served_name}"
        hires_url = f"{scene['id']}/{orig_name}" if served_name != orig_name else None
        preview_name = scene.get("preview") or ensure_scene_preview(project_id, scene['id'], orig_name, force=bool(force_previews))
        preview_url = f"{scene['id']}/{preview_name}" if preview_name else None
        # Provide widths so the player can slightly zoom preview to hide low-res blur without a "jump".
        pano_w = None
        preview_w = None
        try:
            proc_dir = os.path.join(app.config['PROCESSED_FOLDER'], project_id, scene['id'])
            pano_path = os.path.join(proc_dir, served_name)
            if os.path.exists(pano_path):
                with Image.open(pano_path) as im:
                    pano_w = int(im.width)
            if preview_name:
                prev_path = os.path.join(proc_dir, preview_name)
                if os.path.exists(prev_path):
                    with Image.open(prev_path) as im:
                        preview_w = int(im.width)
        except Exception:
            pano_w = pano_w
            preview_w = preview_w
        hotspots = []
        for hs in scene.get('hotspots', []):
            hotspots.append({
                "pitch": hs['pitch'], "yaw": hs['yaw'], "type": "info", "text": f"Go to {hs['target_name']}", "cssClass": "custom-hotspot",
                "clickHandlerFunc": "smoothSwitch",
                "clickHandlerArgs": {
                    "targetSceneId": hs['target_id'],
                    "viaPitch": hs['pitch'],
                    "viaYaw": hs['yaw'],
                    "entryPitch": hs.get('entry_pitch', 0.0),
                    "entryYaw": hs.get('entry_yaw', 0.0)
                }
            })
        safe_haov = float(scene.get('haov', 360) or 360)
        safe_vaov = float(scene.get('vaov', 180) or 180)
        safe_haov = min(360.0, max(30.0, safe_haov))
        safe_vaov = min(180.0, max(30.0, safe_vaov))
        cfg = {
            "title": scene['name'],
            "type": "equirectangular",
            "haov": safe_haov,
            "vaov": safe_vaov,
            "panorama": panorama_url,
            "hotSpots": hotspots,
            "minHfov": 30,
            "maxHfov": 120
        }
        if hires_url:
            cfg["hires"] = hires_url
        if preview_url:
            cfg["preview"] = preview_url
        if pano_w:
            cfg["pano_w"] = pano_w
        if preview_w:
            cfg["preview_w"] = preview_w
        tour_config["scenes"][scene['id']] = cfg
    config_json = json.dumps(tour_config, indent=4)
    watermark_html = '<div class="wm-badge">Created with Pan-o-Rama Free</div>' if watermark_enabled else ""
    html_content = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8"><title>Virtual Tour</title>
    <meta name="lokalny_obiektyw_gallery_template" content="v{GALLERY_TEMPLATE_VERSION}">
    <link rel="stylesheet" href="https://cdn.jsdelivr.net/npm/pannellum@2.5.6/build/pannellum.css"/>
    <script src="https://cdn.jsdelivr.net/npm/pannellum@2.5.6/build/pannellum.js"></script>
	    <style>
	        body {{ margin: 0; padding: 0; background: #000; overflow: hidden; }}
	        #panorama {{ width: 100vw; height: 100vh; background: #0b0b0b; }}
	        .custom-hotspot {{ height: 50px; width: 50px; background: rgba(0, 123, 255, 0.4); border: 3px solid #fff; border-radius: 50%; cursor: pointer; box-shadow: 0 0 15px rgba(0,0,0,0.5); transition: all 0.3s ease; display: flex; align-items: center; justify-content: center; }}
	        .custom-hotspot::after {{ content: ''; width: 15px; height: 15px; border-top: 5px solid #fff; border-right: 5px solid #fff; transform: rotate(-45deg) translate(-2px, 2px); }}
	        .custom-hotspot:hover {{ background: rgba(0, 123, 255, 0.8); transform: scale(1.2); box-shadow: 0 0 20px #007bff; }}
	        .pnlm-load-box, .pnlm-loading, .pnlm-about-msg {{ display: none !important; }}
	        .loading-overlay {{ position: fixed; top: 0; left: 0; right: 0; bottom: 0; background: rgba(0,0,0,0.45); display: none; align-items: center; justify-content: center; z-index: 10000; flex-direction: column; pointer-events: none; }}
	        /* When a low-res preview is visible, avoid a full-screen dimmer: show a small non-blocking HUD instead. */
	        .loading-overlay.preview-mode {{ background: transparent; align-items: flex-end; justify-content: center; padding: 0 0 18px 0; }}
	        .loading-overlay.preview-mode .loading-progress {{ width: min(420px, 72vw); margin-top: 8px; }}
	        .loading-overlay.preview-mode .loading-title {{ font-size: 13px; color: rgba(219,234,255,0.95); text-shadow: 0 2px 12px rgba(0,0,0,0.6); }}
	        .loading-overlay.preview-mode .loading-progress .bar {{ height: 8px; background: rgba(0,0,0,0.25); }}
	        .loading-overlay.preview-mode .loading-progress .pct {{ display: none; }}
        .loading-title {{ font: 700 16px/1.2 'Segoe UI', sans-serif; color: #dbeaff; }}
        .loading-progress {{ width: min(520px, 78vw); margin-top: 12px; }}
        .loading-progress .bar {{ height: 10px; background: rgba(255,255,255,0.14); border: 1px solid rgba(255,255,255,0.18); border-radius: 999px; overflow: hidden; }}
        .loading-progress .fill {{ height: 100%; width: 0%; background: linear-gradient(90deg, #4da3ff, #0d6efd); transition: width 160ms ease; }}
        .loading-progress .pct {{ margin-top: 8px; font: 600 13px/1.2 'Segoe UI', sans-serif; color: #bcd7ff; text-align: center; }}
	        .scene-nav {{ position: fixed; top: 14px; right: 14px; z-index: 9999; pointer-events: auto; }}
	        .quality-nav {{ position: fixed; left: 14px; bottom: 14px; z-index: 9999; pointer-events: auto; }}
        .scene-nav-btn {{ width: 56px; height: 44px; border-radius: 10px; border: 1px solid rgba(255,255,255,0.25); background: rgba(0,0,0,0.55); color: #e6f1ff; font: 800 14px/1 'Segoe UI', sans-serif; cursor: pointer; letter-spacing: 0.4px; }}
        .scene-nav-btn.compact {{ width: 44px; }}
        .scene-nav-btn:hover {{ background: rgba(0,0,0,0.7); border-color: rgba(77,163,255,0.55); }}
        .scene-nav-btn svg {{ width: 22px; height: 22px; }}
        .scene-nav-btn svg path {{ fill: rgba(230,241,255,0.92); }}
        .scene-nav-btn svg circle {{ fill: rgba(10,10,10,0.55); }}
        .scene-nav-menu {{ position: absolute; top: 52px; right: 0; width: min(320px, 78vw); max-height: 50vh; overflow: auto; display: none; background: rgba(10,10,10,0.9); border: 1px solid rgba(255,255,255,0.18); border-radius: 12px; padding: 8px; box-shadow: 0 12px 30px rgba(0,0,0,0.6); }}
        .scene-nav-item {{ display: flex; align-items: center; justify-content: space-between; gap: 10px; padding: 10px 10px; border-radius: 10px; cursor: pointer; color: #d8e8ff; font: 600 14px/1.2 'Segoe UI', sans-serif; }}
        .scene-nav-item:hover {{ background: rgba(77,163,255,0.16); }}
        .scene-nav-item .pill {{ font: 700 11px/1 'Segoe UI', sans-serif; color: #9fc7ff; border: 1px solid rgba(77,163,255,0.35); background: rgba(0,0,0,0.35); padding: 4px 8px; border-radius: 999px; }}
        .wm-badge {{ position: fixed; bottom: 14px; right: 14px; background: rgba(0,0,0,0.55); color: #d8e8ff; border: 1px solid rgba(77,163,255,0.5); border-radius: 999px; padding: 7px 11px; font: 600 12px/1.2 'Segoe UI', sans-serif; z-index: 9999; pointer-events: none; }}
    </style>
</head>
	<body>
		    <div id="panorama"></div>
		    <div class="quality-nav" id="qualityNav" style="display:none">
	        <button class="scene-nav-btn compact" id="qualityBtn" type="button" aria-label="Quality">Web</button>
	    </div>
		    <div class="scene-nav" id="sceneNav">
	        <button class="scene-nav-btn compact" id="sceneNavBtn" type="button" aria-label="Points of Interest">
	            <svg viewBox="0 0 24 24" aria-hidden="true" focusable="false">
	                <path d="M12 21s7-5.2 7-11a7 7 0 1 0-14 0c0 5.8 7 11 7 11z"></path>
	                <circle cx="12" cy="10" r="2.6"></circle>
	            </svg>
	        </button>
	        <div class="scene-nav-menu" id="sceneNavMenu"></div>
	    </div>
    <div class="loading-overlay" id="tourLoader">
        <div class="loading-title" id="tourLoaderText">Loading scene...</div>
        <div class="loading-progress">
            <div class="bar"><div class="fill" id="tourLoaderBar"></div></div>
            <div class="pct" id="tourLoaderPct">0%</div>
        </div>
    </div>
	    {watermark_html}
		    <script>
		    try {{
		        console.log('[tour] template', 'v{GALLERY_TEMPLATE_VERSION}');
		    }} catch (_) {{}}
		    const DEBUG = (() => {{
		        try {{
		            const sp = new URLSearchParams(location.search);
		            return sp.get('debug') === '1' || sp.get('debug') === 'true';
		        }} catch (_) {{ return false; }}
		    }})();
		    function dbg(msg) {{
		        if (!DEBUG) return;
		        try {{
		            console.log('[tour]', `[${{new Date().toISOString().slice(11, 19)}}]`, msg);
		        }} catch (_) {{}}
		    }}

			    const prefetchCache = new Set();
			    const prefetchPromises = new Map(); // sceneId -> Promise
			    const fullReady = new Set(); // base sceneId -> boolean
			    const prefetchState = new Map(); // base sceneId -> 'pending' | 'ok' | 'error'
			    const prefetchQueue = [];
			    const PREFETCH_CONCURRENCY = 1;
			    let prefetchInflight = 0;
			    let lastUserInputAt = Date.now();
			    let userInteracting = false;
			    const hiresMap = {{}}; // base sceneId -> hires url (if available)
			    const QUALITY_STANDARD = 'standard'; // web.jpg or original if no web
			    const QUALITY_ULTRA = 'ultra'; // original hi-res (opt-in, may be slow)
			    let qualityMode = QUALITY_STANDARD;
			    // Do not change HFOV for preview: any "compensation" is perceived as a scale jump.
			    const PREVIEW_ZOOM_MAX = 1.0;
		    let loadTimer = null;
		    let loadPct = 0;
		    let transitioning = false;
		    let pendingRestore = null; // {{ sceneId, hfov }}
		    const sceneState = {{}}; // sceneId -> {{ hfov }}
		    let loaderShowTimer = null;
		    let loaderPreviewMode = false;
		    let loaderWatchdogTimer = null;
		    let activeSceneLoadToken = 0;
		    const previewMap = {{}}; // sceneId -> preview url (kept out of Pannellum config)
			    let activeLoadTarget = null; // base sceneId we're waiting full-res for
			    let loaderActive = false;
			    let switchState = 'idle'; // idle|loading_preview|preview_ready|loading_full
			    // switchReq fields: baseId, previewId, pitch, yaw, hfov, prefetchP, token, previewLoadedAt
			    let switchReq = null;
			    // We synthesize dedicated preview scenes (id: baseId + '__preview') so switching is immediate and interactive.

			    function isPreviewSceneId(sceneId) {{
			        return !!sceneId && sceneId.endsWith('__preview');
			    }}
			    function isHiresSceneId(sceneId) {{
			        return !!sceneId && sceneId.endsWith('__hires');
			    }}
			    function baseSceneId(sceneId) {{
			        if (isPreviewSceneId(sceneId)) return sceneId.slice(0, -9);
			        if (isHiresSceneId(sceneId)) return sceneId.slice(0, -7);
			        return sceneId;
			    }}
			    function previewSceneId(sceneId) {{
			        return `${{sceneId}}__preview`;
			    }}
			    function hiresSceneId(sceneId) {{
			        return `${{sceneId}}__hires`;
			    }}

			    function previewCompensatedHfov(_baseId, hfov) {{ return hfov; }}

				    async function prefetchUrl(url) {{
				        // Warm the browser cache in the background.
				        // Important: avoid blob/image-bitmap decoding here; it can create huge memory spikes
				        // and trigger "page unresponsive" warnings on large panoramas.
				        try {{
				            const ctrl = new AbortController();
				            const t = setTimeout(() => {{ try {{ ctrl.abort(); }} catch (_) {{}} }}, 60000);
				            const r = await fetch(url, {{ cache: 'force-cache', signal: ctrl.signal }});
				            clearTimeout(t);
				            if (!r || !r.ok) {{
				                dbg(`prefetch fetch not ok url=${{url}} status=${{r ? r.status : '(noresp)'}}`);
				                return false;
				            }}
				            try {{
				                if (r.body && r.body.getReader) {{
				                    const reader = r.body.getReader();
				                    while (true) {{
				                        const {{ done }} = await reader.read();
				                        if (done) break;
				                    }}
				                }} else {{
				                    await r.arrayBuffer();
				                }}
				                return true;
				            }} catch (e) {{
				                dbg(`prefetch drain error url=${{url}} err=${{(e && e.message) ? e.message : e}}`);
				                return false;
				            }}
				        }} catch (e) {{
				            dbg(`prefetch fetch error url=${{url}} err=${{(e && e.message) ? e.message : e}}`);
				            return false;
				        }}
				    }}

			    function pumpPrefetch() {{
			        while (prefetchInflight < PREFETCH_CONCURRENCY && prefetchQueue.length) {{
			            const sceneId = prefetchQueue.shift();
			            const scene = tourConfig.scenes[sceneId];
			            if (!scene || !scene.panorama) continue;
			            prefetchInflight++;
			            prefetchState.set(sceneId, 'pending');
			            const p = prefetchUrl(scene.panorama).then((ok) => {{
			                if (ok) {{
			                    fullReady.add(baseSceneId(sceneId));
			                    prefetchState.set(sceneId, 'ok');
			                }} else {{
			                    fullReady.delete(baseSceneId(sceneId));
			                    prefetchState.set(sceneId, 'error');
			                }}
			                return !!ok;
			            }}).finally(() => {{
			                prefetchInflight--;
			                pumpPrefetch();
			            }});
			            prefetchPromises.set(sceneId, p);
			        }}
			    }}

		    function ensureFullPrefetch(baseId) {{
		        const st = prefetchState.get(baseId);
		        if (st === 'ok' || st === 'pending') return prefetchPromises.get(baseId) || Promise.resolve(true);
		        const scene = (tourConfig && tourConfig.scenes) ? tourConfig.scenes[baseId] : null;
		        if (!scene || !scene.panorama) return Promise.resolve(false);
		        if (!prefetchPromises.has(baseId)) {{
		            prefetchQueue.push(baseId);
		            pumpPrefetch();
		        }}
		        return prefetchPromises.get(baseId) || Promise.resolve(false);
		    }}

				    function requestSwitch(targetSceneId, entryPitch, entryYaw, hfov) {{
				        const baseId = baseSceneId(targetSceneId);
				        if (!baseId || !window.viewer) return;
			        if (switchState !== 'idle') {{
			            // Allow users to keep navigating while we're showing a preview by canceling the
			            // in-flight full-res switch and starting a new one.
			            try {{
			                const cur = window.viewer.getScene ? window.viewer.getScene() : null;
			                if (cur && isPreviewSceneId(cur)) {{
			                    dbg(`switch cancel (busy=${{switchState}}) -> ${{baseId}}`);
			                    switchReq = null;
			                    switchState = 'idle';
			                    activeLoadTarget = null;
			                    endSceneLoading();
			                }} else {{
			                    dbg(`switch ignored (busy): ${{switchState}} -> ${{baseId}}`);
			                    return;
			                }}
			            }} catch (_) {{
			                dbg(`switch ignored (busy): ${{switchState}} -> ${{baseId}}`);
			                return;
			            }}
			        }}
				        const previewId = previewSceneId(baseId);
				        const havePreview = !!(tourConfig && tourConfig.scenes && tourConfig.scenes[previewId]);
				        const myTok = (activeSceneLoadToken + 1); // optimistic token before beginSceneLoadingSmart bumps it
				        const hfovForPreview = (havePreview && typeof hfov === 'number') ? previewCompensatedHfov(baseId, hfov) : hfov;
				        switchReq = {{ baseId: baseId, previewId: previewId, pitch: entryPitch, yaw: entryYaw, hfov: hfovForPreview, prefetchP: null, token: myTok, previewLoadedAt: 0 }};
			        activeLoadTarget = baseId;
			        switchState = havePreview ? 'loading_preview' : 'loading_full';
		        dbg(`switch target=${{baseId}} havePreview=${{havePreview}} state=${{prefetchState.get(baseId) || '(none)'}}`);

		        const delay = havePreview ? 1800 : 420;
		        beginSceneLoadingSmart(baseId, havePreview ? 'Loading HD...' : 'Loading scene...', {{
		            delayMs: delay,
		            allowPreview: havePreview
		        }});

		        try {{ switchReq.prefetchP = ensureFullPrefetch(baseId); }} catch (_) {{ switchReq.prefetchP = Promise.resolve(false); }}

			        if (havePreview) {{
			            dbg(`switch: load preview -> ${{previewId}}`);
			            safeLoadSceneWithView(previewId, entryPitch, entryYaw, hfovForPreview);
			        }} else {{
			            dbg(`switch: load full -> ${{baseId}}`);
			            safeLoadSceneWithView(baseId, entryPitch, entryYaw, hfov);
			        }}
			    }}

			    function maybeStartFullAfterPreview() {{
			        if (!switchReq || switchState !== 'preview_ready') return;
			        const baseId = switchReq.baseId;
			        const previewId = switchReq.previewId;
			        const p = switchReq.prefetchP || Promise.resolve(false);
			        const previewAt = switchReq.previewLoadedAt || Date.now();

			        const sleep = (ms) => new Promise((r) => setTimeout(r, ms));
			        const timeout = (ms) => new Promise((r) => setTimeout(() => r('timeout'), ms));
			        const raf2 = () => new Promise((r) => requestAnimationFrame(() => requestAnimationFrame(r)));
			        const waitForIdle = async (idleMs, maxWaitMs) => {{
			            const start = Date.now();
			            while (true) {{
			                if (!switchReq || switchState !== 'preview_ready') return false;
			                if (!window.viewer || !window.viewer.getScene) return false;
			                if (window.viewer.getScene() !== previewId) return false;
			                const since = Date.now() - (lastUserInputAt || 0);
			                const ok = (!userInteracting) && (since >= idleMs);
			                if (ok) return true;
			                if ((Date.now() - start) >= maxWaitMs) return true; // force eventually
			                await sleep(120);
			            }}
			        }};

			        // Keep preview interactive while HD warms up. Starting loadScene(full) too early causes
			        // a black canvas during the full-res download/decode.
			        const MAX_WAIT_MS = 25000;
			        const MIN_PREVIEW_DWELL_MS = 450;
			        const IDLE_UPGRADE_MS = 800;

			        Promise.race([p.then((ok) => ok ? 'ok' : 'error'), timeout(MAX_WAIT_MS)]).then((res) => {{
			            try {{
			                if (!switchReq || switchState !== 'preview_ready') return;
			                if (!window.viewer || !window.viewer.getScene) return;
			                if (window.viewer.getScene() !== previewId) return;

			                const dwell = Date.now() - previewAt;
			                const extra = Math.max(0, MIN_PREVIEW_DWELL_MS - dwell);
			                dbg(`switch: prefetch result=${{res}} dwellMs=${{dwell}} extraWaitMs=${{extra}}`);

			                // Preserve the user's current view while waiting in preview (feels seamless).
			                const curPitch = (typeof window.viewer.getPitch === 'function') ? window.viewer.getPitch() : switchReq.pitch;
			                const curYaw = (typeof window.viewer.getYaw === 'function') ? window.viewer.getYaw() : switchReq.yaw;
			                const curHfov = (typeof window.viewer.getHfov === 'function') ? window.viewer.getHfov() : switchReq.hfov;
			                switchReq.pitch = curPitch;
			                switchReq.yaw = curYaw;
			                switchReq.hfov = curHfov;

			                // Even if prefetch failed/timed out, we eventually need to try loading full.
			                Promise.resolve()
			                    .then(() => extra ? sleep(extra) : null)
			                    .then(() => raf2())
			                    .then(() => waitForIdle(IDLE_UPGRADE_MS, 15000))
			                    .then(() => {{
			                        if (!switchReq || switchState !== 'preview_ready') return;
			                        if (!window.viewer || !window.viewer.getScene) return;
			                        if (window.viewer.getScene() !== previewId) return;
			                        switchState = 'loading_full';
			                        dbg(`switch: load full after preview -> ${{baseId}}`);
			                        safeLoadSceneWithView(baseId, curPitch, curYaw, curHfov);
			                    }});
			            }} catch (_) {{}}
			        }});
			    }}

		    function setLoader(show, text) {{
		        const el = document.getElementById('tourLoader');
		        const t = document.getElementById('tourLoaderText');
		        if (t && text) t.textContent = text;
		        if (!el) return;
		        if (loaderPreviewMode) el.classList.add('preview-mode');
		        else el.classList.remove('preview-mode');
		        el.style.display = show ? 'flex' : 'none';
		        if (!show) {{
		            if (loadTimer) {{ clearInterval(loadTimer); loadTimer = null; }}
		            loadPct = 0;
		            setLoaderProgress(0);
		        }}
		    }}

    function setLoaderProgress(pct) {{
        const bar = document.getElementById('tourLoaderBar');
        const label = document.getElementById('tourLoaderPct');
        if (!bar || !label) return;
        const v = Math.max(0, Math.min(100, Math.round(pct)));
        bar.style.width = `${{v}}%`;
        label.textContent = `${{v}}%`;
    }}

	    function startFakeProgress() {{
	        if (loadTimer) {{ clearInterval(loadTimer); loadTimer = null; }}
	        loadPct = Math.max(loadPct, 6);
	        setLoaderProgress(loadPct);
	        loadTimer = setInterval(() => {{
	            loadPct += (loadPct < 70 ? 6 : (loadPct < 90 ? 2 : 1));
	            if (loadPct >= 92) loadPct = 92;
	            setLoaderProgress(loadPct);
	        }}, 160);
	    }}

		    function beginSceneLoadingSmart(sceneId, text, opts) {{
	        // Show loader only if load takes "long enough" to be noticeable.
	        // Never fully suppress: even with previews, full-res may still take long and looks like a hang.
		        const o = opts || {{}};
		        const delayMs = (typeof o.delayMs === 'number') ? o.delayMs : 420;
		        const allowPreview = (typeof o.allowPreview === 'boolean') ? o.allowPreview : true;
		        if (loaderShowTimer) {{ clearTimeout(loaderShowTimer); loaderShowTimer = null; }}
		        if (loaderWatchdogTimer) {{ clearTimeout(loaderWatchdogTimer); loaderWatchdogTimer = null; }}
		        activeSceneLoadToken++;
		        const myToken = activeSceneLoadToken;
		        // Reset visuals but don't show yet.
		        loaderActive = true;
		        setLoader(false);
		        loadPct = 0;
		        setLoaderProgress(0);
		        loaderPreviewMode = !!(allowPreview && sceneId && previewMap[sceneId]);
		        dbg(`begin-load scene=${{sceneId || '(unknown)'}} delayMs=${{delayMs}} allowPreview=${{allowPreview}} previewMode=${{loaderPreviewMode}} token=${{myToken}}`);
		        loaderShowTimer = setTimeout(() => {{
		            if (myToken !== activeSceneLoadToken) return;
		            // If the load finished quickly, 'load' handler may have cleared this already.
		            loaderShowTimer = null;
		            setLoader(true, text || 'Loading scene...');
		            startFakeProgress();
		            dbg(`loader shown (delay=${{delayMs}}ms)`);
		        }}, delayMs);

		        // Absolute safety net: never keep a loader alive forever (bad caches / missing timings).
		        // Full-res can legitimately take a long time on slow devices / huge panos, so be generous.
		        const watchdogMs = allowPreview ? 60000 : 30000;
		        loaderWatchdogTimer = setTimeout(() => {{
		            if (myToken !== activeSceneLoadToken) return;
		            dbg('watchdog: forcing endSceneLoading()');
		            endSceneLoading();
		        }}, watchdogMs);
		    }}

	    function beginSceneLoading(text) {{
	        beginSceneLoadingSmart(null, text);
	    }}

		    function endSceneLoading() {{
		        if (!loaderActive) return;
		        loaderActive = false;
		        if (loaderShowTimer) {{ clearTimeout(loaderShowTimer); loaderShowTimer = null; }}
		        if (loaderWatchdogTimer) {{ clearTimeout(loaderWatchdogTimer); loaderWatchdogTimer = null; }}
		        if (loadTimer) {{ clearInterval(loadTimer); loadTimer = null; }}
		        loadPct = 100;
		        setLoaderProgress(100);
		        setTimeout(() => setLoader(false), 120);
		        dbg('end-load');
		    }}

    function defaultHfov() {{
        return (tourConfig.default && typeof tourConfig.default.hfov === 'number') ? tourConfig.default.hfov : 70;
    }}

	    function storeBaseHfov() {{
	        if (!window.viewer || transitioning) return;
	        const sid = baseSceneId(window.viewer.getScene());
	        if (!sid) return;
	        sceneState[sid] = {{ hfov: window.viewer.getHfov() }};
	    }}

		    function prefetchScene(sceneId) {{
		        if (!sceneId) return;
		        const baseId = baseSceneId(sceneId);
		        const st = prefetchState.get(baseId);
		        if (st === 'ok' || st === 'pending' || prefetchCache.has(baseId)) return;
		        const scene = tourConfig.scenes[baseId];
		        if (!scene || !scene.panorama) return;
		        if (previewMap[baseId]) {{
		            const img = new Image();
		            img.src = previewMap[baseId];
		            dbg(`prefetch preview ${{baseId}}`);
		        }}
		        if (!prefetchPromises.has(baseId)) {{
		            prefetchQueue.push(baseId);
		            pumpPrefetch();
		            dbg(`prefetch full queued ${{baseId}}`);
		        }}
		        prefetchCache.add(baseId);
		    }}
	    function prefetchLinkedScenes(sceneId) {{
	        const scene = tourConfig.scenes[baseSceneId(sceneId)];
	        if (!scene || !scene.hotSpots) return;
	        scene.hotSpots.forEach(hs => prefetchScene(hs.clickHandlerArgs.targetSceneId));
	    }}

		    function safeLoadScene(sceneId, pitch, yaw, hfov, fadeMs) {{
		        try {{
		            const args = [sceneId];
		            // Only pass view params when at least one is a number; otherwise keep default view.
		            // Passing (undefined, undefined, undefined, 0) can behave inconsistently across browsers.
		            if (typeof pitch === 'number' || typeof yaw === 'number' || typeof hfov === 'number') {{
		                args.push(pitch, yaw, hfov);
		                if (typeof fadeMs === 'number') args.push(fadeMs);
		            }}
		            window.viewer.loadScene.apply(window.viewer, args);
		        }} catch (_) {{}}
		    }}

			    function safeLoadSceneWithView(sceneId, pitch, yaw, hfov) {{
			        if (typeof pitch === 'number' || typeof yaw === 'number' || typeof hfov === 'number') {{
			            safeLoadScene(sceneId, pitch, yaw, hfov, 0);
			        }} else {{
			            safeLoadScene(sceneId);
			        }}
			    }}

			    function getCurrentView() {{
			        try {{
			            const v = window.viewer;
			            if (!v) return {{}};
			            return {{
			                pitch: (typeof v.getPitch === 'function') ? v.getPitch() : undefined,
			                yaw: (typeof v.getYaw === 'function') ? v.getYaw() : undefined,
			                hfov: (typeof v.getHfov === 'function') ? v.getHfov() : undefined,
			            }};
			        }} catch (_) {{
			            return {{}};
			        }}
			    }}

			    function applyQualityMode(mode) {{
			        qualityMode = (mode === QUALITY_ULTRA) ? QUALITY_ULTRA : QUALITY_STANDARD;
			        try {{ localStorage.setItem('tourQualityMode', qualityMode); }} catch (_) {{}}
			        dbg(`quality: set mode=${{qualityMode}}`);
			        // If switching down to standard while on a hires scene, return to base.
			        try {{
			            if (!window.viewer) return;
			            const cur = window.viewer.getScene();
			            if (qualityMode === QUALITY_STANDARD && isHiresSceneId(cur)) {{
			                const baseId = baseSceneId(cur);
			                const view = getCurrentView();
			                beginSceneLoadingSmart(baseId, 'Switching quality...', {{ delayMs: 0, allowPreview: true }});
			                safeLoadSceneWithView(baseId, view.pitch, view.yaw, view.hfov);
			            }}
			        }} catch (_) {{}}
			    }}

			    function maybeUpgradeToUltra() {{
			        try {{
			            if (qualityMode !== QUALITY_ULTRA) return;
			            if (!window.viewer) return;
			            if (switchState !== 'idle') return; // don't interfere with scene switching
			            const cur = window.viewer.getScene();
			            if (!cur) return;
			            if (isPreviewSceneId(cur)) return;
			            if (isHiresSceneId(cur)) return; // already ultra
			            const baseId = baseSceneId(cur);
			            if (!hiresMap[baseId]) {{
			                dbg(`quality: no hires for ${{baseId}}`);
			                return;
			            }}
			            const view = getCurrentView();
			            // Only start the heavy upgrade when user is idle (avoid freezing mid-interaction).
			            const since = Date.now() - (lastUserInputAt || 0);
			            if (userInteracting || since < 800) {{
			                setTimeout(maybeUpgradeToUltra, 250);
			                return;
			            }}
				            dbg(`quality: upgrading to hd for ${{baseId}}`);
				            beginSceneLoadingSmart(baseId, 'Loading HD...', {{ delayMs: 650, allowPreview: true }});
				            safeLoadSceneWithView(hiresSceneId(baseId), view.pitch, view.yaw, view.hfov);
				        }} catch (_) {{}}
				    }}

				    function loadTargetWithPreview(targetSceneId, entryPitch, entryYaw, hfov) {{
				        requestSwitch(targetSceneId, entryPitch, entryYaw, hfov);
				    }}

			    function smoothSwitch(e, args) {{
			        const viewer = window.viewer;
			        const fromScene = baseSceneId(viewer.getScene());
			        const baseFrom = (sceneState[fromScene] && typeof sceneState[fromScene].hfov === 'number') ? sceneState[fromScene].hfov : viewer.getHfov();
			        const zoomTarget = Math.max(30, baseFrom - 40);
			        const targetSceneId = args.targetSceneId;
			        const baseTarget = (sceneState[targetSceneId] && typeof sceneState[targetSceneId].hfov === 'number') ? sceneState[targetSceneId].hfov : defaultHfov();
			        pendingRestore = {{ sceneId: targetSceneId, hfov: baseTarget }};
			        transitioning = true;
			        viewer.setHfov(zoomTarget, 800);
			        viewer.lookAt(args.viaPitch, args.viaYaw, zoomTarget, 800);
		        // Start scene switch while zoom animation is still running for a smoother transition.
		        setTimeout(() => {{
		            // Important: load the target at its intended HFOV (baseTarget). Otherwise users see a
		            // "zoom jump" during preview because the kinetic zoom HFOV is much smaller.
		            loadTargetWithPreview(targetSceneId, args.entryPitch, args.entryYaw, baseTarget);
		        }}, 260);
		    }}
			    const tourConfig = {config_json};
			    // Build explicit preview scenes (interactive spherical) and remove base-scene `preview`
			    // to avoid the "previous scene freeze" behavior seen with Pannellum's built-in preview.
			    try {{
			        // Pull hires URLs out of config so we can build dedicated hires scenes.
			        Object.keys(tourConfig.scenes || {{}}).forEach((sid) => {{
			            const sc = tourConfig.scenes[sid];
			            if (sc && sc.hires) {{
			                hiresMap[sid] = sc.hires;
			                delete sc.hires;
			            }}
			        }});
			        Object.keys(tourConfig.scenes || {{}}).forEach((sid) => {{
			            const sc = tourConfig.scenes[sid];
			            if (sc && sc.preview) {{
			                previewMap[sid] = sc.preview;
			                delete sc.preview;
			            }}
			        }});
				        Object.keys(previewMap).forEach((sid) => {{
				            const sc = tourConfig.scenes[sid];
				            if (!sc) return;
				            const pid = previewSceneId(sid);
				            if (tourConfig.scenes[pid]) return;
				            const hs = Array.isArray(sc.hotSpots) ? sc.hotSpots.map(h => Object.assign({{}}, h)) : [];
				            tourConfig.scenes[pid] = Object.assign({{}}, sc, {{
				                title: (sc.title || sid) + ' (preview)',
				                panorama: previewMap[sid],
				                hotSpots: hs,
				            }});
				        }});
			        Object.keys(hiresMap).forEach((sid) => {{
			            const sc = tourConfig.scenes[sid];
			            if (!sc) return;
			            const hid = hiresSceneId(sid);
			            if (tourConfig.scenes[hid]) return;
			            tourConfig.scenes[hid] = Object.assign({{}}, sc, {{
			                title: (sc.title || sid) + ' (ultra)',
			                panorama: hiresMap[sid],
			                // keep hotspots in ultra so navigation still works
			            }});
			        }});
			        dbg(`previewMap initialized: ${{Object.keys(previewMap).length}} previews`);
			    }} catch (_) {{}}
		    Object.values(tourConfig.scenes).forEach(scene => {{
		        if (scene.hotSpots) {{
		            scene.hotSpots.forEach(hs => {{
		                if (hs.clickHandlerFunc === "smoothSwitch") hs.clickHandlerFunc = smoothSwitch;
		            }});
		        }}
		    }});
		    window.viewer = pannellum.viewer('panorama', tourConfig);
		    prefetchLinkedScenes(tourConfig.default.firstScene);
				    window.viewer.on('scenechange', (sceneId) => {{
				        prefetchLinkedScenes(sceneId);
				        dbg(`scenechange -> ${{sceneId}}`);
				        try {{ updateQualityUI(); }} catch (_) {{}}
				        // When user opted into ultra, upgrade after arriving and becoming idle.
				        if (qualityMode === QUALITY_ULTRA) {{
				            setTimeout(maybeUpgradeToUltra, 50);
				        }}
				    }});
				    window.viewer.on('load', () => {{
				        const rawSid = (() => {{ try {{ return window.viewer.getScene(); }} catch (_) {{ return null; }} }})();
				        const baseSid = baseSceneId(rawSid);
				        const isPreviewLoaded = !!(switchReq && rawSid && rawSid === switchReq.previewId);
				        let preserveView = null;

		        // If a preview scene loaded, keep the loader timer alive: we still want to show "Loading HD..."
		        // if full-res takes long. We'll cancel/end only when the base scene finishes loading.
		        if (loaderShowTimer && !isPreviewLoaded) {{
		            clearTimeout(loaderShowTimer);
		            loaderShowTimer = null;
		            dbg('load: canceled pending loaderShowTimer');
		        }}

				        if (switchReq && rawSid) {{
					            if (rawSid === switchReq.previewId) {{
					                dbg(`load: preview loaded for ${{switchReq.baseId}}`);
					                try {{ switchReq.previewLoadedAt = Date.now(); }} catch (_) {{}}
					                switchState = 'preview_ready';
					                transitioning = false; // allow panning in preview
					                maybeStartFullAfterPreview();
					                return;
					            }}
					            if (baseSid && baseSid === switchReq.baseId && !isPreviewSceneId(rawSid)) {{
					                dbg(`load: full loaded for ${{switchReq.baseId}}`);
					                preserveView = {{
					                    pitch: switchReq.pitch,
					                    yaw: switchReq.yaw,
					                    hfov: switchReq.hfov,
					                }};
					                switchReq = null;
					                switchState = 'idle';
					                try {{ activeLoadTarget = null; }} catch (_) {{}}
					                // fallthrough to common post-load behavior
					            }}
				        }}
				        try {{
			            const sid = baseSid;
			            if (sid) {{
			                let target = defaultHfov();
			                if (preserveView && typeof preserveView.hfov === 'number') {{
			                    // Keep the user's view from preview so the switch is seamless.
			                    target = preserveView.hfov;
			                    if (pendingRestore && pendingRestore.sceneId === sid) pendingRestore = null;
			                }} else if (pendingRestore && pendingRestore.sceneId === sid) {{
			                    target = pendingRestore.hfov;
			                    pendingRestore = null;
			                }} else if (sceneState[sid] && typeof sceneState[sid].hfov === 'number') {{
			                    target = sceneState[sid].hfov;
			                }}
			                sceneState[sid] = {{ hfov: target }};
			                window.viewer.setHfov(target, 0);
			                if (
			                    preserveView &&
			                    typeof preserveView.pitch === 'number' &&
			                    typeof preserveView.yaw === 'number' &&
			                    typeof window.viewer.lookAt === 'function'
			                ) {{
			                    window.viewer.lookAt(preserveView.pitch, preserveView.yaw, target, 0);
			                }}
			            }}
				        }} catch (_) {{}}
				        try {{ updateQualityUI(); }} catch (_) {{}}
					        transitioning = false;
					        setTimeout(() => endSceneLoading(), 120);
						    }});
		    // Kick off initial load after handlers are attached (avoids missing the first 'load' event).
		    beginSceneLoadingSmart(tourConfig.default.firstScene, 'Loading scene...', {{ delayMs: 0, allowPreview: false }});
		    safeLoadScene(tourConfig.default.firstScene);

			    // Track user zoom so we don't "lock" them to one hfov.
			    const panoEl = document.getElementById('panorama');
			    if (panoEl) {{
			        const bump = () => {{ lastUserInputAt = Date.now(); }};
			        const markDown = () => {{ userInteracting = true; bump(); }};
			        const markUp = () => {{ userInteracting = false; bump(); }};
			        panoEl.addEventListener('pointerdown', markDown);
			        panoEl.addEventListener('pointerup', markUp);
			        panoEl.addEventListener('pointercancel', markUp);
			        panoEl.addEventListener('pointermove', bump, {{ passive: true }});
			        panoEl.addEventListener('touchstart', markDown, {{ passive: true }});
			        panoEl.addEventListener('touchend', markUp, {{ passive: true }});
			        panoEl.addEventListener('touchcancel', markUp, {{ passive: true }});
			        panoEl.addEventListener('mousedown', markDown);
			        panoEl.addEventListener('mouseup', markUp);
			        panoEl.addEventListener('mousemove', bump, {{ passive: true }});
			        panoEl.addEventListener('mouseup', () => setTimeout(storeBaseHfov, 0));
			        panoEl.addEventListener('touchend', () => setTimeout(storeBaseHfov, 0));
			        let wheelT = null;
			        panoEl.addEventListener('wheel', () => {{
			            bump();
			            if (wheelT) clearTimeout(wheelT);
			            wheelT = setTimeout(storeBaseHfov, 120);
			        }}, {{ passive: true }});
			    }}

				    // POI dropdown navigation (base scenes only; hide preview/hires variants)
				    const navBtn = document.getElementById('sceneNavBtn');
				    const navMenu = document.getElementById('sceneNavMenu');
				    function baseSceneIds() {{
				        return Object.keys(tourConfig.scenes || {{}})
				            .filter((sid) => !isPreviewSceneId(sid) && !isHiresSceneId(sid));
				    }}
				    function renderSceneMenu() {{
				        if (!navMenu) return;
				        navMenu.innerHTML = '';
				        const ids = baseSceneIds();
				        ids.forEach((sid, idx) => {{
				            const sc = tourConfig.scenes[sid] || {{}};
				            const item = document.createElement('div');
				            item.className = 'scene-nav-item';
				            item.innerHTML = `<span>${{(sc.title || sid)}}</span><span class="pill">#${{idx + 1}}</span>`;
				            item.addEventListener('click', (ev) => {{
				                ev.preventDefault();
				                ev.stopPropagation();
				                navMenu.style.display = 'none';
				                if (!window.viewer) return;
				                const curBase = baseSceneId(window.viewer.getScene && window.viewer.getScene());
				                if (curBase === sid) return;
				                transitioning = true;
				                loadTargetWithPreview(sid, undefined, undefined, undefined);
				            }});
				            navMenu.appendChild(item);
				        }});
				    }}
				    function toggleSceneMenu(show) {{
				        if (!navMenu) return;
				        const want = (show === undefined) ? (navMenu.style.display !== 'block') : !!show;
				        if (want) {{
				            renderSceneMenu();
				            navMenu.style.display = 'block';
				        }} else {{
				            navMenu.style.display = 'none';
				        }}
				    }}
				    if (navBtn && navMenu) {{
				        navBtn.addEventListener('click', (ev) => {{
				            ev.preventDefault();
				            ev.stopPropagation();
				            toggleSceneMenu();
				        }});
				        document.addEventListener('click', () => toggleSceneMenu(false));
				        document.addEventListener('keydown', (ev) => {{
				            if (ev.key === 'Escape') toggleSceneMenu(false);
				        }});
				    }}

				    // Quality toggle (Web <-> HD), only shown when this scene has a true hi-res variant.
				    const qNav = document.getElementById('qualityNav');
				    const qBtn = document.getElementById('qualityBtn');
				    function updateQualityUI() {{
				        try {{
				            if (!qNav || !qBtn || !window.viewer) return;
				            const cur = window.viewer.getScene ? window.viewer.getScene() : null;
				            const baseId = baseSceneId(cur);
				            const has = !!(baseId && hiresMap[baseId]);
				            if (!has) {{
				                qNav.style.display = 'none';
				                qualityMode = QUALITY_STANDARD;
				                qBtn.textContent = 'Web';
				                return;
				            }}
				            qNav.style.display = 'block';
				            qBtn.textContent = (qualityMode === QUALITY_ULTRA) ? 'HD' : 'Web';
				        }} catch (_) {{}}
				    }}
				    // Restore saved mode (optional) but only apply when available for the current scene.
				    try {{
				        const saved = localStorage.getItem('tourQualityMode');
				        if (saved === QUALITY_ULTRA) qualityMode = QUALITY_ULTRA;
				    }} catch (_) {{}}
				    updateQualityUI();
				    if (qBtn) {{
				        qBtn.addEventListener('click', (ev) => {{
				            ev.preventDefault();
				            ev.stopPropagation();
				            const cur = (qualityMode === QUALITY_ULTRA) ? QUALITY_STANDARD : QUALITY_ULTRA;
				            applyQualityMode(cur);
				            updateQualityUI();
				            if (qualityMode === QUALITY_ULTRA) setTimeout(maybeUpgradeToUltra, 50);
				        }});
				    }}
    </script>
</body>
</html>"""
    with open(os.path.join(app.config['PROCESSED_FOLDER'], project_id, 'index.html'), 'w', encoding='utf-8') as f: f.write(html_content)
    return f'/galleries/{project_id}/index.html'

@app.route('/api/project/<project_id>/hotspots', methods=['POST'])
def update_hotspots(project_id):
    try:
        data = request.json; p_dir = os.path.join(app.config['PROCESSED_FOLDER'], project_id); m_path = os.path.join(p_dir, 'metadata.json')
        with open(m_path, 'r') as f: meta = json.load(f)
        for scene in meta['scenes']:
            if scene['id'] in data: scene['hotspots'] = data[scene['id']]
        with open(m_path, 'w') as f: json.dump(meta, f)
        generate_tour(project_id, meta['scenes'])
        return jsonify({'message': 'OK'}), 200
    except Exception as e: return jsonify({'error': str(e)}), 500

@app.route('/api/project/<project_id>/finalize', methods=['POST'])
def finalize_project(project_id):
    try:
        p_dir = os.path.join(app.config['PROCESSED_FOLDER'], project_id)
        with open(os.path.join(p_dir, 'metadata.json'), 'r') as f: meta = json.load(f)
        url = generate_tour(project_id, meta['scenes'])
        return jsonify({'gallery_url': url}), 200
    except Exception as e: return jsonify({'error': str(e)}), 500

@app.route('/galleries/<project_id>/<path:filename>')
def serve_gallery_files(project_id, filename):
    db = get_db()
    tour = db.execute(
        "SELECT owner_id, visibility FROM tours WHERE id = ? AND deleted_at IS NULL",
        (project_id,),
    ).fetchone()
    if tour is not None and tour["visibility"] == "private":
        user = g.current_user
        if user is None or user["id"] != tour["owner_id"]:
            return jsonify({"error": "Forbidden"}), 403
    if filename == "index.html":
        # Backward-compatible: older published tours may have a static index.html without the latest
        # UX (loader, scene menu). Regenerate on-demand.
        try:
            pdir = os.path.join(app.config["PROCESSED_FOLDER"], project_id)
            ipath = os.path.join(pdir, "index.html")
            # Force regen for debugging sessions so we never fight browser or disk cache while iterating.
            force_regen = request.args.get("regen") in ("1", "true") or request.args.get("debug") in ("1", "true")
            needs_regen = True
            if os.path.exists(ipath):
                try:
                    with open(ipath, "r", encoding="utf-8", errors="ignore") as f:
                        head = f.read(8192)
                    # Regenerate if template marker is missing or outdated.
                    marker = f"lokalny_obiektyw_gallery_template\" content=\"v{GALLERY_TEMPLATE_VERSION}"
                    if (not force_regen) and marker in head:
                        needs_regen = False
                except Exception:
                    needs_regen = True
            if needs_regen:
                scenes = load_tour_scenes_and_hotspots(project_id)
                if not scenes:
                    scenes = load_disk_metadata_scenes(project_id)
                if not scenes:
                    scenes = load_disk_scenes_from_index_html(project_id)
                if scenes:
                    watermark_enabled = False
                    if tour is not None:
                        try:
                            owner_ent = get_user_entitlements(tour["owner_id"])
                            watermark_enabled = bool(owner_ent.get("watermark_enabled"))
                        except Exception:
                            watermark_enabled = False
                    # Note: some older galleries may exist on disk without a DB row in `tours`.
                    # We still regenerate `index.html` so players get current UX; access control is enforced above.
                    generate_tour(project_id, scenes, watermark_enabled=watermark_enabled, force_previews=bool(force_regen))
        except Exception as e:
            app.logger.warning(f"On-demand gallery regen failed: {e}")
    base_dir = os.path.join(app.config['PROCESSED_FOLDER'], project_id)
    # Avoid stale cached HTML in browsers/in-app webviews (Facebook, etc.).
    # send_from_directory/send_file uses conditional responses and may end up with Cache-Control: no-cache.
    # For the main HTML, return a plain response with no-store to force refresh.
    if filename.endswith(".html"):
        fpath = os.path.join(base_dir, filename)
        try:
            with open(fpath, "rb") as f:
                body = f.read()
        except Exception:
            body = None
        if body is None:
            return jsonify({"error": "Not found"}), 404
        resp = app.response_class(body, mimetype="text/html")
        resp.headers["Cache-Control"] = "no-store"
        return resp
    resp = send_from_directory(base_dir, filename)
    # Cache images aggressively; names are stable and served per-tour.
    if filename.lower().endswith((".jpg", ".jpeg", ".png", ".webp", ".gif")):
        resp.headers["Cache-Control"] = "public, max-age=31536000, immutable"
    return resp

init_db()

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)
