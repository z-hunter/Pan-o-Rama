from flask import Flask, request, jsonify, send_from_directory
import os
from werkzeug.utils import secure_filename
from PIL import Image, ExifTags, ImageOps
import uuid
import datetime
import cv2
import logging
from flask_cors import CORS
import json
import math
import numpy as np
import re

app = Flask(__name__)
CORS(app)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, '..', 'data')
UPLOAD_FOLDER = os.path.join(DATA_DIR, 'raw_uploads')
PROCESSED_FOLDER = os.path.join(DATA_DIR, 'processed_galleries')
FRONTEND_FOLDER = os.path.join(BASE_DIR, '..', 'frontend')

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

@app.route('/')
def index(): return send_from_directory(FRONTEND_FOLDER, 'index.html')

@app.route('/api/projects', methods=['GET'])
def list_projects():
    projects = []
    try:
        if os.path.exists(app.config['PROCESSED_FOLDER']):
            for pid in os.listdir(app.config['PROCESSED_FOLDER']):
                p_dir = os.path.join(app.config['PROCESSED_FOLDER'], pid)
                m_path = os.path.join(p_dir, 'metadata.json')
                if os.path.exists(m_path):
                    with open(m_path, 'r') as f: meta = json.load(f)
                    preview = None
                    if meta.get('scenes'):
                        s = meta['scenes'][0]
                        preview = f"{s['id']}/{s['panorama'] if s['panorama'] else s['images'][0]}"
                    projects.append({'project_id': pid, 'date': meta.get('created_at', 'Unknown'), 'preview_url': f'/galleries/{pid}/{preview}' if preview else None, 'gallery_url': f'/galleries/{pid}/index.html', 'timestamp': os.path.getctime(m_path)})
        return jsonify(projects), 200
    except Exception as e: return jsonify({'error': str(e)}), 500

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
                crop_offset = None
                with Image.open(img_p) as img:
                    aspect = img.width / img.height
                    vaov_c = 2 * math.degrees(math.atan(18.0 / focal_35))
                    if crop_offset is not None:
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

def generate_tour(project_id, scenes):
    tour_config = {"default": {"firstScene": scenes[0]['id'], "sceneFadeDuration": 1000, "autoLoad": True, "autoRotate": -2, "hfov": 70}, "scenes": {}}
    for scene in scenes:
        panorama_url = f"{scene['id']}/{scene['panorama'] if scene['panorama'] else scene['images'][0]}"
        hotspots = []
        for hs in scene.get('hotspots', []):
            hotspots.append({
                "pitch": hs['pitch'], "yaw": hs['yaw'], "type": "info", "text": f"Go to {hs['target_name']}", "cssClass": "custom-hotspot",
                "clickHandlerFunc": "smoothSwitch", "clickHandlerArgs": {"targetSceneId": hs['target_id'], "targetPitch": hs['pitch'], "targetYaw": hs['yaw']}
            })
        tour_config["scenes"][scene['id']] = {
            "title": scene['name'], "type": "equirectangular", "haov": scene.get('haov', 360), "vaov": scene.get('vaov', 180),
            "panorama": panorama_url, "hotSpots": hotspots, "minHfov": 30, "maxHfov": 120
        }
    config_json = json.dumps(tour_config, indent=4)
    html_content = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8"><title>Virtual Tour</title>
    <link rel="stylesheet" href="https://cdn.jsdelivr.net/npm/pannellum@2.5.6/build/pannellum.css"/>
    <script src="https://cdn.jsdelivr.net/npm/pannellum@2.5.6/build/pannellum.js"></script>
    <style>
        body {{ margin: 0; padding: 0; background: #000; overflow: hidden; }}
        #panorama {{ width: 100vw; height: 100vh; }}
        .custom-hotspot {{ height: 50px; width: 50px; background: rgba(0, 123, 255, 0.4); border: 3px solid #fff; border-radius: 50%; cursor: pointer; box-shadow: 0 0 15px rgba(0,0,0,0.5); transition: all 0.3s ease; display: flex; align-items: center; justify-content: center; }}
        .custom-hotspot::after {{ content: ''; width: 15px; height: 15px; border-top: 5px solid #fff; border-right: 5px solid #fff; transform: rotate(-45deg) translate(-2px, 2px); }}
        .custom-hotspot:hover {{ background: rgba(0, 123, 255, 0.8); transform: scale(1.2); box-shadow: 0 0 20px #007bff; }}
    </style>
</head>
<body>
    <div id="panorama"></div>
    <script>
    function smoothSwitch(e, args) {{
        const viewer = window.viewer;
        const currentHfov = viewer.getHfov();
        viewer.setHfov(currentHfov - 40, 800);
        viewer.lookAt(args.targetPitch, args.targetYaw, currentHfov - 40, 800, () => {{
            viewer.loadScene(args.targetSceneId);
            setTimeout(() => {{ viewer.setHfov(currentHfov, 1200); }}, 500);
        }});
    }}
    const tourConfig = {config_json};
    Object.values(tourConfig.scenes).forEach(scene => {{
        if (scene.hotSpots) {{
            scene.hotSpots.forEach(hs => {{
                if (hs.clickHandlerFunc === "smoothSwitch") hs.clickHandlerFunc = smoothSwitch;
            }});
        }}
    }});
    window.viewer = pannellum.viewer('panorama', tourConfig);
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
    return send_from_directory(os.path.join(app.config['PROCESSED_FOLDER'], project_id), filename)

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)
