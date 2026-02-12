from flask import Flask, request, jsonify, send_from_directory, g, redirect
import os
from werkzeug.utils import secure_filename
from werkzeug.security import generate_password_hash, check_password_hash
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
import sqlite3
import hashlib
import secrets
import functools

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
            CREATE INDEX IF NOT EXISTS idx_tours_owner ON tours(owner_id);
            CREATE INDEX IF NOT EXISTS idx_tours_slug ON tours(slug);
            CREATE INDEX IF NOT EXISTS idx_scenes_tour ON scenes(tour_id);
            CREATE INDEX IF NOT EXISTS idx_hotspots_scene ON hotspots(from_scene_id);
            """
        )
        cols = [r[1] for r in db.execute("PRAGMA table_info(hotspots)").fetchall()]
        if "entry_yaw" not in cols:
            db.execute("ALTER TABLE hotspots ADD COLUMN entry_yaw REAL")
        if "entry_pitch" not in cols:
            db.execute("ALTER TABLE hotspots ADD COLUMN entry_pitch REAL")
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
        "images": json.loads(row["images_json"] or "[]"),
        "haov": row["haov"],
        "vaov": row["vaov"],
        "type": row["scene_type"],
        "order_index": row["order_index"],
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
    return jsonify({"id": u["id"], "email": u["email"], "display_name": u["display_name"]}), 200

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
    sid = f"scene_{order_index}"
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
            if is_pano and aspect >= 2.0:
                detect_and_crop_overlap_wide(img_p)
                with Image.open(img_p) as img_c:
                    aspect = img_c.width / img_c.height
                    haov = 360
                    vaov = 360 / aspect
            else:
                haov = vaov_c * aspect
                vaov = vaov_c

    ts = now_iso()
    db.execute(
        """
        INSERT INTO scenes (id, tour_id, title, panorama_path, images_json, order_index, haov, vaov, scene_type, created_at, updated_at)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, 'equirectangular', ?, ?)
        """,
        (sid, tour["id"], name, pano_file, json.dumps(processed), order_index, round(haov, 2), round(vaov, 2), ts, ts),
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
    gallery_url = generate_tour(tour["id"], scenes)
    db = get_db()
    db.execute("UPDATE tours SET status = 'published', updated_at = ? WHERE id = ?", (now_iso(), tour["id"]))
    db.commit()
    share_url = f"/t/{tour['slug']}"
    return jsonify({"gallery_url": gallery_url, "share_url": share_url, "visibility": tour["visibility"]}), 200

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

def generate_tour(project_id, scenes):
    tour_config = {"default": {"firstScene": scenes[0]['id'], "sceneFadeDuration": 1000, "autoLoad": True, "autoRotate": -2, "hfov": 70}, "scenes": {}}
    for scene in scenes:
        panorama_url = f"{scene['id']}/{scene['panorama'] if scene['panorama'] else scene['images'][0]}"
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
        .pnlm-load-box, .pnlm-loading, .pnlm-about-msg {{ display: none !important; }}
    </style>
</head>
<body>
    <div id="panorama"></div>
    <script>
    const prefetchCache = new Set();
    function prefetchScene(sceneId) {{
        if (!sceneId || prefetchCache.has(sceneId)) return;
        const scene = tourConfig.scenes[sceneId];
        if (!scene || !scene.panorama) return;
        const img = new Image();
        img.src = scene.panorama;
        prefetchCache.add(sceneId);
    }}
    function prefetchLinkedScenes(sceneId) {{
        const scene = tourConfig.scenes[sceneId];
        if (!scene || !scene.hotSpots) return;
        scene.hotSpots.forEach(hs => prefetchScene(hs.clickHandlerArgs.targetSceneId));
    }}

    function smoothSwitch(e, args) {{
        const viewer = window.viewer;
        const currentHfov = viewer.getHfov();
        const zoomTarget = currentHfov - 40;
        viewer.setHfov(zoomTarget, 800);
        viewer.lookAt(args.viaPitch, args.viaYaw, zoomTarget, 800);
        // Start scene switch while zoom animation is still running for a smoother transition.
        setTimeout(() => {{
            viewer.loadScene(args.targetSceneId, args.entryPitch, args.entryYaw, zoomTarget);
            setTimeout(() => {{ viewer.setHfov(currentHfov, 1000); }}, 120);
        }}, 260);
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
    prefetchLinkedScenes(tourConfig.default.firstScene);
    window.viewer.on('scenechange', (sceneId) => prefetchLinkedScenes(sceneId));
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
    return send_from_directory(os.path.join(app.config['PROCESSED_FOLDER'], project_id), filename)

init_db()

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)
