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

def stitch_panorama(image_paths, output_path, is_360=True):
    """Robust stitching with verification."""
    app.logger.info(f"Stitching {len(image_paths)} images...")
    images = []
    total_in_w = 0
    # Lower resolution for stitching to improve match stability with handheld captures
    target_h = 1500
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
                with Image.open(ppano) as img:
                    aspect = img.width / img.height
                    vaov_c = 2 * math.degrees(math.atan(18.0 / focal_35))
                    if is_pano:
                        crop_offset = detect_and_crop_overlap(ppano)
                        if crop_offset is not None:
                            # Refresh after crop and only then force full 360
                            with Image.open(ppano) as img_c:
                                haov = 360
                                vaov = 360 / (img_c.width / img_c.height)
                        else:
                            # Not a full loop; avoid forcing 360 which causes wrap seams/black gaps
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
                crop_offset = detect_and_crop_overlap(img_p)
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
