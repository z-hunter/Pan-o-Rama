from flask import Flask, request, jsonify, send_from_directory
import os
from werkzeug.utils import secure_filename
from PIL import Image
import uuid
import datetime
import cv2 # Import OpenCV
import logging
from flask_cors import CORS
import json

app = Flask(__name__)
CORS(app)

# Configuration
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, '..', 'data')
UPLOAD_FOLDER = os.path.join(DATA_DIR, 'raw_uploads')
PROCESSED_FOLDER = os.path.join(DATA_DIR, 'processed_galleries')
FRONTEND_FOLDER = os.path.join(BASE_DIR, '..', 'frontend')

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['PROCESSED_FOLDER'] = PROCESSED_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 512 * 1024 * 1024 

# Configure logging
logging.basicConfig(filename=os.path.join(BASE_DIR, 'flask_app.log'), level=logging.DEBUG,
                    format='%(asctime)s %(levelname)s:%(message)s')

ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif'}

os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(PROCESSED_FOLDER, exist_ok=True)

def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def process_image(input_path, output_path, max_size=(1920, 1080), quality=85):
    try:
        with Image.open(input_path) as img:
            if img.mode in ("RGBA", "P"):
                img = img.convert("RGB")
            img.thumbnail(max_size, Image.Resampling.LANCZOS)
            img.save(output_path, "JPEG", quality=quality, optimize=True)
            return True
    except Exception as e:
        app.logger.error(f"Error processing image {input_path}: {e}")
        return False

def stitch_panorama(image_paths, output_path):
    images = []
    for path in image_paths:
        img = cv2.imread(path)
        if img is not None:
            images.append(img)
    if len(images) < 2:
        return False
    stitcher = cv2.Stitcher_create()
    status, stitched = stitcher.stitch(images)
    if status == cv2.Stitcher_OK:
        cv2.imwrite(output_path, stitched)
        return True
    else:
        app.logger.error(f"Panorama stitching failed with status {status}")
        return False

@app.route('/')
def index():
    return send_from_directory(FRONTEND_FOLDER, 'index.html')

@app.route('/projects')
@app.route('/all-projects')
def projects_page():
    return send_from_directory(FRONTEND_FOLDER, 'projects.html')

@app.route('/api/projects', methods=['GET'])
def list_projects():
    projects = []
    try:
        if os.path.exists(app.config['PROCESSED_FOLDER']):
            for project_id in os.listdir(app.config['PROCESSED_FOLDER']):
                project_path = os.path.join(app.config['PROCESSED_FOLDER'], project_id)
                meta_path = os.path.join(project_path, 'metadata.json')
                if os.path.isdir(project_path) and os.path.exists(meta_path):
                    with open(meta_path, 'r') as f:
                        meta = json.load(f)
                    
                    preview_image = None
                    if meta.get('scenes'):
                        s = meta['scenes'][0]
                        preview_image = f"{s['id']}/{s['panorama'] if s['panorama'] else s['images'][0]}"
                    
                    projects.append({
                        'project_id': project_id,
                        'date': meta.get('created_at', 'Unknown'),
                        'preview_url': f'/galleries/{project_id}/{preview_image}' if preview_image else None,
                        'gallery_url': f'/galleries/{project_id}/index.html',
                        'timestamp': os.path.getctime(meta_path)
                    })
        projects.sort(key=lambda x: x['timestamp'], reverse=True)
        return jsonify(projects), 200
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/project/create', methods=['POST'])
def create_project():
    try:
        project_id = str(uuid.uuid4())
        p_dir = os.path.join(app.config['PROCESSED_FOLDER'], project_id)
        os.makedirs(p_dir, exist_ok=True)
        metadata = {'project_id': project_id, 'created_at': datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S'), 'scenes': []}
        with open(os.path.join(p_dir, 'metadata.json'), 'w') as f:
            json.dump(metadata, f)
        return jsonify({'project_id': project_id}), 200
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/project/<project_id>/scene/add', methods=['POST'])
def add_scene(project_id):
    try:
        scene_name = request.form.get('scene_name', 'Unnamed Scene')
        is_panorama = request.form.get('is_panorama') == 'true'
        files = request.files.getlist('files[]')
        
        p_dir = os.path.join(app.config['PROCESSED_FOLDER'], project_id)
        meta_path = os.path.join(p_dir, 'metadata.json')
        with open(meta_path, 'r') as f:
            metadata = json.load(f)

        scene_id = f"scene_{len(metadata['scenes'])}"
        raw_dir = os.path.join(app.config['UPLOAD_FOLDER'], project_id, scene_id)
        proc_dir = os.path.join(p_dir, scene_id)
        os.makedirs(raw_dir, exist_ok=True); os.makedirs(proc_dir, exist_ok=True)

        saved_raw = []
        for file in files:
            if file and allowed_file(file.filename):
                fn = secure_filename(file.filename)
                rp = os.path.join(raw_dir, fn)
                file.save(rp); saved_raw.append(rp)

        processed = []
        for rp in saved_raw:
            fn = f"proc_{os.path.basename(rp)}"
            pp = os.path.join(proc_dir, fn)
            if process_image(rp, pp): processed.append(fn)

        pano = None
        if is_panorama and len(saved_raw) >= 2:
            ppano = os.path.join(proc_dir, "panorama.jpg")
            if stitch_panorama(saved_raw, ppano): pano = "panorama.jpg"

        scene_data = {'id': scene_id, 'name': scene_name, 'panorama': pano, 'images': processed, 'hotspots': []}
        metadata['scenes'].append(scene_data)
        with open(meta_path, 'w') as f: json.dump(metadata, f)
        return jsonify({'scene': scene_data}), 200
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/project/<project_id>/hotspots', methods=['POST'])
def update_hotspots(project_id):
    try:
        data = request.json # Expects { scene_id: [hotspots] }
        p_dir = os.path.join(app.config['PROCESSED_FOLDER'], project_id)
        meta_path = os.path.join(p_dir, 'metadata.json')
        with open(meta_path, 'r') as f:
            metadata = json.load(f)
        
        for scene in metadata['scenes']:
            if scene['id'] in data:
                scene['hotspots'] = data[scene['id']]
        
        with open(meta_path, 'w') as f:
            json.dump(metadata, f)
        
        # Regenerate tour file
        generate_tour(project_id, metadata['scenes'])
        return jsonify({'message': 'Hotspots updated'}), 200
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/project/<project_id>/finalize', methods=['POST'])
def finalize_project(project_id):
    try:
        p_dir = os.path.join(app.config['PROCESSED_FOLDER'], project_id)
        with open(os.path.join(p_dir, 'metadata.json'), 'r') as f:
            metadata = json.load(f)
        url = generate_tour(project_id, metadata['scenes'])
        return jsonify({'gallery_url': url}), 200
    except Exception as e:
        return jsonify({'error': str(e)}), 500

def generate_tour(project_id, scenes):
    tour_config = {
        "default": {"firstScene": scenes[0]['id'], "sceneFadeDuration": 800, "autoLoad": True, "autoRotate": -2},
        "scenes": {}
    }
    for scene in scenes:
        panorama_url = f"{scene['id']}/{scene['panorama'] if scene['panorama'] else scene['images'][0]}"
        hotspots = []
        for hs in scene.get('hotspots', []):
            hotspots.append({
                "pitch": hs['pitch'], "yaw": hs['yaw'], "type": "scene",
                "text": f"Go to {hs['target_name']}", "sceneId": hs['target_id'],
                "cssClass": "custom-hotspot"
            })
        tour_config["scenes"][scene['id']] = {
            "title": scene['name'], "type": "equirectangular",
            "panorama": panorama_url, "hotSpots": hotspots
        }

    config_json = json.dumps(tour_config, indent=4)
    html_content = f"""
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8"><title>Virtual Tour</title>
    <link rel="stylesheet" href="https://cdn.jsdelivr.net/npm/pannellum@2.5.6/build/pannellum.css"/>
    <script src="https://cdn.jsdelivr.net/npm/pannellum@2.5.6/build/pannellum.js"></script>
    <style>
        body {{ margin: 0; padding: 0; background: #000; overflow: hidden; }}
        #panorama {{ width: 100vw; height: 100vh; }}
        .custom-hotspot {{ height: 50px; width: 50px; background: url('/img/logo.png'); background-size: contain; background-repeat: no-repeat; cursor: pointer; filter: drop-shadow(0 0 5px rgba(255,255,255,0.5)); transition: transform 0.2s; }}
        .custom-hotspot:hover {{ transform: scale(1.2); }}
    </style>
</head>
<body>
    <div id="panorama"></div>
    <script>
    const viewer = pannellum.viewer('panorama', {config_json});
    
    // Smooth transition engine
    viewer.on('scenechange', (sceneId) => {{
        console.log('Moved to ' + sceneId);
    }});
    </script>
</body>
</html>
"""
    with open(os.path.join(app.config['PROCESSED_FOLDER'], project_id, 'index.html'), 'w', encoding='utf-8') as f:
        f.write(html_content)
    return f'/galleries/{project_id}/index.html'

@app.route('/galleries/<project_id>/<path:filename>')
def serve_gallery_files(project_id, filename):
    return send_from_directory(os.path.join(app.config['PROCESSED_FOLDER'], project_id), filename)

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)
