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
app.config['MAX_CONTENT_LENGTH'] = 512 * 1024 * 1024 # Increased to 512 MB

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

@app.errorhandler(413)
def request_entity_too_large(error):
    return jsonify({'error': 'File too large', 'details': 'The total size of uploaded files exceeds the limit (512MB).'}), 413

@app.route('/')
def index():
    return send_from_directory(FRONTEND_FOLDER, 'index.html')

@app.route('/projects')
def projects_page():
    return send_from_directory(FRONTEND_FOLDER, 'projects.html')

@app.route('/api/projects', methods=['GET'])
def list_projects():
    projects = []
    try:
        if os.path.exists(app.config['PROCESSED_FOLDER']):
            for project_id in os.listdir(app.config['PROCESSED_FOLDER']):
                project_path = os.path.join(app.config['PROCESSED_FOLDER'], project_id)
                if os.path.isdir(project_path):
                    ctime = os.path.getctime(project_path)
                    date_str = datetime.datetime.fromtimestamp(ctime).strftime('%Y-%m-%d %H:%M:%S')
                    
                    preview_image = None
                    # Try to find a preview in any scene
                    for scene_dir in os.listdir(project_path):
                        scene_path = os.path.join(project_path, scene_dir)
                        if os.path.isdir(scene_path):
                            files = os.listdir(scene_path)
                            if 'panorama.jpg' in files:
                                preview_image = f"{scene_dir}/panorama.jpg"
                                break
                            else:
                                proc_files = [f for f in files if f.startswith('proc_')]
                                if proc_files:
                                    preview_image = f"{scene_dir}/{proc_files[0]}"
                                    break
                    
                    projects.append({
                        'project_id': project_id,
                        'date': date_str,
                        'preview_url': f'/galleries/{project_id}/{preview_image}' if preview_image else None,
                        'gallery_url': f'/galleries/{project_id}/index.html',
                        'timestamp': ctime
                    })
        
        projects.sort(key=lambda x: x['timestamp'], reverse=True)
        return jsonify(projects), 200
    except Exception as e:
        app.logger.error(f"Error listing projects: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/project/create', methods=['POST'])
def create_project():
    try:
        project_id = str(uuid.uuid4())
        project_processed_dir = os.path.join(app.config['PROCESSED_FOLDER'], project_id)
        os.makedirs(project_processed_dir, exist_ok=True)
        
        metadata = {
            'project_id': project_id,
            'created_at': datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'scenes': []
        }
        with open(os.path.join(project_processed_dir, 'metadata.json'), 'w') as f:
            json.dump(metadata, f)
            
        return jsonify({'project_id': project_id}), 200
    except Exception as e:
        app.logger.error(f"Error creating project: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/project/<project_id>/scene/add', methods=['POST'])
def add_scene(project_id):
    try:
        scene_name = request.form.get('scene_name', 'Unnamed Scene')
        is_panorama = request.form.get('is_panorama') == 'true'
        files = request.files.getlist('files[]')
        
        project_processed_dir = os.path.join(app.config['PROCESSED_FOLDER'], project_id)
        metadata_path = os.path.join(project_processed_dir, 'metadata.json')
        
        with open(metadata_path, 'r') as f:
            metadata = json.load(f)

        scene_id = f"scene_{len(metadata['scenes'])}"
        scene_raw_dir = os.path.join(app.config['UPLOAD_FOLDER'], project_id, scene_id)
        scene_processed_dir = os.path.join(project_processed_dir, scene_id)
        os.makedirs(scene_raw_dir, exist_ok=True)
        os.makedirs(scene_processed_dir, exist_ok=True)

        saved_raw_paths = []
        for file in files:
            if file and allowed_file(file.filename):
                filename = secure_filename(file.filename)
                raw_path = os.path.join(scene_raw_dir, filename)
                file.save(raw_path)
                saved_raw_paths.append(raw_path)

        processed_individual_files = []
        for raw_path in saved_raw_paths:
            filename = os.path.basename(raw_path)
            processed_filename = f"proc_{filename}"
            processed_path = os.path.join(scene_processed_dir, processed_filename)
            process_image(raw_path, processed_path)
            processed_individual_files.append(processed_filename)

        panorama_filename = None
        if is_panorama and len(saved_raw_paths) >= 2:
            panorama_path = os.path.join(scene_processed_dir, "panorama.jpg")
            if stitch_panorama(saved_raw_paths, panorama_path):
                panorama_filename = "panorama.jpg"

        scene_data = {
            'id': scene_id,
            'name': scene_name,
            'panorama': panorama_filename,
            'images': processed_individual_files
        }
        metadata['scenes'].append(scene_data)
        
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f)

        return jsonify({'scene': scene_data}), 200
    except Exception as e:
        app.logger.error(f"Error adding scene: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/project/<project_id>/finalize', methods=['POST'])
def finalize_project(project_id):
    try:
        project_processed_dir = os.path.join(app.config['PROCESSED_FOLDER'], project_id)
        metadata_path = os.path.join(project_processed_dir, 'metadata.json')
        with open(metadata_path, 'r') as f:
            metadata = json.load(f)
        
        gallery_url = generate_tour(project_id, metadata['scenes'])
        return jsonify({'gallery_url': gallery_url}), 200
    except Exception as e:
        app.logger.error(f"Error finalizing: {e}")
        return jsonify({'error': str(e)}), 500

def generate_tour(project_id, scenes):
    tour_config = {
        "default": {"firstScene": scenes[0]['id'], "sceneFadeDuration": 1000, "autoLoad": True},
        "scenes": {}
    }

    for i, scene in enumerate(scenes):
        scene_id = scene['id']
        panorama_url = f"{scene_id}/{scene['panorama'] if scene['panorama'] else scene['images'][0]}"
        hotspots = []
        if i < len(scenes) - 1:
            hotspots.append({"pitch": 0, "yaw": 0, "type": "scene", "text": f"Next: {scenes[i+1]['name']}", "sceneId": scenes[i+1]['id']})
        if i > 0:
            hotspots.append({"pitch": 0, "yaw": 180, "type": "scene", "text": f"Back: {scenes[i-1]['name']}", "sceneId": scenes[i-1]['id']})

        tour_config["scenes"][scene_id] = {
            "title": scene['name'],
            "type": "equirectangular",
            "panorama": panorama_url,
            "hotSpots": hotspots
        }

    config_json = json.dumps(tour_config, indent=4)
    html_content = f"""
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Pan-o-Rama Tour</title>
    <link rel="stylesheet" href="https://cdn.jsdelivr.net/npm/pannellum@2.5.6/build/pannellum.css"/>
    <script type="text/javascript" src="https://cdn.jsdelivr.net/npm/pannellum@2.5.6/build/pannellum.js"></script>
    <style>body {{ margin: 0; padding: 0; background: #000; }} #panorama {{ width: 100vw; height: 100vh; }}</style>
</head>
<body>
    <div id="panorama"></div>
    <script>pannellum.viewer('panorama', {config_json});</script>
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
