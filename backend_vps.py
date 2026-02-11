from flask import Flask, request, jsonify, send_from_directory
import os
from werkzeug.utils import secure_filename
from PIL import Image
import uuid
import datetime
import cv2 # Import OpenCV
import logging
from flask_cors import CORS

app = Flask(__name__)
CORS(app)

# Configuration
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, '..', 'data')
UPLOAD_FOLDER = os.path.join(DATA_DIR, 'raw_uploads')
PROCESSED_FOLDER = os.path.join(DATA_DIR, 'processed_galleries')

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['PROCESSED_FOLDER'] = PROCESSED_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 512 * 1024 * 1024

# Configure logging
logging.basicConfig(filename='/tmp/flask_app.log', level=logging.DEBUG,
                    format='%(asctime)s %(levelname)s:%(message)s')

ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif'}

os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(PROCESSED_FOLDER, exist_ok=True)

def allowed_file(filename):
    return '.' in filename and 
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def process_image(input_path, output_path, max_size=(1920, 1080), quality=85):
    try:
        with Image.open(input_path) as img:
            if img.mode in ('RGBA', 'P'):
                img = img.convert('RGB')
            img.thumbnail(max_size, Image.Resampling.LANCZOS)
            img.save(output_path, 'JPEG', quality=quality, optimize=True)
            return True
    except Exception as e:
        app.logger.error(f'Error processing image {input_path}: {e}')
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
        app.logger.error(f'Panorama stitching failed with status {status}')
        return False

@app.errorhandler(413)
def request_entity_too_large(error):
    return jsonify({'error': 'File too large', 'details': 'The total size of uploaded files exceeds the limit (512MB).'}), 413

@app.route('/api/status')
def hello_world():
    return 'Hello, this is the Lokalny Obiektyw Backend!'

@app.route('/api/upload', methods=['POST'])
def upload_files():
    try:
        if 'files[]' not in request.files:
            return jsonify({'error': 'No file part in the request'}), 400
        
        uploaded_files = request.files.getlist('files[]')
        is_panorama = request.form.get('is_panorama') == 'true'
        
        if not uploaded_files or uploaded_files[0].filename == '':
            return jsonify({'error': 'No selected file(s)'}), 400

        project_id = str(uuid.uuid4())
        project_raw_dir = os.path.join(app.config['UPLOAD_FOLDER'], project_id)
        project_processed_dir = os.path.join(app.config['PROCESSED_FOLDER'], project_id)

        os.makedirs(project_raw_dir, exist_ok=True)
        os.makedirs(project_processed_dir, exist_ok=True)

        saved_raw_paths = []
        for file in uploaded_files:
            if file and allowed_file(file.filename):
                filename = secure_filename(file.filename)
                raw_path = os.path.join(project_raw_dir, filename)
                file.save(raw_path)
                saved_raw_paths.append(raw_path)

        processed_files = []
        processed_individual_files = []
        for raw_path in saved_raw_paths:
            filename = os.path.basename(raw_path)
            processed_filename = f'proc_{filename}'
            processed_path = os.path.join(project_processed_dir, processed_filename)
            if process_image(raw_path, processed_path):
                processed_individual_files.append(processed_filename)

        if is_panorama and len(saved_raw_paths) >= 2:
            panorama_filename = 'panorama.jpg'
            panorama_path = os.path.join(project_processed_dir, panorama_filename)
            if stitch_panorama(saved_raw_paths, panorama_path):
                processed_files.append(panorama_filename)
                processed_files.extend(processed_individual_files)
            else:
                app.logger.warning('Stitching failed, falling back to individual images.')
                processed_files = processed_individual_files
        else:
            processed_files = processed_individual_files

        gallery_url = generate_gallery(project_id, processed_files)

        return jsonify({
            'message': 'Success',
            'project_id': project_id,
            'processed_files': processed_files,
            'gallery_url': gallery_url
        }), 200

    except Exception as e:
        app.logger.error(f'Unhandled exception: {e}', exc_info=True)
        return jsonify({'error': 'Internal server error', 'details': str(e)}), 500

def generate_gallery(project_id, image_filenames):
    panorama_file = next((f for f in image_filenames if f == 'panorama.jpg'), None)
    pannellum_css = '<link rel="stylesheet" href="https://cdn.jsdelivr.net/npm/pannellum@2.5.6/build/pannellum.css"/>'
    pannellum_js = '<script type="text/javascript" src="https://cdn.jsdelivr.net/npm/pannellum@2.5.6/build/pannellum.js"></script>'
    
    gallery_html = ""
    if panorama_file:
        gallery_html += f"""
        <div id="panorama" style="width: 100%; height: 600px; margin-bottom: 20px;"></div>
        <script>
        pannellum.viewer('panorama', {{
            "type": "equirectangular",
            "panorama": "{panorama_file}",
            "autoLoad": true
        }});
        </script>
        <h2>Source Images</h2>
        """
    
    other_images = [img for img in image_filenames if img != 'panorama.jpg']
    gallery_html += '<div class="gallery-grid">' + " ".join([f'<div class="item"><img src="{name}" loading="lazy" /></div>' for name in other_images]) + '</div>'

    html_content = f"""
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Gallery: {project_id}</title>
    {pannellum_css if panorama_file else ''}
    {pannellum_js if panorama_file else ''}
    <style>
        body {{ font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif; padding: 20px; background: #1a1a1a; color: #eee; }}
        h1 {{ color: #fff; border-bottom: 1px solid #444; padding-bottom: 10px; }}
        h2 {{ color: #ccc; margin-top: 30px; }}
        .gallery-grid {{ display: flex; flex-wrap: wrap; gap: 15px; }}
        img {{ max-width: 100%; height: auto; border-radius: 4px; box-shadow: 0 4px 6px rgba(0,0,0,0.3); transition: transform 0.2s; }}
        img:hover {{ transform: scale(1.02); }}
        .item {{ flex: 1 1 300px; max-width: 400px; }}
    </style>
</head>
<body>
    <div style="width: 100%;"><h1>Lokalny Obiektyw Project: {project_id}</h1></div>
    {gallery_html}
</body>
</html>
"""
    gallery_path = os.path.join(app.config['PROCESSED_FOLDER'], project_id, 'index.html')
    with open(gallery_path, 'w', encoding='utf-8') as f:
        f.write(html_content)
    return f'/galleries/{project_id}/index.html'

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)
