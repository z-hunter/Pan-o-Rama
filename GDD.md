# GDD

## Purpose
Pan-o-Rama is a virtual-tour platform for creating, editing, publishing, and sharing panorama tours.

Core user flows:
- create a tour
- upload/process panorama scenes
- link scenes with hotspots
- set start view and incoming-link entry views
- publish a shareable tour player
- browse published tours in the gallery

## Stack
- Backend: Flask
- Frontend: static HTML/CSS/JS
- Database: SQLite at `data/app.db`
- Image pipeline: Pillow + OpenCV
- Storage: local filesystem under `data/processed_galleries/`
- Deployment: OVH VPS, Gunicorn behind Nginx

## Main Components

### Backend
- Entry point: [backend/app.py](C:\Users\Michael\lokalny_obiektyw\backend\app.py)
- Blueprints:
  - [backend/blueprints/auth.py](C:\Users\Michael\lokalny_obiektyw\backend\blueprints\auth.py)
  - [backend/blueprints/users.py](C:\Users\Michael\lokalny_obiektyw\backend\blueprints\users.py)
  - [backend/blueprints/tours.py](C:\Users\Michael\lokalny_obiektyw\backend\blueprints\tours.py)
  - [backend/blueprints/scenes.py](C:\Users\Michael\lokalny_obiektyw\backend\blueprints\scenes.py)
  - [backend/blueprints/analytics.py](C:\Users\Michael\lokalny_obiektyw\backend\blueprints\analytics.py)
- Shared services:
  - [backend/services/tour_service.py](C:\Users\Michael\lokalny_obiektyw\backend\services\tour_service.py)
  - [backend/services/scene_service.py](C:\Users\Michael\lokalny_obiektyw\backend\services\scene_service.py)
  - [backend/services/billing_service.py](C:\Users\Michael\lokalny_obiektyw\backend\services\billing_service.py)
- Core helpers:
  - [backend/core/config.py](C:\Users\Michael\lokalny_obiektyw\backend\core\config.py)
  - [backend/core/database.py](C:\Users\Michael\lokalny_obiektyw\backend\core\database.py)
  - [backend/core/models.py](C:\Users\Michael\lokalny_obiektyw\backend\core\models.py)

### Frontend
- Studio editor: [frontend/index.html](C:\Users\Michael\lokalny_obiektyw\frontend\index.html)
- Public gallery: [frontend/projects.html](C:\Users\Michael\lokalny_obiektyw\frontend\projects.html)
- Auth/account/dashboard pages live in `frontend/`
- Shared API wrapper: [frontend/js/api.js](C:\Users\Michael\lokalny_obiektyw\frontend\js\api.js)

### Tour Players
- Production player template: [frontend/player_template_legacy.html](C:\Users\Michael\lokalny_obiektyw\frontend\player_template_legacy.html)
- Experimental viewer kept in repo: [frontend/player_template.html](C:\Users\Michael\lokalny_obiektyw\frontend\player_template.html)
- Published tours are generated as static `index.html` files per tour in `data/processed_galleries/<tour_id>/`

## Data Model

### Tour
Represents one virtual tour.

Important fields:
- `id`
- `owner_id`
- `title`
- `slug`
- `visibility`
- `status`
- `start_scene_id`
- `start_pitch`
- `start_yaw`
- `default_hfov`

### Scene
Represents one panorama in a tour.

Important fields:
- `id`
- `tour_id`
- `title`
- `panorama_path`
- `preview_path`
- `images_json`
- `order_index`
- `haov`
- `vaov`
- `scene_type`
- `processing_status`

Derived assets per scene folder:
- full panorama, usually original `panorama_path`
- `web.jpg`
- `studio_web.jpg`
- `preview.jpg`

### Hotspot
Directional link from one scene to another.

Important fields:
- `from_scene_id`
- `to_scene_id`
- `pitch`
- `yaw`
- `entry_pitch`
- `entry_yaw`
- `label`

### User / Plan / Access
- users own tours
- plans control limits and watermarking
- private tours can be shared to specific users through access grants

### Jobs
Async processing jobs track stitching and scene processing state.

## File Layout

### Runtime Data
- Raw uploads: `data/raw_uploads/`
- Processed tours: `data/processed_galleries/<tour_id>/`
- Database: `data/app.db`

### Published Tour Folder
Typical contents:
- `index.html`
- `cover.jpg`
- one folder per scene
- each scene folder contains panorama assets and previews

## Important Flows

### 1. Tour Editing
- Studio loads a tour via `GET /tours/<id>`
- scenes and hotspots come from DB
- Pannellum is used inside Studio for editing
- user can set:
  - start scene and start camera view
  - incoming-link entry view
  - hotspot links

### 2. Scene Processing
- scene upload goes through `POST /tours/<id>/scenes`
- backend stitches or stores panorama
- helper functions generate `web.jpg`, `studio_web.jpg`, and previews
- processing state is stored on the scene row

### 3. Publishing
- `POST /tours/<id>/finalize` calls `generate_tour(...)`
- backend injects `window.TOUR_DATA` into the player template
- output is written to `data/processed_galleries/<tour_id>/index.html`
- public share URL is `/t/<slug>`

### 4. Start View Sync
- Studio button `Set Current View as Start` sends `PATCH /tours/<id>`
- backend updates `start_scene_id`, `start_pitch`, `start_yaw`, `default_hfov`
- if the tour is already published, backend now also regenerates the published `index.html`
- result: gallery player reflects Studio start-view changes without manual re-finalize

### 5. Public Player
- share route `/t/<slug>` serves the generated static `index.html`
- scene assets are loaded relative to that folder
- legacy player supports:
  - Web / HD quality toggle
  - preview-to-full transitions
  - hotspot navigation
  - start scene camera settings
  - incoming-link entry camera settings

## Routing Summary
- `/` -> Studio
- `/dashboard.html` -> dashboard
- `/projects.html` -> public gallery
- `/t/<slug>` -> published player
- `/tours/...` -> authenticated tour API
- `/scenes/...` -> scene and hotspot API
- `/auth/...` -> auth API

## Current Architecture Decisions
- Published tours are static HTML outputs, not server-rendered pages.
- Production uses the legacy Pannellum-based player.
- The experimental Three.js player stays in the repo but is not used for published tours.
- Tour player behavior is controlled by data injected into `window.TOUR_DATA`.

## Developer Notes
- If a Studio change should affect the public player, verify both DB state and generated `data/processed_galleries/<tour_id>/index.html`.
- If a share page looks stale after code deploy, existing published tours may need regeneration.
- When debugging private/public access, check both DB visibility and static-file serving route behavior.
- For real-world viewer verification, test against `https://pan-o-rama.online` or server IP `57.128.228.75`.

## Production
- VPS IP: `57.128.228.75`
- Web domain: `pan-o-rama.online`
- Project root on server: `/home/debian/lokalny_obiektyw`
- Nginx fronts Gunicorn
- Main services:
  - `lokalny_obiektyw.service`
  - `lokalny_obiektyw_worker.service`
