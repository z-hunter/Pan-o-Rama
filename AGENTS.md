# Repository Guidelines
- Make active use of MCP Memory to remember important information and discussion summaries.
- Use the browser MCP (Playwright) to independently navigate the web and test the application

## Project Structure & Module Organization
- `backend/app.py` contains the Flask API, image processing pipeline (Pillow/OpenCV), and gallery generation.
- `frontend/index.html` and `frontend/projects.html` are the static UI entry points.
- `data/raw_uploads` stores incoming files; `data/processed_galleries` stores generated scenes and tours.
- `img/` holds shared static assets (for example `img/logo.png`).
- `debug/` contains local troubleshooting artifacts and sample outputs.
- The core backend (Flask/Gunicorn) and frontend (static HTML/JS) are deployed on the OVHcloud VPS

## Key Information / Configuration Details
*   **VPS IP Address:** `57.128.228.75`
*   **VPS SSH User:** `debian`
*   **SSH Key:** `C:\Users\Michael\.ssh\gemini_cli_id_rsa`
*   **Project Root on VPS:** `/home/debian/lokalny_obiektyw`
*   **Backend Code Path:** `/home/debian/lokalny_obiektyw/backend/app.py`
*   **Frontend Code Path:** `/home/debian/lokalny_obiektyw/frontend/index.html`
*   **Data Storage Paths:** `/home/debian/lokalny_obiektyw/data/{raw_uploads,processed_galleries}`
*   **Gunicorn Address:** `127.0.0.1:8000` (TCP)
*   **Nginx Configuration File:** `/etc/nginx/sites-available/lokalny_obiektyw`
*   **Flask Route for Uploads:** `/api/upload`


## Build, Test, and Development Commands
- `python -m venv .venv && .\.venv\Scripts\activate` creates and activates a local virtual environment.
- `pip install -r backend/requirements.txt` installs Flask, Pillow, OpenCV, and related dependencies.
- `python backend/app.py` runs local development server on `http://localhost:5000`.
- `python backend_vps.py` runs deployment helper logic used for VPS workflows.
- `codex mcp list` verifies configured MCP servers (used for local agent tooling).

## Coding Style & Naming Conventions
- Python: follow PEP 8, 4-space indentation, `snake_case` for functions/variables, `UPPER_CASE` for constants.
- Keep Flask routes concise; move heavy image logic into helper functions (as in `stitch_panorama_*`).
- Frontend JS: use `camelCase` for variables/functions; keep DOM IDs descriptive (`sceneName`, `linkingStatus`).
- Prefer explicit logging (`app.logger.info/warning/error`) for image pipeline branches.

## Testing Guidelines
- No automated test suite is committed yet; use manual API and UI validation for changes.
- Minimum check before PR:
  - create project (`POST /api/project/create`)
  - add scene (`POST /api/project/<id>/scene/add`)
  - finalize (`POST /api/project/<id>/finalize`)
- Validate panorama output in browser and confirm files appear under `data/processed_galleries/<project_id>/`.

## Commit & Pull Request Guidelines
- Commit style in this repo is imperative and outcome-focused (for example: `Stabilize panorama stitching and remove black gaps`).
- Keep commits scoped: one functional change per commit.
- PRs should include:
  - concise problem/solution summary
  - affected paths (for example `backend/app.py`, `frontend/index.html`)
  - manual verification steps and result
  - screenshots/GIFs for UI or panorama behavior changes.

## Security & Configuration Tips
- Do not commit VPS secrets, SSH keys, or runtime logs with sensitive data.
- Keep large generated files out of Git unless needed for reproducible debugging.
- Verify write permissions for `data/` directories before deployment to avoid upload/runtime failures.
