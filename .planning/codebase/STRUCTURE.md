# Directory Structure

## Key Locations
- `backend/` - Python API and core logic
  - `app.py` - Core Monolithic Flask web server (routing and application logic)
  - `worker.py` - Worker initialization script for RQ tasks
  - `requirements.txt` - Python project dependencies
  - `test_mvp_api.py` - Basic manual test script
- `frontend/` - Static HTML/JS frontend Views
  - `index.html`, `dashboard.html`, `projects.html`, `account.html`, `login.html`
- `data/` - User content storage paths (Stateful)
  - `raw_uploads/` - Incoming images
  - `processed_galleries/` - Processed generation configurations and imagery
- `img/` - Global static visual assets
- `debug/` - Artifacts and logs useful for troubleshooting locally
- `scripts/` - Standalone utility files (e.g., `cubemap_to_equirect.py`)
- `AGENTS.md` - Technical specification and AI agent operational guidelines
- `ProductDesign.md` - MVP requirements and specifications
