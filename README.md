# Pan-o-Rama (MVP)

Pan-o-Rama is a local-first virtual tour service built with Flask + static frontend.
Users can register, log in, create tours from panoramas, add hotspots, publish share links, and keep tours private (owner-only).

## Stack
- Backend: Flask, Pillow, OpenCV, SQLite
- Frontend: static HTML/CSS/JS
- Storage: local filesystem (`data/`)

## Local Setup
1. Create and activate venv:
   - Windows PowerShell:
     - `python -m venv .venv`
     - `.\.venv\Scripts\Activate.ps1`
2. Install dependencies:
   - `pip install -r backend/requirements.txt`
3. Run app:
   - `python backend/app.py`
4. Open:
   - Editor: `http://localhost:5000/`
   - Login: `http://localhost:5000/login`
   - Dashboard: `http://localhost:5000/dashboard`

## Main API (MVP)
- Auth: `POST /auth/register`, `POST /auth/login`, `POST /auth/logout`
- Profile: `GET /me`, `PATCH /me`, `PATCH /me/password`
- Tours: `POST /tours`, `GET /tours/my`, `GET/PATCH/DELETE /tours/<id>`
- Scenes: `POST /tours/<id>/scenes`, `PATCH/DELETE /scenes/<id>`
- Hotspots: `POST /scenes/<id>/hotspots`, `PATCH/DELETE /hotspots/<id>`
- Publish: `POST /tours/<id>/finalize`
- Public gallery: `GET /gallery`
- Share link: `GET /t/<slug>`

## Privacy Rules
- `public` tours are listed in `/gallery` and openable via `/t/<slug>`.
- `private` tours are owner-only.
- Direct access to `/galleries/<tour_id>/...` is blocked for private tours unless owner is authenticated.

## Quick Manual Test
1. Register on `/login`.
2. Create tour in `/dashboard` (`public` and `private`).
3. Add scenes via API or editor flow.
4. Finalize tour and open `share_url`.
5. Verify private share URL returns `403` in logged-out browser.
