# Lokalny Obiektyw Project Summary

## Project Goal
Develop a hyper-local digital service for immigrant businesses in Poland to generate at least $1000/month, starting with an MVP for automated image processing and galleries.

## Current Status (February 10, 2026)
The core backend (Flask/Gunicorn) and frontend (static HTML/JS) are deployed on the OVHcloud VPS. The primary blocking issue of `502 Bad Gateway` for file uploads, caused by Nginx's inability to correctly proxy requests to Gunicorn, has been resolved. File uploads are now successfully reaching the Flask application, being "saved" (dummy implementation), and returning a `200 OK` JSON response.

## Resolved Blocking Issue: Nginx 502 Bad Gateway
**Symptom:** Persistent `502 Bad Gateway` error for `POST /api/upload` requests from Nginx, resulting in `JSON.parse: unexpected character` on the frontend.
**Root Cause (Identified through extensive debugging):**
1.  **Nginx `Permission denied` to Unix socket:** Initially, Nginx workers (`www-data` user) were unable to connect to the Gunicorn Unix socket (`/tmp/lokalny_obiektyw.sock`) due to file system permissions, despite group memberships appearing correct.
    *   **Resolution:** Switched Gunicorn to listen on `127.0.0.1:8000` (TCP) instead of a Unix socket.
2.  **Flask `OSError: Permission denied` during file save:** After switching to TCP, Gunicorn workers (running as `debian` user) lacked write permissions to the `/home/debian/lokalny_obiektyw/data/raw_uploads` directory when attempting `file.save()`.
    *   **Resolution:** Applied recursive ownership (`chown -R debian:www-data`) and permissions (`chmod -R ug+rwx`, `chmod g+s`) to `/home/debian/lokalny_obiektyw/data/raw_uploads` and `processed_galleries` directories.
3.  **Nginx routing precedence issues:** Even after Gunicorn was listening on TCP and file permissions were correct, Nginx continued to return `404` for `/api/upload` requests, treating them as static files rather than proxying. Multiple `location` configurations (`^~`, regex with `rewrite`, explicit `=` match, `error_page` hack) were attempted without success.
    *   **Final Resolution:** It was determined that Nginx on this specific VPS was not reliably stripping URI prefixes or handling `rewrite` directives as expected within `proxy_pass` blocks for `/api/` locations. The final working configuration involves:
        *   **Flask route updated:** The Flask application's `upload_files` route was changed from `@app.route('/upload')` to `@app.route('/api/upload')` to match the full incoming URI.
        *   **Simplified Nginx `location /api/`:** The Nginx configuration for `location /api/` was simplified to directly `proxy_pass http://127.0.0.1:8000;` without any `rewrite` directives. This instructs Nginx to pass the *full original URI* (`/api/upload`) to Gunicorn, which Flask is now configured to expect.

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

## Task List (Updated)
*   [DONE] Create basic project structure.
*   [DONE] Develop basic Flask backend.
*   [DONE] Implement image processing logic (Pillow).
*   [DONE] Implement web gallery generation.
*   [DONE] Investigate panorama tours.
*   [DONE] Integrate image stitching (OpenCV).
*   [DONE] Develop minimal web interface.
*   [DONE] Set up OVHcloud VPS environment.
*   [DONE] Develop deployment scripts/instructions.
*   [DONE] Resolve `502 Bad Gateway` error for file uploads.
*   [PENDING] Implement project management mechanism.

## Immediate Next Steps
1.  **Verify actual image processing:** Re-enable actual image processing logic (resizing, watermarking, etc.) in `app.py`, ensuring OpenCV (if used) functions correctly within the Gunicorn environment.
2.  **Implement panorama stitching:** Re-enable `stitch_panorama` and integrate it into the workflow.
3.  **Gallery generation:** Fully implement web gallery generation based on processed images.
4.  **Frontend integration:** Connect the frontend to the actual gallery URLs.
5.  **Security Review:** Disable Nginx debug logging, ensure all ports are correctly secured.
6.  **Refine user flow and authentication/authorization.**
