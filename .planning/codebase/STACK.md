# Tech Stack

## General
- Main runtime: Python 3
- Deployment: OVHcloud VPS (Debian, Nginx, Gunicorn)

## Backend
- Framework: Flask
- Data manipulation: Pillow, OpenCV (`opencv-python`) for image processing and stitching
- Message Queue / Async tasks: RQ (Redis Queue)
- Worker Cache & Queue Backend: Redis
- Dependency Management: `requirements.txt` via `pip`

## Frontend
- Vanilla HTML/CSS/JS structure
- No complex bundler or JS framework is used for the main views
- Static assets and views served directly or via Nginx
