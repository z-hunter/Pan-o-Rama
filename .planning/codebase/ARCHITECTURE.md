# System Architecture

## Overview
A two-tier modular application composed of a static HTML/JS frontend and a Python Flask backend. The heavy computational load for panoramic stitching is offloaded to a background worker process.

## Key Architectures & Flow
1. **Frontend Layer (`frontend/`)**: Static views such as `index.html` and `dashboard.html` that communicate with the backend via REST APIs.
2. **API Backend (`backend/app.py`)**: The primary access point and web server. It handles route management, authentication, Stripe calls, and kicks off asynchronous background jobs.
3. **Async Workers (`backend/worker.py`)**: Powered by RQ (Redis Queue), this parses job tasks to perform intensive image generation/stitching (Pillow/OpenCV).
4. **Data Layer (`data/`)**: Acts as a stateful blob and user data store, managing incoming files in `raw_uploads` and outputting verified scenes into `processed_galleries`.
