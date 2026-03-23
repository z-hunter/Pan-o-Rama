# Product Requirements

## 1. Core User Flows (User Stories)

### 1.1 Virtual Tour Creation
- As a user, I want to upload multiple 360° panoramas.
- As a user, I want to link panoramas together by placing directional arrows to enable intuitive "walking" navigation between spaces (similar to Google Maps).
- As a user, I want my generated tour to be playable directly in the web browser with seamlessly animated pseudo-3D transitions between scenes.

### 1.2 Monetization & Export
- As a subscribed user, I want to download my completed tour as a standalone archive (HTML/JS/Images) to self-host it independently.
- As an un-subscribed user or guest, I should not be able to download the standalone archive.

### 1.3 B2B / Admin Workflow
- As a platform admin (agency), I want the system to automatically remove camera tripods from the bottom (nadir) of the uploaded 360° panoramas to save processing time.
- As a platform admin (agency), I want the system to automatically patch a custom logo at the nadir of the panoramas.
- As a platform admin, I want a "God Mode" view where I can browse and edit all projects across the platform, regardless of user privacy settings.

## 2. Functional Requirements

### 2.1 Backend / API
- **Authentication**: JWT-based session/auth management for users and admins.
- **Image Processing Queues**: Tripod removal and logo patching MUST run asynchronously via Redis/RQ to prevent blocking Flask web threads.
- **Tour Export Generator**: Endpoint that takes project details, zips static HTML template and associated images, and serves the download.

### 2.2 Frontend / UI
- **Design System**: A modern, highly-polished UI/UX that drops the current "technical prototype" feel in favor of a commercial-grade, engaging, and stylish aesthetic.
- **Editing Tool**: Interactive canvas where users can visually place directional arrows (rather than basic pin markers) to link panoramas.
- **Viewer Player**: WebGL-based robust viewer featuring intuitive navigation, custom shader pseudo-3D transitions, and robust memory leak prevention.
- **Admin Dashboard**: Specialized view showing a tabular list of all global tours with quick-links to the editor for each.

## 3. Non-Functional Requirements
- **Performance**: High-resolution panoramas must load efficiently (consider tiled progressive loading if practical, or at least optimized JPEG compression).
- **Scalability**: Heavy mathematical operations (OpenCV blending) strictly restricted to background tasks.
- **Hosting Constraints**: Standalone exports MUST work out-of-the-box (no CORS errors on static file serving on generic CDNs).

## 4. Constraints & Out of Scope
- No alternative/one-off payment setups; strictly subscription via Stripe.
- No photogrammetry or 3D model generation. Standard point-to-point panorama jumping only.
