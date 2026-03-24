# Project Stack Research

## Current & Recommended Stack
Since this is a brownfield project, we are evaluating the existing stack against domain standards for 360° virtual tour platforms.

### 1. Viewers (Frontend)
- **Standard**: Most custom 360° applications use a dedicated library like **Marzipano** (by Google), **Pannellum**, or a custom **Three.js / WebGL** wrapper. 
- **Recommendation**: If not already using a standard robust viewer, integrating Marzipano or Pannellum on the frontend will save immense development time.

### 2. Image Processing Pipeline (Backend)
- **Standard**: Operations like nadir patching and tripod removal require heavy matrix manipulations. Typically handled via **OpenCV**, **Pillow**, or specialized CV packages.
- **Recommendation**: Keep leveraging Python with OpenCV and Pillow via the existing Redis/RQ workers to handle these tasks asynchronously without blocking web threads.

### 3. Monetization & Access
- **Standard**: **Stripe Billing** for subscriptions.
- **Recommendation**: Utilize Stripe's Webhooks via Flask to sync user state (e.g. `subscription_active=True`), which gates the "export" and "white-label" features.

### 4. Storage & Delivery
- **Standard**: **AWS S3 or Cloudflare R2** combined with a CDN is mandatory for serving heavily tiled panoramic data or large 8K images quickly globally.
- **Recommendation**: The current local `data/` setup is adequate for MVP on OVHcloud, but a transition to S3/CDN should be planned for scale to avoid VPS bandwidth saturation.
