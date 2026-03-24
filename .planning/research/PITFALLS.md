# Project Pitfalls Research

## Common Mistakes in Virtual Tour Development

### 1. Memory Leaks in Browser (WebGL)
- **Warning Sign:** Browsers crashing or refreshing when jumping between 20+ scenes on mobile devices.
- **Prevention Strategy:** Explicitly dispose of WebGL textures from memory when a scene is exited. Do not load all 20 panoramas simultaneously; use lazy-loading or tiled progressive loading.
- **Phase Targeting:** Addressed in the core frontend viewer implementation.

### 2. Main Thread Blocking on Server
- **Warning Sign:** Requests timing out while someone uploads/processes a heavy 50MB 8K panorama.
- **Prevention Strategy:** Keep all image processing 100% inside RQ workers. Ensure Flask never runs `cv2` inline in a request block.
- **Phase Targeting:** Architecture review / worker pipeline implementation.

### 3. Cross-Origin Resource Sharing (CORS) Issues for Exports
- **Warning Sign:** Users embedding exported tours facing `Origin blocked by CORS policy` errors because canvases try to read cross-domain texture data.
- **Prevention Strategy:** If serving statically from Nginx, enforce explicit CORS wildcard headers (`Access-Control-Allow-Origin: *`) specifically on the `processed_galleries` route for image files.
- **Phase Targeting:** Server configuration phase.
