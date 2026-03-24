# Project Architecture Research

## System Structure
A SaaS virtual tour platform with advanced CV (Computer Vision) features typically employs an asynchronous offloading structure.

### Core Components
1. **API Web Gateway (Flask)**: Handles light web events, routing, Stripe API webhooks, and JWT/Session validation.
2. **Process Queue (RQ/Redis)**: Acts as the shock-absorber for intensive tasks.
3. **Computer Vision Worker (OpenCV/Python)**: Dedicated background processes executing heavy math operations (tripod removal via inpainting, blending nadir logos).
4. **Blob Store (`data/`)**: Stores source images (`raw_uploads`) and static output assets (`processed_galleries`).

### Data Flow for 360 Processing
1. User requests "Publish/Process" on a project.
2. Flask places an `ImageProcessJob` onto the Redis queue and returns `202 Accepted`.
3. Worker retrieves job, grabs raw images from Blob Store.
4. Worker applies nadir patch and scales the image.
5. Worker saves output to `processed_galleries/`, updates Project state to `COMPLETED` in DB.
6. Frontend polls or receives WebSocket event to display the final tour.
