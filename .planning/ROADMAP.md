# Project Roadmap

> **Current Milestone**: 1. Core Platform Overhaul & Commercial UI

| Phase | Description | Status |
|---|---|---|
| **Phase 1** | Refactor Infrastructure & Modularize App | Not Started |
| **Phase 2** | UI/UX Complete Redesign & Dashboard Implementation | Not Started |
| **Phase 3** | Admin "God Mode" & Advanced Permissions | Not Started |
| **Phase 4** | Advanced WebGL Viewer & "Walking" Navigation | Not Started |
| **Phase 5** | Seamless Pseudo-3D Scene Transitions | Not Started |
| **Phase 6** | Computer Vision: Auto-Nadir & Tripod Removal Worker | Not Started |
| **Phase 7** | Subscriptions via Stripe Billing | Not Started |
| **Phase 8** | Standalone Autonomous Tour Export | Not Started |

---

## Phase 1: Refactor Infrastructure & Modularize App
**Goal:** Address current tech debt by breaking down the 217KB Flask `app.py` monolith into modular blueprints, enforcing standard python directory models to prepare for complex features.
**Key Deliverables:**
- Extraction of authentication, user routing, and API routing.
- Established clean routing structures.
- Strict definitions for Redis queue backgrounds.
**UAT (Acceptance Criteria):**
- [ ] Application boots successfully natively and via Gunicorn.
- [ ] Existing endpoints complete basic tour generation flawlessly without `app.py` logic errors.

## Phase 2: UI/UX Complete Redesign & Dashboard Implementation
**Goal:** Drop the "technical prototype" look and apply a commercially viable, stylish, and highly-polished frontend system across all views (HTML/Vanilla JS).
**Key Deliverables:**
- Styling system (CSS variables / minimal framework) implemented across index, dashboard, registration, accounts.
- Redesigned "My Tours" dashboard supporting thumbnail previews and slick modal forms.
**UAT:**
- [ ] The entire site looks coherent, stylish, and responds immediately to layout changes on mobile and desktop.

## Phase 3: Admin "God Mode" & Advanced Permissions
**Goal:** Create explicit roles mapping and an Admin dashboard to observe and forcibly edit all domains on the platform for agency intervention.
**Key Deliverables:**
- Role differentiation logic in backend (JWT / Session markers limit).
- System-wide search and view-all list within a specific Admin dashboard route.
**UAT:**
- [ ] Test admin token successfully views and enters standard user's private projects natively without 403 blocks.

## Phase 4: Advanced WebGL Viewer & "Walking" Navigation
**Goal:** Upgrade the tour playback mechanism. Replace legacy pin hotspots with a Google Maps-inspired interactive "walking" path logic utilizing planar arrows.
**Key Deliverables:**
- WebGL / Three.js base wrapper implementation supporting nadir-placed arrow entities.
- Click raycasting mapped accurately to adjacent rooms logic.
**UAT:**
- [ ] Walking arrow entities appear flattened toward the floor. Clicking navigates correctly between linked scenes.

## Phase 5: Seamless Pseudo-3D Scene Transitions
**Goal:** Prevent visual hard-cuts by manipulating shaders to drag/warp the image smoothly into the next locale.
**Key Deliverables:**
- Shader logic implementation for frame transformation during scene leaps.
- WebGL texture disposal cycle enforcement (garbage collection) to prevent Mobile RAM leaks on large tours.
**UAT:**
- [ ] Pushing the navigation arrow provides an animated warp into the new panoramic texture without blocking the main event thread or crashing after 10 jumps.

## Phase 6: Computer Vision: Auto-Nadir & Auto-Tripod
**Goal:** Reduce manual processing time by automating the patch over the tripod via server-side OpenCV workers.
**Key Deliverables:**
- Backend RQ task that manipulates the bottom pole of panoramic images and injects a transparent PNG patch logo dynamically.
**UAT:**
- [ ] Uploading a standard 360 photo generates a processed preview with a clean, branded nadir patch centered globally.

## Phase 7: Subscriptions via Stripe Billing
**Goal:** Control access to premium B2B features (Exports, removal of default watermarks) via Stripe-gated limits.
**Key Deliverables:**
- Checkout integration with Flask Stripe wrapper.
- Active webhook listener syncing subscription status automatically.
**UAT:**
- [ ] Purchasing a test plan modifies the internal profile limits logic synchronously.

## Phase 8: Standalone Autonomous Tour Export
**Goal:** Allow users to download a standalone archive (ZIP) containing an HTML view that works on any remote CDN without CORS errors or DB deps.
**Key Deliverables:**
- Zipping worker task that compiles the finalized WebGL source, project JSON, and compressed panorama assets.
- Endpoint serving the generated `.zip` dynamically to active premium users.
**UAT:**
- [ ] Unzipping the provided package and running it via simple HTTPServers proves it operates autonomously.

## Future (Icebox)
- Photogrammetry logic
- Advanced floor-plan generation
