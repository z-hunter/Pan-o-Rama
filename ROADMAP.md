# Roadmap (2 / 6 / 12 Weeks)

Audience focus: **Realtors + SMB venues**  
Primary goal: **Activation** (users reach “Published Tour” faster)  
Product focus: **Ops/scale** (reliable processing + predictable storage)

## Success Criteria

### Activation (Aha)
- Conversion: `signup -> first_published_tour`
- Median time: `signup -> first_published_tour`
- Funnel events: `tour_created`, `scene_processed`, `hotspot_created`, `tour_published`

### Ops/Scale
- Job success rate (scene processing) + error breakdown
- P95 processing time per scene (web + preview + stitch when needed)
- Viewer reliability: 5xx/timeout rate on tour open and scene switches

## 2 Weeks (Sprint 1): Async Processing + Transparent Progress

### Epic A: Background Jobs (Redis + RQ)
Goal: move heavy work off request/response to prevent timeouts and “server crash” UX.

Backend changes:
- Introduce a job model + status tracking.
- Run an RQ worker as a separate process/service.
- Processing pipeline as a job:
  - Save upload
  - Validate (type/size/quota)
  - Process image(s) / stitch (if needed)
  - Generate derivatives (`preview.jpg`, `web.jpg`, keep original as HD source)
  - DB update + gallery regen

API additions (compat preserved):
- `POST /tours/:id/scenes?async=1` -> `202 { job_id, scene_id }`
- `GET /jobs/:job_id` -> `{ status, stage, progress_pct, message, result, error }`
- Keep current sync path as fallback (`async=0` or missing flag).

Job stages (fixed vocabulary):
- `queued`, `upload_saved`, `processing`, `stitching`, `derivatives`, `db_update`, `done`, `failed`

Acceptance:
- Large uploads do not break the UI; failures return structured `failed` with actionable error codes.

### Epic B: Studio Processing UX (Polling)
Goal: user sees progress and can keep working.

Frontend changes:
- After upload, show a scene row/card with job status, stage, %.
- Poll `/jobs/:id` every 1-2s (backoff after 30s).
- Provide Retry for failed jobs and clear error text.

Acceptance:
- User can continue placing links / editing other scenes while jobs run.

### Epic C: Minimal Product Analytics for Activation
Goal: measure funnel and guide iteration.

DB:
- Add `events` table: `id, user_id, tour_id, event_name, props_json, created_at`.

Events to emit:
- `signup_complete`, `tour_created`, `scene_upload_started`, `scene_job_done`, `hotspot_created`, `tour_published`, `share_opened`

Acceptance:
- Basic funnel can be computed from DB for the last N days.

## 6 Weeks (Sprint 2-3): Quotas + Backups + Predictable Output

### Epic D: Quotas / Limits by Plan
Goal: prevent abuse and keep operations predictable.

Add plan limits (examples; tune later):
- `max_storage_mb`
- `max_scenes_per_tour`
- `max_scene_pixels`
- `max_upload_mb`

Enforcement:
- Validate limits before queueing a job.
- Return structured errors: `plan_limit_exceeded`, `storage_quota_exceeded`, `scene_too_large` with upgrade CTA.

Acceptance:
- Cannot overload the service with extreme uploads; UI messages are clear.

### Epic E: Backups + Restore + Garbage Collection
Goal: reduce risk of losing user content.

Ops additions:
- Daily backups for:
  - `data/app.db`
  - `data/processed_galleries/`
  - optionally `data/raw_uploads/`
- Restore procedure + script (testable).
- Retention policy (e.g. 7/30 days).
- GC for:
  - soft-deleted tours
  - old raw uploads
  - orphaned processed directories

Acceptance:
- Restore tested on a separate directory and can open existing tours.

### Epic F: Quality + “Web-safe by Default”
Goal: stop browser freezes and reduce “why is my pano broken” support.

Preflight checks:
- Heuristics for “likely equirectangular 360” (aspect ~2:1, metadata hints).
- Warn in Studio when input is suspicious.

Derivatives policy:
- Use `web.jpg` by default in viewer to avoid huge textures.
- Keep an opt-in HD toggle for original hires (already implemented in player).
- Cap max “web” size for Facebook/in-app webviews.

Acceptance:
- Fewer reports of blank scenes / freezing tabs; consistent outputs.

## 12 Weeks (Sprint 4-6): Share Moment + Paid Value Hooks + Admin Diagnostics

### Epic G: “Publish & Share” Moment
Goal: user immediately shares after publish (activation/retention).

After finalize, show:
- QR code
- short share link (slug)
- embed code (iframe)
- “Open on phone”
- “Share to Facebook” entry point

Acceptance:
- Users can share without hunting for URLs.

### Epic H: Branding + CTA + Leads (Pro/Business)
Goal: provide obvious paid value for Realtors/SMB.

Viewer additions (plan-gated):
- Branding: logo/colors/contact block
- CTA button: “Call / Book viewing / Reserve”
- Lead form (email/phone) saved in DB + export CSV

Acceptance:
- Viewer can generate a lead; easy to demonstrate ROI.

### Epic I: Admin / Diagnostics
Goal: faster debugging, safer scaling.

Add:
- `/healthz` (db ok, redis ok, disk free, worker heartbeat)
- Simple admin page for jobs/errors/events
- Rate limiting for uploads

Acceptance:
- Can detect failures quickly and understand causes.

## Implementation Defaults (Decisions Locked)
- Queue: **Redis + RQ** (1 worker initially; scale up later).
- Progress transport: **polling** (`GET /jobs/:id`).
- Storage: **VPS disk + backups** for 12 weeks; prepare future migration to object storage.
- Backward compatibility: keep current sync endpoints working while UI migrates to async.

## Test Scenarios (Acceptance-Level)
1. New user: signup -> create tour -> upload 1 scene -> job done -> add POI -> publish -> open share.
2. Very large pano: job runs without hanging UI; `web.jpg` served; HD toggle works.
3. Stitch failure: job becomes `failed` with code; UI shows Retry; second attempt can succeed.
4. Quota exceeded: refused before queueing job; clear upgrade CTA.
5. Backup/restore: restore DB + processed_galleries; existing tours open correctly.

## Assumptions
- VPS can run Redis and an RQ worker under systemd.
- SQLite remains acceptable for 12 weeks if DB writes are serialized by worker jobs.
- Plan limits can be adjusted via `plans` table without migrations.

