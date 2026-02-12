# Product Design: Minimal Virtual Tour Service

## 1. Product Goal
Build an MVP where users can create virtual tours from panoramas, connect scenes with transitions, manage tours, and share public links.

The service must also support private tours that:
- are not listed in the public gallery,
- are not accessible by direct link without authentication and permissions.

## 2. User Roles
- **Guest**: can browse the public gallery and open public tours.
- **User**: can register/login, create and manage own tours.
- **Admin**: basic moderation of users and content.

## 3. Core Entities
- **User**: `id`, `email`, `password_hash`, `status`, `created_at`.
- **Tour**: `id`, `owner_id`, `title`, `description`, `visibility` (`public`/`private`), `slug`, `created_at`, `updated_at`.
- **Scene**: `id`, `tour_id`, `title`, `panorama_url`, `order_index`.
- **Hotspot**: `id`, `from_scene_id`, `to_scene_id`, `yaw`, `pitch`, `label`.
- **AccessGrant**: `id`, `tour_id`, `user_id`, `role` (`viewer`/`editor`).

## 4. MVP User Flows
1. **Auth**
- Register with email/password.
- Login/logout.
- Basic account settings (change password, profile info).

2. **Tour Creation**
- Create empty tour.
- Upload panoramas as scenes.
- Add transitions (hotspots) between scenes.
- Set visibility (`public` or `private`).

3. **Tour Management**
- Edit title/description/scenes/hotspots/visibility.
- Delete tour (soft delete recommended).

4. **Sharing**
- Public tour: accessible via share link and visible in public gallery.
- Private tour: hidden from gallery; link returns `403` for unauthorized users.

## 5. Required Screens
- Landing + Public Gallery.
- Register / Login / Forgot Password.
- Dashboard: “My Tours”.
- Tour Editor: scene list + panorama viewer + hotspot placement mode.
- Tour Settings: metadata, visibility, delete.
- Account Settings.

## 6. Access & Privacy Rules
- Only owner/editor can modify tour content.
- Private tours are excluded from public listing and search indexing.
- Server-side authorization required for all private tour read/write endpoints.

## 7. Minimal API Surface
- `POST /auth/register`, `POST /auth/login`, `POST /auth/logout`
- `GET /me`, `PATCH /me`
- `POST /tours`, `GET /tours/my`, `GET /tours/:id`
- `PATCH /tours/:id`, `DELETE /tours/:id`
- `POST /tours/:id/scenes`, `PATCH /scenes/:id`, `DELETE /scenes/:id`
- `POST /scenes/:id/hotspots`, `PATCH /hotspots/:id`, `DELETE /hotspots/:id`
- `GET /gallery` (public tours only)

## 8. Non-Functional Requirements (MVP)
- Password hashing with Argon2/Bcrypt.
- Auth rate limiting.
- PostgreSQL for metadata; object storage for panoramas.
- Centralized logs and error tracking.
- Backup policy for DB and media.

## 9. Success Metrics
- Time-to-first-tour.
- Percentage of users who publish at least one tour.
- Public vs private tour usage ratio.
- Viewer engagement: average scenes viewed per session.
