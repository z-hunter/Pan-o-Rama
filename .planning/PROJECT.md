# Project: Lokalny Obiektyw Virtual Tours

## Core Value
A hybrid platform that empowers individual users to create, share, and export 360° virtual tours via a SaaS subscription, while simultaneously providing a powerful backend toolkit for a custom interior digitization agency (B2B).

## What This Is
- A web application for uploading and stitching panoramic photos into interactive virtual tours.
- A commercial subscription service for tour creators and photographers.
- An administrative platform that supports white-label and advanced features (tripod removal, nadir patching) specifically tailored for a digitization agency.

## Target Audience
- **Photographers & Enthusiasts**: Users who want an easy subscription-based platform to build, share, or export fully self-hostable tours.
- **Agency Administration**: Internal staff needing to manage white-label projects, patch nadir logos, and oversee user content seamlessly.

## Requirements

### Validated
- ✓ Interactive virtual tour creation from panoramic scenes
- ✓ Hotspot generation mapping for scene-to-scene transitions
- ✓ Registration, Authentication, and User Dashboard
- ✓ Standalone export/download of an autonomous tour (HTML/JS/Images) for external hosting
- ✓ Basic public gallery and permission architecture system

### Active
- [ ] **Subscription Monetization**: Implement Stripe subscription tiers.
- [ ] **Tripod Removal**: Automated image processing to remove tripods from the nadir of 360° files.
- [ ] **Custom Nadir Logo**: Auto-placement of a specific logo/patch in the bottom area of a panorama.
- [ ] **God Mode (Admin View)**: Administrative ability to view *all* user files across the platform, bypassing privacy locks.
- [ ] **Admin Editing**: Administrative ability to edit other users' projects or configurations.

### Out of Scope
- [ ] Alternative non-subscription payment models (like direct one-off DLC purchases) — sticking strictly to subscription plans for predictable MRR.

## Key Decisions
| Decision | Rationale | Outcome |
|----------|-----------|---------|
| Subscription-only business model | Focuses the business loop on predictable recurring revenue | — Pending |
| Admin bypass for viewing/editing | Agency requires access to everything for B2B moderation and intervention capabilities | — Pending |
| Auto nadir-patching | Significant time-saver for repetitive processing of 360° tour deliveries | — Pending |

## Evolution
This document evolves at phase transitions and milestone boundaries.

**After each phase transition** (via `/gsd-transition`):
1. Requirements invalidated? → Move to Out of Scope with reason
2. Requirements validated? → Move to Validated with phase reference
3. New requirements emerged? → Add to Active
4. Decisions to log? → Add to Key Decisions
5. "What This Is" still accurate? → Update if drifted

**After each milestone** (via `/gsd-complete-milestone`):
1. Full review of all sections
2. Core Value check — still the right priority?
3. Audit Out of Scope — reasons still valid?
4. Update Context with current state

---
*Last updated: 2026-03-24 after initialization*
