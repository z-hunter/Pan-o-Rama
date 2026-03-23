# Testing Strategy

## Primary Method: Manual Validation
- Currently, **no automated test suite** operates on the repository.
- Rely solely on manual API execution and UI interaction for confirming code changes.
- Pre-merge basic check pattern:
  - Create project (`POST /api/project/create`)
  - Add scene (`POST /api/project/<id>/scene/add`)
  - Finalize (`POST /api/project/<id>/finalize`)
- Verifications extend into ensuring panorama output correctly parses in the browser, and output media lives correctly in `data/processed_galleries/<project_id>/`.

## Mock Scripts
- Occasional use of isolated helper scripts under `backend/test_mvp_api.py` and `scripts/` exist that mimic workflow or validate components manually in the local testbed.
