# Technical Concerns

## Technical Debt & Maintenance
- **Massive Monolith**: `backend/app.py` has grown exceptionally large (~217 KB). Refactoring business logic, file uploads, image transformations, and authentication down to smaller maintainable routers is critical.
- **No Automated Test Coverage**: Lacking an automated test suite substantially limits confidence during active development, forcing long manual feedback loops.

## Deployment & Production Resiliency
- Due to strict manual deployment, ensure secrets, SSH keys, or server logs remain out of version control and git commits. Large output files in `data/` shouldn't be committed unless absolutely necessary for reproducible debugging.
- Verify read/write permissions precisely over `data/` directories as local environments often differ from server `www-data` setups, which causes common upload failures.

## Stripe & Payment Resiliency
- A Stripe integration exists but depends on manual validation without an automated stub; changes in API versions could impact transactions unexpectedly.
- File processing is bound behind a Redis server connection. A dropped connection or overloaded worker immediately locks up queue times.
