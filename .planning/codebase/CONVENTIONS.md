# Coding Conventions

## Python
- **Style**: Follow PEP 8 guidelines.
- **Indentation**: 4-space indentation.
- **Naming**: 
  - `snake_case` for functions and variables.
  - `UPPER_CASE` for constants.
- **Best Practices**: Keep Flask routes concise. Offload heavy image logic into helper functions (e.g., `stitch_panorama_*`). Focus heavily on branching explicit logs like `app.logger.info/warning/error` during pipelines.

## JavaScript
- **Style**: Vanilla ES6+.
- **Naming**: 
  - `camelCase` for variable and function names.
- **DOM IDs**: Keep DOM elements descriptive, e.g., `sceneName` or `linkingStatus`.

## Git Commits & PRs
- **Style**: Imperative, outcome-focused (e.g., "Stabilize panorama stitching and remove black gaps").
- **Scope**: Narrow, with one functional change per commit.
- **PR Expectation**: Require a problem/solution summary, details of affected paths, and steps to manually verify (along with visual proofs such as GIFs or screenshots).
