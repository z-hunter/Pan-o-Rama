# Project Features Research

## Domain: Virtual Tour Platform (SaaS / B2B)

### Table Stakes (Must-Haves)
- **Equirectangular Viewer:** Smooth drag-to-look, gyroscope support on mobile devices.
- **Interconnected Scenes (Hotspots):** Clickable arrows/icons to jump between rooms.
- **Embeddability:** `<iframe>` generated links allowing tours to be placed on external sites.
- **Basic Auth & User Dashboard:** Space for users to see, edit, and delete their tours.

### Differentiators (Competitive Advantage)
- **Auto-Tripod Removal & Nadir Patching:** Significantly reduces turnaround time for virtual tour agencies.
- **Autonomous Tour Export:** Ability for users to download an archive containing independent HTML/JS/images to self-host.
- **"God Mode" Platform Moderation:** Advanced tools for the agency to intervene and perform QA on client tours.

### Anti-Features (Do Not Build)
- **Built-in 3D model generation (Photogrammetry):** Too complex and computationally expensive. Stick strictly to 2D equirectangular panoramic jumps.
- **Native iOS/Android App wrapper:** Unnecessary for MVP. Modern WebGL works flawlessly in mobile browsers.
