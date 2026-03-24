# Research Summary

## Domain: 360° Virtual Tour Platform & Agency B2B SaaS

### Key Findings
**Stack:** The existing Flask + Worker architecture combined with an off-the-shelf WebGL panoramic viewer (like Marzipano or Pannellum) is optimal. Moving storage towards a CDN later will be essential.
**Table Stakes:** Intuitive equirectangular viewing, hotspot jumping, basic auth, and an embeddable delivery format.
**Watch Out For:** WebGL memory leaks during scene transitions (must garbage-collect old textures), and keeping Flask asynchronous (CV2 modifications like tripod removal MUST sit in Redis/RQ background workers).

### Differentiators
The B2B aspects (autonomous export, automated nadir patching, administration "God mode") are the true value drivers over basic tour hosting services. These justify the subscription price and act as workflow boosters for agencies.

*This research directly shapes the upcoming requirements planning.*
