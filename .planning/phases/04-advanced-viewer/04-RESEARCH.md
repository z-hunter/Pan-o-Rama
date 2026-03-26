# Phase 4: Advanced WebGL Viewer - Research

**Researched:** 2026-03-25
**Domain:** Three.js, WebGL, 360-degree Panoramas, CSS3DRenderer
**Confidence:** HIGH

## Summary

The current panorama viewer uses Pannellum, which is limited in terms of custom 3D overlays and seamless transitions. To achieve a modern, "walking" navigation experience with floor arrows and warp transitions, we will migrate to a custom Three.js-based engine. This allows for higher-performance rendering, advanced shaders, and the integration of CSS3DRenderer for UI elements that perspectively match the 3D scene.

**Primary recommendation:** Use a single Three.js `SphereGeometry` with a custom `ShaderMaterial` for transitions, and a separate `CSS3DRenderer` for floor-based navigation hotspots.

## User Constraints (from CONTEXT.md)

### Locked Decisions
- **D-01: No heavy frameworks**: Use Vanilla CSS (prefer CSS variables) and Vanilla JS.
- **D-02: Consistent Header**: All pages must share a consistent header/navigation component.
- **D-03: Thumbnail Previews**: Dashboard must show a visual preview for each tour.
- **D-04: Mobile First**: High responsiveness is critical.

### the agent's Discretion
- Best approach for floor arrows: **Ray-Plane Intersection** with a virtual floor plane at `y = -100`.
- Animation patterns for scene transitions: **Warp/Displacement Shader** with a noise map.
- Optimization for mobile: **Manual texture disposal** and texture resolution management.

## Standard Stack

### Core
| Library | Version | Purpose | Why Standard |
|---------|---------|---------|--------------|
| Three.js | 0.183.2 | Core WebGL Rendering | Industry standard for 3D on the web. |
| CSS3DRenderer | 0.183.2 | 3D Perspective HTML Overlays | Allows using HTML/CSS for hotspots while matching 3D perspective. |

### Supporting
| Library | Version | Purpose | When to Use |
|---------|---------|---------|--------------|
| GSAP | 3.12.x | Animation / Tweening | Smoothest engine for uniform/camera animations. |

**Installation:**
```html
<script type="importmap">
{
  "imports": {
    "three": "https://cdn.jsdelivr.net/npm/three@0.183.2/build/three.module.js",
    "three/addons/": "https://cdn.jsdelivr.net/npm/three@0.183.2/examples/jsm/"
  }
}
</script>
```

## Architecture Patterns

### Floor Arrows Implementation
To make navigation icons look like they are "on the floor":
1.  **Raycasting:** Project a ray from the camera center through the hotspot's `pitch` and `yaw`.
2.  **Intersection:** Find where this ray intersects a horizontal plane at `y = -H` (e.g., `y = -100`).
3.  **Placement:** Place a `CSS3DObject` at this intersection point.
4.  **Rotation:** Rotate the object `x = -Math.PI / 2` so it lies flat on the ground.

### Smooth Transitions (Warp)
Use a `ShaderMaterial` on the panorama sphere:
1.  **Uniforms:** `uTex1` (from), `uTex2` (to), `uProgress` (0.0 to 1.0), `uDisp` (Noise Map).
2.  **Logic:** Interpolate UVs by offsetting them with noise values, weighted by `uProgress`. This creates a "dissolving" or "warping" effect rather than a simple cross-fade.

### Optimization for Mobile
- **Texture Disposal:** Always call `texture.dispose()`, `material.dispose()`, and `geometry.dispose()` when removing objects or switching scenes.
- **Resolution:** Limit panorama textures to 4096px width on mobile.
- **Preloading:** Load the "to" texture in the background before starting the transition.

## Don't Hand-Roll

| Problem | Don't Build | Use Instead | Why |
|---------|-------------|-------------|-----|
| CSS3D Layering | Custom DOM syncing | CSS3DRenderer | Synchronizes a separate DOM tree with Three.js camera perfectly. |
| Animation Curves | Custom `requestAnimationFrame` lerps | GSAP | Handles easing, interruptions, and timeline management reliably. |
| 360 Projections | Custom Sphere UV math | Three.js SphereGeometry | Built-in equirectangular mapping is well-tested and performant. |

## Common Pitfalls

### Pitfall 1: Texture Memory Leak
**What goes wrong:** Switching between many scenes crashes mobile browsers.
**How to avoid:** Explicitly dispose of the "from" texture AFTER the transition completes.

### Pitfall 2: CSS3D Click-Through
**What goes wrong:** Users can't drag the panorama when the mouse is over a hotspot.
**How to avoid:** Use `pointer-events: none` on the CSS3D renderer container, but `pointer-events: auto` on the individual hotspot elements.

### Pitfall 3: Coordinate System Mismatch
**What goes wrong:** Hotspots appearing in the wrong location compared to Pannellum.
**How to avoid:** Ensure the `yaw` and `pitch` interpretation matches (Pannellum uses degrees, Three.js usually radians; Pannellum $0,0$ is center, Three.js depends on sphere rotation).

## Code Examples

### Transition Fragment Shader
```glsl
varying vec2 vUv;
uniform sampler2D uTex1;
uniform sampler2D uTex2;
uniform sampler2D uDisp;
uniform float uProgress;

void main() {
    vec4 disp = texture2D(uDisp, vUv);
    vec2 distortedUv1 = vUv + uProgress * (disp.rg - 0.5) * 0.1;
    vec2 distortedUv2 = vUv - (1.0 - uProgress) * (disp.rg - 0.5) * 0.1;
    
    vec4 tex1 = texture2D(uTex1, distortedUv1);
    vec4 tex2 = texture2D(uTex2, distortedUv2);
    
    gl_FragColor = mix(tex1, tex2, uProgress);
}
```

### Floor Hotspot Projection
```javascript
function projectToFloor(pitch, yaw, H = 100) {
    const p = THREE.MathUtils.degToRad(pitch);
    const y = THREE.MathUtils.degToRad(yaw);
    
    // Check if below horizon
    if (p >= 0) return null; 
    
    const r = H / Math.tan(-p);
    return new THREE.Vector3(
        r * Math.sin(y),
        -H,
        r * Math.cos(y)
    );
}
```

## Sources

### Primary (HIGH confidence)
- Three.js Documentation - [SphereGeometry](https://threejs.org/docs/#api/en/geometries/SphereGeometry)
- Three.js Examples - [CSS3DRenderer](https://threejs.org/examples/?q=css3d#css3d_sprites)
- MDN - [WebGL Shader Programming](https://developer.mozilla.org/en-US/docs/Web/API/WebGL_API/Tutorial/Using_shaders_to_apply_color_in_WebGL)

### Metadata
**Confidence breakdown:**
- Standard stack: HIGH - Verified current versions.
- Architecture: HIGH - Ray-plane intersection and CSS3D are standard for high-end tours.
- Pitfalls: HIGH - Common issues in mobile WebGL development.

**Research date:** 2026-03-25
**Valid until:** 2026-04-25
