# Pan-o-Rama Pro: Product Design Document

## 1. Vision
Transform static 360° panoramas into an immersive, interconnected virtual environment that feels like a seamless 3D space.

## 2. Key Challenges & Solutions
### A. The "Blind Linking" Problem
*   **Current:** Randomly placed buttons.
*   **Pro Solution:** **Visual Portal Editor**. Users can rotate the camera, point at a door or path, and "drop" a portal exactly where it belongs in the 3D space.

### B. The "Jump Cut" Disorientation
*   **Current:** Abrupt image swapping.
*   **Pro Solution:** **Kinetic Transitions**. 
    1.  On click: Move camera to align perfectly with the portal.
    2.  Zoom-in (increase FOV) to create a sense of forward motion.
    3.  Cross-fade to the new scene.
    4.  New scene starts with a slight zoom-out to settle the view.

### C. Latency & Loading
*   **Current:** Load image only when needed.
*   **Pro Solution:** **Predictive Prefetching**. Automatically load low-res versions of all connected scenes in the background as soon as the current scene is idle.

## 3. UI/UX Design (Commercial Standard)

### Phase 1: The Studio (Editor)
*   **Sidebar:** List of scenes with "Edit" buttons.
*   **Viewer Window:** The interactive panorama.
*   **Action HUD:**
    *   `[+] Add Portal` button.
    *   When active: "Click on the scene to place a portal".
*   **Portal Configuration Modal:**
    *   Title: "Configure Portal"
    *   Dropdown: "Destination Scene" (List of all other rooms).
    *   Toggle: "Bidirectional Link" (Creates a return portal automatically).
    *   Icon Picker: (Standard Arrow, Door, Information).

### Phase 2: The Experience (Viewer)
*   **Minimalist Interface:** HUD fades out after 3 seconds of inactivity.
*   **Floating Navigation:** Sleek, semi-transparent animated arrows that pulse slightly to invite interaction.
*   **Compass & Map (Future):** A small radar in the corner showing orientation.

## 4. Technical Architecture
### Data Structure (Metadata.json)
```json
{
  "scenes": [
    {
      "id": "scene_0",
      "name": "Entrance",
      "panorama": "panorama.jpg",
      "portals": [
        { "pitch": -10.5, "yaw": 45.2, "target": "scene_1", "label": "Go to Office" }
      ]
    }
  ]
}
```

### Transition Logic (Pseudo-code)
1. `viewer.lookAt(portal.pitch, portal.yaw, 50, 1000)` // Aim and zoom in over 1s
2. `viewer.loadScene(target, { yaw: portal.entryYaw })` // Swap with calculated entry angle
3. `viewer.setHfov(100, 500)` // Settle zoom
