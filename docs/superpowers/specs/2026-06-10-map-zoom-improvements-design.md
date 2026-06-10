# Design Spec: Map Scroll Zoom Improvements for macOS

## Goal
Fix map scrolling zoom behavior in `exif_pipeline.py` when running on macOS. Prevent the map from jumping between minimum and maximum zoom in one scroll tick (standard mouse wheel), while preserving default library behavior for Apple Magic Mouse / Trackpads.

---

## 1. Diagnostics & Refinement
In the `tkintermapview` library:
- A multiplier of `0.1` is applied directly to `event.delta`.
- Standard scroll wheel sends `delta = 120`, changing zoom by `12` levels.
- Apple Magic Mouse sends very small delta values (like `1` or `2`). The user preferred to keep the default library handling for these.

To fix the standard scroll wheel without affecting other scroll events:
- If `abs(event.delta) >= 120` (standard scroll wheel mouse), divide by `120.0` to change the zoom level by exactly `1.0`.
- Else, fall back to the default library behavior (`step = event.delta * 0.1`).

---

## 2. Solution: Canvas Re-binding
We will intercept the mouse wheel events on the map's canvas component (`self._map.canvas`) and bind them to a custom `custom_mouse_zoom` method:
- In `custom_mouse_zoom(event)`, check `event.delta`. If `0`, return.
- If `sys.platform == "darwin"` and `abs(event.delta) >= 120`, set `step = event.delta / 120.0`.
- Else if `sys.platform == "darwin"`, set `step = event.delta * 0.1`.
- For other platforms/buttons, use standard `tkintermapview` values.
- Call `self._map.set_zoom(self._map.zoom + step, ...)`.

---

## 3. Verification and Testing
- Add a new unit test `test_tagger_app_map_scroll_zoom` that:
  - Instantiates `TaggerApp`.
  - Mocks `self._map.set_zoom` to record calls.
  - Simulates a scroll event (`delta = 120`).
  - Verifies that standard mouse wheel zoom changes zoom by exactly `1.0`.
  - Simulates a Magic Mouse scroll event (`delta = 1`).
  - Verifies it zooms by `0.1`.
- Run the full test suite (`python -m pytest tests/test_exif_pipeline.py`).
