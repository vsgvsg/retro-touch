# Design Spec: Map Scroll Zoom Improvements for macOS (Throttled Zoom)

## Goal
Fix map scrolling zoom behavior in `exif_pipeline.py` when running on macOS. Prevent the map from jumping between minimum and maximum zoom in one scroll tick (standard mouse wheel) and make it responsive to trackpads and Apple Magic Mouse.

---

## 1. Diagnostics & Refinement
In the `tkintermapview` library:
- A multiplier of `0.1` is applied directly to `event.delta`.
- Standard scroll wheel sends `delta = 120`, changing zoom by `12` levels.
- Apple Magic Mouse sends very small delta values (like `1` or `2`), changing zoom by `0.1` or `0.2` levels, which doesn't cross integer rounding boundaries and gets ignored.

Rather than scaling delta values (which can vary wildly across different mouse devices and OS configurations), we will use a time-based throttling mechanism:
- Ignore any scroll events where the time since the last registered zoom is less than `0.15` seconds (150ms cooldown).
- Zoom in by exactly `+1.0` if `delta > 0`, and zoom out by `-1.0` if `delta < 0`.

---

## 2. Solution: Canvas Re-binding and Cooldown
We will intercept the mouse wheel events on the map's canvas component (`self._map.canvas`) and bind them to a custom `custom_mouse_zoom` method:
- Initialize `self._last_scroll_time = 0.0` in `TaggerApp.__init__`.
- In `custom_mouse_zoom(event)`, check `time.time() - self._last_scroll_time < 0.15`. If true, return.
- Check `event.delta`. If `0`, return.
- Set `step = 1.0 if event.delta > 0 else -1.0`.
- Update `self._last_scroll_time = time.time()`.
- Call `self._map.set_zoom(self._map.zoom + step, ...)`.

---

## 3. Verification and Testing
- Add a new unit test `test_tagger_app_map_scroll_zoom` that:
  - Instantiates `TaggerApp`.
  - Mocks `self._map.set_zoom` to record calls.
  - Simulates a scroll event (`delta = 120`).
  - Simulates another scroll event immediately (`delta = 120`) and verifies it is ignored due to the cooldown.
  - Mocks time to bypass the cooldown, simulates a Magic Mouse scroll event (`delta = 1`), and verifies it zooms by exactly `1.0`.
- Run the full test suite (`python -m pytest tests/test_exif_pipeline.py`).
