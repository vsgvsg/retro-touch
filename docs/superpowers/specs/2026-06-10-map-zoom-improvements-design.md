# Design Spec: Map Scroll Zoom Improvements for macOS

## Goal
Fix map scrolling zoom behavior in `exif_pipeline.py` when running on macOS. Prevent the map from jumping between minimum and maximum zoom in one scroll tick (standard mouse wheel) and make it responsive to trackpads and Apple Magic Mouse.

---

## 1. Issue Diagnosis
In the `tkintermapview` library:
- A multiplier of `0.1` is applied directly to `event.delta`.
- On macOS, a standard mouse wheel scroll tick returns `event.delta = 120` or `-120`, resulting in a sudden change of `12` zoom levels.
- Trackpads and Magic Mouse return small delta values like `1` or `-1`, resulting in a tiny change of `0.1` zoom levels, which is ignored by the map widget's integer-rounding draw code.

---

## 2. Solution: Canvas Re-binding
We will intercept the mouse wheel events on the map's canvas component (`self._map.canvas`) and bind them to a custom `custom_mouse_zoom` method:
- Standard scroll wheel events (`abs(delta) >= 120`) will be normalized by dividing by `120.0` (changing zoom by exactly `1.0` level per tick).
- Small scroll gesture events (`abs(delta) < 120` on macOS) will be scaled using a multiplier of `0.2` to provide a smooth, responsive zoom curve for Magic Mouse/Trackpads.
- Bind the custom zoom function to `<MouseWheel>`, `<Button-4>`, and `<Button-5>` on the `self._map.canvas` widget.

---

## 3. Verification and Testing
- Add a new unit test `test_tagger_app_map_scroll_zoom` that:
  - Instantiates `TaggerApp`.
  - Mocks `self._map.set_zoom` to record calls.
  - Simulates a standard mouse wheel scroll event (`delta = 120`).
  - Simulates a Magic Mouse scroll event (`delta = 1`).
  - Verifies that the correct relative zoom levels are passed to `set_zoom`.
- Run the full test suite (`python -m pytest tests/test_exif_pipeline.py`).
