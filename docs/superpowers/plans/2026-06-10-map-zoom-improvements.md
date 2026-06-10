# Map Zoom macOS Compatibility Fix Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Fix macOS scrolling zoom on standard scroll wheel mice by dividing `delta` by `120`, while keeping default library behavior for other events.

**Architecture:** Intercept canvas mouse wheel bindings in the map widget and apply division logic when `abs(delta) >= 120` on macOS.

**Tech Stack:** Python 3, Tkinter, tkintermapview

---

### Task 1: Map zoom macOS compatibility fix

**Files:**
- Modify: `exif_pipeline.py`
- Modify: `tests/test_exif_pipeline.py`

- [ ] **Step 1: Write a failing test for macOS scroll wheel zoom**

  Replace the `test_tagger_app_map_scroll_zoom` test at the end of `tests/test_exif_pipeline.py` with:

  ```python
  def test_tagger_app_map_scroll_zoom(tmp_path, monkeypatch):
      from PIL import Image
      import json
      import sys
      
      jpg_path = tmp_path / "1990-penza-00001.jpg"
      img = Image.new("RGB", (100, 100), color="red")
      img.save(jpg_path, "JPEG")
      
      locs_file = tmp_path / "locations.json"
      locs_file.write_text(json.dumps({"locations": []}), encoding="utf-8")
      
      sc_file = tmp_path / "1990-penza-00001.faces.json"
      sc_file.write_text(json.dumps({}), encoding="utf-8")
      
      monkeypatch.setattr(ep.NominatimClient, "_get", lambda self, url: [])
      
      app = ep.TaggerApp([str(jpg_path)], extracted_dir=str(tmp_path))
      try:
          # Mock set_zoom to capture zoom steps
          zoom_calls = []
          def mock_set_zoom(new_zoom, relative_pointer_x=0.5, relative_pointer_y=0.5):
              zoom_calls.append(new_zoom)
          
          app._map.set_zoom = mock_set_zoom
          app._map.zoom = 10.0
          
          class MockEvent:
              def __init__(self, delta, x, y, num=0):
                  self.delta = delta
                  self.x = x
                  self.y = y
                  self.num = num
                  
          # 1. Simulate standard scroll wheel event (delta=120)
          app._map._custom_mouse_zoom(MockEvent(120, 180, 180))
          
          # Verify standard mouse wheel zoom changes zoom by exactly 1 level
          if sys.platform == "darwin":
              assert len(zoom_calls) == 1
              assert abs(zoom_calls[0] - 11.0) < 0.001
              
          zoom_calls.clear()
          
          # 2. Simulate Magic Mouse scroll event (delta=1)
          app._map._custom_mouse_zoom(MockEvent(1, 180, 180))
          
          # Verify Magic Mouse uses default library behavior (zoom by 0.1 levels)
          if sys.platform == "darwin":
              assert len(zoom_calls) == 1
              assert abs(zoom_calls[0] - 10.1) < 0.001
      finally:
          app.destroy()
  ```

- [ ] **Step 2: Run test to verify it fails**

  Run:
  ```bash
  ~/.venv/bin/python -m pytest tests/test_exif_pipeline.py::test_tagger_app_map_scroll_zoom -v
  ```
  Expected: FAIL on macOS (since it currently does throttled integer zoom instead of fallback logic).

- [ ] **Step 3: Modify implementation**

  In `exif_pipeline.py`:

  1. Remove `self._last_scroll_time = 0.0` from `TaggerApp.__init__`.
  
  2. Update `custom_mouse_zoom` logic inside `TaggerApp._build_sidebar` (around lines 605-625):
     ```python
              # Custom macOS scroll-zoom fix
              def custom_mouse_zoom(event):
                  relative_mouse_x = event.x / self._map.width
                  relative_mouse_y = event.y / self._map.height
                  
                  raw_delta = event.delta
                  if raw_delta == 0:
                      return
                      
                  if sys.platform == "darwin":
                      if abs(raw_delta) >= 120:
                          step = raw_delta / 120.0
                      else:
                          step = raw_delta * 0.1
                  elif sys.platform.startswith("win"):
                      step = raw_delta * 0.01
                  elif event.num == 4:
                      step = 1.0
                  elif event.num == 5:
                      step = -1.0
                  else:
                      step = raw_delta * 0.1
                      
                  new_zoom = self._map.zoom + step
                  self._map.set_zoom(new_zoom, relative_pointer_x=relative_mouse_x, relative_pointer_y=relative_mouse_y)
     ```

- [ ] **Step 4: Run test to verify it passes**

  Run:
  ```bash
  ~/.venv/bin/python -m pytest tests/test_exif_pipeline.py::test_tagger_app_map_scroll_zoom -v
  ```
  Expected: PASS

- [ ] **Step 5: Run full test suite**

  Run:
  ```bash
  ~/.venv/bin/python -m pytest tests/test_exif_pipeline.py -v
  ```
  Expected: All 64 tests pass.

- [ ] **Step 6: Commit**

  Run:
  ```bash
  git add exif_pipeline.py tests/test_exif_pipeline.py
  git commit -m "fix: normalize macOS standard scroll wheel zoom while keeping default magic mouse behavior"
  ```
