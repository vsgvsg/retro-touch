# Map Zoom macOS Compatibility Fix Implementation Plan (Throttled Zoom)

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Fix macOS scrolling zoom on the map widget by using a time-based 150ms cooldown and directional indicators to support both standard scroll wheel mice and Apple Magic Mouse/trackpads.

**Architecture:** Initialize a scroll timestamp variable, intercept canvas mouse wheel bindings in the map widget, and apply a 150ms throttle with `+1.0`/`-1.0` zoom steps.

**Tech Stack:** Python 3, Tkinter, tkintermapview

---

### Task 1: Map zoom macOS compatibility fix

**Files:**
- Modify: `exif_pipeline.py`
- Modify: `tests/test_exif_pipeline.py`

- [ ] **Step 1: Write a failing test for throttled custom mouse zoom**

  Replace the `test_tagger_app_map_scroll_zoom` test at the end of `tests/test_exif_pipeline.py` with:

  ```python
  def test_tagger_app_map_scroll_zoom(tmp_path, monkeypatch):
      from PIL import Image
      import json
      import sys
      import time
      
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
                  
          # 1. First scroll event (standard mouse wheel, delta=120)
          app._map._custom_mouse_zoom(MockEvent(120, 180, 180))
          
          # Verify standard mouse wheel zoom changes zoom by exactly 1 level
          if sys.platform == "darwin":
              assert len(zoom_calls) == 1
              assert abs(zoom_calls[0] - 11.0) < 0.001
              
          # 2. Immediate second scroll event (delta=120) -> should be throttled/ignored
          app._map._custom_mouse_zoom(MockEvent(120, 180, 180))
          if sys.platform == "darwin":
              assert len(zoom_calls) == 1  # Still only 1 call
              
          # 3. Bypass cooldown using monkeypatch on time.time
          zoom_calls.clear()
          current_time = time.time()
          monkeypatch.setattr(time, "time", lambda: current_time + 1.0)
          
          # Simulate Magic Mouse/Trackpad scroll event (delta=1)
          app._map._custom_mouse_zoom(MockEvent(1, 180, 180))
          
          # Verify magic mouse zoom changes zoom by exactly 1.0 level as well
          if sys.platform == "darwin":
              assert len(zoom_calls) == 1
              assert abs(zoom_calls[0] - 11.0) < 0.001
      finally:
          app.destroy()
  ```

- [ ] **Step 2: Run test to verify it fails**

  Run:
  ```bash
  ~/.venv/bin/python -m pytest tests/test_exif_pipeline.py::test_tagger_app_map_scroll_zoom -v
  ```
  Expected: FAIL on macOS (since it doesn't throttle or use the new zoom steps).

- [ ] **Step 3: Modify implementation**

  In `exif_pipeline.py`:

  1. Initialize `_last_scroll_time` in `TaggerApp.__init__` (around lines 507-512):
     ```python
              self._last_saved_year = None
              self._last_saved_month = None
              self._last_scroll_time = 0.0
     ```

  2. Update `custom_mouse_zoom` logic inside `TaggerApp._build_sidebar` (around lines 605-630):
     ```python
              # Custom macOS scroll-zoom fix
              def custom_mouse_zoom(event):
                  now = time.time()
                  if now - self._last_scroll_time < 0.15:
                      return  # Cooldown active
                      
                  relative_mouse_x = event.x / self._map.width
                  relative_mouse_y = event.y / self._map.height
                  
                  raw_delta = event.delta
                  if raw_delta == 0:
                      return
                      
                  step = 1.0 if raw_delta > 0 else -1.0
                  self._last_scroll_time = now
                  
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
  git commit -m "fix: throttle macOS map zoom events to support Apple Magic Mouse and trackpads"
  ```
