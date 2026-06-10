# Map Zoom macOS Compatibility Fix Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Fix macOS scrolling zoom on the map widget, preventing excessive zoom jumps on standard scroll wheels and enabling responsiveness for Apple Magic Mouse and trackpads.

**Architecture:** Intercept canvas mouse wheel bindings in the map widget and apply a normalized delta calculation in a custom event handler function inside the tagger UI.

**Tech Stack:** Python 3, Tkinter, tkintermapview

---

### Task 1: Map zoom macOS compatibility fix

**Files:**
- Modify: `exif_pipeline.py`
- Modify: `tests/test_exif_pipeline.py`

- [ ] **Step 1: Write a failing test for custom mouse zoom**

  Append this test at the end of `tests/test_exif_pipeline.py`:

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
          
          # Simulate scroll event
          class MockEvent:
              def __init__(self, delta, x, y, num=0):
                  self.delta = delta
                  self.x = x
                  self.y = y
                  self.num = num
                  
          # We need to get the bound function for <MouseWheel> on the canvas
          # To invoke it, we can call the function bound to "<MouseWheel>" directly or trigger it.
          # To trigger, we can call the handler registered on canvas.bind("<MouseWheel>")
          bindings = app._map.canvas.bind("<MouseWheel>")
          # We can also call the mouse_zoom method directly if we overrode it or registered our custom one.
          # Let's inspect the binding or invoke the callback directly.
          # Since tkinter bindings are Tcl functions, it's easier to invoke our custom function if we find it
          # or we can mock/call the event handler directly if we expose it, or just invoke the callback if it's stored.
          # Wait, in the implementation we will bind the custom function.
          # Let's call the actual custom function. We can find it because it's bound.
          # Alternatively, we can verify that the custom zoom function behaves correctly by triggering it
          # via event_generate or calling the Tkinter binding function.
          # Let's use event_generate to trigger a real Tkinter scroll event!
          # This is the most authentic test.
          app._map.zoom = 10.0
          
          # Generate MouseWheel event with delta=120 (standard mouse wheel scroll on mac)
          app._map.canvas.event_generate("<MouseWheel>", x=50, y=50, delta=120)
          app.root.update_idletasks()
          
          # Verify standard mouse wheel zoom changes zoom by exactly 1 level
          # (On mac, original code would do 10.0 + 120*0.1 = 22.0)
          # With our fix, it should do 10.0 + 1.0 = 11.0
          if sys.platform == "darwin":
              assert len(zoom_calls) == 1
              assert abs(zoom_calls[0] - 11.0) < 0.001
              
          zoom_calls.clear()
          
          # Generate MouseWheel event with delta=1 (Apple magic mouse/trackpad scroll)
          app._map.canvas.event_generate("<MouseWheel>", x=50, y=50, delta=1)
          app.root.update_idletasks()
          
          # Verify magic mouse zoom changes zoom by 0.2 levels
          if sys.platform == "darwin":
              assert len(zoom_calls) == 1
              assert abs(zoom_calls[0] - 10.2) < 0.001
      finally:
          app.destroy()
  ```

- [ ] **Step 2: Run test to verify it fails**

  Run:
  ```bash
  ~/.venv/bin/python -m pytest tests/test_exif_pipeline.py::test_tagger_app_map_scroll_zoom -v
  ```
  Expected: FAIL on macOS (since it doesn't scale delta correctly).

- [ ] **Step 3: Modify implementation**

  In `exif_pipeline.py` inside `TaggerApp._build_sidebar` (around lines 595-601, right after creating the map widget):

  Add the custom mouse zoom method and re-bind the scroll events:

  ```python
          # Map widget
          self._map = tkintermapview.TkinterMapView(loc_frame, width=360, height=360,
                                                    corner_radius=0)
          self._map.pack(fill=tk.BOTH, expand=True, pady=(4, 0))
          self._map.set_tile_server("https://tile.openstreetmap.org/{z}/{x}/{y}.png")
          self._map.set_position(53.2007, 45.0046)  # default: Penza
          self._map.set_zoom(6)
          self._map.add_left_click_map_command(self._on_map_click)

          # Custom macOS scroll-zoom fix
          def custom_mouse_zoom(event):
              relative_mouse_x = event.x / self._map.width
              relative_mouse_y = event.y / self._map.height
              
              raw_delta = event.delta
              if sys.platform == "darwin":
                  if abs(raw_delta) >= 120:
                      step = raw_delta / 120.0
                  else:
                      step = raw_delta * 0.2
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

          self._map.canvas.bind("<MouseWheel>", custom_mouse_zoom)
          self._map.canvas.bind("<Button-4>", custom_mouse_zoom)
          self._map.canvas.bind("<Button-5>", custom_mouse_zoom)
  ```

  (Ensure `import sys` is present at the top of the file - it is).

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
  git commit -m "fix: resolve macOS map scrolling zoom jumps and lack of trackpad responsiveness"
  ```
