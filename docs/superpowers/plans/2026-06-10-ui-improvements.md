# UI Improvements Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Improve RetroTouch EXIF tagger UI map sizing, Nominatim search candidate limit, and add keyboard shortcuts helper button.

**Architecture:** Modify existing Tkinter widgets configuration and packing rules in `exif_pipeline.py` to allow vertical resizing and larger default map dimensions, increase Nominatim search limits to 10 candidates, and add a Shortcuts button to the bottom sidebar row.

**Tech Stack:** Python 3, Tkinter, tkintermapview

---

### Task 1: Increase Nominatim search candidate limit to 10

**Files:**
- Modify: `exif_pipeline.py`
- Modify: `tests/test_exif_pipeline.py`

- [ ] **Step 1: Write a failing test for Nominatim search candidate limit**

  Append this test at the end of `tests/test_exif_pipeline.py`:

  ```python
  def test_nominatim_client_search_limit_10(monkeypatch):
      from exif_pipeline import NominatimClient
      client = NominatimClient()
      requested_url = ""
      def mock_get(url):
          nonlocal requested_url
          requested_url = url
          return []
      monkeypatch.setattr(client, "_get", mock_get)
      client.search("Rome")
      assert "limit=10" in requested_url
  ```

- [ ] **Step 2: Run test to verify it fails**

  Run:
  ```bash
  ~/.venv/bin/python -m pytest tests/test_exif_pipeline.py::test_nominatim_client_search_limit_10 -v
  ```
  Expected: FAIL (assertion error: "limit=10" not in requested_url, since it is currently limit=5).

- [ ] **Step 3: Modify implementation**

  Change `NominatimClient.search()` in `exif_pipeline.py` (around lines 163-164) to set limit to 10:

  ```python
          params = urllib.parse.urlencode({"q": query, "format": "json",
                                           "addressdetails": 1, "limit": 10})
  ```

- [ ] **Step 4: Run test to verify it passes**

  Run:
  ```bash
  ~/.venv/bin/python -m pytest tests/test_exif_pipeline.py::test_nominatim_client_search_limit_10 -v
  ```
  Expected: PASS

- [ ] **Step 5: Commit**

  Run:
  ```bash
  git add exif_pipeline.py tests/test_exif_pipeline.py
  git commit -m "feat: increase Nominatim search candidate limit to 10"
  ```

---

### Task 2: Map widget sizing and vertical expansion

**Files:**
- Modify: `exif_pipeline.py`
- Modify: `tests/test_exif_pipeline.py`

- [ ] **Step 1: Write a failing test for map sizing and location frame pack configuration**

  Append this test at the end of `tests/test_exif_pipeline.py`:

  ```python
  def test_tagger_app_map_sizing_and_vertical_expansion(tmp_path, monkeypatch):
      from PIL import Image
      import json
      
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
          # Verify map default height is 360
          assert app._map.height == 360
          
          # Verify loc_frame is packed with expansion (expand=True)
          # We can check pack_info of its master
          info = app._map.master.pack_info()
          assert info.get("expand") == "1" or info.get("expand") is True
          assert info.get("fill") == "both"
      finally:
          app.destroy()
  ```

- [ ] **Step 2: Run test to verify it fails**

  Run:
  ```bash
  ~/.venv/bin/python -m pytest tests/test_exif_pipeline.py::test_tagger_app_map_sizing_and_vertical_expansion -v
  ```
  Expected: FAIL (assertion error: app._map.height == 260 != 360).

- [ ] **Step 3: Modify implementation**

  In `exif_pipeline.py`:
  
  1. Change packing options of `loc_frame` (around line 579):
     ```python
             # ── LOCATION section ──
             loc_frame = ttk.LabelFrame(sb, text="Location", padding=8)
             loc_frame.pack(fill=tk.BOTH, expand=True, padx=8, pady=4)
     ```
  
  2. Change map widget height (around line 595):
     ```python
             # Map widget
             self._map = tkintermapview.TkinterMapView(loc_frame, width=360, height=360,
                                                       corner_radius=0)
     ```

- [ ] **Step 4: Run test to verify it passes**

  Run:
  ```bash
  ~/.venv/bin/python -m pytest tests/test_exif_pipeline.py::test_tagger_app_map_sizing_and_vertical_expansion -v
  ```
  Expected: PASS

- [ ] **Step 5: Commit**

  Run:
  ```bash
  git add exif_pipeline.py tests/test_exif_pipeline.py
  git commit -m "style: increase default map height to 360 and enable vertical resizing"
  ```

---

### Task 3: Shortcuts helper button

**Files:**
- Modify: `exif_pipeline.py`
- Modify: `tests/test_exif_pipeline.py`

- [ ] **Step 1: Write a failing test for the shortcuts helper button**

  Append this test at the end of `tests/test_exif_pipeline.py`:

  ```python
  def test_tagger_app_shortcuts_button(tmp_path, monkeypatch):
      from PIL import Image
      import json
      
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
          # Verify the shortcuts button exists
          assert hasattr(app, "_shortcuts_btn")
          assert app._shortcuts_btn.cget("text") == "?"
          
          # Verify click invokes show_shortcuts
          called = False
          def mock_show_shortcuts():
              nonlocal called
              called = True
          app._show_shortcuts = mock_show_shortcuts
          app._shortcuts_btn.invoke()
          assert called is True
      finally:
          app.destroy()
  ```

- [ ] **Step 2: Run test to verify it fails**

  Run:
  ```bash
  ~/.venv/bin/python -m pytest tests/test_exif_pipeline.py::test_tagger_app_shortcuts_button -v
  ```
  Expected: FAIL (AttributeError: 'TaggerApp' object has no attribute '_shortcuts_btn').

- [ ] **Step 3: Modify implementation**

  In `exif_pipeline.py` (around lines 608-614), add the shortcuts button packed on the right side of the bottom button row of the sidebar:

  ```python
          # Buttons
          btn_row = ttk.Frame(sb)
          btn_row.pack(fill=tk.X, padx=8, pady=8)
          self._shortcuts_btn = ttk.Button(btn_row, text="?", width=3,
                                           command=self._show_shortcuts)
          self._shortcuts_btn.pack(side=tk.RIGHT, padx=(4, 0))
          ttk.Button(btn_row, text="Skip →",
                     command=self._next).pack(side=tk.RIGHT, padx=(4, 0))
          ttk.Button(btn_row, text="✓ Save & Next", style="Accent.TButton",
                     command=self._save_and_next).pack(side=tk.LEFT, expand=True, fill=tk.X)
  ```

- [ ] **Step 4: Run test to verify it passes**

  Run:
  ```bash
  ~/.venv/bin/python -m pytest tests/test_exif_pipeline.py::test_tagger_app_shortcuts_button -v
  ```
  Expected: PASS

- [ ] **Step 5: Run full test suite**

  Run:
  ```bash
  ~/.venv/bin/python -m pytest tests/test_exif_pipeline.py -v
  ```
  Expected: All 62 tests pass.

- [ ] **Step 6: Commit**

  Run:
  ```bash
  git add exif_pipeline.py tests/test_exif_pipeline.py
  git commit -m "feat: add shortcuts button next to Skip button in sidebar"
  ```
