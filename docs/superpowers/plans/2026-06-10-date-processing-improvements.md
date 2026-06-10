# Date Processing Propagation Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Use the previous manual year and month as the default values for the next untagged photo.

**Architecture:** Initialize `_last_saved_year` and `_last_saved_month` variables in `TaggerApp.__init__`. Set these when a photo is successfully saved via `_save_and_next`. Check and use these in `_autofill` when a photo does not have `taken` info in its sidecar.

**Tech Stack:** Python 3, Tkinter

---

### Task 1: Date propagation state and logic

**Files:**
- Modify: `exif_pipeline.py`
- Modify: `tests/test_exif_pipeline.py`

- [ ] **Step 1: Write a failing test for date propagation**

  Append this test at the end of `tests/test_exif_pipeline.py`:

  ```python
  def test_tagger_app_date_propagation(tmp_path, monkeypatch):
      from PIL import Image
      import json
      
      # Create two photos
      jpg1 = tmp_path / "img1.jpg"
      jpg2 = tmp_path / "img2.jpg"
      for p in (jpg1, jpg2):
          img = Image.new("RGB", (100, 100), color="blue")
          img.save(p, "JPEG")
          
      locs_file = tmp_path / "locations.json"
      locs_file.write_text(json.dumps({"locations": []}), encoding="utf-8")
      
      # Mock geocoder and reverse geocode
      monkeypatch.setattr(ep.NominatimClient, "_get", lambda self, url: [])
      
      app = ep.TaggerApp([str(jpg1), str(jpg2)], extracted_dir=str(tmp_path))
      try:
          # Set manual date
          app._year_var.set("1995")
          app._month_var.set("Jan")
          
          # Set location so it can be saved
          app._lat = 1.0
          app._lng = 2.0
          app._city = "City"
          app._state = "State"
          app._country = "Country"
          app._display_name = "City, State, Country"
          
          # Save & Next
          app._save_and_next()
          
          # Verify the app advanced to photo 2
          assert app.idx == 1
          
          # Verify photo 2 defaults to the previous manually entered year and month
          assert app._year_var.get() == "1995"
          assert app._month_var.get() == "Jan"
      finally:
          app.destroy()
  ```

- [ ] **Step 2: Run test to verify it fails**

  Run:
  ```bash
  ~/.venv/bin/python -m pytest tests/test_exif_pipeline.py::test_tagger_app_date_propagation -v
  ```
  Expected: FAIL (assertion error: app._year_var.get() == "" != "1995").

- [ ] **Step 3: Modify implementation**

  In `exif_pipeline.py`:

  1. Initialize date propagation fields in `TaggerApp.__init__` (around lines 507-508):
     ```python
              self._photo_img = None     # keep reference to avoid GC
              self._pin = None           # current map marker
              self._last_saved_year = None
              self._last_saved_month = None
     ```

  2. Capture saved date in `TaggerApp._save_and_next` (around line 893):
     ```python
             month = months.index(month_str) if month_str in months[1:] else None
             self._last_saved_year = year
             self._last_saved_month = month_str
     ```

  3. Update `TaggerApp._autofill` to use these defaults (around line 795):
     ```python
             self._month_var.set(months[month] if month else "")
         elif getattr(self, "_last_saved_year", None) is not None:
             self._year_var.set(str(self._last_saved_year))
             self._month_var.set(self._last_saved_month or "")
             # Still parse location hint from filename if present
             year, hint = parse_filename(pathlib.Path(self.photos[self.idx]).name)
             if hint:
                 self._search_var.set(hint)
                 self._bg_run(self._search_and_fly, hint)
         else:
             year, hint = parse_filename(pathlib.Path(self.photos[self.idx]).name)
             self._year_var.set(str(year) if year else "")
             self._month_var.set("")
             if hint:
                 self._search_var.set(hint)
                 self._bg_run(self._search_and_fly, hint)
     ```

- [ ] **Step 4: Run test to verify it passes**

  Run:
  ```bash
  ~/.venv/bin/python -m pytest tests/test_exif_pipeline.py::test_tagger_app_date_propagation -v
  ```
  Expected: PASS

- [ ] **Step 5: Run full test suite**

  Run:
  ```bash
  ~/.venv/bin/python -m pytest tests/test_exif_pipeline.py -v
  ```
  Expected: All 63 tests pass.

- [ ] **Step 6: Commit**

  Run:
  ```bash
  git add exif_pipeline.py tests/test_exif_pipeline.py
  git commit -m "feat: propagate last saved date as default for next untagged photo"
  ```
