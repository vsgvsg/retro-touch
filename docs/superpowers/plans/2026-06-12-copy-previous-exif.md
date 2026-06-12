# Copy Previous Photo EXIF Metadata Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a button and a shortcut (`c`) to copy date and location metadata from the previous photo in RetroTouch's EXIF tagging flow, overwriting existing fields in the UI.

**Architecture:** Update `TaggerApp` in `exif_pipeline.py` to add the button, bind the shortcut key, check previous photo sidecar data on photo load to dynamically enable/disable the button, and copy metadata fields when triggered.

**Tech Stack:** Python 3, Tkinter / ttk, pytest

---

### Task 1: Write TDD tests for copying metadata and button states

**Files:**
- Modify: `tests/test_exif_pipeline.py`

- [ ] **Step 1: Write the failing test**

Add `test_tagger_app_copy_previous` at the end of `tests/test_exif_pipeline.py`:

```python
def test_tagger_app_copy_previous(tmp_path, monkeypatch):
    from PIL import Image
    import json
    import tkinter as tk
    
    # Create three photos
    jpg1 = tmp_path / "img1.jpg"
    jpg2 = tmp_path / "img2.jpg"
    jpg3 = tmp_path / "img3.jpg"
    for p in (jpg1, jpg2, jpg3):
        img = Image.new("RGB", (100, 100), color="blue")
        img.save(p, "JPEG")
        
    locs_file = tmp_path / "locations.json"
    locs_file.write_text(json.dumps({"locations": []}), encoding="utf-8")
    
    # Write metadata sidecar for photo 1
    sc1 = {
        "taken": {"year": 1955, "month": 6, "source": "manual"},
        "location": {
            "lat": 40.7128, "lng": -74.0060,
            "display_name": "New York, USA",
            "city": "New York", "state": "New York", "country": "USA",
            "source": "manual"
        }
    }
    sc1_path = tmp_path / "img1.faces.json"
    sc1_path.write_text(json.dumps(sc1), encoding="utf-8")
    
    # Write metadata sidecar for photo 2 (empty/none)
    sc2 = {}
    sc2_path = tmp_path / "img2.faces.json"
    sc2_path.write_text(json.dumps(sc2), encoding="utf-8")
    
    # Write metadata sidecar for photo 3 (partial, location only)
    sc3 = {
        "location": {
            "lat": 34.0522, "lng": -118.2437,
            "display_name": "Los Angeles, USA",
            "city": "Los Angeles", "state": "California", "country": "USA",
            "source": "manual"
        }
    }
    sc3_path = tmp_path / "img3.faces.json"
    sc3_path.write_text(json.dumps(sc3), encoding="utf-8")

    monkeypatch.setattr(ep.NominatimClient, "_get", lambda self, url: [])
    
    app = ep.TaggerApp([str(jpg1), str(jpg2), str(jpg3)], extracted_dir=str(tmp_path))
    try:
        # 1. First photo (idx=0)
        assert app.idx == 0
        # Button should be disabled
        assert app._copy_prev_btn.cget("state") == tk.DISABLED
        
        # 2. Advance to photo 2 (idx=1)
        app._next()
        assert app.idx == 1
        # Button should be enabled since photo 1 has metadata
        assert app._copy_prev_btn.cget("state") == tk.NORMAL
        
        # Populate current fields with dummy data to verify overwrite
        app._year_var.set("2000")
        app._month_var.set("Oct")
        app._set_location(0.0, 0.0, "", "", "", "")
        
        # Click copy previous
        app._copy_previous()
        
        # Verify year and location were copied
        assert app._year_var.get() == "1955"
        assert app._month_var.get() == "Jun"
        assert app._lat == 40.7128
        assert app._lng == -74.0060
        assert app._city == "New York"
        
        # 3. Advance to photo 3 (idx=2)
        app._next()
        assert app.idx == 2
        # Button should be disabled since photo 2 sidecar is empty
        assert app._copy_prev_btn.cget("state") == tk.DISABLED
        
        # Fill in current fields
        app._year_var.set("2010")
        app._month_var.set("Dec")
        
        # Manually enable and call copy previous (simulating call from photo 3 where prev has only location)
        # Note: prev photo (photo 2) has NO metadata, but let's test if we copy from photo 1
        # Let's set index back to 1 (which will copy from photo 1 which has date + location)
        app.idx = 1
        app._copy_previous()
        assert app._year_var.get() == "1955"
        
        # Let's set idx to 2 (which copies from photo 2, but photo 2 has no metadata in sidecar, so nothing copies)
        app.idx = 2
        app._year_var.set("2010")
        app._copy_previous()
        assert app._year_var.get() == "2010"  # Unchanged
        
    finally:
        app.destroy()
```

- [ ] **Step 2: Run test to verify it fails**

Run: `~/.venv/bin/python -m pytest tests/test_exif_pipeline.py -k test_tagger_app_copy_previous`
Expected: FAIL (AttributeError: 'TaggerApp' object has no attribute '_copy_prev_btn')

- [ ] **Step 3: Write minimal implementation in `exif_pipeline.py`**

- Modify: `exif_pipeline.py`
  - In `_build_sidebar()`, add the button to `btn_row`.
  - In `_bind_keys()`, bind key `c` to call `self._copy_previous()`.
  - In `_show_shortcuts()`, add `c` key mapping to the popover lines.
  - In `_load_photo(self, idx)`, enable/disable the button depending on whether previous photo has metadata.
  - Add the `_copy_previous(self)` helper.

- [ ] **Step 4: Run test to verify it passes**

Run: `~/.venv/bin/python -m pytest tests/test_exif_pipeline.py -k test_tagger_app_copy_previous`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add tests/test_exif_pipeline.py exif_pipeline.py
git commit -m "feat: add copy previous photo exif metadata button and shortcut"
```
