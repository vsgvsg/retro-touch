# Scrollable Places and Deletion Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make the memorized places list scrollable, displaying all cache entries sorted by usage count, and allow removing entries from the list via Cmd+click (or Ctrl+click).

**Architecture:**
- Add a `remove` method to `LocationCache` in `exif_pipeline.py`.
- Wrap the chips frame in a horizontal scrollable `tk.Canvas` with mousewheel scrolling propagation.
- Update `_refresh_chips` to display all entries, bind mousewheel to each chip button, and bind Cmd/Ctrl+click to trigger removal and return `"break"`.

**Tech Stack:** Python 3, Tkinter / ttk, pytest

---

### Task 1: Write TDD tests for LocationCache.remove and Cmd+click deletion

**Files:**
- Modify: `tests/test_exif_pipeline.py`

- [ ] **Step 1: Write the failing test**

Add `test_location_cache_remove` and `test_tagger_app_remove_chip` at the end of `tests/test_exif_pipeline.py`:

```python
def test_location_cache_remove(tmp_path):
    import json
    locs_file = tmp_path / "locations.json"
    locs_file.write_text(json.dumps({
        "locations": [
            {"lat": 1.0, "lng": 2.0, "city": "A", "state": "", "country": "X", "use_count": 5},
            {"lat": 3.0, "lng": 4.0, "city": "B", "state": "", "country": "Y", "use_count": 2}
        ]
    }), encoding="utf-8")
    
    cache = ep.LocationCache(locs_file)
    assert len(cache.all_entries()) == 2
    
    # Remove entry A
    cache.remove({"lat": 1.0, "lng": 2.0})
    
    # Reload and verify
    cache2 = ep.LocationCache(locs_file)
    entries = cache2.all_entries()
    assert len(entries) == 1
    assert entries[0]["city"] == "B"


def test_tagger_app_remove_chip(tmp_path, monkeypatch):
    from PIL import Image
    import json
    
    jpg_path = tmp_path / "img1.jpg"
    img = Image.new("RGB", (100, 100), color="blue")
    img.save(jpg_path, "JPEG")
    
    locs_file = tmp_path / "locations.json"
    locs_file.write_text(json.dumps({
        "locations": [
            {"lat": 10.0, "lng": 20.0, "display_name": "Place X", "city": "Place X", "state": "", "country": "US", "use_count": 1}
        ]
    }), encoding="utf-8")
    
    monkeypatch.setattr(ep.NominatimClient, "_get", lambda self, url: [])
    
    import tkintermapview
    class MockMarker:
        def delete(self):
            pass
    monkeypatch.setattr(tkintermapview.TkinterMapView, "set_marker", lambda self, lat, lng, **kwargs: MockMarker())
    
    app = ep.TaggerApp([str(jpg_path)], extracted_dir=str(tmp_path))
    try:
        # Check initial chips count (should be 1)
        children = app._chips_frame.winfo_children()
        assert len(children) == 1
        assert children[0].cget("text") == "Place X"
        
        # Simulate Cmd+Click removal
        app._remove_cache_entry({"lat": 10.0, "lng": 20.0})
        
        # Check chips count again (should be 0)
        children = app._chips_frame.winfo_children()
        assert len(children) == 0
        
        # Verify locations.json has no entries
        assert len(app.cache.all_entries()) == 0
    finally:
        app.destroy()
```

- [ ] **Step 2: Run test to verify it fails**

Run: `~/.venv/bin/python -m pytest tests/test_exif_pipeline.py -k "test_location_cache_remove or test_tagger_app_remove_chip"`
Expected: FAIL (AttributeError: 'LocationCache' object has no attribute 'remove' or similar)

- [ ] **Step 3: Write minimal implementation in `exif_pipeline.py`**

- Modify: `exif_pipeline.py`
  - In `LocationCache`, add the `remove(self, entry)` method.
  - In `TaggerApp._build_sidebar`, wrap `self._chips_frame` in a horizontal scrollable canvas `self._chips_canvas`.
  - In `TaggerApp._refresh_chips`, display all entries, bind mousewheel event, and bind Command/Control clicks to delete entries.
  - In `TaggerApp`, implement `_remove_cache_entry(self, entry)`.

- [ ] **Step 4: Run test to verify it passes**

Run: `~/.venv/bin/python -m pytest tests/test_exif_pipeline.py -k "test_location_cache_remove or test_tagger_app_remove_chip"`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add tests/test_exif_pipeline.py exif_pipeline.py
git commit -m "feat: make memorized places scrollable and add Cmd+click deletion"
```
