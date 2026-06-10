# Design Spec: UI Layout and Search Improvements for Exif Pipeline Tagger

## Goal
Improve user experience in the `exif_pipeline.py` Tkinter TaggerApp by making the map more responsive to window resizing, increasing the chances of finding smaller cities via geocoding search, and adding an easily accessible button to display keyboard shortcuts.

---

## 1. Map Sizing & Expansion
### Problem
The map widget is constrained to a fixed 260px height and does not grow or shrink when the main application window is resized. This limits visibility when screen space is abundant.

### Solution
- Set `loc_frame` to pack with `fill=tk.BOTH, expand=True` inside the sidebar.
- Change the default height of the `TkinterMapView` from 260 to 360.
- When the window is expanded, the map will grow both horizontally and vertically, filling the available space.

---

## 2. Nominatim Search for Smaller Cities
### Problem
Searching for smaller towns or villages often yields no results in the top 5 matches because they are overshadowed by larger cities/regions with similar names.

### Solution
- Increase the `limit` query parameter from `5` to `10` in `NominatimClient.search()`.
- Return more candidates, improving the likelihood of finding the intended smaller city.

---

## 3. Keyboard Shortcuts Button
### Problem
Users cannot easily see keyboard shortcuts without pressing `?` or `F1` on the keyboard, which they may not know about.

### Solution
- Add a small `?` button next to the "Skip →" button in the bottom button row of the sidebar.
- When clicked, this button invokes `self._show_shortcuts()`, displaying the shortcuts dialog window.

---

## 4. Verification and Testing
- Add unit tests verifying `NominatimClient.search` calls utilize `limit=10`.
- Verify the GUI widget structure updates do not break existing Tkinter initialization and cleanup flows.
- Run the full test suite (`python -m pytest tests/test_exif_pipeline.py`).
