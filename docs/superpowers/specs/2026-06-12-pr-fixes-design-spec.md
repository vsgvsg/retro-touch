# Design Spec: PR Review Fixes

- **Date**: 2026-06-12
- **Author**: Antigravity
- **Status**: Approved

## Requirements

1. **Clear Location Leak**: Reset the location variables (`self._lat`, `self._lng`, `self._city`, `self._state`, `self._country`, `self._display_name`), map marker, and display label before loading any new photo (respecting the filename hint autofill rule if present).
2. **Thread-Safe Geocoder**: Enforce thread safety in `NominatimClient` using a reentrant lock (`threading.RLock()`) for access to `self._last` and `self._cache`.
3. **App Teardown Safety**: Add a `self._destroyed` check to `_set_location()` to prevent `TclError` from background threads during teardown.
4. **Minor Enhancements**:
   * Prune completed thread objects inside `_bg_run()`.
   * Support case-insensitive keybind check for shortcut `c` (by testing `c.lower() == "c"`).
   * Bind Linux horizontal scroll events (`<Button-6>`/`<Button-7>`) to the chips scrollable container and buttons.

## Proposed Changes

### `exif_pipeline.py`

#### 1. Clear Location State
Implement `_clear_location_state(self)`:
```python
    def _clear_location_state(self):
        self._lat = None
        self._lng = None
        self._city = ""
        self._state = ""
        self._country = ""
        self._display_name = ""
        self._loc_display.set("")
        if self._pin:
            try:
                self._pin.delete()
            except Exception:
                pass
            self._pin = None
```
And invoke it at the start of `_load_photo(self, idx)`:
```python
    def _load_photo(self, idx: int):
        if not self.photos:
            return
        self._clear_location_state()
        self.idx = idx % len(self.photos)
        jpg = self.photos[self.idx]
```

#### 2. Thread-Safe `NominatimClient`
Add lock support to `NominatimClient`:
```python
class NominatimClient:
    """Thin Nominatim geocoder — enforces 1.1 s between requests, caches by query."""

    BASE = "https://nominatim.openstreetmap.org"
    UA = "retro-touch/1.0 (photo-archival-tool)"

    def __init__(self):
        self._last = 0.0
        self._cache: dict[str, list] = {}
        self._lock = threading.RLock()

    def _get(self, url: str) -> dict | list:
        with self._lock:
            elapsed = time.monotonic() - self._last
            if elapsed < 1.1:
                time.sleep(1.1 - elapsed)
            try:
                req = urllib.request.Request(url, headers={"User-Agent": self.UA})
                with urllib.request.urlopen(req, timeout=8) as r:
                    return _json.loads(r.read())
            finally:
                self._last = time.monotonic()
```

#### 3. Teardown Safety in `_set_location`
```python
    def _set_location(self, lat: float, lng: float, city: str, state: str,
                      country: str, display_name: str, fly: bool = False):
        if self._destroyed:
            return
        self._lat, self._lng = lat, lng
        # ... rest of method ...
```

#### 4. Enhancements
* In `_bg_run(self, target, *args)`:
  ```python
    def _bg_run(self, target, *args):
        """Spawn a daemon thread, track it for clean teardown."""
        self._bg_threads = [t for t in self._bg_threads if t.is_alive()]
        t = threading.Thread(target=target, args=args, daemon=True)
        self._bg_threads.append(t)
        t.start()
  ```
* In `_bind_keys(self)`:
  ```python
                elif c.lower() == "c":
                    if self.idx > 0:
                        self._copy_previous()
  ```
* Horizontal Linux scrollbinds:
  Bind `<Button-6>` and `<Button-7>` events in `_build_sidebar()` and `_refresh_chips()`.

## Testing Strategy

Add tests to `tests/test_exif_pipeline.py` verifying:
1. `_clear_location_state` properly resets all UI/state variables.
2. Photo navigation clears location state between loads.
3. Thread safety logic executes without errors.
