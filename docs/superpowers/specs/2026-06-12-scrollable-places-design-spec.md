# Design Spec: Scrollable Memorized Places and Deletion

- **Date**: 2026-06-12
- **Author**: Antigravity
- **Status**: Approved

## Requirements

1. **Scrollable Places List**: Make the memorized places list scrollable. It should display all entries in the location cache sorted by usage frequency.
2. **Horizontal Scrolling**: Keep it horizontal and let it scroll using mouse wheel / trackpad.
3. **Deletion (Cmd+click)**: Clicking an item with Cmd (or Ctrl) held down should remove it from the cache and locations list.

## Proposed Changes

### `exif_pipeline.py`

#### 1. LocationCache
Add `remove(self, entry)`:
```python
    def remove(self, entry: dict):
        """Remove a location entry from the cache by matching coordinates."""
        self._data = [
            e for e in self._data
            if not (isinstance(e, dict) and
                    e.get("lat") == entry.get("lat") and
                    e.get("lng") == entry.get("lng"))
        ]
        self._save()
```

#### 2. Scrollable Chips Frame UI
In `_build_sidebar()`, change how `self._chips_frame` is created. Wrap it in a scrollable canvas:
```python
        # Scrollable frequent chips container
        self._chips_container = ttk.Frame(loc_frame)
        self._chips_container.pack(fill=tk.X, pady=(4, 0))

        self._chips_canvas = tk.Canvas(self._chips_container, height=36, bg=BG, highlightthickness=0)
        self._chips_canvas.pack(fill=tk.X, side=tk.TOP, expand=True)

        self._chips_frame = ttk.Frame(self._chips_canvas)
        self._chips_canvas.create_window((0, 0), window=self._chips_frame, anchor="nw")

        def _on_chips_configure(event):
            self._chips_canvas.configure(scrollregion=self._chips_canvas.bbox("all"))
        self._chips_frame.bind("<Configure>", _on_chips_configure)

        def _on_chips_wheel(event):
            if sys.platform == "darwin":
                self._chips_canvas.xview_scroll(-1 * event.delta, "units")
            else:
                self._chips_canvas.xview_scroll(-1 * (event.delta // 120), "units")
        self._chips_canvas.bind("<MouseWheel>", _on_chips_wheel)
        self._chips_frame.bind("<MouseWheel>", _on_chips_wheel)
```

#### 3. Refreshing and Rendering Chips
Update `_refresh_chips()`:
* Sort and fetch all entries: `sorted(self.cache.all_entries(), key=lambda e: e.get("use_count", 1), reverse=True)`
* Bind mouse wheel to each chip button.
* Bind `<Command-Button-1>` and `<Control-Button-1>` to call `_remove_cache_entry(entry)` and return `"break"`.

```python
    def _refresh_chips(self):
        for w in self._chips_frame.winfo_children():
            w.destroy()
        
        # Display all entries sorted by frequency
        entries = sorted(self.cache.all_entries(), key=lambda e: e.get("use_count", 1), reverse=True)
        
        def _on_chips_wheel(event):
            if sys.platform == "darwin":
                self._chips_canvas.xview_scroll(-1 * event.delta, "units")
            else:
                self._chips_canvas.xview_scroll(-1 * (event.delta // 120), "units")

        for entry in entries:
            name = entry.get("city") or entry.get("display_name", "?")
            btn = ttk.Button(self._chips_frame, text=name,
                             command=lambda e=entry: self._apply_cache_entry(e))
            btn.pack(side=tk.LEFT, padx=2, pady=2)
            
            # Bind scrolling to the button
            btn.bind("<MouseWheel>", _on_chips_wheel)
            
            # Bind Cmd/Ctrl+Click to delete
            btn.bind("<Command-Button-1>", lambda e, entry=entry: self._remove_cache_entry(entry))
            btn.bind("<Control-Button-1>", lambda e, entry=entry: self._remove_cache_entry(entry))
```

#### 4. Removal Handler
Implement `_remove_cache_entry()` in `TaggerApp`:
```python
    def _remove_cache_entry(self, entry: dict):
        self.cache.remove(entry)
        self._refresh_chips()
        return "break"
```

## Testing Strategy

### Unit / Integration Tests
We will add automated tests in `tests/test_exif_pipeline.py` to verify:
1. Deleting a location from `LocationCache` removes it and persists to file.
2. Clicking a chip without modifiers applies it.
3. Simulating a Cmd/Ctrl+Click on a chip removes the entry from the cache.
