# Design Spec: Copy Previous Photo EXIF Metadata

- **Date**: 2026-06-12
- **Author**: Antigravity
- **Status**: Approved

## Requirements

1. **Copy Button**: Add a button to copy date (year and month) and location info from the previous photo in the sorted sequence.
2. **Placement**: Place the button to the left of the "✓ Save & Next" button in the bottom sidebar row.
3. **Keyboard Shortcut**: Bind the `c` key (when not focused in a text entry) to execute the copy action.
4. **Overwrite Behavior**: Overwrite any existing date and/or location data in the current photo's input fields/map/variables with the copied values.
5. **Partial Data**: If the previous photo only has date or location, copy what is present and do not clear/overwrite the other fields.
6. **Dynamic Enablement**: Disable the button for the first photo (`self.idx == 0`) or if the previous photo's sidecar has no date or location metadata. Otherwise, enable it.
7. **Map Sync**: When location is copied, center/fly the map to the copied coordinates and place a marker.

## Proposed Changes

### `exif_pipeline.py`

#### 1. Button UI
Add the `_copy_prev_btn` instance variable during sidebar creation in `_build_sidebar()`:
```python
self._copy_prev_btn = ttk.Button(btn_row, text="Copy Prev", command=self._copy_previous)
self._copy_prev_btn.pack(side=tk.LEFT, padx=(0, 4))
```

#### 2. Key Bindings
Bind the `c` key in `_bind_keys()`:
```python
elif c == "c" and not in_entry:
    if self.idx > 0:
        self._copy_previous()
```
And add it to `_show_shortcuts()`:
```python
("c",           "Copy from previous photo"),
```

#### 3. State Update Logic
In `_load_photo(self, idx)`:
* Disable the button if `idx == 0`.
* If `idx > 0`, inspect the previous sidecar file (`{prev_stem}.faces.json`).
* If `taken` or `location` fields exist in the sidecar, enable the button. Otherwise, disable it.

```python
# Enable/disable "Copy Prev" button based on previous photo's sidecar presence
if self.idx <= 0:
    self._copy_prev_btn.configure(state=tk.DISABLED)
else:
    prev_jpg = self.photos[self.idx - 1]
    prev_stem = pathlib.Path(prev_jpg).stem
    prev_sc_path = str(self.extracted_dir / f"{prev_stem}.faces.json")
    prev_sc = load_sidecar(prev_sc_path)
    if prev_sc and (prev_sc.get("taken") or prev_sc.get("location")):
        self._copy_prev_btn.configure(state=tk.NORMAL)
    else:
        self._copy_prev_btn.configure(state=tk.DISABLED)
```

#### 4. Copying Logic
Implement `_copy_previous(self)`:
* Load previous sidecar.
* Copy year/month (if present).
* Copy location details and center the map (if present).

```python
def _copy_previous(self):
    if self.idx <= 0:
        return
    prev_jpg = self.photos[self.idx - 1]
    prev_stem = pathlib.Path(prev_jpg).stem
    prev_sc_path = str(self.extracted_dir / f"{prev_stem}.faces.json")
    prev_sc = load_sidecar(prev_sc_path)
    if not prev_sc:
        return

    # Overwrite taken info if present
    if prev_sc.get("taken"):
        taken = prev_sc["taken"]
        if taken.get("year"):
            self._year_var.set(str(taken["year"]))
        month = taken.get("month")
        months = ["", "Jan", "Feb", "Mar", "Apr", "May", "Jun",
                  "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
        if month is not None and (1 <= month <= 12):
            self._month_var.set(months[month])
        else:
            self._month_var.set("")

    # Overwrite location info if present
    if prev_sc.get("location"):
        loc = prev_sc["location"]
        self._set_location(
            loc["lat"], loc["lng"], loc["city"],
            loc.get("state", ""), loc["country"],
            loc["display_name"], fly=True
        )
```

## Testing Strategy

### Unit / Integration Tests
We will add automated tests in `tests/test_exif_pipeline.py` to verify:
1. The button is disabled on the first photo.
2. The button is enabled on the second photo if the first photo has saved metadata.
3. Clicking the button or simulating the `c` key successfully copies metadata.
4. Verification that fields are NOT overwritten/cleared if the previous photo lacks that specific piece of metadata.
