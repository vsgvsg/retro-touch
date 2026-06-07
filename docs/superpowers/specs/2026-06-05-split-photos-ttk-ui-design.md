# Split Photos — Tkinter/ttk UI Port

**Date:** 2026-06-05
**Branch:** `feat/split-photos-ttk-ui`
**Goal:** Rework `split_photos.py`'s interactive editor from OpenCV HighGUI to a
Tkinter/ttk GUI that matches the look and interaction patterns of
`face_pipeline.py`'s `LabelerApp` / `AgeLabelerApp` / `PhotoReviewApp` (label,
ages, match).

## Why

`split_photos.py` is the only one of the three tools still on raw OpenCV
HighGUI: a status banner strip + boxes painted onto a numpy image, keyboard-only
chrome. The face pipeline GUIs share a polished ttk theme (clam, `#5a6cf0`
accent, `#fafaff` bg), rounded photo crops, a clickable progress bar, themed
buttons, and color-coded state. This port brings the splitter up to that bar so
the whole pipeline feels like one product.

## Chosen direction (from brainstorming)

- **Full Tkinter port** (not a hybrid or restyled-OpenCV). The editor becomes a
  ttk app like the others.
- **Layout A — canvas + right sidebar.** Scan editing canvas on the left; a ttk
  sidebar on the right with a per-box list + an active-box control panel.
  Mirrors `PhotoReviewApp`'s photo-pane + scrollable-rows split.
- **Native Tk Canvas items.** The scan is one background `PhotoImage`; each box
  is drawn as native Canvas items (polygon outline, corner-handle ovals,
  orientation arrow). No per-frame full-image redraw; crisp, easy hit-testing.
- **Inline preview.** The crop of the active box renders as a rounded photo
  (`crop_to_round_photo`) inside the sidebar's ACTIVE BOX panel, live as you
  edit. No separate preview window.

## Architecture

### Preserved (pure layer — untouched)

These stay exactly as-is; existing tests are the regression guard:

- `Box` dataclass + `to_dict`/`from_dict`
- `detect_photos`, `normalize_rect`, `_estimate_background`
- `crop_box`
- Sidecar I/O: `sidecar_path`, `save_metadata`, `load_metadata`
- Geometry helpers: `disp_to_full`, `full_to_disp`, `point_in_box`, `_to_local`,
  `grab_handle`, `resize_box`, `_orient_arrow`

Invariant kept: **box geometry is always stored in full-resolution scan
coordinates; display scale is applied only at render/mouse time.** The `boxes`
list is the single source of truth — no display-coord state is persisted.

### Replaced

- OpenCV `Editor` class → `SplitterApp` + `CanvasEditor`
- `render()`, `scale_base()` → native Canvas drawing in `CanvasEditor`
- OpenCV `main()` chrome / HighGUI key constants (`WINDOW`, `BANNER_H`,
  `HANDLE_R` semantics, `ARROW_*`) → Tk event bindings

### Theme reuse without cross-import

Per the project's "one file per tool, never cross-import" rule, the small shared
helpers are **copied** into `split_photos.py` (not imported from
`face_pipeline.py`):

- theme constants: `ACCENT`, `BG`, `CARD_BORDER`, `STATE_COLORS`
- `_install_theme(root)` (clam + Title/Sub/Primary styles)
- `crop_to_round_photo(crop, cell, radius)` (with its degrade-to-square
  try/except)

## Components (new)

### `SplitterApp`

The `tk.Tk()` root, themed via the local `_install_theme`. Responsibilities:

- Owns the scan list and current index.
- Top header: title (`Title.TLabel`, scan filename), subtitle (`Sub.TLabel`,
  e.g. "4 photos · 2 cropped"), and a **clickable** `ttk.Progressbar` showing
  `scan i / n` — clicking jumps scans (mirrors `LabelerApp._on_progress_click`,
  committing/saving current edits first).
- Bottom action bar: `+ Re-detect`, `Crop all`, `← Prev`, `Next →`
  (Primary-styled). Prev/Next save metadata first (today's `=`/`-` semantics);
  a separate crop-then-next path preserves today's Enter behavior.

### `CanvasEditor`

A `tk.Canvas` embedded in the left pane. Responsibilities:

- Background = the scan as one `PhotoImage` at the fit-to-window scale
  (longest side ~1000px default, same as today).
- Each box rendered as native Canvas items: polygon outline (accent when active,
  amber/grey otherwise), corner-handle ovals on the active box, and the
  orientation arrow (`_orient_arrow`).
- Mouse: drag inside a box = move, drag edge/corner = resize, drag empty = new
  box. Reuses `point_in_box`, `grab_handle`, `resize_box`. Display↔full coord
  conversion via the existing helpers.
- Keyboard (bound on the canvas/root): arrows + `hjkl` nudge active box,
  `[`/`]` orient, `,` `.` `<` `>` tilt, `x`/Delete delete, `n`/Tab next box.
- Selection syncs both ways with the sidebar PHOTOS list.

### Sidebar (ttk)

- **PHOTOS list** — one row per box: a state dot + `#id` + size (e.g. `4×6`-ish
  WxH) + a short status word. Dot color from `box_state` via `STATE_COLORS`
  palette. Clicking a row selects that box on the canvas. Scrollable on demand
  (the `_sync_scrollbar` pack/forget pattern) if many boxes.
- **ACTIVE BOX panel** — angle stepper (◀ value ▶, ±0.5° / ±5°), orientation
  stepper (◀ ▲top ▶), inline rounded-crop **preview** (live), `Delete box`
  button. Steppers call the same `_nudge_angle` / orientation logic as the keys.

### `box_state` (new pure helper, TDD-tested)

```
box_state(box, scan_shape, active=False) -> "cropped" | "attention" | "editing"
```

Analogous to `face_pipeline.face_state`. Precedence:

- `"editing"` if `active` is True.
- `"attention"` if the box is zero/negative size OR extends off-canvas (any
  corner outside `scan_shape`) — i.e. would fail or mis-crop.
- `"cropped"` if `box.output` is set (already exported) and otherwise fine.
- default `"cropped"`-vs-neutral: a normal not-yet-cropped box is the neutral/
  base color. (Map states to colors: editing→`ACCENT`, attention→amber
  `#d8a23a`, cropped→green `#2faf6a`, neutral→grey.)

This single helper drives both the row dot and any canvas tint so they always
agree (same discipline as `face_state`).

## Data flow

Unchanged contract — the `*.photos.json` sidecar is the only persistence:

1. **Load:** `main()` lists scans; per scan `load_metadata()` else
   `detect_photos()`. Never re-detect over existing manual edits.
2. **Edit:** mouse/keyboard/steppers mutate `Box` objects in full-res coords;
   canvas items, the matching sidebar row, and the inline preview all recompute
   from the same `Box`.
3. **Save:** `save_metadata()` on save/Prev/Next; `crop_all()` writes
   `extracted/<stem>_NN.jpg` + sidecar via `crop_box`, exactly as today.

## Error handling

- **No-display / Tk-less env:** the system `/usr/bin/python3` Tk can't open a
  window on this macOS. `main()` wraps `tk.Tk()` in try/except `tk.TclError`,
  prints the "run via `.venv/bin/python`" hint, and exits non-zero instead of
  crashing.
- **Zero-size / off-canvas box:** `crop_box` already raises `ValueError`; the
  preview shows an "invalid box" placeholder and `crop_all` skips + warns
  (today's behavior). Such boxes read amber "attention" in the list.
- **Unreadable scan:** skip + warn (today's behavior).
- **`crop_to_round_photo` failure:** degrades to a square (copied verbatim).

## Testing

Follows the project convention: pure functions get TDD tests; the interactive
GUI is verified manually.

- **New TDD test** for `box_state` (cropped / attention / editing / neutral),
  added to `tests/test_split_photos.py`.
- **All existing pure tests** (Box roundtrip, detector, cropper, geometry) must
  keep passing untouched — the port's regression guard.
- **GUI wiring smoke test** (headless, no human), per the CLAUDE.md pattern:
  ```
  .venv/bin/python -c "import split_photos as sp; \
    app=sp.SplitterApp(...); app._show(); \
    app.root.after(100, app.root.destroy); app.root.mainloop()"
  ```
  Builds + renders the first screen then auto-closes, catching wiring errors.
  Full drag/resize/preview interaction still needs a human at the GUI.
- **Run GUI via the venv** (Homebrew Python 3.13 + Tk 9.0), like the rest of the
  pipeline.

## Docs to update

- `CLAUDE.md` "Photo Scan Splitter" + Commands: describe the ttk GUI, the venv
  requirement, and the new mouse/stepper/sidebar model (drop the OpenCV HighGUI
  gotchas that no longer apply to the editor; keep detector/crop notes).
- Controls help text printed by `main()`.

## Out of scope (YAGNI)

- No change to the detector, cropper, or sidecar format.
- No filmstrip / gallery of all crops (Layout C was rejected).
- No separate preview window (inline only).
- No cross-import between the tools.
