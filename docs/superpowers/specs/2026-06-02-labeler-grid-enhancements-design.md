# Labeler Grid Enhancements — Design

**Date:** 2026-06-02
**Status:** Approved (pending spec review)

## Goal

Three enhancements to the interactive `LabelerApp` (`face_pipeline.py`):

1. **Show all crops** of a cluster in a scrollable grid (not just the top 9).
2. **Exclude a misclustered face** via Ctrl-Click on its crop — removes it from
   the cluster by setting its sidecar `cluster` to `"unassigned"` (persistent).
3. **Full-photo preview** — left-click a crop to open the whole uncropped source
   photo with a rectangle around that face, so small/blurry crops are
   identifiable in context.

## Non-goals

- No change to `detect`, `cluster`, `match`, `run_label`'s orchestration, or the
  `labels.json` / sidecar schemas (exclude reuses the existing `cluster` field).
- No undo for exclude (consistent with Skip / delete elsewhere in the tool).
- The Canvas/Toplevel widget wiring stays manual-verification (human-operated UI,
  like `split_photos.py`'s `Editor`). New pure helpers are TDD-tested.

## Architecture

All in `face_pipeline.py`. The change is concentrated in `LabelerApp`'s view
layer. The data model (`cluster_index`, `labels_map`, sidecars) is unchanged
except that **exclude mutates sidecars**.

### New pure helpers (TDD-tested)

- `grid_positions(n, cols=3) -> list[tuple[int, int]]`
  The `(row, col)` for each of n crops, row-major. Replaces `build_montage`'s
  tiling math.

- `scale_to_fit(w, h, max_dim) -> tuple[int, int, float]`
  Uniform downscale so the longer side ≤ `max_dim`. Never upscales
  (`scale <= 1.0`; if already small, returns `(w, h, 1.0)`). Returns
  `(new_w, new_h, scale)`. Used to fit the full photo into the preview window and
  to scale the bbox rectangle onto it.

- `exclude_face(images_dir, image, face_id) -> bool`
  Set that face's `cluster` to `"unassigned"` in its sidecar and write it
  immediately. Returns True if a matching face was found and updated, False
  otherwise. Reuses `read_faces_json` / `write_faces_json`.

### Removed (now dead)

`build_montage` and `pick_montage_faces` — plus their unit tests. The grid now
shows every crop as an individual widget, so the single-image tiler and the top-k
selector are unused. (`crop_face`, `cluster_face_index`, `previous_names`,
`write_labels` stay.)

`_fit_cell`'s per-cell letterbox math is **kept**, but moves into the new UI
helper `_crop_to_photo(crop, cell)` (which letterboxes a BGR crop to a `cell`×
`cell` box and returns a Tk `PhotoImage`). `_fit_cell` as a standalone function
goes away; its logic lives on inside `_crop_to_photo`. It has no unit test of its
own to remove (it was only exercised indirectly via `build_montage` tests).

### LabelerApp view rebuild

The single montage `Label` becomes a scrollable `Canvas` holding an inner
`Frame` of per-crop `Label` widgets:

- 3 columns, **all** faces of the cluster, in index-row order.
- A vertical `Scrollbar`; the canvas `scrollregion` tracks the inner frame size;
  mouse-wheel scrolls.
- Each crop widget retains its `(image, face_id, bbox)` and binds:
  - `<Button-1>` → full-photo preview.
  - `<Control-Button-1>` → exclude. macOS fallback: also bind `<Button-2>`
    (Tk can report Control-click as Button-2). The manual smoke test confirms
    which fires; drop the dead binding.
- `PhotoImage` refs held in a list so Tk doesn't garbage-collect them.

`crop_face` and `_source` (the per-session BGR image cache) are reused as-is. The
existing letterbox-into-a-cell logic moves into a small UI helper
`_crop_to_photo(crop, cell)` (BGR crop → fitted `PhotoImage`), used per crop.

## Grid build & exclude flow

**Build (per cluster), in `_show`:** clear the inner frame; for every face in
`cluster_index[cid]` (all of them, index-row order): `crop_face(_source(image),
bbox)`; if `_source` is None use a gray placeholder; fit to a ~128px cell via
`_crop_to_photo`; create a `Label`, grid it at `grid_positions[i]`, bind the
click handlers, and store `(image, face_id, bbox)` on/with the widget. Update the
canvas scrollregion. Title shows `Cluster <id> (i of n) — N faces`.

**Exclude (Ctrl-Click / Button-2 fallback):**
1. `exclude_face(images_dir, image, face_id)` → sidecar face `cluster =
   "unassigned"`, written now.
2. Remove that face from the in-memory `cluster_index[cid]` list (so it won't
   reappear if you revisit via Back).
3. Re-render the current cluster's grid and update the title count.

A placeholder cell (unreadable image) is a no-op on both click and Ctrl-click.
Excluding the last face leaves an empty grid; the cluster stays in the ordered
`cluster_ids` list so Next/Back indices don't shift. An entered name still
applies to whatever faces remain; exclude and naming are order-independent.

## Full-photo preview

Left-click a crop → open a `Toplevel` titled with the source filename:

- Load the full source via `_source`; `scale_to_fit(w, h, 900)`; downscale a copy.
- Draw a rectangle at the face's bbox scaled by the same factor (skip drawing if
  the bbox is degenerate / collapses).
- BGR→RGB→`PhotoImage`, shown in a `Label`.
- Closes on click, Escape, or the window close button.
- Only one preview at a time: opening a new one destroys the previous `Toplevel`
  (mirrors the `Editor`'s single-preview guard). Guard `destroyWindow`/destroy
  against an already-closed window.

## Error handling

- Unreadable source image → gray placeholder cell; click and Ctrl-click are
  no-ops on it.
- Degenerate bbox in preview → show the whole photo without a rectangle.
- `exclude_face` on a face_id not present in the sidecar → returns False; the UI
  still removes it from the in-memory list and re-renders (defensive).

## Testing strategy

Pure helpers get TDD tests (`tests/test_face_pipeline.py`, synthetic data):

- `grid_positions` — row-major positions for full and partial last row; n=0 → [].
- `scale_to_fit` — downscales an oversized dimension (longer side hits max_dim);
  returns scale 1.0 and unchanged dims when already within max_dim; square and
  non-square cases.
- `exclude_face` — flips a fixture sidecar face's `cluster` to `"unassigned"`,
  persists to disk; returns True; returns False for an unknown face_id; leaves
  other faces untouched.

Removed with their code: the `build_montage` tests (3) and the
`pick_montage_faces` tests (2). `_fit_cell` had no direct tests; its letterbox
logic survives inside `_crop_to_photo` (UI, manually verified).

Verified manually (human-operated UI):
- Scrollable all-crops grid; mouse-wheel + scrollbar.
- Left-click → full-photo preview with bbox rectangle; close behaviors; single
  preview at a time.
- Ctrl-Click → crop disappears, count updates, sidecar shows `"unassigned"`;
  confirm which binding fires on macOS and keep it.
- Naming still works alongside exclude; resume still works.

Run with `.venv/bin/python -m pytest tests/ -q` (the venv on Homebrew Python 3.13
with working Tk; the system Python's Tk 8.5 cannot open a window on this macOS).

## Code conventions

- One file (`face_pipeline.py`). Pure helpers TDD-tested; `LabelerApp` Tk widget
  code verified manually, like `Editor`.
- Bboxes/crops in full-resolution image coords; the cell-fit and preview
  downscale are display-only.
- Exclude reuses the existing sidecar `cluster` field and the
  `read_faces_json`/`write_faces_json` round-trip — no schema change.
