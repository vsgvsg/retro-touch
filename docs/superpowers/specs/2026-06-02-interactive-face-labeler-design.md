# Interactive Face Labeler — Design

**Date:** 2026-06-02
**Status:** Approved (pending spec review)

## Goal

Replace the headless `label` stage of `face_pipeline.py` with an interactive
**Tkinter** labeler. For each face cluster, it shows a montage of that cluster's
face crops, lets the human type a name (or reuse a previously-entered one from a
list), and writes the result to the same `extracted/labels.json` the rest of the
pipeline already consumes.

## Non-goals

- No change to `detect`, `cluster`, or `match`, or to the `labels.json` schema
  (`{person_id: name}`).
- Not a web UI; not a per-face labeler. Labeling unit is the **cluster**.
- The Tkinter event loop is not unit-tested (human-operated UI, like the
  existing `Editor` HighGUI class). Pure helpers around it are TDD-tested.
- No new pip dependencies. Tkinter and PIL/Pillow are already available
  (Tk 8.5; Pillow installed transitively via insightface).

## Architecture

Everything stays in `face_pipeline.py` (one-file convention). A thin Tkinter UI
shell sits over pure, tested core functions.

### Pure helpers (TDD-tested)

- `cluster_face_index(images_dir, index) -> dict[str, list[dict]]`
  Gathers each real cluster's faces from the sidecars. Returns
  `{person_id: [{"image", "face_id", "bbox", "det_score"}, ...]}`.
  `"unassigned"` is excluded. Reuses existing `read_faces_json`.

- `crop_face(image, bbox) -> np.ndarray`
  `image[y1:y2, x1:x2]` with the box clamped to image bounds so a slightly
  out-of-range bbox cannot raise. Returns a BGR array.

- `pick_montage_faces(faces, k=9) -> list[dict]`
  The k highest-`det_score` faces for a cluster (best-quality / most-frontal
  first). Returns all of them if fewer than k.

- `build_montage(crops, cols=3, cell=128, pad=4) -> np.ndarray`
  Resize each crop to fit a `cell×cell` box (preserve aspect, letterbox on a
  gray background), tile left-to-right / top-to-bottom into a single numpy
  image with `pad` px between cells. Fewer than `cols²` crops → partial last
  row. Zero crops → a small gray placeholder.

- `previous_names(labels_map) -> list[str]`
  Sorted unique non-empty names already entered.

- `write_labels(images_dir, labels_map) -> str`
  Writes `extracted/labels.json` (the after-each save). Returns the path.

### UI shell (manual verification)

- `LabelerApp` (Tkinter) — owns the window, montage canvas, name `Entry`,
  previous-names `Listbox`, and Back/Skip/Next/Done buttons. Calls the pure
  helpers; holds no pixel logic of its own. The BGR→`ImageTk.PhotoImage`
  conversion (via PIL) lives here, not in `build_montage`.

### Orchestration

- `run_label(images_dir) -> int`
  1. `load_cache(images_dir)` (raises FileNotFoundError with guidance if absent).
  2. `cluster_face_index(...)`; if no real clusters → print
     "No clusters found. Run 'cluster' first." and return 1 (no window).
  3. Load any existing `labels.json` (resume / pre-fill names).
  4. Launch `LabelerApp`; return 0 on clean exit.
  `import tkinter` failure → `SystemExit` with an actionable message, consistent
  with the existing `hdbscan` / `insightface` guards.

## Crop & montage pipeline

Each face crop comes from its source `extracted/<image>`, sliced by the sidecar
`bbox`. The labeler reads each needed source image once and caches it in memory
for the session. Crops are cut with `crop_face`. `pick_montage_faces` selects
the best k; `build_montage` tiles them into one image. The montage array is
converted to a Tkinter image via PIL (`cv2` BGR→RGB → `PIL.Image` →
`ImageTk.PhotoImage`) inside `LabelerApp`. Every pixel operation upstream of that
conversion is pure and unit-testable on shapes and tiling without a display.

## UI & session flow

Window layout, top to bottom:
- **Title strip:** e.g. `Cluster person_001  (2 of 5) — 30 faces`.
- **Montage canvas:** the 3×3 (or partial) grid for the current cluster.
- **Name entry:** an `Entry`, pre-filled with this cluster's existing name if
  `labels.json` already had one. Focused on show.
- **Previous-names list:** a `Listbox` of `previous_names(labels_map)`. Single
  click fills the entry (reuse a name without retyping).
- **Buttons:** `Back`, `Skip`, `Next` (`Done` on the last cluster). Enter = Next.

State: an ordered list of cluster ids + a current index.

- **Next:** read entry. Non-empty → `labels_map[cluster] = name`. Empty → treat
  as skip (ensure cluster absent from `labels_map`). **Write `labels.json`
  immediately.** Advance; on the last cluster, finish and close.
- **Back:** save current entry as in Next, then step to the previous cluster
  (its entry shows the saved name).
- **Skip:** remove any name for this cluster from `labels_map`, write, advance.
- **Done / window close:** save current entry, write, exit cleanly.

Because the app saves after each step and loads `labels.json` on launch, closing
or crashing mid-session resumes where the names left off. Skipped / empty
clusters are simply absent from `labels.json`.

## Error handling

- Unreadable source image → that crop becomes a gray placeholder cell; the
  montage still renders and the session continues.
- Missing embedding cache → `run_label` exits before opening the window with the
  "Run 'detect' first" message (existing `load_cache` behavior).
- No real clusters → "No clusters found. Run 'cluster' first.", return 1.
- `import tkinter` failure → `SystemExit` with install/enable guidance.

## Testing strategy

Pure helpers get TDD tests (`tests/test_face_pipeline.py`, synthetic data):

- `cluster_face_index` — groups faces by cluster from fixture sidecars; excludes
  `unassigned`; carries bbox + det_score.
- `crop_face` — correct sub-array for an in-bounds bbox; clamps an out-of-bounds
  bbox without raising.
- `pick_montage_faces` — returns top-k by det_score, descending; returns all when
  fewer than k.
- `build_montage` — output shape for full grid and partial last row; zero crops →
  placeholder; cell count / padding math.
- `previous_names` — sorted unique non-empty names; ignores empty values.
- `write_labels` — writes the expected `{person_id: name}` JSON; round-trips.

Verified manually (per repo convention for human-operated UI):
- `LabelerApp` window, montage display, entry/listbox interaction, Back/Skip/
  Next/Done navigation, after-each save, and resume-from-existing-`labels.json`.

Runnable with the existing `python3 -m pytest tests/ -q`.

## Code conventions

- One file (`face_pipeline.py`).
- Pure functions (cluster gathering, cropping, montage tiling, name list, label
  I/O) get TDD tests; the `LabelerApp` Tkinter class is verified manually, like
  `split_photos.py`'s `Editor`.
- Bboxes and crops in full-resolution image coords; montage is the only resized
  representation and is display-only.
- Only the body of `run_label` is replaced. The old skeleton-writing helpers
  `scaffold_labels` and `_collect_persons_and_examples` are left in place (their
  tests still pass) but are no longer called by `run_label` — they are not the
  right shape for the labeler, which needs every face per cluster with bbox +
  det_score (hence the new `cluster_face_index`). Removing the now-unused helpers
  is out of scope for this change.
