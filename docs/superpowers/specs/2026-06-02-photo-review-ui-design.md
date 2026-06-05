# Per-Photo Match Review UI — Design

**Date:** 2026-06-02
**Status:** Approved (pending spec review)

## Goal

After `match` scores faces, open an interactive **per-photo** review window
(`PhotoReviewApp`) so a human can confirm or correct who is in each photo:

1. For each photo, draw the detected faces' bounding boxes (from the
   `.faces.json` sidecar) as numbered rectangles, with a matching numbered text
   input per face. Editing a name and leaving the photo writes the override into
   the sidecar (`face["label"]`).
2. If a face is `unassigned` (no cluster) and the human types a name, assign it a
   cluster: reuse the cluster of a same-named person if one exists in
   `labels.json`, otherwise mint a new `person_NNN`. The face's sidecar
   `cluster` is updated and `labels.json` records the name.

The review window launches automatically at the end of `match` (unless
`--no-review` is passed for headless/scripted use).

## Non-goals

- No change to `detect`, `cluster`, `label`, the sidecar/`labels.json` schemas,
  or `faces.npy`/`faces_index.json` layout.
- No new persisted cluster structure — cluster assignment continues to live in
  the sidecars (see "The faces.npy question").
- The `PhotoReviewApp` Tkinter event loop is manual-verification (human at the
  GUI), like `LabelerApp` and `split_photos.py`'s `Editor`. Pure helpers are
  TDD-tested.

## Architecture

All in `face_pipeline.py`. `match` keeps its scoring and `match_report.json`
write (still useful and testable), then launches `PhotoReviewApp` seeded with the
per-face top-1 suggestions it just computed — unless `--no-review` is given.

### Pure helpers (TDD-tested)

- `existing_cluster_ids(images_dir) -> set[str]`
  Scan all `extracted/*.faces.json` for `cluster` values matching `person_NNN`.

- `next_cluster_id(existing_ids) -> str`
  `person_{max+1:03d}` over the numeric suffixes; `person_000` when none exist.

- `resolve_or_create_cluster(name, labels_map, existing_ids) -> tuple[str, dict]`
  If `name` already maps to a cluster id in `labels_map`, return
  `(that_id, labels_map)` unchanged. Otherwise mint `next_cluster_id(existing_ids)`,
  return `(new_id, {**labels_map, new_id: name})`. (Caller adds new_id to
  `existing_ids` so repeated calls within one photo mint distinct ids.)

- `prefill_name(face, suggestion, labels_map) -> str`
  Precedence: existing `face["label"]` (if non-empty) → `suggestion` (if
  non-empty) → name of `face["cluster"]` in `labels_map` → `""`.

- `apply_photo_edits(images_dir, image, edits, labels_map, existing_ids) -> dict`
  For one photo: load its sidecar; for each `face_id -> name` in `edits`:
  - set `face["label"] = name` (empty name clears it);
  - if `name` is non-empty AND the face's current `cluster` is `"unassigned"`
    (or empty), call `resolve_or_create_cluster` and set `face["cluster"]` to the
    resolved id, updating `labels_map` and `existing_ids` in place for subsequent
    faces;
  write the sidecar. Return the (possibly updated) `labels_map`. The caller
  persists `labels.json`.

  Note: editing the name of a face that is *already* in a named cluster writes
  only its `label` and leaves `cluster` untouched — the cluster is clustering
  provenance, the label is the human truth. Cluster reassignment happens *only*
  for faces that were `unassigned`/empty.

### Scoring (kept, lightly refactored)

`run_match` still computes `sims` and the `report` and writes
`match_report.json`. The per-face top-1 suggestion (`report[i].candidates[0].name`
or `""`) is collected into a `{(image, face_id): name}` map handed to the UI.

### UI shell (manual verification)

`PhotoReviewApp` — walks the photos that have at least one face, in sorted order.
Reuses `_source` (BGR image cache), `scale_to_fit`, `read_faces_json`,
`write_faces_json`.

## Data model & the faces.npy question

No new persisted structures. Everything rides the existing schema:
- `face["label"]` — the human name the inputs edit.
- `face["cluster"]` — changed only when naming a previously-`unassigned` face.
- `labels.json` — `{cluster_id: name}`, updated when a cluster gains a name.

**`faces.npy` is never rewritten.** Embeddings don't change; only a face's
*cluster assignment* changes, and that lives in the sidecar.
`faces_index.json` already maps embedding row → `(image, face_id)`, so the cache's
view of "which person is row N" is derived through the sidecar — no parallel
cluster array is added (it would duplicate the sidecar truth and risk drift).

**Caveat (documented):** a future `cluster` run re-clusters from scratch and would
overwrite these manual assignments — the same caveat the pipeline already states
for cluster ids. Also, reusing a cluster by name means two distinct people who
share a name collapse into one cluster; acceptable since the human controls names.

## UI layout & flow

Window, top to bottom:
- **Title strip:** `original-002_01.jpg  (3 of 18) — 26 faces`.
- **Photo canvas:** full source downscaled via `scale_to_fit(w, h, 900)`; each
  face's bbox drawn as a rectangle scaled by the same factor, with its **number**
  painted at a box corner. Faces numbered in sidecar `id` order; on-image number
  `N` = input row `N` = (position in `faces` list) + 1.
- **Numbered inputs:** a scrollable column of rows — `N.` + an `Entry` prefilled
  via `prefill_name`, plus a small tag showing the face's current cluster
  (`unassigned` / `person_NNN`) so unclustered faces are visible.
- **Buttons:** `Back`, `Next` (`Done` on the last photo).

**Per-photo commit** (Next / Back / Done / window close): gather each row's entry
into `edits = {face_id: name}`, call `apply_photo_edits(...)`, then persist
`labels.json`. New cluster ids minted during the photo are added to the
in-memory `existing_ids` so two newly-named people on one photo get distinct ids.

## Error handling

- Unreadable source image → skip that photo with a console warning (not shown in
  the walk; no blank window).
- Photos with zero faces → excluded from the walk list.
- Degenerate bbox → still paint the number; skip the rectangle.
- `import tkinter` failure → `SystemExit` with guidance (same guard as
  `run_label`).
- `--no-review` on `match` → old headless behavior (report + optional `--apply`),
  no window; used by tests and scripts.

## Testing strategy

Pure helpers get TDD tests (`tests/test_face_pipeline.py`, synthetic sidecars):

- `existing_cluster_ids` — collects `person_NNN` across fixture sidecars; ignores
  `unassigned`/empty.
- `next_cluster_id` — `person_000` on empty; `max+1` with gaps (e.g. {000,002} →
  003); zero-padded width.
- `resolve_or_create_cluster` — reuses id for an existing name; mints a new id +
  extends labels_map for a new name.
- `prefill_name` — each precedence rung: existing label wins; else suggestion;
  else cluster's labels_map name; else "".
- `apply_photo_edits` — writes labels to the sidecar; assigns a new cluster to a
  named unassigned face and updates labels_map; leaves an already-clustered
  face's cluster untouched; clears label on empty input; two unassigned faces
  named on one photo get distinct minted ids.

Verified manually (human-operated UI): `PhotoReviewApp` window, numbered bbox
overlays matching input rows, scroll, prefill, per-photo save on
Next/Back/Done/close, new-cluster assignment reflected in the sidecar.

Run via `.venv/bin/python -m pytest tests/ -q` (system Python's Tk can't open a
window on this macOS).

## Code conventions

- One file (`face_pipeline.py`). Pure helpers TDD-tested; `PhotoReviewApp` Tk
  code verified manually, like `LabelerApp`/`Editor`.
- Bboxes in full-resolution coords; the preview downscale is display-only and
  uses the existing `scale_to_fit`.
- Cluster/label writes reuse `read_faces_json`/`write_faces_json`; no schema or
  cache-format change.
