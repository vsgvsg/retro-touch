# Photo Scan Splitter

Three tools: `split_photos.py` detects, lets a human adjust, and crops
multiple photos out of flatbed scan images in `images/` into `extracted/`.
`face_pipeline.py` then detects/clusters/tags faces in those `extracted/` crops.
`restore_photos.py` restores old photos, reconstructing blurry faces grounded
on a sharper reference of the same person at a similar age (uses the pipeline's
labels + ages). The three are one-off tools; they never cross-import — the JSON
artifacts under `extracted/` are the only interface between them.

## Commands
- `python3 split_photos.py` - launch the interactive editor (needs a human at the GUI; Claude can't drive the OpenCV window)
- `python3 -m pytest tests/ -q` - run all tests (deps installed on system Python 3.9 AND in `~/.venv`). For anything touching the Tk GUI, use `~/.venv/bin/python` (see face pipeline notes).

## Face pipeline (face_pipeline.py)
- `python3 face_pipeline.py detect` - detect faces + embeddings in extracted/ (downloads buffalo_l on first run). Now also stores a per-face `age`/`age_source: "auto"` (buffalo_l's estimate — rough on old scans).
- `python3 face_pipeline.py detect --backfill-age` - re-run the model on already-detected photos and merge ONLY `age` into existing sidecars, matching faces by bbox IoU. Preserves `cluster`/`label`/`embedding_ref` and never overwrites a manual age. Safe to re-run; use it to add age to sidecars detected before the age field existed.
- `python3 face_pipeline.py ages` - manual age-entry GUI (ttk, reuses the shared helpers): per-persona scrollable crop grid, each crop with an age field prefilled from the auto estimate; click a crop for the full-photo preview. Saving writes `age` + `age_source: "manual"` (authoritative — `--backfill-age` won't clobber it). Needs a human at the GUI.
- `python3 face_pipeline.py cluster` - HDBSCAN-cluster embeddings into person_NNN groups. If manual work exists (labels.json or assigned clusters) it warns and requires typing `yes` first (refuses non-interactively); `--yes`/`-y` skips the prompt.
- `python3 face_pipeline.py label` - interactive Tkinter labeler (ttk-themed): progress bar + scrollable grid of ALL of a cluster's face crops (larger, rounded); left-click a crop = full-photo preview (face boxed), Ctrl-click a crop = exclude it (sets sidecar cluster to "unassigned" and dims the crop with an "excluded" overlay — it stays visible for the session); name field with typeahead-filtered reuse chips (top-recent names as chips; typing filters them and offers `+ Create "<text>"`) replaces the old listbox; writes extracted/labels.json after each step. Needs a human at the GUI.
- `python3 face_pipeline.py match --gallery extracted/labels.json` - rank candidates vs labeled centroids, then open a per-photo review UI: photo with numbered bbox overlays colored by face state (green = confident match ≥ threshold, blue = labeled, amber = unassigned), and a scrollable column of per-face cards (face thumbnail + color-matched number badge + name input + confidence meter + clickable hint pill). Edits write face label; naming an unassigned face assigns/mints a cluster. `--no-review` for headless (report + optional `--apply`). Needs a human at the GUI.
- `python3 face_pipeline.py report` - print per-labeled-person centroid coverage (faces + unique source images), sorted by coverage. Read-only; surfaces thin (1-2 image) personas that match weakly.
- Embeddings cached L2-normalized in extracted/faces.npy + faces_index.json; per-photo extracted/<name>.faces.json sidecars.
- Pure helpers are TDD-tested; the LabelerApp Tkinter UI, FaceModel, and HDBSCAN library call are verified manually.
- Match mostly reads "unknown" at the default threshold on this scan set — that's correct (few same-person repeats), not a bug.
- A person's match centroid = mean of embeddings whose face `cluster` (NOT `label`) maps to that name; it strengthens only as more faces are assigned to the cluster. An unconfirmed suggestion or a label on a still-`unassigned` face does not feed it.
- Review always shows each face's best candidate: at/above threshold it prefills the box; below threshold (or when it disagrees with an existing label) it appears as a clickable `→ Name? (score) — click to use` hint pill you can click to accept. Lets a 1-face persona (scores ~0.06) still be surfaced and grown; hints never auto-commit. The `face_state` helper (matched/confident/unassigned, mirroring prefill precedence) drives both the photo box color and the row meter color so they always agree.
- Adding photos later (incremental): run `detect` (idempotent — skips cached photos) then `match`; do NOT re-run `cluster` (it re-clusters from scratch and wipes manual cluster assignments). `match`/review never overwrite an existing face `label` (box prefill precedence: existing label → best candidate if ≥threshold → cluster name → blank), so re-running is safe for already-corrected faces.
- The system /usr/bin/python3 ships a Tk that can't open a window on this macOS; run the GUI (and ideally the whole pipeline) via the venv: `~/.venv/bin/python face_pipeline.py ...` (Homebrew Python 3.13 + Tk 9.0). `~/.venv` is gitignored; recreate with `python3.13 -m venv ~/.venv && ~/.venv/bin/pip install -r requirements.txt`.
- All three GUIs (LabelerApp, PhotoReviewApp, AgeLabelerApp) share module-level helpers: `_install_theme` (ttk "clam" + color constants/`STATE_COLORS`), `crop_to_round_photo` (rounded crop→PhotoImage, degrades to square), and `face_state`. Each app's vertical scrollbar is shown on demand via `_sync_scrollbar` (pack/forget by comparing content reqheight vs canvas height), and the top progress bar is clickable (`_on_progress_click` jumps to the photo/cluster at the click x-fraction, committing current edits first).
- Catch Tk GUI *wiring* errors without a human: `~/.venv/bin/python -c "...; app=fp.SomeApp(...); app._show(); app.root.after(100, app.root.destroy); app.root.mainloop()"` builds + renders the first screen then auto-closes (full interaction still needs a human). InsightFace logs noise to stderr — filter with `grep -v -E "find model|Applied providers|set det-size|recognition|detection|landmark|genderage"`.

## Photo restoration (restore_photos.py)
- `python3 restore_photos.py face <photo>` - restore the face region(s) only and composite them back into the otherwise-original photo (feathered, color-matched). `python3 restore_photos.py photo <photo>` - enhance the whole image, then identity-ground its faces. Both write to `extracted/restored/<name>.jpg` + a `extracted/restored/<name>.restore.json` provenance sidecar.
- Reads `labels.json` + the `*.faces.json` sidecars (incl. `age`); NEVER imports `face_pipeline.py`. Run via `~/.venv/bin/python` like the rest.
- Pipeline = deterministic-first escalation: every face gets a safe Stage-1 enhance; only faces that fail the sharpness/size gate (`needs_escalation`) escalate to Stage-2 identity-grounded generation, conditioned on the best same-person reference. Faces with no persona or no usable reference stay Stage-1 only (never grounded against the wrong person).
- Reference selection (`select_reference`): among the persona's other faces, prefer one within `--age-window` years (default 5) of the target by quality (bbox area · det_score · sharpness); if none in-window, fall back to the closest age and flag it (`age_fallback`) in provenance; with no age info, fall back to best quality (`no_age`). Age comes from `detect --backfill-age` + the `ages` GUI — so do those first for the age-grounding to work.
- Provenance is first-class: each face records the exact reference used (image/face_id/age/quality), the stage, the model, and `ai_reconstructed: true` for Stage-2 — the honesty layer for what's real vs. synthesized.
- `--dry-run` prints the per-face plan (stage + chosen reference + reason) and writes nothing — use it to preview before spending API calls. Flags: `--age-window`, `--sharpness-thresh`, `--min-area`, `--provider replicate|fake`, `--out`.
- Generative work runs on a cloud GPU (no CUDA on this M4): `ReplicateProvider` (default; needs `REPLICATE_API_TOKEN`) — its model slugs/input keys are the manual-verification surface and may need updating against the live Replicate pages. `FakeProvider` (echoes input, records calls) backs the tests and `--dry-run`. Pure helpers are TDD-tested; the provider, the buffalo_l age pass, and the GUIs are verified manually.

## Code conventions
- One file per tool (`split_photos.py`, `face_pipeline.py`, `restore_photos.py`); they're one-off tools, not a package. Don't cross-import between them — pass data through the `extracted/` JSON artifacts.
- Pure functions (detector, cropper, metadata I/O, geometry helpers) get TDD tests; the `Editor` HighGUI class is verified manually, not unit-tested.
- Box geometry is always stored in FULL-resolution scan coords; display scale is applied only at render/mouse time.

## Gotchas (OpenCV HighGUI on macOS)
- Use `cv2.waitKeyEx()` (not `waitKey() & 0xFF`) so arrow keys survive; no-key sentinel is `-1`.
- Keys only register when an OpenCV window has focus, not the terminal.
- `cv2.getWindowProperty` on a never-created window raises (doesn't return -1); guard `destroyWindow` with try/except `cv2.error`.
- Detected box `angle` is normalized to (-45, 45] via `normalize_rect`; deliberate quarter-turns go in `Box.orientation`, not the deskew angle.
- `crop_box` orientation = which edge is the photo's real top (matches the on-screen arrow): 90=right→CCW, 270=left→CW.
- Auto-detect is deliberately best-effort (non-white beds, touching photos mis-detect); the human fixes those in the editor. Don't over-tune the detector.

## Persistence
- Metadata saved per-scan as `images/<scan>.photos.json`; on restart, if it exists, boxes load from it and detection is skipped (never override manual edits).
