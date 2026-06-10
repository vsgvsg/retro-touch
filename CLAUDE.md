# RetroTouch

Four tools for digitizing, organizing, and restoring scanned photos:
1. `split_photos.py` detects, lets a human adjust (in a Tkinter/ttk GUI), and crops multiple photos out of flatbed scan images in `images/` into `extracted/`.
2. `detect_duplicates.py` detects and resolves duplicate photos in the extracted crops using perceptual hashing (dhash) and union-find grouping.
3. `face_pipeline.py` detects, clusters, labels, and ages faces in those `extracted/` crops.
4. `restore_photos.py` restores old/blurry photos, reconstructing faces grounded on a sharper photo of the same person at a similar age (using labels and ages from the face pipeline).

The four are one-off tools; they never cross-import — the JSON and image artifacts under `extracted/` are the only interface between them.

## Commands
- `.venv/bin/python split_photos.py` - launch the Tkinter editor (canvas + sidebar, ttk-themed like the face pipeline GUIs). Needs a human at the GUI; run via the venv (system Python's Tk can't open a window on this macOS).
- `.venv/bin/python detect_duplicates.py [--dir DIR] [--threshold THRESHOLD] [--dry-run]` - run the duplicate detection and cleanup tool.
- `.venv/bin/python face_pipeline.py [detect|cluster|label|ages|match|report|merge]` - run the face metadata detection, clustering, labeling, or matching commands.
- `.venv/bin/python restore_photos.py [face|photo] <photo>` - run identity-grounded photo restoration.
- `.venv/bin/python -m pytest tests/ -q` - run all tests. Use the venv (Homebrew Python 3.13 + Tk 9.0): the GUI tests run there. The system `python3` (3.9) ships Tk 8.5, which SIGABRTs on `Tk()` and pops a macOS crash dialog — under it those tests are auto-skipped (gated on `TkVersion >= 8.6`), so `python3 -m pytest` still passes but doesn't exercise the GUI. NEVER detect display by calling `Tk()` (even in a subprocess); read `tkinter.TkVersion` instead (opens no window).

## Duplicate Photo Detection (detect_duplicates.py)
- Computes a 64-bit dhash for every photo in the directory.
- Groups duplicate images whose Hamming distance is within `--threshold` (default: 2) using Union-Find.
- Primary resolution: keeps the highest quality photo (sorting by resolution desc, file size desc, filename asc) as the primary, and moves all other duplicates to a nested `duplicates/` folder.
- Preserves manual tagging: automatically moves matching `.faces.json` sidecar files along with the duplicate images.
- Implements collision resolution (safely appends incremental suffixes like `_1.jpg` if name already exists in target directory).
- Pure helpers (`compute_dhash`, `UnionFind`, `find_duplicate_groups`, `resolve_duplicates`) are TDD-tested.


## Photo Scan Splitter GUI (split_photos.py)
- ttk app (clam theme, shared look with face_pipeline): header (filename + "N photos · M cropped" + clickable progress bar to jump scans), a `CanvasEditor` (tk.Canvas: scan as a background PhotoImage, each box drawn as native canvas items — polygon + corner handles + orientation arrow), and a right sidebar (PHOTOS list w/ thumbnail + state-colored dot per box, click a row to select; ACTIVE BOX panel: top-edge word via `orientation_label` (top/right/bottom/left) + inline rounded-crop preview + `↻ Rotate` and `Delete` buttons — tilt is keyboard-only).
- Mouse: drag inside=move, drag edge/corner=resize, drag empty=new box. Keyboard mirrors the old editor (arrows/hjkl nudge, [ ]=orient, , . < >=tilt, x/Del=delete, n/Tab=next box, Enter=crop all + next, ?/F1=shortcuts popover).
- Tk keyboard gotchas (cost debugging this session, apply to all the Tk GUIs):
  - Bind app-global shortcuts with `root.bind_all("<Key>", ...)`, NOT `root.bind` — a root bind dies the moment a ttk Button takes focus (its class bindtag shadows the key). `focus_set()` a widget after building so keys work before the first click.
  - Match PUNCTUATION keys on `event.char` (`"]"`, `","`, `"?"`), not the X11 keysym name (`"bracketright"`) — this Tk reports the char, so name matching silently never fires. Named keys (arrows, `n`, `Return`, `F1`) match on `event.keysym` fine.
  - `Tab` is eaten by focus-traversal on the CLASS bindtag before `bind_all` runs; bind `<Tab>`/`<ISO_Left_Tab>` on the widget INSTANCE and `return "break"`.
- `box_state(box, scan_shape, active)` is the pure helper (TDD-tested) that color-codes both the sidebar dot and the canvas box: editing(accent) > attention(amber: zero-size/off-canvas) > cropped(green) > neutral(grey). Mirrors face_pipeline's `face_state`. Box movement goes through the pure `nudge_box` helper (also TDD-tested).
- Theme helpers (`_install_theme`, `crop_to_round_photo`, color constants) are COPIED from face_pipeline.py — the tools never cross-import; the `extracted/`/`*.photos.json` artifacts remain the only interface.
- Catch GUI wiring errors without a human (same pattern as the face pipeline): build `SplitterApp(scans, out_dir)`, call `app._show()`, then `app.root.after(150, app.root.destroy); app.root.mainloop()`.

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
- The system /usr/bin/python3 ships a Tk that can't open a window on this macOS; run the GUI (and ideally the whole pipeline) via the venv: `.venv/bin/python face_pipeline.py ...` (Homebrew Python 3.13 + Tk 9.0). `.venv` is gitignored; recreate with `brew install exempi && python3.13 -m venv .venv && .venv/bin/pip install -r requirements.txt` (`exempi` is a system library required by `python-xmp-toolkit` for XMP face region writing).
- All three GUIs (LabelerApp, PhotoReviewApp, AgeLabelerApp) share module-level helpers: `_install_theme` (ttk "clam" + color constants/`STATE_COLORS`), `crop_to_round_photo` (rounded crop→PhotoImage, degrades to square), and `face_state`. Each app's vertical scrollbar is shown on demand via `_sync_scrollbar` (pack/forget by comparing content reqheight vs canvas height), and the top progress bar is clickable (`_on_progress_click` jumps to the photo/cluster at the click x-fraction, committing current edits first).
- Catch Tk GUI *wiring* errors without a human: `.venv/bin/python -c "...; app=fp.SomeApp(...); app._show(); app.root.after(100, app.root.destroy); app.root.mainloop()"` builds + renders the first screen then auto-closes (full interaction still needs a human). InsightFace logs noise to stderr — filter with `grep -v -E "find model|Applied providers|set det-size|recognition|detection|landmark|genderage"`.

## Photo restoration (restore_photos.py)
- `python3 restore_photos.py face <photo>` - restore the face region(s) only and composite them back into the otherwise-original photo (feathered, color-matched). `python3 restore_photos.py photo <photo>` - enhance the whole image, then identity-ground its faces. Both write to `extracted/restored/<name>.jpg` + a `extracted/restored/<name>.restore.json` provenance sidecar.
- Reads `labels.json` + the `*.faces.json` sidecars (incl. `age`); NEVER imports `face_pipeline.py`. Run via `.venv/bin/python` like the rest.
- Pipeline = deterministic-first escalation: every face gets a safe Stage-1 enhance; only faces that fail the sharpness/size gate (`needs_escalation`) escalate to Stage-2 identity-grounded generation, conditioned on the best same-person reference. Faces with no persona or no usable reference stay Stage-1 only (never grounded against the wrong person).
- Reference selection (`select_reference`): among the persona's other faces, prefer one within `--age-window` years (default 5) of the target by quality (bbox area · det_score · sharpness); if none in-window, fall back to the closest age and flag it (`age_fallback`) in provenance; with no age info, fall back to best quality (`no_age`). Age comes from `detect --backfill-age` + the `ages` GUI — so do those first for the age-grounding to work.
- Provenance is first-class: each face records the exact reference used (image/face_id/age/quality), the stage, the model, and `ai_reconstructed: true` for Stage-2 — the honesty layer for what's real vs. synthesized.
- `--dry-run` prints the per-face plan (stage + chosen reference + reason) and writes nothing — use it to preview before spending API calls. Flags: `--age-window`, `--sharpness-thresh`, `--min-area`, `--provider replicate|fake`, `--out`.
- Generative work runs on a cloud GPU (no CUDA on this M4): `ReplicateProvider` (default; needs `REPLICATE_API_TOKEN`) — its model slugs/input keys are the manual-verification surface and may need updating against the live Replicate pages. `FakeProvider` (echoes input, records calls) backs the tests and `--dry-run`. Pure helpers are TDD-tested; the provider, the buffalo_l age pass, and the GUIs are verified manually.

## EXIF Pipeline (exif_pipeline.py)
- `python exif_pipeline.py tag` - interactive GUI to tag each extracted photo with year (required), month (optional), and location (lat/lng + city/state/country). Auto-fills year and location from filename (e.g. `1960-penza-00004_04.jpg` → year=1960, map flies to Penza).
- `python exif_pipeline.py report` - print tagging coverage stats (read-only).
- Date Propagation: Successfully saving a photo with a manually entered year and optional month propagates those values as the default year/month for the next untagged photo.
- Shortcuts Button: A small `?` button next to "Skip →" in the sidebar bottom row opens the keyboard shortcuts helper window.
- Map Layout & Zoom:
  - The map is `360`px high by default and is configured with vertical and horizontal expansion (`fill=tk.BOTH, expand=True`) to grow when the window is resized.
  - macOS Zoom Fix: Custom event bindings on the canvas normalize standard mouse wheel zoom (`abs(delta) >= 120` delta divided by `120.0` to change zoom by exactly 1 level), while preserving the default library handling for trackpads and Apple Magic Mouse.
- Extends the existing `*.faces.json` sidecar with top-level `taken` and `location` keys; also writes EXIF DateTimeOriginal, GPS tags, IPTC Keywords (face names), and XMP MWG Regions (face rectangles) to the `.jpg` file.
- Maintains `extracted/locations.json` — a cache of lat/lng → human name mappings with use counts. Entries within 1000m are coalesced. Top 8 by use count appear as quick-select chips above the map.
- Map: `tkintermapview` (OpenStreetMap tiles). Geocoding: Nominatim (free, no API key, 1 req/sec rate limit enforced, limit increased to `10` candidates to find smaller cities, and `accept-language=en` set to fetch geocoding results in English). EXIF: `piexif`. XMP: `python-xmp-toolkit` (requires `brew install exempi`).
- Safe write: EXIF/XMP written to `.jpg.tmp`, verified with Pillow, then `os.replace()` — original never corrupted.
- Sidecar `taken` dict: `{"year": 1960, "month": 4, "source": "manual"}` — `month` key omitted entirely when unknown.
- Sidecar `location` dict: `{"lat": 53.2, "lng": 45.0, "display_name": "...", "city": "...", "state": "...", "country": "...", "source": "manual"}`.
- `exif_written: true` added to sidecar after successful EXIF write.
- Pure helpers TDD-tested: `parse_filename`, `haversine`, `coalesce_location`, `format_taken`, `parse_nominatim_address`, `decimal_to_dms`, `normalize_bbox`, `load_sidecar`, `save_sidecar`, `sidecar_is_tagged`.
- GUI verified manually. System dependency: `brew install exempi` (for `python-xmp-toolkit`).

## Code conventions
- One file per tool (`split_photos.py`, `face_pipeline.py`, `restore_photos.py`); they're one-off tools, not a package. Don't cross-import between them — pass data through the `extracted/` JSON artifacts.
- Pure functions (detector, cropper, metadata I/O, geometry helpers) get TDD tests; the `Editor` HighGUI class is verified manually, not unit-tested.
- Box geometry is always stored in FULL-resolution scan coords; display scale is applied only at render/mouse time.

## Gotchas (detector & crop geometry)
- Detected box `angle` is normalized to (-45, 45] via `normalize_rect`; deliberate quarter-turns go in `Box.orientation`, not the deskew angle.
- `crop_box` orientation = which edge is the photo's real top (matches the on-screen arrow): 90=right→CCW, 270=left→CW.
- Auto-detect is deliberately best-effort (non-white beds, touching photos mis-detect); the human fixes those in the editor. Don't over-tune the detector.

## Persistence
- Metadata saved per-scan as `images/<scan>.photos.json`; on restart, if it exists, boxes load from it and detection is skipped (never override manual edits).
