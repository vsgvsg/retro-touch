# Identity-Grounded Photo Restoration — Design

**Date:** 2026-06-04
**Status:** Approved (pending spec review)

## Goal

Restore/enhance old scanned photos, with face reconstruction **grounded on the
person's real identity** already captured by the face pipeline. When a face is
blurry but its persona is known, the tool uses a sharper photo of the *same
person at a similar age* (drawn from the tagged collection) to guide the
enhancement — rather than inventing a face from a generic prior.

Concretely:

- A new tool **`restore_photos.py`** that consumes the face pipeline's artifacts
  (`labels.json`, `faces.npy`, per-photo `*.faces.json`) and produces restored
  images under `restored/`.
- Two additions to **`face_pipeline.py`** that produce the **age** signal the
  reference selection needs: a merge-only auto age backfill, and a dedicated
  manual age-entry GUI.

## Non-goals

- Not a package and no cross-imports — `restore_photos.py` reads the pipeline's
  JSON/npy artifacts as its interface, never imports `face_pipeline.py` or
  `split_photos.py`. (Mirrors the existing two-tool separation.)
- No local model-weight management — the generative/enhancement work runs via a
  cloud GPU API (the machine is an Apple M4 Pro: Metal, no CUDA). The local
  footprint is an API client + image I/O only.
- Not a SaaS wrapper — no mainstream product does identity-grounding from the
  user's own reference library; that gap is the reason to build.
- Per-person trained priors (DreamBooth/LoRA/MyStyle) are out of scope — the
  dataset is thin (≈15 personas, few repeats, often 1–2 images each), so
  identity-grounding is **zero-shot from a single reference**, not training.
- The tool never overwrites the source images in `extracted/`; it writes new
  files to `restored/`.

## Background: why this is feasible here

`buffalo_l` already produces 512-d **ArcFace** embeddings (cached in
`faces.npy`). That is exactly the identity vector consumed by zero-shot
identity-conditioning models (**InstantID**, **IP-Adapter-FaceID**). Detection +
tagging — the hard part — is already done. What is missing is (a) a reliable
per-face **age**, and (b) the restoration tool itself.

## Part 1 — Age signal (`face_pipeline.py` changes)

Reference selection's "similar age" criterion needs a per-face age. `buffalo_l`
computes age in the same forward pass, but its estimates are unreliable on
faded/blurry scans, so auto-estimation is a **seed** and the human can correct
it.

### Storage (sidecar additions)

Each face in `extracted/<name>.faces.json` gains:

```json
{
  "id": 0,
  "bbox": [x1, y1, x2, y2],
  "det_score": 0.94,
  "embedding_ref": 137,
  "cluster": "person_003",
  "label": "Alice",
  "age": 34,
  "age_source": "manual"
}
```

- `age`: the effective age used downstream (integer years), or `null`.
- `age_source`: `"manual"` | `"auto"` | `null`. Manual is authoritative and is
  never overwritten by an auto pass.

### `detect --backfill-age` (merge-only)

A dedicated backfill that, for each already-processed photo, re-runs `buffalo_l`
to obtain `age` and writes it back **by matching faces to the existing sidecar**
(by `id`, validated against `bbox` IoU), updating **only** `age`/`age_source`
(→ `"auto"`) and preserving `cluster`, `label`, `embedding_ref`, and any
existing `age_source: "manual"`. This avoids the full-rewrite hazard that makes
a plain re-`detect` dangerous (it would clobber manual cluster/label work).

New `detect` runs going forward also populate `age`/`age_source: "auto"`.

### `face_pipeline.py ages` (new GUI subcommand)

A focused Tkinter screen for manually entering/correcting ages, reusing the
shared GUI helpers (`_install_theme`, `crop_to_round_photo`, `_sync_scrollbar`,
the clickable top progress bar, and click-a-crop → full-photo preview).

- Iterates **per persona** (you judge someone's age best by seeing their faces
  across photos together). Progress bar tracks persona position.
- Shows a scrollable grid of that persona's face crops; each crop has a small
  **age input prefilled from the auto estimate** (correct a number, don't type
  from blank). Click a crop → full-photo preview (face boxed) for scene context.
- Saving a value sets `age` + `age_source: "manual"` in that face's sidecar.
- Verified manually (like the existing labeler/review GUIs).

## Part 2 — Restoration tool (`restore_photos.py`)

New single file + `tests/test_restore_photos.py`. Outputs to `restored/`.

### Reference selection (the core of the idea)

For a target face whose `cluster` → persona is known, gather every **other**
face across the collection mapping to the same persona and score each candidate
reference by:

- **Quality**: bbox resolution (area), `det_score`, and sharpness
  (variance-of-Laplacian on the crop).
- **Age proximity**: `|age_ref − age_target|`, preferring candidates within a
  configurable window (`--age-window N`, default e.g. 5 years). If none fall in
  the window, fall back to the closest available age **and flag the gap** in
  provenance.

Pick the top-scoring reference. If the persona has no higher-quality reference
than the target (or no other face at all), skip identity-grounding for that face
and record the reason.

### Pipeline (deterministic-first escalation)

1. **Stage 1 — deterministic (always):** upscale + denoise + mild face restore
   (Real-ESRGAN + CodeFormer at high-fidelity weight). No invented identity.
2. **Escalation gate (computed locally, cheap):** a face qualifies for Stage 2
   when its sharpness/size falls below threshold. Otherwise Stage 1 is its final
   result.
3. **Stage 2 — identity-grounded (escalated faces only):** call a cloud model
   (**Replicate** default — hosts InstantID / IP-Adapter-FaceID) with the chosen
   reference + the degraded region as structure, using **ControlNet to preserve
   the original pose & expression** so it sharpens identity rather than
   regenerating the person.
4. **Composite:** *face-only* mode blends restored faces back into the original
   (feather + color-match at the bbox); *whole-photo* mode returns the fully
   enhanced image.

Both stages run via the cloud API by default; the escalation gate is local.

### Faces without identity

Unlabeled/`unassigned` faces, and faces where no suitable reference exists, get
**Stage 1 only** — never identity-grounded against the wrong person.

### Interface

```
restore_photos.py face  <photo> [--persona NAME]   # face-only composite mode
restore_photos.py photo <photo>                    # whole-photo mode
```

Shared flags:

- `--dry-run` — print the chosen reference per face + the plan + a cost estimate;
  **no API calls**.
- `--age-window N` — reference age tolerance (years).
- `--provider replicate|fal` — cloud backend (default `replicate`).
- `--sharpness-thresh`, `--out restored` — escalation + output tuning.

API credentials come from an env var (e.g. `REPLICATE_API_TOKEN`); missing →
clear, actionable error before any work begins.

### Provenance (first-class)

Every output gets `restored/<name>.restore.json` recording, per face: the
source photo, persona, the **exact reference image used** (filename, its age,
its scores), the model + version, the params, the stage reached, and an explicit
**`ai_reconstructed: true`** flag when Stage 2 ran. For family/genealogy photos
this is the honesty layer — what is original vs. synthesized is always
traceable.

## Error handling

- Missing `faces.npy` / sidecars / `labels.json` → message naming the
  `face_pipeline.py` step to run first.
- Missing API token or provider error → fail fast with the env-var name; in
  `--dry-run` no token is required.
- A face with no usable reference → Stage 1 only, reason recorded (not an error).
- Unreadable image → warn + skip (matches splitter/pipeline behavior).
- `--backfill-age` on a photo whose faces no longer match the sidecar (bbox IoU
  too low) → skip that face with a warning, never silently mis-assign an age.

## Testing strategy

Pure helpers get TDD tests (`tests/test_restore_photos.py`, plus additions to
`tests/test_face_pipeline.py`), no model/API load — using small synthetic data:

- Reference scoring: quality score, age-proximity ranking, window vs. fallback,
  exclusion of the target face / same-persona-only filtering.
- Sharpness metric (variance-of-Laplacian) on synthetic crops.
- Escalation-gate decision from sharpness/size + threshold.
- Composite geometry: bbox placement, feather mask, no out-of-bounds writes.
- Provenance writer: required fields incl. `ai_reconstructed`, reference record.
- Age backfill: bbox-IoU face matching, merge-only update, manual-preserving.

Verified manually (per repo convention for heavy/IO/GUI parts):

- The cloud API calls and actual restoration quality (the API client is mocked
  in tests).
- The `ages` Tkinter GUI.
- End-to-end on real `extracted/` photos.

Runnable with the existing `python3 -m pytest tests/ -q` (GUI-touching tests via
`.venv/bin/python`, per the face-pipeline notes).

## Dependencies

Add to `requirements.txt`:

```
replicate>=0.25      # default cloud provider client
# fal-client        # optional alt provider
```

Reuse existing `opencv-python` / `numpy` / `Pillow`. No new local model weights.
The first implementation step confirms the client imports and a `--dry-run`
plan renders before any paid API calls.

## Code conventions (consistent with the repo)

- One file per tool (`restore_photos.py`); not a package; no cross-imports —
  artifacts are the interface.
- Box/bbox geometry stored and reasoned about in full-resolution image coords;
  scaling applied only at render/composite time.
- Pure functions (reference scoring, sharpness, escalation gate, compositing
  geometry, provenance I/O, age-backfill matching) get TDD tests; the cloud API
  client, the `buffalo_l` age pass, and the `ages` GUI are verified manually.
- Sidecar-everywhere: restoration provenance mirrors the `.faces.json` /
  `.photos.json` sidecar pattern.
```
