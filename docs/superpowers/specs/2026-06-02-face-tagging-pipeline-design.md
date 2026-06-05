# Face Detection & Tagging Pipeline — Design

**Date:** 2026-06-02
**Status:** Approved (pending spec review)

## Goal

A standalone, headless CLI tool that detects faces in the cropped photos under
`extracted/`, computes face embeddings with InsightFace, clusters the unlabeled
embeddings into per-person groups, and (optionally) matches new faces against a
human-labeled gallery via cosine similarity.

The tool is independent of `split_photos.py` — it consumes `split_photos.py`'s
output (`extracted/`) but never imports or modifies it.

## Non-goals

- No GUI. Labeling is done by editing a JSON file, not in an OpenCV window.
- No GPU assumption — CPU inference via onnxruntime.
- Not a package — one file, mirroring the repo's single-file convention.
- `match` does not silently relabel data; it produces a report for human review
  (with an explicit opt-in `--apply` for top-1 writeback).

## Architecture

New single file **`face_pipeline.py`** + tests in `tests/test_face_pipeline.py`.

A thin `FaceModel` wrapper isolates InsightFace (model load + inference) so the
rest of the code is pure and unit-testable without loading the heavy model. The
model is loaded only by the `detect` stage.

Four CLI subcommands, normally run in order:

```
python3 face_pipeline.py detect  [--images extracted] [--det-thresh 0.5]
python3 face_pipeline.py cluster [--min-cluster-size 3] [--min-samples N]
python3 face_pipeline.py label                          # scaffold labels.json
python3 face_pipeline.py match   --gallery labels.json [--top 3] [--threshold 0.5] [--apply]
```

### Model

InsightFace `buffalo_l` pack (RetinaFace detector + ArcFace `w600k_r50`,
512-d embeddings), CPU via onnxruntime. First `detect` run auto-downloads the
pack (~300 MB) to `~/.insightface`.

## Storage formats

### Per-photo sidecar — `extracted/<name>.faces.json`

Mirrors the existing `.photos.json` sidecar convention.

```json
{
  "image": "original-001_02.jpg",
  "image_size": [width, height],
  "model": "buffalo_l",
  "faces": [
    {
      "id": 0,
      "bbox": [x1, y1, x2, y2],
      "det_score": 0.94,
      "embedding_ref": 137,
      "cluster": "person_003",
      "label": "Alice"
    }
  ]
}
```

- `bbox` / `det_score`: from detection (full-resolution image coords).
- `embedding_ref`: row index into the shared embeddings cache. Embeddings are
  **not** inlined (512 float32 each).
- `cluster`: filled by `cluster` (`person_NNN`, or `"unassigned"` for noise).
- `label`: real name, resolved cluster → name (filled only by `match --apply`,
  otherwise left as written/empty).

### Embedding cache (at the images-dir root)

- `faces.npy` — `(N, 512)` float32 array, **L2-normalized at write time** so
  cosine similarity is a plain dot product everywhere downstream.
- `faces_index.json` — `{"model": "buffalo_l", "rows": [{"image": "...",
  "face_id": 0}, ...]}`. Lets `cluster`/`match` work from the cache alone and
  lets `detect` skip already-processed photos (idempotent re-runs).

### Gallery / labels file — `labels.json` (human-edited)

```json
{ "person_003": "Alice", "person_007": "Bob" }
```

Maps cluster ids → real names. `match` builds a centroid per *named* person
from the cached embeddings of faces in those clusters.

## Stage behavior & data flow

### `detect`
Loads `FaceModel`. For each image in `--images` (default `extracted/`):
- If a `.faces.json` exists for it and its `model` matches, skip (idempotent).
- Otherwise detect faces; drop those below `--det-thresh` (default 0.5).
- For each kept face: record bbox + det_score; append its L2-normalized 512-d
  embedding to `faces.npy`, append a row to `faces_index.json`, set
  `embedding_ref`.
- Write the sidecar. Print per-image face counts and a total.

### `cluster`
Loads `faces.npy` + index (no model). Runs **HDBSCAN** (primary algorithm;
`metric="euclidean"` on L2-normalized vectors ≈ cosine; `min_cluster_size`
default 3, `min_samples` configurable). Maps each face to `person_NNN`; HDBSCAN
noise (`-1`) → `"unassigned"`. Writes `cluster` back into every sidecar by
(image, face_id) via the index.

Cluster ids are **not stable** across runs. If `labels.json` exists, print a
remap warning plus a summary so the human knows labels may need re-checking.
Print cluster sizes.

### `match`
Loads cache + `labels.json`. Builds a centroid per *named* person (mean of that
person's normalized embeddings, re-normalized). For each face: cosine similarity
to every centroid; emit ranked **top-N** candidates above `--threshold`
(default 0.5), else `unknown`.

Output is a **report** — printed table + written `match_report.json` — for
human review. It does NOT overwrite sidecar `label` fields by default. Optional
`--apply` writes the top-1 label back into sidecars when the human trusts the
results.

### `label` (helper)
Scaffolds `labels.json` from current clusters (every cluster id mapped to `""`)
and lists one representative face's source image per cluster, so the human knows
who to name. Pure file munging, no model.

## Error handling

- Missing embedding cache → clear message to run `detect` first.
- Missing/empty `labels.json` → `match`/`label` explain the prior step.
- Unreadable image → warn + skip (matches splitter behavior).
- `insightface` / `hdbscan` import errors → actionable install hint.

## Testing strategy

Pure functions get TDD tests (`tests/test_face_pipeline.py`, no model load),
using small synthetic numpy embeddings — fast, no model download:

- `l2_normalize` — unit norm; zero-vector guard.
- `cosine_sim` / `cosine_sim_matrix` — known-vector cases.
- `build_centroids` — mean + renormalize per label; ignores unlabeled.
- `rank_candidates` — top-N ordering, threshold cutoff, `unknown` fallback.
- Sidecar + index I/O — `read/write_faces_json`, `append_embeddings`,
  `load_cache`; round-trip and idempotent-skip logic.
- `clusters_to_persons` — HDBSCAN label array (incl. `-1` noise) →
  `person_NNN` / `unassigned` mapping.
- `scaffold_labels` — clusters → `labels.json` skeleton.

Verified manually / light integration (per repo convention for heavy/IO parts):
- `FaceModel` wrapper (real InsightFace inference).
- The HDBSCAN library call itself (the `clusters_to_persons` mapping is tested;
  the library call is not).
- End-to-end `detect` → `cluster` → `match` on the real `extracted/` photos.

Runnable with the existing `python3 -m pytest tests/ -q`.

## Dependencies

Add to `requirements.txt`:

```
insightface>=0.7
onnxruntime>=1.16     # CPU inference
hdbscan>=0.8
scikit-learn>=1.3     # used for normalize/metrics; also pulled in by hdbscan
```

None are currently installed. The first implementation step installs them and
confirms import + a one-face smoke test before building further.

## Code conventions (consistent with the repo)

- One file (`face_pipeline.py`); not a package.
- Pure functions (normalization, cosine sim, centroid build, candidate ranking,
  cluster→person mapping, sidecar/index I/O, label scaffolding) get TDD tests.
- The `FaceModel` wrapper and the HDBSCAN library call are verified manually.
- Embeddings stored L2-normalized; all geometry/coords in full image resolution.
