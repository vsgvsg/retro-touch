# Face Detection & Tagging Pipeline Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a standalone headless CLI tool (`face_pipeline.py`) that detects faces in `extracted/` photos, computes InsightFace embeddings, clusters them into per-person groups with HDBSCAN, and matches faces against a labeled gallery via cosine similarity.

**Architecture:** One file `face_pipeline.py` with four CLI subcommands (`detect`, `cluster`, `label`, `match`). Pure helpers (normalization, cosine, centroids, ranking, sidecar/index I/O, cluster→person mapping, label scaffolding) are TDD-tested with synthetic numpy embeddings. A thin `FaceModel` wrapper isolates InsightFace so the model loads only in `detect`. Embeddings cached L2-normalized in `faces.npy` + `faces_index.json`; per-photo `<name>.faces.json` sidecars mirror the existing `.photos.json` idiom.

**Tech Stack:** Python 3.9, numpy, InsightFace (buffalo_l), onnxruntime (CPU), hdbscan, scikit-learn, pytest.

---

## File Structure

- **Create `face_pipeline.py`** — the whole tool: dataclasses, pure helpers, `FaceModel` wrapper, stage functions (`run_detect`, `run_cluster`, `run_label`, `run_match`), `argparse` CLI in `main()`.
- **Create `tests/test_face_pipeline.py`** — unit tests for pure helpers, using synthetic embeddings (no model load).
- **Modify `requirements.txt`** — add insightface, onnxruntime, hdbscan, scikit-learn.

All geometry/coords in full image resolution; embeddings stored L2-normalized so cosine == dot product downstream.

---

## Task 1: Dependencies & smoke test

**Files:**
- Modify: `requirements.txt`

- [ ] **Step 1: Add dependencies to requirements.txt**

Append these lines to `requirements.txt`:

```
insightface>=0.7
onnxruntime>=1.16
hdbscan>=0.8
scikit-learn>=1.3
```

- [ ] **Step 2: Install**

Run: `python3 -m pip install -r requirements.txt`
Expected: installs succeed (insightface + onnxruntime + hdbscan + scikit-learn).

- [ ] **Step 3: Verify imports**

Run: `python3 -c "import insightface, onnxruntime, hdbscan, sklearn; print('ok')"`
Expected: prints `ok` (downloads of model packs happen later, in detect).

- [ ] **Step 4: Commit**

```bash
git add requirements.txt
git commit -m "build: add face pipeline dependencies"
```

---

## Task 2: Test scaffold + l2_normalize

**Files:**
- Create: `face_pipeline.py`
- Create: `tests/test_face_pipeline.py`

- [ ] **Step 1: Write the failing test**

Create `tests/test_face_pipeline.py`:

```python
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import face_pipeline as fp


def test_l2_normalize_unit_norm():
    v = np.array([3.0, 4.0], dtype=np.float32)
    out = fp.l2_normalize(v)
    assert np.isclose(np.linalg.norm(out), 1.0)
    assert np.allclose(out, [0.6, 0.8])


def test_l2_normalize_zero_vector_is_safe():
    v = np.zeros(4, dtype=np.float32)
    out = fp.l2_normalize(v)
    assert out.shape == v.shape
    assert np.all(np.isfinite(out))
    assert np.isclose(np.linalg.norm(out), 0.0)


def test_l2_normalize_rows_of_matrix():
    m = np.array([[3.0, 4.0], [0.0, 2.0]], dtype=np.float32)
    out = fp.l2_normalize(m)
    norms = np.linalg.norm(out, axis=1)
    assert np.allclose(norms, [1.0, 1.0])
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python3 -m pytest tests/test_face_pipeline.py -q`
Expected: FAIL — `ModuleNotFoundError: No module named 'face_pipeline'` (or `AttributeError: l2_normalize`).

- [ ] **Step 3: Write minimal implementation**

Create `face_pipeline.py`:

```python
"""Face detection & tagging pipeline — detect, embed, cluster, match faces."""
from __future__ import annotations

import numpy as np


def l2_normalize(x: np.ndarray) -> np.ndarray:
    """L2-normalize a vector, or each row of a 2-D array. Zero vectors stay zero."""
    x = np.asarray(x, dtype=np.float32)
    if x.ndim == 1:
        n = np.linalg.norm(x)
        return x / n if n > 0 else x
    norms = np.linalg.norm(x, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return x / norms
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python3 -m pytest tests/test_face_pipeline.py -q`
Expected: PASS (3 passed).

- [ ] **Step 5: Commit**

```bash
git add face_pipeline.py tests/test_face_pipeline.py
git commit -m "feat: add l2_normalize helper"
```

---

## Task 3: Cosine similarity helpers

**Files:**
- Modify: `face_pipeline.py`
- Modify: `tests/test_face_pipeline.py`

- [ ] **Step 1: Write the failing test**

Append to `tests/test_face_pipeline.py`:

```python
def test_cosine_sim_identical_is_one():
    a = np.array([1.0, 0.0, 0.0], dtype=np.float32)
    assert np.isclose(fp.cosine_sim(a, a), 1.0)


def test_cosine_sim_orthogonal_is_zero():
    a = np.array([1.0, 0.0], dtype=np.float32)
    b = np.array([0.0, 1.0], dtype=np.float32)
    assert np.isclose(fp.cosine_sim(a, b), 0.0)


def test_cosine_sim_matrix_shape_and_values():
    faces = np.array([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32)
    centroids = np.array([[1.0, 0.0]], dtype=np.float32)
    sims = fp.cosine_sim_matrix(faces, centroids)
    assert sims.shape == (2, 1)
    assert np.isclose(sims[0, 0], 1.0)
    assert np.isclose(sims[1, 0], 0.0)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python3 -m pytest tests/test_face_pipeline.py -k cosine -q`
Expected: FAIL — `AttributeError: module 'face_pipeline' has no attribute 'cosine_sim'`.

- [ ] **Step 3: Write minimal implementation**

Add to `face_pipeline.py` (after `l2_normalize`):

```python
def cosine_sim(a: np.ndarray, b: np.ndarray) -> float:
    """Cosine similarity between two vectors."""
    a = l2_normalize(np.asarray(a, dtype=np.float32))
    b = l2_normalize(np.asarray(b, dtype=np.float32))
    return float(np.dot(a, b))


def cosine_sim_matrix(faces: np.ndarray, centroids: np.ndarray) -> np.ndarray:
    """(F,D) faces x (C,D) centroids -> (F,C) cosine similarities."""
    faces = l2_normalize(np.atleast_2d(np.asarray(faces, dtype=np.float32)))
    centroids = l2_normalize(np.atleast_2d(np.asarray(centroids, dtype=np.float32)))
    return faces @ centroids.T
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python3 -m pytest tests/test_face_pipeline.py -k cosine -q`
Expected: PASS (3 passed).

- [ ] **Step 5: Commit**

```bash
git add face_pipeline.py tests/test_face_pipeline.py
git commit -m "feat: add cosine similarity helpers"
```

---

## Task 4: build_centroids

**Files:**
- Modify: `face_pipeline.py`
- Modify: `tests/test_face_pipeline.py`

- [ ] **Step 1: Write the failing test**

Append to `tests/test_face_pipeline.py`:

```python
def test_build_centroids_means_and_normalizes():
    # two faces for Alice along +x, one for Bob along +y
    embeddings = np.array([
        [2.0, 0.0],
        [4.0, 0.0],
        [0.0, 3.0],
    ], dtype=np.float32)
    labels = ["Alice", "Alice", "Bob"]
    names, cents = fp.build_centroids(embeddings, labels)
    assert names == ["Alice", "Bob"]
    # centroids are L2-normalized
    assert np.allclose(np.linalg.norm(cents, axis=1), [1.0, 1.0])
    assert np.allclose(cents[0], [1.0, 0.0])
    assert np.allclose(cents[1], [0.0, 1.0])


def test_build_centroids_ignores_empty_labels():
    embeddings = np.array([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32)
    labels = ["Alice", ""]  # second face unlabeled -> excluded
    names, cents = fp.build_centroids(embeddings, labels)
    assert names == ["Alice"]
    assert cents.shape == (1, 2)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python3 -m pytest tests/test_face_pipeline.py -k centroid -q`
Expected: FAIL — `AttributeError: ... build_centroids`.

- [ ] **Step 3: Write minimal implementation**

Add to `face_pipeline.py`:

```python
def build_centroids(embeddings: np.ndarray, labels: list[str]):
    """Mean (then re-normalized) embedding per non-empty label.

    Returns (names_sorted, centroids) where centroids[i] corresponds to names[i].
    """
    embeddings = np.asarray(embeddings, dtype=np.float32)
    by_name: dict[str, list[int]] = {}
    for i, lab in enumerate(labels):
        if lab:
            by_name.setdefault(lab, []).append(i)
    names = sorted(by_name)
    if not names:
        return [], np.zeros((0, embeddings.shape[1]), dtype=np.float32)
    cents = np.stack([embeddings[by_name[n]].mean(axis=0) for n in names])
    return names, l2_normalize(cents)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python3 -m pytest tests/test_face_pipeline.py -k centroid -q`
Expected: PASS (2 passed).

- [ ] **Step 5: Commit**

```bash
git add face_pipeline.py tests/test_face_pipeline.py
git commit -m "feat: add build_centroids"
```

---

## Task 5: rank_candidates

**Files:**
- Modify: `face_pipeline.py`
- Modify: `tests/test_face_pipeline.py`

- [ ] **Step 1: Write the failing test**

Append to `tests/test_face_pipeline.py`:

```python
def test_rank_candidates_orders_by_score_and_caps_top_n():
    names = ["Alice", "Bob", "Carol"]
    sims = np.array([0.9, 0.7, 0.4])
    out = fp.rank_candidates(names, sims, top=2, threshold=0.5)
    assert [c["name"] for c in out] == ["Alice", "Bob"]
    assert np.isclose(out[0]["score"], 0.9)


def test_rank_candidates_threshold_filters():
    names = ["Alice", "Bob"]
    sims = np.array([0.45, 0.30])  # both below 0.5
    out = fp.rank_candidates(names, sims, top=3, threshold=0.5)
    assert out == []  # caller treats empty as "unknown"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python3 -m pytest tests/test_face_pipeline.py -k rank -q`
Expected: FAIL — `AttributeError: ... rank_candidates`.

- [ ] **Step 3: Write minimal implementation**

Add to `face_pipeline.py`:

```python
def rank_candidates(names: list[str], sims: np.ndarray, top: int, threshold: float):
    """Top-N (name, score) dicts above threshold, highest first. Empty => unknown."""
    sims = np.asarray(sims, dtype=np.float32)
    order = np.argsort(-sims)
    out = []
    for j in order:
        if sims[j] < threshold:
            break
        out.append({"name": names[j], "score": round(float(sims[j]), 4)})
        if len(out) >= top:
            break
    return out
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python3 -m pytest tests/test_face_pipeline.py -k rank -q`
Expected: PASS (2 passed).

- [ ] **Step 5: Commit**

```bash
git add face_pipeline.py tests/test_face_pipeline.py
git commit -m "feat: add rank_candidates"
```

---

## Task 6: clusters_to_persons

**Files:**
- Modify: `face_pipeline.py`
- Modify: `tests/test_face_pipeline.py`

- [ ] **Step 1: Write the failing test**

Append to `tests/test_face_pipeline.py`:

```python
def test_clusters_to_persons_maps_ids_and_noise():
    labels = np.array([0, 0, 1, -1, 1])
    out = fp.clusters_to_persons(labels)
    assert out == ["person_000", "person_000", "person_001",
                   "unassigned", "person_001"]


def test_clusters_to_persons_all_noise():
    labels = np.array([-1, -1])
    out = fp.clusters_to_persons(labels)
    assert out == ["unassigned", "unassigned"]
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python3 -m pytest tests/test_face_pipeline.py -k persons -q`
Expected: FAIL — `AttributeError: ... clusters_to_persons`.

- [ ] **Step 3: Write minimal implementation**

Add to `face_pipeline.py`:

```python
def clusters_to_persons(labels) -> list[str]:
    """HDBSCAN integer labels -> person_NNN strings; -1 (noise) -> 'unassigned'."""
    out = []
    for lab in labels:
        lab = int(lab)
        out.append("unassigned" if lab < 0 else f"person_{lab:03d}")
    return out
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python3 -m pytest tests/test_face_pipeline.py -k persons -q`
Expected: PASS (2 passed).

- [ ] **Step 5: Commit**

```bash
git add face_pipeline.py tests/test_face_pipeline.py
git commit -m "feat: add clusters_to_persons mapping"
```

---

## Task 7: Cache paths + embedding cache I/O

**Files:**
- Modify: `face_pipeline.py`
- Modify: `tests/test_face_pipeline.py`

- [ ] **Step 1: Write the failing test**

Append to `tests/test_face_pipeline.py`:

```python
def test_cache_paths(tmp_path):
    npy, idx = fp.cache_paths(str(tmp_path))
    assert npy.endswith("faces.npy")
    assert idx.endswith("faces_index.json")


def test_append_and_load_embeddings_roundtrip(tmp_path):
    d = str(tmp_path)
    e1 = np.array([[1.0, 0.0]], dtype=np.float32)
    refs = fp.append_embeddings(d, e1, [{"image": "a.jpg", "face_id": 0}], "buffalo_l")
    assert refs == [0]
    e2 = np.array([[0.0, 1.0], [1.0, 1.0]], dtype=np.float32)
    refs2 = fp.append_embeddings(d, e2,
        [{"image": "b.jpg", "face_id": 0}, {"image": "b.jpg", "face_id": 1}], "buffalo_l")
    assert refs2 == [1, 2]
    emb, index = fp.load_cache(d)
    assert emb.shape == (3, 2)
    assert index["model"] == "buffalo_l"
    assert index["rows"][2] == {"image": "b.jpg", "face_id": 1}


def test_load_cache_missing_raises(tmp_path):
    import pytest
    with pytest.raises(FileNotFoundError):
        fp.load_cache(str(tmp_path))
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python3 -m pytest tests/test_face_pipeline.py -k "cache or embeddings" -q`
Expected: FAIL — `AttributeError: ... cache_paths`.

- [ ] **Step 3: Write minimal implementation**

Add to `face_pipeline.py` (add `import json`, `import os` to the imports at top):

```python
def cache_paths(images_dir: str) -> tuple[str, str]:
    return (os.path.join(images_dir, "faces.npy"),
            os.path.join(images_dir, "faces_index.json"))


def load_cache(images_dir: str) -> tuple[np.ndarray, dict]:
    """Load (embeddings, index). Raises FileNotFoundError if not built yet."""
    npy, idx = cache_paths(images_dir)
    if not (os.path.exists(npy) and os.path.exists(idx)):
        raise FileNotFoundError(
            f"No embedding cache in {images_dir}. Run 'detect' first.")
    emb = np.load(npy)
    with open(idx) as f:
        index = json.load(f)
    return emb, index


def append_embeddings(images_dir: str, new_emb: np.ndarray,
                      new_rows: list[dict], model: str) -> list[int]:
    """Append L2-normalized embeddings + index rows; return assigned row refs."""
    npy, idx = cache_paths(images_dir)
    new_emb = l2_normalize(np.atleast_2d(np.asarray(new_emb, dtype=np.float32)))
    if os.path.exists(npy) and os.path.exists(idx):
        emb = np.load(npy)
        with open(idx) as f:
            index = json.load(f)
    else:
        emb = np.zeros((0, new_emb.shape[1]), dtype=np.float32)
        index = {"model": model, "rows": []}
    start = emb.shape[0]
    emb = np.vstack([emb, new_emb]) if start else new_emb
    index["rows"].extend(new_rows)
    index["model"] = model
    np.save(npy, emb)
    with open(idx, "w") as f:
        json.dump(index, f, indent=2)
    return list(range(start, start + len(new_rows)))
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python3 -m pytest tests/test_face_pipeline.py -k "cache or embeddings" -q`
Expected: PASS (3 passed).

- [ ] **Step 5: Commit**

```bash
git add face_pipeline.py tests/test_face_pipeline.py
git commit -m "feat: add embedding cache I/O"
```

---

## Task 8: Sidecar JSON I/O

**Files:**
- Modify: `face_pipeline.py`
- Modify: `tests/test_face_pipeline.py`

- [ ] **Step 1: Write the failing test**

Append to `tests/test_face_pipeline.py`:

```python
def test_faces_sidecar_path():
    p = fp.faces_sidecar_path("extracted/original-001_02.jpg")
    assert p == "extracted/original-001_02.faces.json"


def test_write_read_faces_json_roundtrip(tmp_path):
    img_path = str(tmp_path / "x.jpg")
    faces = [{"id": 0, "bbox": [1, 2, 3, 4], "det_score": 0.9,
              "embedding_ref": 5, "cluster": "person_000", "label": "Alice"}]
    fp.write_faces_json(img_path, (640, 480), "buffalo_l", faces)
    data = fp.read_faces_json(img_path)
    assert data["image"] == "x.jpg"
    assert data["image_size"] == [640, 480]
    assert data["model"] == "buffalo_l"
    assert data["faces"][0]["label"] == "Alice"


def test_read_faces_json_missing_returns_none(tmp_path):
    assert fp.read_faces_json(str(tmp_path / "nope.jpg")) is None
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python3 -m pytest tests/test_face_pipeline.py -k faces_json -q`
Expected: FAIL — `AttributeError: ... faces_sidecar_path`.

- [ ] **Step 3: Write minimal implementation**

Add to `face_pipeline.py`:

```python
def faces_sidecar_path(image_path: str) -> str:
    stem, _ = os.path.splitext(image_path)
    return stem + ".faces.json"


def write_faces_json(image_path: str, image_size: tuple[int, int],
                     model: str, faces: list[dict]) -> str:
    data = {
        "image": os.path.basename(image_path),
        "image_size": [int(image_size[0]), int(image_size[1])],
        "model": model,
        "faces": faces,
    }
    path = faces_sidecar_path(image_path)
    with open(path, "w") as f:
        json.dump(data, f, indent=2)
    return path


def read_faces_json(image_path: str) -> dict | None:
    path = faces_sidecar_path(image_path)
    if not os.path.exists(path):
        return None
    with open(path) as f:
        return json.load(f)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python3 -m pytest tests/test_face_pipeline.py -k faces_json -q`
Expected: PASS (3 passed).

- [ ] **Step 5: Commit**

```bash
git add face_pipeline.py tests/test_face_pipeline.py
git commit -m "feat: add faces sidecar I/O"
```

---

## Task 9: scaffold_labels

**Files:**
- Modify: `face_pipeline.py`
- Modify: `tests/test_face_pipeline.py`

- [ ] **Step 1: Write the failing test**

Append to `tests/test_face_pipeline.py`:

```python
def test_scaffold_labels_one_entry_per_real_cluster():
    persons = ["person_000", "person_000", "unassigned", "person_001"]
    out = fp.scaffold_labels(persons)
    assert out == {"person_000": "", "person_001": ""}  # unassigned excluded


def test_scaffold_labels_empty():
    assert fp.scaffold_labels(["unassigned", "unassigned"]) == {}
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python3 -m pytest tests/test_face_pipeline.py -k scaffold -q`
Expected: FAIL — `AttributeError: ... scaffold_labels`.

- [ ] **Step 3: Write minimal implementation**

Add to `face_pipeline.py`:

```python
def scaffold_labels(persons: list[str]) -> dict:
    """Build a labels.json skeleton: every real cluster id -> empty name."""
    ids = sorted({p for p in persons if p != "unassigned"})
    return {pid: "" for pid in ids}
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python3 -m pytest tests/test_face_pipeline.py -k scaffold -q`
Expected: PASS (2 passed).

- [ ] **Step 5: Commit**

```bash
git add face_pipeline.py tests/test_face_pipeline.py
git commit -m "feat: add scaffold_labels"
```

---

## Task 10: FaceModel wrapper (manual verification)

**Files:**
- Modify: `face_pipeline.py`

This wraps InsightFace. Not unit-tested (heavy model load), verified manually per repo convention.

- [ ] **Step 1: Add the wrapper**

Add to `face_pipeline.py`:

```python
MODEL_NAME = "buffalo_l"


class FaceModel:
    """Thin InsightFace wrapper. Loads buffalo_l (RetinaFace + ArcFace r50)."""

    def __init__(self, name: str = MODEL_NAME, det_size: int = 640):
        try:
            from insightface.app import FaceAnalysis
        except ImportError as e:
            raise SystemExit(
                "insightface not installed. Run: pip install -r requirements.txt") from e
        self.name = name
        self.app = FaceAnalysis(name=name, providers=["CPUExecutionProvider"])
        self.app.prepare(ctx_id=-1, det_size=(det_size, det_size))

    def detect(self, image, det_thresh: float = 0.5) -> list[dict]:
        """Return [{bbox:[x1,y1,x2,y2], det_score, embedding(np.ndarray 512)}]."""
        faces = self.app.get(image)
        out = []
        for f in faces:
            if float(f.det_score) < det_thresh:
                continue
            x1, y1, x2, y2 = [int(round(v)) for v in f.bbox]
            out.append({
                "bbox": [x1, y1, x2, y2],
                "det_score": float(f.det_score),
                "embedding": np.asarray(f.embedding, dtype=np.float32),
            })
        return out
```

- [ ] **Step 2: Manual smoke test**

Run (uses a real cropped photo; first run downloads buffalo_l ~300 MB):

```bash
python3 -c "
import cv2, face_pipeline as fp
m = fp.FaceModel()
img = cv2.imread('extracted/original-001_01.jpg')
faces = m.detect(img)
print('faces:', len(faces))
for f in faces[:3]:
    print(f['bbox'], round(f['det_score'],3), f['embedding'].shape)
"
```

Expected: prints a face count and, for any faces, a 4-int bbox, a score, and `(512,)` embedding shape. (Zero faces is acceptable if that photo has none — try another `extracted/*.jpg`.)

- [ ] **Step 3: Commit**

```bash
git add face_pipeline.py
git commit -m "feat: add InsightFace FaceModel wrapper"
```

---

## Task 11: `detect` stage

**Files:**
- Modify: `face_pipeline.py`

Orchestrates FaceModel + cache + sidecars. The pure I/O it relies on is already
tested (Tasks 7–8); this orchestration is verified by the end-to-end run in Task 14.

- [ ] **Step 1: Add list_images + run_detect**

Add to `face_pipeline.py` (add `import glob` to imports):

```python
def list_images(images_dir: str) -> list[str]:
    files = []
    for ext in ("*.jpg", "*.jpeg", "*.png", "*.tif", "*.tiff"):
        files.extend(glob.glob(os.path.join(images_dir, ext)))
    return sorted(files)


def run_detect(images_dir: str, det_thresh: float = 0.5) -> int:
    import cv2
    images = list_images(images_dir)
    if not images:
        print(f"No images found in {images_dir}")
        return 1
    model = FaceModel()
    total = 0
    for path in images:
        existing = read_faces_json(path)
        if existing is not None and existing.get("model") == model.name:
            print(f"{os.path.basename(path)}: cached, skipping")
            continue
        img = cv2.imread(path)
        if img is None:
            print(f"  ! cannot read {path}, skipping")
            continue
        dets = model.detect(img, det_thresh=det_thresh)
        rows = [{"image": os.path.basename(path), "face_id": i}
                for i in range(len(dets))]
        refs = []
        if dets:
            emb = np.stack([d["embedding"] for d in dets])
            refs = append_embeddings(images_dir, emb, rows, model.name)
        faces = [{
            "id": i,
            "bbox": d["bbox"],
            "det_score": round(d["det_score"], 4),
            "embedding_ref": refs[i],
            "cluster": "",
            "label": "",
        } for i, d in enumerate(dets)]
        h, w = img.shape[:2]
        write_faces_json(path, (w, h), model.name, faces)
        print(f"{os.path.basename(path)}: {len(faces)} face(s)")
        total += len(faces)
    print(f"Detected {total} face(s) across {len(images)} image(s).")
    return 0
```

- [ ] **Step 2: Manual smoke test**

Run: `python3 -c "import face_pipeline as fp; raise SystemExit(fp.run_detect('extracted'))"`
Expected: per-image face counts, a total, and `extracted/faces.npy` + `faces_index.json` + `*.faces.json` created. A second run prints "cached, skipping" for all.

- [ ] **Step 3: Commit**

```bash
git add face_pipeline.py
git commit -m "feat: add detect stage"
```

---

## Task 12: `cluster` stage

**Files:**
- Modify: `face_pipeline.py`

Uses HDBSCAN (library call, manually verified) + `clusters_to_persons` (tested).
Writes `cluster` back into sidecars via the index.

- [ ] **Step 1: Add run_cluster**

Add to `face_pipeline.py`:

```python
def _write_clusters_to_sidecars(images_dir: str, index: dict,
                                persons: list[str]) -> None:
    """Group person assignments by image and patch each sidecar's faces."""
    by_image: dict[str, dict[int, str]] = {}
    for row, person in zip(index["rows"], persons):
        by_image.setdefault(row["image"], {})[row["face_id"]] = person
    for image_name, face_map in by_image.items():
        path = os.path.join(images_dir, image_name)
        data = read_faces_json(path)
        if data is None:
            continue
        for face in data["faces"]:
            if face["id"] in face_map:
                face["cluster"] = face_map[face["id"]]
        write_faces_json(path, tuple(data["image_size"]), data["model"],
                         data["faces"])


def run_cluster(images_dir: str, min_cluster_size: int = 3,
                min_samples: int | None = None) -> int:
    try:
        import hdbscan
    except ImportError as e:
        raise SystemExit(
            "hdbscan not installed. Run: pip install -r requirements.txt") from e
    emb, index = load_cache(images_dir)  # raises FileNotFoundError w/ guidance
    if emb.shape[0] == 0:
        print("No embeddings to cluster. Run 'detect' first.")
        return 1
    clusterer = hdbscan.HDBSCAN(min_cluster_size=min_cluster_size,
                                min_samples=min_samples, metric="euclidean")
    labels = clusterer.fit_predict(emb)  # euclidean on L2-normed ~= cosine
    persons = clusters_to_persons(labels)
    _write_clusters_to_sidecars(images_dir, index, persons)

    if os.path.exists(os.path.join(images_dir, "labels.json")):
        print("! labels.json exists; cluster ids may have changed. "
              "Re-check labels against the new clusters below.")
    from collections import Counter
    counts = Counter(persons)
    for pid in sorted(counts):
        print(f"  {pid}: {counts[pid]} face(s)")
    n_real = len([p for p in set(persons) if p != "unassigned"])
    print(f"Clustered into {n_real} person(s); "
          f"{counts.get('unassigned', 0)} unassigned.")
    return 0
```

- [ ] **Step 2: Manual smoke test**

Run: `python3 -c "import face_pipeline as fp; raise SystemExit(fp.run_cluster('extracted'))"`
Expected: per-cluster face counts and a summary; sidecars now show non-empty `cluster` fields (e.g. `person_000` or `unassigned`).

- [ ] **Step 3: Commit**

```bash
git add face_pipeline.py
git commit -m "feat: add cluster stage"
```

---

## Task 13: `label` and `match` stages

**Files:**
- Modify: `face_pipeline.py`

Both reuse tested helpers (`scaffold_labels`, `build_centroids`,
`cosine_sim_matrix`, `rank_candidates`, cache/sidecar I/O).

- [ ] **Step 1: Add run_label**

Add to `face_pipeline.py`:

```python
def _collect_persons_and_examples(images_dir: str, index: dict):
    """Return (persons_in_row_order, {person_id: example_image_name})."""
    persons, examples = [], {}
    for row in index["rows"]:
        data = read_faces_json(os.path.join(images_dir, row["image"]))
        person = ""
        if data is not None:
            for face in data["faces"]:
                if face["id"] == row["face_id"]:
                    person = face.get("cluster", "")
                    break
        persons.append(person)
        if person and person != "unassigned" and person not in examples:
            examples[person] = row["image"]
    return persons, examples


def run_label(images_dir: str) -> int:
    _, index = load_cache(images_dir)
    persons, examples = _collect_persons_and_examples(images_dir, index)
    skeleton = scaffold_labels(persons)
    if not skeleton:
        print("No clusters found. Run 'cluster' first.")
        return 1
    path = os.path.join(images_dir, "labels.json")
    with open(path, "w") as f:
        json.dump(skeleton, f, indent=2)
    print(f"Wrote {path}. Fill in names for each cluster:")
    for pid in sorted(skeleton):
        print(f"  {pid}: e.g. see {examples.get(pid, '?')}")
    return 0
```

- [ ] **Step 2: Add run_match**

Add to `face_pipeline.py`:

```python
def run_match(images_dir: str, gallery: str, top: int = 3,
              threshold: float = 0.5, apply: bool = False) -> int:
    emb, index = load_cache(images_dir)
    if not os.path.exists(gallery):
        print(f"No gallery file {gallery}. Run 'label' and fill it in first.")
        return 1
    with open(gallery) as f:
        labels_map = json.load(f)  # {person_id: name}

    persons, _ = _collect_persons_and_examples(images_dir, index)
    # gallery labels: row's name = labels_map[row's cluster] (if named)
    gallery_labels = [labels_map.get(p, "") for p in persons]
    names, centroids = build_centroids(emb, gallery_labels)
    if not names:
        print(f"No named clusters in {gallery}. Add names and retry.")
        return 1

    sims = cosine_sim_matrix(emb, centroids)  # (F, C)
    report = []
    for ref, row in enumerate(index["rows"]):
        cands = rank_candidates(names, sims[ref], top=top, threshold=threshold)
        report.append({
            "image": row["image"],
            "face_id": row["face_id"],
            "candidates": cands,  # empty => unknown
        })

    out_path = os.path.join(images_dir, "match_report.json")
    with open(out_path, "w") as f:
        json.dump(report, f, indent=2)
    for r in report:
        top1 = r["candidates"][0] if r["candidates"] else {"name": "unknown", "score": 0}
        print(f"  {r['image']} #{r['face_id']}: "
              f"{top1['name']} ({top1['score']})")
    print(f"Wrote {out_path}.")

    if apply:
        by_image: dict[str, dict[int, str]] = {}
        for r in report:
            if r["candidates"]:
                by_image.setdefault(r["image"], {})[r["face_id"]] = \
                    r["candidates"][0]["name"]
        for image_name, face_map in by_image.items():
            path = os.path.join(images_dir, image_name)
            data = read_faces_json(path)
            if data is None:
                continue
            for face in data["faces"]:
                if face["id"] in face_map:
                    face["label"] = face_map[face["id"]]
            write_faces_json(path, tuple(data["image_size"]), data["model"],
                             data["faces"])
        print("Applied top-1 labels to sidecars.")
    return 0
```

- [ ] **Step 3: Manual smoke test**

Run (after editing `extracted/labels.json` to name at least one cluster):

```bash
python3 -c "import face_pipeline as fp; raise SystemExit(fp.run_label('extracted'))"
# edit extracted/labels.json, give one person a name, then:
python3 -c "import face_pipeline as fp; raise SystemExit(fp.run_match('extracted', 'extracted/labels.json'))"
```

Expected: `run_label` writes a labels.json skeleton with example image hints; `run_match` prints per-face top-1 and writes `match_report.json`.

- [ ] **Step 4: Commit**

```bash
git add face_pipeline.py
git commit -m "feat: add label and match stages"
```

---

## Task 14: CLI wiring + end-to-end

**Files:**
- Modify: `face_pipeline.py`

- [ ] **Step 1: Add argparse main()**

Add to `face_pipeline.py` (add `import argparse`, `import sys` to imports):

```python
def main(argv: list[str]) -> int:
    p = argparse.ArgumentParser(description="Face detection & tagging pipeline")
    sub = p.add_subparsers(dest="cmd", required=True)

    d = sub.add_parser("detect", help="detect faces + embeddings")
    d.add_argument("--images", default="extracted")
    d.add_argument("--det-thresh", type=float, default=0.5)

    c = sub.add_parser("cluster", help="cluster embeddings into persons")
    c.add_argument("--images", default="extracted")
    c.add_argument("--min-cluster-size", type=int, default=3)
    c.add_argument("--min-samples", type=int, default=None)

    l = sub.add_parser("label", help="scaffold labels.json from clusters")
    l.add_argument("--images", default="extracted")

    m = sub.add_parser("match", help="match faces against a labeled gallery")
    m.add_argument("--images", default="extracted")
    m.add_argument("--gallery", required=True)
    m.add_argument("--top", type=int, default=3)
    m.add_argument("--threshold", type=float, default=0.5)
    m.add_argument("--apply", action="store_true")

    args = p.parse_args(argv[1:])
    if args.cmd == "detect":
        return run_detect(args.images, args.det_thresh)
    if args.cmd == "cluster":
        return run_cluster(args.images, args.min_cluster_size, args.min_samples)
    if args.cmd == "label":
        return run_label(args.images)
    if args.cmd == "match":
        return run_match(args.images, args.gallery, args.top,
                         args.threshold, args.apply)
    return 1


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
```

- [ ] **Step 2: Run the full unit suite**

Run: `python3 -m pytest tests/ -q`
Expected: all tests pass (existing split_photos tests + new face_pipeline tests).

- [ ] **Step 3: End-to-end via CLI**

```bash
python3 face_pipeline.py detect --images extracted
python3 face_pipeline.py cluster --images extracted
python3 face_pipeline.py label --images extracted
# edit extracted/labels.json to name a cluster, then:
python3 face_pipeline.py match --images extracted --gallery extracted/labels.json
```

Expected: detect → counts + cache files; cluster → person groups; label → skeleton; match → printed top-1 per face + `match_report.json`.

- [ ] **Step 4: Commit**

```bash
git add face_pipeline.py
git commit -m "feat: wire up face_pipeline CLI"
```

---

## Task 15: Update project docs

**Files:**
- Modify: `CLAUDE.md`

- [ ] **Step 1: Document the new tool**

Add a section to `CLAUDE.md` after the existing `## Commands` section:

```markdown
## Face pipeline (face_pipeline.py)
- `python3 face_pipeline.py detect` - detect faces + embeddings in extracted/ (downloads buffalo_l on first run)
- `python3 face_pipeline.py cluster` - HDBSCAN-cluster embeddings into person_NNN groups
- `python3 face_pipeline.py label` - scaffold extracted/labels.json (cluster id -> name)
- `python3 face_pipeline.py match --gallery extracted/labels.json` - rank candidates vs labeled centroids
- Embeddings cached L2-normalized in extracted/faces.npy + faces_index.json; per-photo extracted/<name>.faces.json sidecars.
- Pure helpers are TDD-tested; FaceModel + HDBSCAN library call are verified manually.
```

- [ ] **Step 2: Commit**

```bash
git add CLAUDE.md
git commit -m "docs: document face_pipeline in CLAUDE.md"
```

---

## Self-Review Notes

- **Spec coverage:** detect (T11), cluster/HDBSCAN (T12), label (T13), match centroid+top-N report+`--apply` (T13), sidecar+`.npy` cache storage (T7–T8), gallery/labels.json (T9/T13), normalization-at-write (T7), idempotent detect (T11), error guidance (T7/T12/T13), testing strategy (T2–T9 pure; T10–T14 manual), deps (T1). All spec sections map to tasks.
- **Type consistency:** `cosine_sim_matrix(faces, centroids)->(F,C)` used consistently in T13; `build_centroids->(names, centroids)`; `clusters_to_persons` and `scaffold_labels` operate on the same `person_NNN`/`unassigned` vocabulary; sidecar face keys (`id, bbox, det_score, embedding_ref, cluster, label`) identical across T8/T11/T12/T13.
- **Placeholders:** none — every code step shows complete code.
