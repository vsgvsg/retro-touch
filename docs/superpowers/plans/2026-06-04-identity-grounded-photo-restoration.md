# Identity-Grounded Photo Restoration Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a per-face age signal to the face pipeline and a new `restore_photos.py` tool that restores old photos, reconstructing blurry faces grounded on a sharper reference image of the *same person at a similar age*.

**Architecture:** Two parts. **Part 1** extends `face_pipeline.py` with an `age`/`age_source` field per face, a merge-only `detect --backfill-age` pass, and an `ages` Tkinter GUI for manual correction. **Part 2** is a standalone `restore_photos.py` that reads the pipeline's JSON artifacts (never imports it), selects the best same-person reference, runs a deterministic-first / identity-grounded-escalation pipeline through a cloud GPU provider (Replicate), composites faces back, and writes provenance sidecars.

**Tech Stack:** Python 3 (system 3.9 + `.venv` 3.13 for Tk), numpy, opencv-python, insightface/`buffalo_l`, Tkinter/ttk, the `replicate` API client, pytest.

**Conventions to honor (from CLAUDE.md):**
- One file per tool; **no cross-imports** between `split_photos.py`, `face_pipeline.py`, `restore_photos.py`. Artifacts (JSON/npy) are the interface.
- Pure functions get TDD tests; model calls, the API client, and Tk GUIs are verified manually.
- All bbox/box geometry stored in full-resolution image coords; scaling applied only at render/composite time.
- Tests live in `tests/`, import the module via the `sys.path.insert` pattern already at the top of `tests/test_face_pipeline.py`.
- GUI-touching tests / runs use `.venv/bin/python` (system 3.9 Tk can't open a window on this macOS).
- Run the suite with `python3 -m pytest tests/ -q`.

---

# Part 1 — Age signal in `face_pipeline.py`

## Task 1: `bbox_iou` pure helper

Used by the age backfill to match freshly-detected faces to existing sidecar faces without trusting array order.

**Files:**
- Modify: `face_pipeline.py` (add helper near `crop_face`, ~line 374)
- Test: `tests/test_face_pipeline.py`

- [ ] **Step 1: Write the failing tests**

Add to `tests/test_face_pipeline.py`:

```python
def test_bbox_iou_identical_is_one():
    assert fp.bbox_iou([0, 0, 10, 10], [0, 0, 10, 10]) == 1.0


def test_bbox_iou_disjoint_is_zero():
    assert fp.bbox_iou([0, 0, 10, 10], [20, 20, 30, 30]) == 0.0


def test_bbox_iou_half_overlap():
    # [0,0,10,10] vs [5,0,15,10]: inter=50, union=150
    assert abs(fp.bbox_iou([0, 0, 10, 10], [5, 0, 15, 10]) - (50 / 150)) < 1e-6


def test_bbox_iou_degenerate_box_is_zero():
    assert fp.bbox_iou([0, 0, 0, 0], [0, 0, 10, 10]) == 0.0
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python3 -m pytest tests/test_face_pipeline.py -k bbox_iou -q`
Expected: FAIL with `AttributeError: module 'face_pipeline' has no attribute 'bbox_iou'`

- [ ] **Step 3: Write minimal implementation**

Add to `face_pipeline.py` (immediately after `crop_face`):

```python
def bbox_iou(a, b) -> float:
    """Intersection-over-union of two [x1,y1,x2,y2] boxes. 0 for degenerate."""
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    iw, ih = max(0, ix2 - ix1), max(0, iy2 - iy1)
    inter = iw * ih
    area_a = max(0, ax2 - ax1) * max(0, ay2 - ay1)
    area_b = max(0, bx2 - bx1) * max(0, by2 - by1)
    union = area_a + area_b - inter
    return float(inter / union) if union > 0 else 0.0
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python3 -m pytest tests/test_face_pipeline.py -k bbox_iou -q`
Expected: PASS (4 passed)

- [ ] **Step 5: Commit**

```bash
git add face_pipeline.py tests/test_face_pipeline.py
git commit -m "feat: bbox_iou helper for age-backfill face matching"
```

---

## Task 2: `match_faces_by_bbox` pure helper

Maps each existing sidecar face id → index of the best-overlapping freshly-detected face (≥ threshold). Greedy, one detection per existing face.

**Files:**
- Modify: `face_pipeline.py` (after `bbox_iou`)
- Test: `tests/test_face_pipeline.py`

- [ ] **Step 1: Write the failing tests**

```python
def test_match_faces_by_bbox_pairs_by_overlap():
    existing = [{"id": 0, "bbox": [0, 0, 10, 10]},
                {"id": 1, "bbox": [100, 100, 110, 110]}]
    detected = [{"bbox": [100, 100, 110, 110]},   # idx 0 -> id 1
                {"bbox": [0, 0, 10, 10]}]          # idx 1 -> id 0
    assert fp.match_faces_by_bbox(existing, detected) == {0: 1, 1: 0}


def test_match_faces_by_bbox_skips_below_threshold():
    existing = [{"id": 0, "bbox": [0, 0, 10, 10]}]
    detected = [{"bbox": [50, 50, 60, 60]}]        # no overlap
    assert fp.match_faces_by_bbox(existing, detected) == {}


def test_match_faces_by_bbox_no_double_assignment():
    existing = [{"id": 0, "bbox": [0, 0, 10, 10]},
                {"id": 1, "bbox": [0, 0, 10, 9]}]
    detected = [{"bbox": [0, 0, 10, 10]}]          # only one detection
    out = fp.match_faces_by_bbox(existing, detected)
    assert list(out.values()) == [0]               # one face claims it
    assert len(out) == 1
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python3 -m pytest tests/test_face_pipeline.py -k match_faces_by_bbox -q`
Expected: FAIL with `AttributeError: ... 'match_faces_by_bbox'`

- [ ] **Step 3: Write minimal implementation**

```python
def match_faces_by_bbox(existing_faces, detected, iou_thresh: float = 0.5) -> dict:
    """{existing face id: detected index} by best IoU >= thresh; 1:1, greedy."""
    out: dict = {}
    used: set = set()
    for face in existing_faces:
        cands = [(bbox_iou(face["bbox"], d["bbox"]), j)
                 for j, d in enumerate(detected) if j not in used]
        cands = [(v, j) for v, j in cands if v >= iou_thresh]
        if cands:
            _, j = max(cands)
            out[face["id"]] = j
            used.add(j)
    return out
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python3 -m pytest tests/test_face_pipeline.py -k match_faces_by_bbox -q`
Expected: PASS (3 passed)

- [ ] **Step 5: Commit**

```bash
git add face_pipeline.py tests/test_face_pipeline.py
git commit -m "feat: match_faces_by_bbox for merge-only age backfill"
```

---

## Task 3: `merge_age_into_faces` pure helper

Writes ages into a face list, **never** overwriting a `manual` value.

**Files:**
- Modify: `face_pipeline.py` (after `match_faces_by_bbox`)
- Test: `tests/test_face_pipeline.py`

- [ ] **Step 1: Write the failing tests**

```python
def test_merge_age_sets_auto_age_by_id():
    faces = [{"id": 0}, {"id": 1}]
    out = fp.merge_age_into_faces(faces, {0: 34.6, 1: 7.2})
    assert out[0]["age"] == 35 and out[0]["age_source"] == "auto"
    assert out[1]["age"] == 7 and out[1]["age_source"] == "auto"


def test_merge_age_preserves_manual():
    faces = [{"id": 0, "age": 40, "age_source": "manual"}]
    out = fp.merge_age_into_faces(faces, {0: 12.0})
    assert out[0]["age"] == 40 and out[0]["age_source"] == "manual"


def test_merge_age_ignores_unmatched_and_none():
    faces = [{"id": 0}, {"id": 1}]
    out = fp.merge_age_into_faces(faces, {0: None})
    assert "age" not in out[0] or out[0].get("age") is None
    assert "age" not in out[1]
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python3 -m pytest tests/test_face_pipeline.py -k merge_age -q`
Expected: FAIL with `AttributeError: ... 'merge_age_into_faces'`

- [ ] **Step 3: Write minimal implementation**

```python
def merge_age_into_faces(faces, age_by_id, source: str = "auto"):
    """Set face['age']/['age_source'] from {id: age}, skipping manual faces."""
    for face in faces:
        if face.get("age_source") == "manual":
            continue
        age = age_by_id.get(face["id"])
        if age is not None:
            face["age"] = int(round(age))
            face["age_source"] = source
    return faces
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python3 -m pytest tests/test_face_pipeline.py -k merge_age -q`
Expected: PASS (3 passed)

- [ ] **Step 5: Commit**

```bash
git add face_pipeline.py tests/test_face_pipeline.py
git commit -m "feat: merge_age_into_faces (manual-preserving age writeback)"
```

---

## Task 4: `parse_age` + `set_face_age`

`parse_age` validates GUI text; `set_face_age` writes a single face's manual age to its sidecar (mirrors the existing `exclude_face` IO pattern).

**Files:**
- Modify: `face_pipeline.py` (add after `exclude_face`, ~line 425)
- Test: `tests/test_face_pipeline.py`

- [ ] **Step 1: Write the failing tests**

```python
def test_parse_age_valid_and_invalid():
    assert fp.parse_age("34") == 34
    assert fp.parse_age("  7 ") == 7
    assert fp.parse_age("") is None
    assert fp.parse_age("abc") is None
    assert fp.parse_age("-3") is None
    assert fp.parse_age("999") is None


def test_set_face_age_writes_manual(tmp_path):
    d = str(tmp_path)
    img = os.path.join(d, "p.jpg")
    fp.write_faces_json(img, (100, 100), "buffalo_l",
                        [{"id": 0, "bbox": [0, 0, 10, 10], "cluster": "person_000"}])
    assert fp.set_face_age(d, "p.jpg", 0, 41) is True
    data = fp.read_faces_json(img)
    assert data["faces"][0]["age"] == 41
    assert data["faces"][0]["age_source"] == "manual"


def test_set_face_age_missing_face_returns_false(tmp_path):
    d = str(tmp_path)
    img = os.path.join(d, "p.jpg")
    fp.write_faces_json(img, (100, 100), "buffalo_l", [{"id": 0, "bbox": [0, 0, 1, 1]}])
    assert fp.set_face_age(d, "p.jpg", 9, 20) is False
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python3 -m pytest tests/test_face_pipeline.py -k "parse_age or set_face_age" -q`
Expected: FAIL with `AttributeError: ... 'parse_age'`

- [ ] **Step 3: Write minimal implementation**

```python
def parse_age(text):
    """GUI text -> int age in [0,120], or None if blank/invalid."""
    text = (text or "").strip()
    if not text:
        return None
    try:
        v = int(text)
    except ValueError:
        return None
    return v if 0 <= v <= 120 else None


def set_face_age(images_dir, image, face_id, age) -> bool:
    """Set one face's manual age in its sidecar (age=None clears). Returns found."""
    path = os.path.join(images_dir, image)
    data = read_faces_json(path)
    if data is None:
        return False
    found = False
    for face in data["faces"]:
        if face["id"] == face_id:
            if age is None:
                face["age"] = None
                face["age_source"] = None
            else:
                face["age"] = int(age)
                face["age_source"] = "manual"
            found = True
            break
    if found:
        write_faces_json(path, tuple(data["image_size"]), data["model"],
                         data["faces"])
    return found
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python3 -m pytest tests/test_face_pipeline.py -k "parse_age or set_face_age" -q`
Expected: PASS (3 passed)

- [ ] **Step 5: Commit**

```bash
git add face_pipeline.py tests/test_face_pipeline.py
git commit -m "feat: parse_age + set_face_age sidecar writer"
```

---

## Task 5: `FaceModel.detect` returns age; `run_detect` persists it

The model call is verified manually (per convention). New `detect` runs write `age`/`age_source: "auto"`.

**Files:**
- Modify: `face_pipeline.py:166-179` (`FaceModel.detect`) and `face_pipeline.py:213-220` (face dict in `run_detect`)

- [ ] **Step 1: Add `age` to `FaceModel.detect` output**

Replace the `out.append({...})` block in `FaceModel.detect` (currently lines 174-178) with:

```python
            age = getattr(f, "age", None)
            out.append({
                "bbox": [x1, y1, x2, y2],
                "det_score": float(f.det_score),
                "embedding": np.asarray(f.embedding, dtype=np.float32),
                "age": (None if age is None else float(age)),
            })
```

- [ ] **Step 2: Persist age in `run_detect`**

In `run_detect`, replace the `faces = [{...}]` comprehension (currently lines 213-220) with:

```python
        faces = [{
            "id": i,
            "bbox": d["bbox"],
            "det_score": round(d["det_score"], 4),
            "embedding_ref": refs[i],
            "cluster": "",
            "label": "",
            "age": (None if d.get("age") is None else int(round(d["age"]))),
            "age_source": (None if d.get("age") is None else "auto"),
        } for i, d in enumerate(dets)]
```

- [ ] **Step 3: Manual verification (model call — no unit test)**

Run on one real photo in a scratch dir (or accept it runs on next full `detect`). Confirm a fresh sidecar now carries `age`/`age_source`:

```bash
.venv/bin/python -c "import cv2, face_pipeline as fp; m=fp.FaceModel(); d=m.detect(cv2.imread('extracted/original-001_02.jpg')); print([{k:x.get(k) for k in ('bbox','age')} for x in d])"
```
Expected: prints faces with non-`None` `age` values (integers-ish floats).

- [ ] **Step 4: Run full suite to confirm no regressions**

Run: `python3 -m pytest tests/ -q`
Expected: PASS (all existing + Task 1-4 tests)

- [ ] **Step 5: Commit**

```bash
git add face_pipeline.py
git commit -m "feat: capture buffalo_l age in detect + sidecars"
```

---

## Task 6: `run_backfill_age` + `detect --backfill-age`

Re-runs the model on already-processed photos and merges **only** age into existing sidecars via IoU matching, preserving cluster/label/manual ages.

**Files:**
- Modify: `face_pipeline.py` (add `run_backfill_age` after `run_detect`, ~line 227; wire CLI in `main`)

- [ ] **Step 1: Implement `run_backfill_age`**

Add after `run_detect`:

```python
def run_backfill_age(images_dir: str, det_thresh: float = 0.5) -> int:
    """Merge auto ages into existing sidecars (preserves cluster/label/manual)."""
    import cv2
    images = list_images(images_dir)
    if not images:
        print(f"No images found in {images_dir}")
        return 1
    model = FaceModel()
    updated = 0
    for path in images:
        data = read_faces_json(path)
        if data is None:
            print(f"{os.path.basename(path)}: no sidecar, skipping")
            continue
        img = cv2.imread(path)
        if img is None:
            print(f"  ! cannot read {path}, skipping")
            continue
        dets = model.detect(img, det_thresh=det_thresh)
        id_to_j = match_faces_by_bbox(data["faces"], dets)
        age_by_id = {fid: dets[j].get("age") for fid, j in id_to_j.items()}
        merge_age_into_faces(data["faces"], age_by_id, source="auto")
        write_faces_json(path, tuple(data["image_size"]), data["model"],
                         data["faces"])
        n = sum(1 for v in age_by_id.values() if v is not None)
        updated += n
        print(f"{os.path.basename(path)}: aged {n}/{len(data['faces'])} face(s)")
    print(f"Backfilled age on {updated} face(s).")
    return 0
```

- [ ] **Step 2: Wire the CLI flag**

In `main`, the `detect` subparser block (currently lines 1317-1319) — add an argument:

```python
    d = sub.add_parser("detect", help="detect faces + embeddings")
    d.add_argument("--images", default="extracted")
    d.add_argument("--det-thresh", type=float, default=0.5)
    d.add_argument("--backfill-age", action="store_true",
                   help="re-run model to merge age into existing sidecars only")
```

And change the `detect` dispatch (currently line 1344-1345):

```python
    if args.cmd == "detect":
        if args.backfill_age:
            return run_backfill_age(args.images, args.det_thresh)
        return run_detect(args.images, args.det_thresh)
```

- [ ] **Step 3: Manual verification on the real `extracted/` set**

```bash
.venv/bin/python face_pipeline.py detect --backfill-age
.venv/bin/python -c "import json; d=json.load(open('extracted/original-001_02.faces.json')); print([(f['id'], f.get('age'), f.get('age_source'), f.get('cluster'), f.get('label')) for f in d['faces']])"
```
Expected: faces now show integer `age` + `"auto"`, and `cluster`/`label` are unchanged from before the run.

- [ ] **Step 4: Run full suite**

Run: `python3 -m pytest tests/ -q`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add face_pipeline.py
git commit -m "feat: detect --backfill-age (merge-only age pass)"
```

---

## Task 7: `ages` GUI subcommand (`AgeLabelerApp`)

A per-persona crop grid where each crop has an age field prefilled from the auto estimate. Manual-verified; reuses shared helpers. Adds the tiny pure `age_prefill` (tested) and an `age` key to `cluster_face_index`.

**Files:**
- Modify: `face_pipeline.py` — add `age` to `cluster_face_index` (line ~351), add `age_prefill`, add `AgeLabelerApp` class, add `run_ages`, wire CLI
- Test: `tests/test_face_pipeline.py`

- [ ] **Step 1: Write the failing tests**

```python
def test_age_prefill_formats_value_and_blank():
    assert fp.age_prefill({"age": 34}) == "34"
    assert fp.age_prefill({"age": None}) == ""
    assert fp.age_prefill({}) == ""


def test_cluster_face_index_includes_age(tmp_path):
    d = str(tmp_path)
    fp.write_faces_json(os.path.join(d, "a.jpg"), (100, 100), "buffalo_l",
                        [{"id": 0, "bbox": [0, 0, 9, 9], "det_score": 0.9,
                          "cluster": "person_000", "age": 30}])
    index = {"model": "buffalo_l", "rows": [{"image": "a.jpg", "face_id": 0}]}
    ci = fp.cluster_face_index(d, index)
    assert ci["person_000"][0]["age"] == 30
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python3 -m pytest tests/test_face_pipeline.py -k "age_prefill or cluster_face_index_includes_age" -q`
Expected: FAIL (`age_prefill` missing; `cluster_face_index` row has no `age`)

- [ ] **Step 3: Add `age_prefill` and extend `cluster_face_index`**

Add `age_prefill` next to `previous_names` (~line 379):

```python
def age_prefill(face) -> str:
    """Age value as an editable string, '' when unset."""
    a = face.get("age")
    return "" if a is None else str(int(a))
```

In `cluster_face_index`, extend the appended dict (currently lines 351-356) to include age:

```python
            out.setdefault(cluster, []).append({
                "image": image,
                "face_id": face["id"],
                "bbox": face["bbox"],
                "det_score": face["det_score"],
                "age": face.get("age"),
            })
```

- [ ] **Step 4: Run those tests to verify they pass**

Run: `python3 -m pytest tests/test_face_pipeline.py -k "age_prefill or cluster_face_index_includes_age" -q`
Expected: PASS (2 passed)

- [ ] **Step 5: Add the `AgeLabelerApp` class**

Add after `LabelerApp` (before `PhotoReviewApp`):

```python
class AgeLabelerApp:
    """Per-persona crop grid; each crop has an age field (prefilled from auto)."""

    def __init__(self, images_dir, cluster_index, labels_map):
        import tkinter as tk
        from tkinter import ttk
        self.tk = tk
        self.ttk = ttk
        self.images_dir = images_dir
        self.cluster_ids = sorted(cluster_index)
        self.cluster_index = cluster_index
        self.labels_map = labels_map
        self.idx = 0
        self._img_cache = {}
        self._cells = []
        self._age_vars = []          # (image, face_id, StringVar)
        self._preview = None
        self._preview_photo = None

        self.root = tk.Tk()
        self.root.title("Face Ages")
        self.root.geometry("520x760")
        _install_theme(self.root)

        head = ttk.Frame(self.root)
        head.pack(fill="x", padx=16, pady=(12, 4))
        self.cluster_var = tk.StringVar()
        self.sub_var = tk.StringVar()
        ttk.Label(head, textvariable=self.cluster_var,
                  style="Title.TLabel").pack(anchor="w")
        ttk.Label(head, textvariable=self.sub_var,
                  style="Sub.TLabel").pack(anchor="w")
        self.progress = ttk.Progressbar(head, maximum=len(self.cluster_ids),
                                         cursor="hand2")
        self.progress.pack(fill="x", pady=(8, 0))
        self.progress.bind("<Button-1>", self._on_progress_click)

        wrap = ttk.Frame(self.root)
        wrap.pack(padx=16, pady=8, fill="both", expand=True)
        self.canvas = tk.Canvas(wrap, height=420, highlightthickness=0, bg=BG)
        self._vbar = ttk.Scrollbar(wrap, orient="vertical",
                                   command=self.canvas.yview)
        self.canvas.configure(yscrollcommand=self._vbar.set)
        self.canvas.pack(side="left", fill="both", expand=True)
        self.grid_frame = ttk.Frame(self.canvas)
        self.canvas.create_window((0, 0), window=self.grid_frame, anchor="nw")
        self.grid_frame.bind(
            "<Configure>",
            lambda e: (self.canvas.configure(
                scrollregion=self.canvas.bbox("all")), self._sync_scrollbar()))
        self.canvas.bind("<Configure>", lambda e: self._sync_scrollbar())
        self.canvas.bind_all(
            "<MouseWheel>",
            lambda e: self.canvas.yview_scroll(int(-1 * (e.delta / 120)), "units"))

        bar = ttk.Frame(self.root)
        bar.pack(fill="x", padx=16, pady=12)
        ttk.Button(bar, text="← Back", command=self._back).pack(side="left")
        self.next_btn = ttk.Button(bar, text="Save & Next →",
                                   style="Primary.TButton", command=self._next)
        self.next_btn.pack(side="right")
        self.root.protocol("WM_DELETE_WINDOW", self._on_close)

    def _source(self, image_name):
        import cv2
        if image_name not in self._img_cache:
            self._img_cache[image_name] = cv2.imread(
                os.path.join(self.images_dir, image_name))
        return self._img_cache[image_name]

    def _sync_scrollbar(self):
        content = self.grid_frame.winfo_reqheight()
        visible = self.canvas.winfo_height()
        if content > visible:
            if not self._vbar.winfo_ismapped():
                self._vbar.pack(side="right", fill="y", before=self.canvas)
        else:
            if self._vbar.winfo_ismapped():
                self._vbar.pack_forget()
            self.canvas.yview_moveto(0)

    def _on_progress_click(self, event):
        width = self.progress.winfo_width()
        if width <= 0 or not self.cluster_ids:
            return
        frac = max(0.0, min(1.0, event.x / width))
        target = min(len(self.cluster_ids) - 1, int(frac * len(self.cluster_ids)))
        if target == self.idx:
            return
        self._commit_current()
        self.idx = target
        self._show()

    def _make_cell(self, face, row, col):
        src = self._source(face["image"])
        crop = crop_face(src, face["bbox"]) if src is not None else \
            np.full((64, 64, 3), 128, np.uint8)
        photo = crop_to_round_photo(crop, cell=120)
        self._cells.append(photo)
        cell = self.ttk.Frame(self.grid_frame)
        cell.grid(row=row, column=col, padx=6, pady=6)
        lbl = self.tk.Label(cell, image=photo, bg=BG, borderwidth=0)
        lbl.pack()
        if src is not None:
            lbl.bind("<Button-1>", lambda e, f=face: self._preview_full(f))
        var = self.tk.StringVar(value=age_prefill(face))
        ent = self.ttk.Entry(cell, textvariable=var, width=6, justify="center")
        ent.pack(pady=(4, 0))
        self._age_vars.append((face["image"], face["face_id"], var))

    def _preview_full(self, face):
        import cv2
        from PIL import Image, ImageTk
        src = self._source(face["image"])
        if src is None:
            return
        self._close_preview()
        h, w = src.shape[:2]
        nw, nh, s = scale_to_fit(w, h, 900)
        disp = cv2.resize(src, (nw, nh), interpolation=cv2.INTER_AREA)
        x1, y1, x2, y2 = [int(round(v * s)) for v in face["bbox"]]
        if x2 > x1 and y2 > y1:
            cv2.rectangle(disp, (x1, y1), (x2, y2), (0, 0, 255), 2)
        rgb = cv2.cvtColor(disp, cv2.COLOR_BGR2RGB)
        top = self.tk.Toplevel(self.root)
        top.title(face["image"])
        self._preview = top
        self._preview_photo = ImageTk.PhotoImage(Image.fromarray(rgb))
        plbl = self.tk.Label(top, image=self._preview_photo)
        plbl.pack()
        plbl.bind("<Button-1>", lambda e: self._close_preview())
        top.bind("<Escape>", lambda e: self._close_preview())
        top.protocol("WM_DELETE_WINDOW", self._close_preview)

    def _close_preview(self):
        if self._preview is not None:
            try:
                self._preview.destroy()
            except self.tk.TclError:
                pass
            self._preview = None
            self._preview_photo = None

    def _show(self):
        cid = self.cluster_ids[self.idx]
        faces = self.cluster_index[cid]
        name = self.labels_map.get(cid, "")
        self.cluster_var.set(f"{name or cid} · {len(faces)} faces")
        self.sub_var.set(f"Person {self.idx + 1} of {len(self.cluster_ids)} · "
                         f"enter age per crop · click a crop for full photo")
        self.progress.configure(value=self.idx)
        for child in self.grid_frame.winfo_children():
            child.destroy()
        self._cells = []
        self._age_vars = []
        positions = grid_positions(len(faces), cols=3)
        for face, (r, c) in zip(faces, positions):
            self._make_cell(face, r, c)
        self.canvas.yview_moveto(0)
        self.next_btn.configure(
            text="Done" if self.idx == len(self.cluster_ids) - 1
            else "Save & Next →")

    def _commit_current(self):
        for image, face_id, var in self._age_vars:
            set_face_age(self.images_dir, image, face_id, parse_age(var.get()))

    def _next(self):
        self._commit_current()
        if self.idx == len(self.cluster_ids) - 1:
            self.root.destroy()
            return
        self.idx += 1
        self._show()

    def _back(self):
        self._commit_current()
        if self.idx > 0:
            self.idx -= 1
            self._show()

    def _on_close(self):
        self._close_preview()
        self._commit_current()
        self.root.destroy()

    def run(self):
        self._show()
        self.root.mainloop()
```

- [ ] **Step 6: Add `run_ages` and wire the CLI**

Add after `run_label`:

```python
def run_ages(images_dir: str) -> int:
    """Launch the manual age-entry GUI over clustered faces."""
    try:
        import tkinter  # noqa: F401
    except ImportError as e:
        raise SystemExit(
            "tkinter not available. Run via a Python with Tk support.") from e
    _, index = load_cache(images_dir)
    cluster_index = cluster_face_index(images_dir, index)
    if not cluster_index:
        print("No clusters found. Run 'cluster' first.")
        return 1
    labels_map = {}
    path = os.path.join(images_dir, "labels.json")
    if os.path.exists(path):
        with open(path) as f:
            labels_map = json.load(f)
    AgeLabelerApp(images_dir, cluster_index, labels_map).run()
    print("Ages saved to sidecars.")
    return 0
```

In `main`, add the subparser (after the `label` block, ~line 1330) and dispatch (after the `label` dispatch, ~line 1350):

```python
    ag = sub.add_parser("ages", help="manually enter/correct per-face ages")
    ag.add_argument("--images", default="extracted")
```
```python
    if args.cmd == "ages":
        return run_ages(args.images)
```

- [ ] **Step 7: Run the unit tests, then manually verify the GUI**

Run: `python3 -m pytest tests/test_face_pipeline.py -q`
Expected: PASS

Manual (human at GUI): `.venv/bin/python face_pipeline.py ages` — confirm per-persona grid shows crops with age fields prefilled from the auto backfill; editing one and clicking "Save & Next" persists `age_source: "manual"` into the sidecar.

- [ ] **Step 8: Commit**

```bash
git add face_pipeline.py tests/test_face_pipeline.py
git commit -m "feat: ages GUI subcommand for manual per-face age entry"
```

---

# Part 2 — `restore_photos.py`

All Part-2 tests live in a new `tests/test_restore_photos.py` that starts with:

```python
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import restore_photos as rp
```

## Task 8: File scaffold + artifact readers

`restore_photos.py` reads the pipeline's artifacts directly — no `face_pipeline` import.

**Files:**
- Create: `restore_photos.py`
- Create: `tests/test_restore_photos.py`

- [ ] **Step 1: Write the failing tests**

Create `tests/test_restore_photos.py` with the import header above, then:

```python
def _sidecar(d, name, faces, size=(100, 100)):
    path = os.path.join(d, name)
    data = {"image": name, "image_size": list(size), "model": "buffalo_l",
            "faces": faces}
    with open(os.path.splitext(path)[0] + ".faces.json", "w") as f:
        json.dump(data, f)


def test_read_faces_json_missing_returns_none(tmp_path):
    assert rp.read_faces_json(str(tmp_path / "nope.jpg")) is None


def test_load_labels_missing_is_empty(tmp_path):
    assert rp.load_labels(str(tmp_path)) == {}


def test_persona_faces_groups_by_name(tmp_path):
    d = str(tmp_path)
    with open(os.path.join(d, "labels.json"), "w") as f:
        json.dump({"person_000": "Alice"}, f)
    _sidecar(d, "a.jpg", [
        {"id": 0, "bbox": [0, 0, 10, 10], "det_score": 0.9,
         "cluster": "person_000", "age": 30},
        {"id": 1, "bbox": [0, 0, 5, 5], "det_score": 0.8,
         "cluster": "unassigned", "age": None}])
    out = rp.persona_faces(d, rp.load_labels(d))
    assert list(out) == ["Alice"]
    assert out["Alice"][0]["image"] == "a.jpg"
    assert out["Alice"][0]["age"] == 30
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python3 -m pytest tests/test_restore_photos.py -q`
Expected: FAIL with `ModuleNotFoundError: No module named 'restore_photos'`

- [ ] **Step 3: Create `restore_photos.py` with the readers**

```python
"""Identity-grounded photo restoration — reads face_pipeline artifacts only."""
from __future__ import annotations

import argparse
import glob
import json
import os
import sys

import numpy as np


def faces_sidecar_path(image_path: str) -> str:
    stem, _ = os.path.splitext(image_path)
    return stem + ".faces.json"


def read_faces_json(image_path: str):
    path = faces_sidecar_path(image_path)
    if not os.path.exists(path):
        return None
    with open(path) as f:
        return json.load(f)


def load_labels(images_dir: str) -> dict:
    path = os.path.join(images_dir, "labels.json")
    if not os.path.exists(path):
        return {}
    with open(path) as f:
        return json.load(f)


def persona_faces(images_dir: str, labels_map: dict) -> dict:
    """{persona name: [{image, face_id, bbox, det_score, age}, ...]}."""
    out: dict = {}
    for path in sorted(glob.glob(os.path.join(images_dir, "*.faces.json"))):
        with open(path) as f:
            data = json.load(f)
        for face in data.get("faces", []):
            name = labels_map.get(face.get("cluster", ""), "")
            if not name:
                continue
            out.setdefault(name, []).append({
                "image": data["image"],
                "face_id": face["id"],
                "bbox": face["bbox"],
                "det_score": face.get("det_score", 0.0),
                "age": face.get("age"),
            })
    return out
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python3 -m pytest tests/test_restore_photos.py -q`
Expected: PASS (3 passed)

- [ ] **Step 5: Commit**

```bash
git add restore_photos.py tests/test_restore_photos.py
git commit -m "feat: restore_photos scaffold + artifact readers"
```

---

## Task 9: `crop_face` + `sharpness`

**Files:**
- Modify: `restore_photos.py`
- Test: `tests/test_restore_photos.py`

- [ ] **Step 1: Write the failing tests**

```python
def test_crop_face_clamps_and_crops():
    img = np.zeros((20, 20, 3), np.uint8)
    img[5:15, 5:15] = 255
    crop = rp.crop_face(img, [5, 5, 15, 15])
    assert crop.shape == (10, 10, 3)
    assert int(crop.mean()) == 255


def test_sharpness_sharp_beats_blurred():
    import cv2
    sharp = np.zeros((40, 40), np.uint8)
    sharp[:, 20:] = 255                      # hard edge
    blurred = cv2.GaussianBlur(sharp, (9, 9), 5)
    assert rp.sharpness(sharp) > rp.sharpness(blurred)


def test_sharpness_empty_is_zero():
    assert rp.sharpness(np.zeros((0, 0, 3), np.uint8)) == 0.0
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python3 -m pytest tests/test_restore_photos.py -k "crop_face or sharpness" -q`
Expected: FAIL with `AttributeError: ... 'crop_face'`

- [ ] **Step 3: Write minimal implementation**

```python
def crop_face(image, bbox):
    """Crop to [x1,y1,x2,y2], clamped to bounds (mirrors face_pipeline)."""
    h, w = image.shape[:2]
    x1, y1, x2, y2 = [int(round(v)) for v in bbox]
    x1, x2 = max(0, min(x1, w)), max(0, min(x2, w))
    y1, y2 = max(0, min(y1, h)), max(0, min(y2, h))
    if x2 <= x1:
        x1, x2 = 0, w
    if y2 <= y1:
        y1, y2 = 0, h
    return image[y1:y2, x1:x2]


def sharpness(crop) -> float:
    """Variance of the Laplacian — higher is sharper. 0 for empty."""
    import cv2
    if crop is None or crop.size == 0:
        return 0.0
    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY) if crop.ndim == 3 else crop
    return float(cv2.Laplacian(gray, cv2.CV_64F).var())
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python3 -m pytest tests/test_restore_photos.py -k "crop_face or sharpness" -q`
Expected: PASS (3 passed)

- [ ] **Step 5: Commit**

```bash
git add restore_photos.py tests/test_restore_photos.py
git commit -m "feat: crop_face + sharpness (variance of Laplacian)"
```

---

## Task 10: `reference_quality` + `enrich_candidate`

**Files:**
- Modify: `restore_photos.py`
- Test: `tests/test_restore_photos.py`

- [ ] **Step 1: Write the failing tests**

```python
def test_reference_quality_monotonic():
    base = rp.reference_quality(100, 0.8, 50)
    assert rp.reference_quality(200, 0.8, 50) > base     # bigger area
    assert rp.reference_quality(100, 0.95, 50) > base    # higher det_score
    assert rp.reference_quality(100, 0.8, 80) > base      # sharper


def test_enrich_candidate_adds_quality_and_area():
    face = {"image": "a.jpg", "face_id": 0, "bbox": [0, 0, 10, 10],
            "det_score": 0.9, "age": 30}
    crop = np.zeros((10, 10, 3), np.uint8)
    crop[:, 5:] = 255
    out = rp.enrich_candidate(face, crop)
    assert out["area"] == 100
    assert out["sharpness"] >= 0.0
    assert out["quality"] >= 0.0
    assert out["age"] == 30 and out["face_id"] == 0
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python3 -m pytest tests/test_restore_photos.py -k "reference_quality or enrich_candidate" -q`
Expected: FAIL with `AttributeError: ... 'reference_quality'`

- [ ] **Step 3: Write minimal implementation**

```python
def reference_quality(area, det_score, sharp) -> float:
    """Monotonic combined quality; sqrt(area) keeps size from dominating."""
    return float((max(0.0, area) ** 0.5) * max(0.0, det_score) *
                 (1.0 + max(0.0, sharp)))


def enrich_candidate(face, crop) -> dict:
    """face dict + computed sharpness, area, quality."""
    s = sharpness(crop)
    x1, y1, x2, y2 = face["bbox"]
    area = max(0, x2 - x1) * max(0, y2 - y1)
    return {**face, "sharpness": s, "area": area,
            "quality": reference_quality(area, face.get("det_score", 0.0), s)}
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python3 -m pytest tests/test_restore_photos.py -k "reference_quality or enrich_candidate" -q`
Expected: PASS (2 passed)

- [ ] **Step 5: Commit**

```bash
git add restore_photos.py tests/test_restore_photos.py
git commit -m "feat: reference_quality + enrich_candidate"
```

---

## Task 11: `select_reference` (the core of the feature)

Picks the best same-person reference: prefer in-age-window by quality, else closest age, else best quality when no age info. Excludes the target face.

**Files:**
- Modify: `restore_photos.py`
- Test: `tests/test_restore_photos.py`

- [ ] **Step 1: Write the failing tests**

```python
def _cand(image, fid, age, quality):
    return {"image": image, "face_id": fid, "age": age, "quality": quality}


def test_select_reference_prefers_in_window_by_quality():
    target = {"image": "t.jpg", "face_id": 0, "age": 30}
    cands = [_cand("a.jpg", 1, 28, 10.0), _cand("b.jpg", 2, 32, 50.0),
             _cand("c.jpg", 3, 5, 99.0)]            # out of window
    best, reason = rp.select_reference(target, cands, age_window=5)
    assert reason == "in_window" and best["image"] == "b.jpg"


def test_select_reference_age_fallback_when_none_in_window():
    target = {"image": "t.jpg", "face_id": 0, "age": 30}
    cands = [_cand("a.jpg", 1, 10, 10.0), _cand("b.jpg", 2, 50, 99.0)]
    best, reason = rp.select_reference(target, cands, age_window=5)
    assert reason == "age_fallback" and best["image"] == "b.jpg"  # |50-30|<|10-30|


def test_select_reference_no_age_uses_quality():
    target = {"image": "t.jpg", "face_id": 0, "age": None}
    cands = [_cand("a.jpg", 1, None, 10.0), _cand("b.jpg", 2, None, 40.0)]
    best, reason = rp.select_reference(target, cands, age_window=5)
    assert reason == "no_age" and best["image"] == "b.jpg"


def test_select_reference_excludes_target_and_empties():
    target = {"image": "t.jpg", "face_id": 0, "age": 30}
    cands = [_cand("t.jpg", 0, 30, 99.0)]           # only the target itself
    best, reason = rp.select_reference(target, cands, age_window=5)
    assert best is None and reason == "no_reference"
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python3 -m pytest tests/test_restore_photos.py -k select_reference -q`
Expected: FAIL with `AttributeError: ... 'select_reference'`

- [ ] **Step 3: Write minimal implementation**

```python
def select_reference(target, candidates, age_window):
    """Best same-person reference. Returns (candidate|None, reason).

    reason: 'in_window' | 'age_fallback' | 'no_age' | 'no_reference'.
    """
    pool = [c for c in candidates
            if not (c["image"] == target["image"]
                    and c["face_id"] == target["face_id"])]
    if not pool:
        return None, "no_reference"
    t_age = target.get("age")
    aged = [c for c in pool if c.get("age") is not None]
    if t_age is not None and aged:
        in_window = [c for c in aged if abs(c["age"] - t_age) <= age_window]
        if in_window:
            return max(in_window, key=lambda c: c["quality"]), "in_window"
        return min(aged, key=lambda c: (abs(c["age"] - t_age), -c["quality"])), \
            "age_fallback"
    return max(pool, key=lambda c: c["quality"]), "no_age"
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python3 -m pytest tests/test_restore_photos.py -k select_reference -q`
Expected: PASS (4 passed)

- [ ] **Step 5: Commit**

```bash
git add restore_photos.py tests/test_restore_photos.py
git commit -m "feat: select_reference (quality + age-proximity ranking)"
```

---

## Task 12: `needs_escalation` gate

**Files:**
- Modify: `restore_photos.py`
- Test: `tests/test_restore_photos.py`

- [ ] **Step 1: Write the failing tests**

```python
def test_needs_escalation_triggers_on_blur_or_small():
    assert rp.needs_escalation(sharp=10.0, area=40000,
                               sharp_thresh=100.0, min_area=6400) is True   # blurry
    assert rp.needs_escalation(sharp=500.0, area=100,
                               sharp_thresh=100.0, min_area=6400) is True   # small
    assert rp.needs_escalation(sharp=500.0, area=40000,
                               sharp_thresh=100.0, min_area=6400) is False  # fine
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python3 -m pytest tests/test_restore_photos.py -k needs_escalation -q`
Expected: FAIL with `AttributeError: ... 'needs_escalation'`

- [ ] **Step 3: Write minimal implementation**

```python
def needs_escalation(sharp, area, sharp_thresh, min_area) -> bool:
    """A face needs identity-grounded Stage 2 when it is blurry or small."""
    return bool(sharp < sharp_thresh or area < min_area)
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python3 -m pytest tests/test_restore_photos.py -k needs_escalation -q`
Expected: PASS (1 passed)

- [ ] **Step 5: Commit**

```bash
git add restore_photos.py tests/test_restore_photos.py
git commit -m "feat: needs_escalation gate"
```

---

## Task 13: `composite_face` (feathered blend-back)

**Files:**
- Modify: `restore_photos.py`
- Test: `tests/test_restore_photos.py`

- [ ] **Step 1: Write the failing tests**

```python
def test_composite_face_replaces_center_keeps_shape():
    base = np.zeros((100, 100, 3), np.uint8)
    restored = np.full((50, 50, 3), 255, np.uint8)
    out = rp.composite_face(base, restored, [25, 25, 75, 75], feather=0.2)
    assert out.shape == base.shape
    assert out[50, 50, 0] > 200                  # center fully replaced
    assert out[0, 0, 0] == 0                      # outside bbox untouched


def test_composite_face_offscreen_bbox_is_noop():
    base = np.zeros((100, 100, 3), np.uint8)
    restored = np.full((10, 10, 3), 255, np.uint8)
    out = rp.composite_face(base, restored, [200, 200, 210, 210])
    assert int(out.sum()) == 0
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python3 -m pytest tests/test_restore_photos.py -k composite_face -q`
Expected: FAIL with `AttributeError: ... 'composite_face'`

- [ ] **Step 3: Write minimal implementation**

```python
def composite_face(base, restored_face, bbox, feather: float = 0.2):
    """Blend restored_face into base at bbox with a feathered edge."""
    import cv2
    out = base.copy()
    h, w = out.shape[:2]
    x1, y1, x2, y2 = [int(round(v)) for v in bbox]
    x1, y1 = max(0, x1), max(0, y1)
    x2, y2 = min(w, x2), min(h, y2)
    bw, bh = x2 - x1, y2 - y1
    if bw <= 0 or bh <= 0:
        return out
    patch = cv2.resize(restored_face, (bw, bh), interpolation=cv2.INTER_AREA)
    mask = np.ones((bh, bw), np.float32)
    fpx = max(1, int(min(bw, bh) * feather))
    if 2 * fpx <= min(bw, bh):
        ramp = np.linspace(0.0, 1.0, fpx, dtype=np.float32)
        mask[:fpx, :] *= ramp[:, None]
        mask[-fpx:, :] *= ramp[::-1][:, None]
        mask[:, :fpx] *= ramp[None, :]
        mask[:, -fpx:] *= ramp[None, ::-1]
    m3 = mask[:, :, None]
    region = out[y1:y2, x1:x2].astype(np.float32)
    blended = patch.astype(np.float32) * m3 + region * (1.0 - m3)
    out[y1:y2, x1:x2] = blended.astype(np.uint8)
    return out
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python3 -m pytest tests/test_restore_photos.py -k composite_face -q`
Expected: PASS (2 passed)

- [ ] **Step 5: Commit**

```bash
git add restore_photos.py tests/test_restore_photos.py
git commit -m "feat: composite_face feathered blend-back"
```

---

## Task 14: Provenance writer

**Files:**
- Modify: `restore_photos.py`
- Test: `tests/test_restore_photos.py`

- [ ] **Step 1: Write the failing tests**

```python
def test_face_provenance_stage2_is_ai_reconstructed():
    ref = {"image": "ref.jpg", "face_id": 2, "age": 31, "quality": 12.3456}
    p = rp.face_provenance(0, "Alice", ref, "stage2", "instant-id",
                           {"reason": "in_window"})
    assert p["ai_reconstructed"] is True
    assert p["reference"]["image"] == "ref.jpg"
    assert p["reference"]["quality"] == 12.3456
    assert p["params"]["reason"] == "in_window"


def test_face_provenance_stage1_has_no_reference_flag_false():
    p = rp.face_provenance(1, "", None, "stage1", "real-esrgan", {})
    assert p["ai_reconstructed"] is False and p["reference"] is None


def test_write_restore_json_roundtrip(tmp_path):
    out_img = str(tmp_path / "restored" / "a.jpg")
    os.makedirs(os.path.dirname(out_img))
    prov = {"source": "a.jpg", "faces": []}
    path = rp.write_restore_json(out_img, prov)
    assert path.endswith("a.restore.json")
    with open(path) as f:
        assert json.load(f)["source"] == "a.jpg"
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python3 -m pytest tests/test_restore_photos.py -k provenance -q`
Expected: FAIL with `AttributeError: ... 'face_provenance'`

- [ ] **Step 3: Write minimal implementation**

```python
def face_provenance(face_id, persona, reference, stage, model, params) -> dict:
    """One face's restoration record. stage in {'stage1','stage2'}."""
    ref = None
    if reference is not None:
        ref = {"image": reference["image"], "face_id": reference["face_id"],
               "age": reference.get("age"),
               "quality": round(float(reference["quality"]), 4)}
    return {"face_id": face_id, "persona": persona, "reference": ref,
            "stage": stage, "model": model,
            "ai_reconstructed": stage == "stage2", "params": params}


def restore_sidecar_path(out_image_path: str) -> str:
    stem, _ = os.path.splitext(out_image_path)
    return stem + ".restore.json"


def write_restore_json(out_image_path: str, provenance: dict) -> str:
    path = restore_sidecar_path(out_image_path)
    with open(path, "w") as f:
        json.dump(provenance, f, indent=2)
    return path
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python3 -m pytest tests/test_restore_photos.py -k provenance -q`
Expected: PASS (3 passed)

- [ ] **Step 5: Commit**

```bash
git add restore_photos.py tests/test_restore_photos.py
git commit -m "feat: restoration provenance writer (ai_reconstructed flag)"
```

---

## Task 15: Provider abstraction (`FakeProvider` + `ReplicateProvider`)

The provider isolates the cloud calls so the pipeline is testable with a fake. `ReplicateProvider` is verified manually; `FakeProvider` drives tests.

**Files:**
- Modify: `restore_photos.py`
- Test: `tests/test_restore_photos.py`

- [ ] **Step 1: Write the failing tests**

```python
def test_fake_provider_records_calls():
    p = rp.FakeProvider()
    img = np.zeros((4, 4, 3), np.uint8)
    assert p.enhance(img).shape == img.shape
    assert p.identity_restore(img, img).shape == img.shape
    assert [c[0] for c in p.calls] == ["enhance", "identity_restore"]
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python3 -m pytest tests/test_restore_photos.py -k fake_provider -q`
Expected: FAIL with `AttributeError: ... 'FakeProvider'`

- [ ] **Step 3: Write the implementation**

```python
class FakeProvider:
    """Test double: echoes the input image, records calls. No network."""
    name = "fake"

    def __init__(self):
        self.calls = []

    def enhance(self, bgr):
        self.calls.append(("enhance", bgr.shape))
        return bgr

    def identity_restore(self, degraded_bgr, reference_bgr):
        self.calls.append(("identity_restore", degraded_bgr.shape))
        return degraded_bgr


class ReplicateProvider:
    """Replicate-backed provider. Verified manually (network + paid)."""
    name = "replicate"

    # Pin specific model versions during manual verification; placeholders here
    # are intentionally swapped for real slugs when first wiring the account.
    ENHANCE_MODEL = "tencentarc/gfpgan"
    IDENTITY_MODEL = "zsxkib/instant-id"

    def __init__(self):
        if not os.environ.get("REPLICATE_API_TOKEN"):
            raise SystemExit(
                "REPLICATE_API_TOKEN not set. Export it or use --dry-run.")
        try:
            import replicate  # noqa: F401
        except ImportError as e:
            raise SystemExit(
                "replicate not installed. Run: pip install -r requirements.txt"
            ) from e

    def _run(self, model, inputs):
        import io
        import cv2
        import replicate
        from PIL import Image
        import urllib.request
        out = replicate.run(model, input=inputs)
        url = out[0] if isinstance(out, list) else out
        with urllib.request.urlopen(str(url)) as r:
            buf = r.read()
        rgb = np.array(Image.open(io.BytesIO(buf)).convert("RGB"))
        return cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)

    def _to_png(self, bgr):
        import cv2
        ok, buf = cv2.imencode(".png", bgr)
        import io
        return io.BytesIO(buf.tobytes())

    def enhance(self, bgr):
        return self._run(self.ENHANCE_MODEL, {"img": self._to_png(bgr)})

    def identity_restore(self, degraded_bgr, reference_bgr):
        return self._run(self.IDENTITY_MODEL, {
            "image": self._to_png(reference_bgr),
            "pose_image": self._to_png(degraded_bgr)})


def make_provider(name: str):
    """Provider factory used by the CLI."""
    if name == "fake":
        return FakeProvider()
    if name == "replicate":
        return ReplicateProvider()
    raise SystemExit(f"unknown provider: {name}")
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python3 -m pytest tests/test_restore_photos.py -k fake_provider -q`
Expected: PASS (1 passed)

- [ ] **Step 5: Commit**

```bash
git add restore_photos.py tests/test_restore_photos.py
git commit -m "feat: provider abstraction (FakeProvider + ReplicateProvider)"
```

> **Note for manual wiring:** `ENHANCE_MODEL`/`IDENTITY_MODEL` slugs and their input keys are confirmed against Replicate's current model pages during manual verification (Task 18 notes), since hosted model signatures change. The exact slug/version is not asserted in tests.

---

## Task 16: `restore_photo` orchestration

Ties the helpers together with dependency-injected provider. Tested with `FakeProvider` + synthetic images.

**Files:**
- Modify: `restore_photos.py`
- Test: `tests/test_restore_photos.py`

- [ ] **Step 1: Write the failing tests**

```python
import cv2  # add near top of test file if not already imported


def _img(d, name, fill, size=(120, 120)):
    img = np.full((size[1], size[0], 3), fill, np.uint8)
    cv2.imwrite(os.path.join(d, name), img)


def _scene(tmp_path):
    """Two photos of 'Alice': a sharp reference and a blurry target."""
    d = str(tmp_path)
    with open(os.path.join(d, "labels.json"), "w") as f:
        json.dump({"person_000": "Alice"}, f)
    # sharp reference photo
    ref = np.zeros((120, 120, 3), np.uint8)
    ref[:, 60:] = 255
    cv2.imwrite(os.path.join(d, "ref.jpg"), ref)
    _sidecar(d, "ref.jpg", [{"id": 0, "bbox": [10, 10, 110, 110],
             "det_score": 0.95, "cluster": "person_000", "age": 30}],
             size=(120, 120))
    # blurry target photo (heavy blur -> low sharpness -> escalates)
    tgt = cv2.GaussianBlur(ref, (21, 21), 12)
    cv2.imwrite(os.path.join(d, "tgt.jpg"), tgt)
    _sidecar(d, "tgt.jpg", [{"id": 0, "bbox": [10, 10, 110, 110],
             "det_score": 0.9, "cluster": "person_000", "age": 31}],
             size=(120, 120))
    return d


def test_restore_photo_escalates_blurry_known_face(tmp_path):
    d = _scene(tmp_path)
    prov = rp.FakeProvider()
    out_img, provenance = rp.restore_photo(
        d, "tgt.jpg", "face", prov, rp.load_labels(d),
        sharp_thresh=10_000.0, min_area=1)        # force escalation
    assert ("identity_restore", (100, 100, 3)) in prov.calls
    assert provenance["faces"][0]["ai_reconstructed"] is True
    assert provenance["faces"][0]["reference"]["image"] == "ref.jpg"
    assert out_img.shape == (120, 120, 3)


def test_restore_photo_stage1_when_sharp(tmp_path):
    d = _scene(tmp_path)
    prov = rp.FakeProvider()
    _, provenance = rp.restore_photo(
        d, "ref.jpg", "face", prov, rp.load_labels(d),
        sharp_thresh=0.0, min_area=1)             # never escalate
    assert provenance["faces"][0]["stage"] == "stage1"
    assert all(c[0] == "enhance" for c in prov.calls)


def test_restore_photo_dry_run_makes_no_calls(tmp_path):
    d = _scene(tmp_path)
    prov = rp.FakeProvider()
    _, provenance = rp.restore_photo(
        d, "tgt.jpg", "face", prov, rp.load_labels(d),
        sharp_thresh=10_000.0, min_area=1, dry_run=True)
    assert prov.calls == []
    assert provenance["faces"][0]["stage"] in ("stage1", "stage2")
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python3 -m pytest tests/test_restore_photos.py -k restore_photo -q`
Expected: FAIL with `AttributeError: ... 'restore_photo'`

- [ ] **Step 3: Write the implementation**

```python
def _enriched_personas(images_dir, by_persona):
    """persona -> [enriched candidate, ...] (loads each source once)."""
    import cv2
    src_cache: dict = {}
    out: dict = {}
    for name, cands in by_persona.items():
        lst = []
        for c in cands:
            if c["image"] not in src_cache:
                src_cache[c["image"]] = cv2.imread(
                    os.path.join(images_dir, c["image"]))
            src = src_cache[c["image"]]
            crop = (crop_face(src, c["bbox"]) if src is not None
                    else np.zeros((1, 1, 3), np.uint8))
            lst.append(enrich_candidate(c, crop))
        out[name] = lst
    return out


def restore_photo(images_dir, image, mode, provider, labels_map,
                  age_window: int = 5, sharp_thresh: float = 100.0,
                  min_area: int = 80 * 80, dry_run: bool = False,
                  log=print):
    """Restore one photo. Returns (out_image|None, provenance dict).

    mode: 'face' (composite restored faces into the original) or
    'photo' (enhance the whole image first, then identity-ground faces).
    """
    import cv2
    src = cv2.imread(os.path.join(images_dir, image))
    if src is None:
        log(f"  ! cannot read {image}")
        return None, {"source": image, "faces": []}
    data = read_faces_json(os.path.join(images_dir, image))
    faces = data["faces"] if data else []
    enriched = _enriched_personas(images_dir, persona_faces(images_dir, labels_map))

    if mode == "photo" and not dry_run:
        out_img = provider.enhance(src)
    else:
        out_img = src.copy()

    face_provs = []
    for face in faces:
        persona = labels_map.get(face.get("cluster", ""), "") or face.get("label", "")
        crop = crop_face(src, face["bbox"])
        s = sharpness(crop)
        x1, y1, x2, y2 = face["bbox"]
        area = max(0, x2 - x1) * max(0, y2 - y1)
        escalate = needs_escalation(s, area, sharp_thresh, min_area)

        reference, reason = None, "no_persona"
        if persona and persona in enriched:
            target = {"image": image, "face_id": face["id"], "age": face.get("age")}
            reference, reason = select_reference(target, enriched[persona], age_window)

        stage = "stage1"
        params = {"reason": reason, "sharpness": round(s, 2), "area": area,
                  "mode": mode}
        if escalate and reference is not None:
            stage = "stage2"
            if not dry_run:
                ref_src = cv2.imread(os.path.join(images_dir, reference["image"]))
                ref_crop = (crop_face(ref_src, reference["bbox"])
                            if ref_src is not None else crop)
                restored = provider.identity_restore(crop, ref_crop)
                out_img = composite_face(out_img, restored, face["bbox"])
        elif mode == "face" and not dry_run:
            enhanced = provider.enhance(crop)
            out_img = composite_face(out_img, enhanced, face["bbox"])

        model = (provider.name if stage == "stage2"
                 else (provider.name if mode == "face" else "whole-photo"))
        face_provs.append(face_provenance(
            face["id"], persona, reference, stage, model, params))
        log(f"  {image} #{face['id']}: {stage} "
            f"({persona or 'unknown'}, {reason})")

    return out_img, {"source": image, "faces": face_provs}
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python3 -m pytest tests/test_restore_photos.py -k restore_photo -q`
Expected: PASS (3 passed)

- [ ] **Step 5: Commit**

```bash
git add restore_photos.py tests/test_restore_photos.py
git commit -m "feat: restore_photo orchestration (escalation + provenance)"
```

---

## Task 17: CLI (`main`) with `face`/`photo`, `--dry-run`, flags

**Files:**
- Modify: `restore_photos.py`
- Test: `tests/test_restore_photos.py`

- [ ] **Step 1: Write the failing tests**

```python
def test_main_dry_run_writes_nothing(tmp_path):
    d = _scene(tmp_path)
    rc = rp.main(["restore_photos.py", "face", "tgt.jpg", "--images", d,
                  "--provider", "fake", "--dry-run",
                  "--sharpness-thresh", "10000", "--min-area", "1"])
    assert rc == 0
    assert not os.path.exists(os.path.join(d, "restored", "tgt.jpg"))


def test_main_face_mode_writes_output_and_provenance(tmp_path):
    d = _scene(tmp_path)
    rc = rp.main(["restore_photos.py", "face", "tgt.jpg", "--images", d,
                  "--provider", "fake",
                  "--sharpness-thresh", "10000", "--min-area", "1"])
    assert rc == 0
    assert os.path.exists(os.path.join(d, "restored", "tgt.jpg"))
    with open(os.path.join(d, "restored", "tgt.restore.json")) as f:
        prov = json.load(f)
    assert prov["faces"][0]["ai_reconstructed"] is True
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python3 -m pytest tests/test_restore_photos.py -k main_ -q`
Expected: FAIL with `AttributeError: ... 'main'`

- [ ] **Step 3: Write the implementation**

```python
def run_restore(images_dir, image, mode, provider_name, age_window,
                sharp_thresh, min_area, out_dir, dry_run) -> int:
    import cv2
    labels_map = load_labels(images_dir)
    provider = None if dry_run else make_provider(provider_name)
    if provider is None:                       # dry-run still needs the fake API shape
        provider = FakeProvider()
    out_img, provenance = restore_photo(
        images_dir, image, mode, provider, labels_map,
        age_window=age_window, sharp_thresh=sharp_thresh,
        min_area=min_area, dry_run=dry_run)
    if dry_run:
        print("[dry-run] no images written.")
        return 0
    if out_img is None:
        return 1
    os.makedirs(os.path.join(images_dir, out_dir), exist_ok=True)
    out_path = os.path.join(images_dir, out_dir, image)
    cv2.imwrite(out_path, out_img)
    write_restore_json(out_path, provenance)
    print(f"Wrote {out_path} (+ provenance).")
    return 0


def main(argv: list[str]) -> int:
    p = argparse.ArgumentParser(description="Identity-grounded photo restoration")
    sub = p.add_subparsers(dest="cmd", required=True)
    for cmd, helptext in (("face", "restore faces only, composite back"),
                          ("photo", "restore the whole photo")):
        sp = sub.add_parser(cmd, help=helptext)
        sp.add_argument("image", help="filename within --images")
        sp.add_argument("--images", default="extracted")
        sp.add_argument("--provider", default="replicate",
                        choices=["replicate", "fake"])
        sp.add_argument("--age-window", type=int, default=5)
        sp.add_argument("--sharpness-thresh", type=float, default=100.0)
        sp.add_argument("--min-area", type=int, default=80 * 80)
        sp.add_argument("--out", default="restored")
        sp.add_argument("--dry-run", action="store_true")
    args = p.parse_args(argv[1:])
    return run_restore(args.images, args.image, args.cmd, args.provider,
                       args.age_window, args.sharpness_thresh, args.min_area,
                       args.out, args.dry_run)


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python3 -m pytest tests/test_restore_photos.py -k main_ -q`
Expected: PASS (2 passed)

- [ ] **Step 5: Commit**

```bash
git add restore_photos.py tests/test_restore_photos.py
git commit -m "feat: restore_photos CLI (face/photo, --dry-run, flags)"
```

---

## Task 18: Dependencies, docs, full verification

**Files:**
- Modify: `requirements.txt`, `CLAUDE.md`

- [ ] **Step 1: Add the dependency**

Append to `requirements.txt`:

```
replicate>=0.25
```

- [ ] **Step 2: Install it in both interpreters (per repo note)**

```bash
python3 -m pip install 'replicate>=0.25'
.venv/bin/pip install 'replicate>=0.25'
```
Expected: installs cleanly.

- [ ] **Step 3: Run the full test suite**

Run: `python3 -m pytest tests/ -q`
Expected: PASS (all face_pipeline + restore_photos tests).

- [ ] **Step 4: `--dry-run` smoke on real data**

```bash
.venv/bin/python restore_photos.py face original-001_02.jpg --provider fake --dry-run
```
Expected: prints a per-face `stage1/stage2 (persona, reason)` plan and `[dry-run] no images written.`; no `restored/` directory created.

- [ ] **Step 5: Manual verification of the real provider (human, paid API)**

With `REPLICATE_API_TOKEN` exported, confirm/adjust the `ReplicateProvider` model slugs + input keys against the current Replicate model pages, then:

```bash
.venv/bin/python restore_photos.py face original-001_02.jpg
```
Confirm `extracted/restored/original-001_02.jpg` and its `.restore.json` are written and the restored faces look reasonable. (Quality + model-slug correctness is the manual-verification surface, per repo convention.)

- [ ] **Step 6: Update `CLAUDE.md`**

Add a `## Photo restoration (restore_photos.py)` section documenting: the `face`/`photo` modes; `--dry-run`/`--age-window`/`--provider`/`--sharpness-thresh` flags; that it reads `labels.json` + sidecars and writes `restored/<name>.jpg` + `restored/<name>.restore.json`; that it never imports `face_pipeline.py`; the deterministic-first/identity-grounded escalation; and that age comes from `detect --backfill-age` + the `ages` GUI. Also add `detect --backfill-age` and the `ages` subcommand to the face-pipeline command list.

- [ ] **Step 7: Commit**

```bash
git add requirements.txt CLAUDE.md
git commit -m "docs: document restore_photos + age tooling; add replicate dep"
```

---

## Self-Review (completed during planning)

**Spec coverage:**
- Age `age`/`age_source` storage → Tasks 3–5. ✓
- Merge-only `detect --backfill-age` → Tasks 1, 2, 6. ✓
- Manual `ages` GUI reusing shared helpers → Task 7. ✓
- Reference selection (quality + age window + flagged fallback) → Tasks 10, 11. ✓
- Deterministic-first escalation → Tasks 12, 16. ✓
- Identity-grounded Stage 2 via cloud provider → Tasks 15, 16. ✓
- Face-only vs whole-photo modes → Tasks 16, 17. ✓
- Provenance with `ai_reconstructed` → Tasks 14, 16. ✓
- `--dry-run` / `--age-window` / `--provider` / thresholds → Task 17. ✓
- Faces without identity → Stage 1 only → covered by `restore_photo` (no reference ⇒ stage1). ✓
- No cross-imports; artifacts as interface → Task 8 readers, enforced throughout. ✓
- Dependencies + docs → Task 18. ✓

**Type consistency:** candidate dicts carry `image`/`face_id`/`bbox`/`det_score`/`age`, enriched with `sharpness`/`area`/`quality`; `select_reference` consumes `quality`/`age`; `face_provenance` consumes `reference["image"|"face_id"|"age"|"quality"]`; provider methods `enhance(bgr)` / `identity_restore(degraded, reference)` consistent across Tasks 15–17. ✓

**Placeholder scan:** Replicate model slugs are explicitly flagged as manual-verification items (hosted signatures change) and are deliberately not asserted in tests — every code step contains runnable code. ✓
