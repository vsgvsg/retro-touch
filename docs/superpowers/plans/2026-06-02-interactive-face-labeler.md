# Interactive Face Labeler Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace the headless `label` stage with an interactive Tkinter labeler that shows a montage of each cluster's face crops, lets the human type or reuse a name, and writes `extracted/labels.json` after every step.

**Architecture:** All in `face_pipeline.py`. A thin Tkinter `LabelerApp` shell sits over pure, TDD-tested core functions (gather cluster faces, crop, pick montage faces, build montage image, previous-names list, write labels). Only the widget wiring and the BGR→PhotoImage conversion are manual-verification.

**Tech Stack:** Python 3.9, numpy, OpenCV (cv2), Tkinter (stdlib, Tk 8.5), PIL/Pillow (already installed), pytest.

---

## File Structure

- **Modify `face_pipeline.py`** — add pure helpers `cluster_face_index`, `crop_face`, `pick_montage_faces`, `build_montage`, `previous_names`, `write_labels`; add the `LabelerApp` Tkinter class; replace the body of `run_label` (currently `face_pipeline.py:294-313`). Leave `scaffold_labels` (`:144`) and `_collect_persons_and_examples` (`:277`) untouched (now unused by run_label, tests still pass; removal out of scope).
- **Modify `tests/test_face_pipeline.py`** — append unit tests for the six pure helpers.
- **Modify `CLAUDE.md`** — update the `label` line to say it's an interactive Tkinter window.

New pure helpers go in `face_pipeline.py` just before `run_label`. `LabelerApp` goes right after the helpers, before `run_label`.

---

## Task 1: cluster_face_index

**Files:**
- Modify: `face_pipeline.py` (add helper before `run_label` at line 294)
- Modify: `tests/test_face_pipeline.py` (append)

- [ ] **Step 1: Write the failing test**

Append to `tests/test_face_pipeline.py`:

```python
def test_cluster_face_index_groups_and_excludes_unassigned(tmp_path):
    d = str(tmp_path)
    fp.write_faces_json(os.path.join(d, "a.jpg"), (100, 100), "buffalo_l", [
        {"id": 0, "bbox": [0, 0, 10, 10], "det_score": 0.9,
         "embedding_ref": 0, "cluster": "person_000", "label": ""},
        {"id": 1, "bbox": [5, 5, 15, 15], "det_score": 0.8,
         "embedding_ref": 1, "cluster": "unassigned", "label": ""},
    ])
    fp.write_faces_json(os.path.join(d, "b.jpg"), (100, 100), "buffalo_l", [
        {"id": 0, "bbox": [1, 1, 9, 9], "det_score": 0.7,
         "embedding_ref": 2, "cluster": "person_000", "label": ""},
    ])
    index = {"model": "buffalo_l", "rows": [
        {"image": "a.jpg", "face_id": 0},
        {"image": "a.jpg", "face_id": 1},
        {"image": "b.jpg", "face_id": 0},
    ]}
    out = fp.cluster_face_index(d, index)
    assert set(out.keys()) == {"person_000"}        # unassigned excluded
    assert len(out["person_000"]) == 2
    f0 = out["person_000"][0]
    assert f0["image"] == "a.jpg" and f0["face_id"] == 0
    assert f0["bbox"] == [0, 0, 10, 10] and f0["det_score"] == 0.9
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python3 -m pytest tests/test_face_pipeline.py -k cluster_face_index -q`
Expected: FAIL — `AttributeError: ... cluster_face_index`.

- [ ] **Step 3: Write minimal implementation**

Insert into `face_pipeline.py` immediately before `def run_label` (currently line 294):

```python
def cluster_face_index(images_dir: str, index: dict) -> dict:
    """Gather each real cluster's faces from the sidecars.

    Returns {person_id: [{"image", "face_id", "bbox", "det_score"}, ...]},
    in index-row order. 'unassigned' is excluded.
    """
    out: dict[str, list[dict]] = {}
    cache: dict[str, dict | None] = {}
    for row in index["rows"]:
        image = row["image"]
        if image not in cache:
            cache[image] = read_faces_json(os.path.join(images_dir, image))
        data = cache[image]
        if data is None:
            continue
        for face in data["faces"]:
            if face["id"] != row["face_id"]:
                continue
            cluster = face.get("cluster", "")
            if not cluster or cluster == "unassigned":
                break
            out.setdefault(cluster, []).append({
                "image": image,
                "face_id": face["id"],
                "bbox": face["bbox"],
                "det_score": face["det_score"],
            })
            break
    return out
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python3 -m pytest tests/test_face_pipeline.py -k cluster_face_index -q`
Expected: PASS (1 passed).

- [ ] **Step 5: Commit**

```bash
git add face_pipeline.py tests/test_face_pipeline.py
git commit -m "feat: add cluster_face_index helper"
```

---

## Task 2: crop_face

**Files:**
- Modify: `face_pipeline.py` (add helper before `run_label`)
- Modify: `tests/test_face_pipeline.py` (append)

- [ ] **Step 1: Write the failing test**

Append to `tests/test_face_pipeline.py`:

```python
def test_crop_face_in_bounds():
    img = np.zeros((100, 100, 3), dtype=np.uint8)
    img[10:20, 30:40] = 255
    crop = fp.crop_face(img, [30, 10, 40, 20])
    assert crop.shape == (10, 10, 3)
    assert (crop == 255).all()


def test_crop_face_clamps_out_of_bounds():
    img = np.zeros((50, 50, 3), dtype=np.uint8)
    # bbox extends past the right/bottom edges and starts negative
    crop = fp.crop_face(img, [-5, -5, 70, 60])
    assert crop.shape[0] == 50 and crop.shape[1] == 50  # clamped to image
    assert crop.ndim == 3
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python3 -m pytest tests/test_face_pipeline.py -k crop_face -q`
Expected: FAIL — `AttributeError: ... crop_face`.

- [ ] **Step 3: Write minimal implementation**

Insert into `face_pipeline.py` before `def run_label`:

```python
def crop_face(image: np.ndarray, bbox: list) -> np.ndarray:
    """Crop image to bbox [x1,y1,x2,y2], clamped to image bounds."""
    h, w = image.shape[:2]
    x1, y1, x2, y2 = [int(round(v)) for v in bbox]
    x1 = max(0, min(x1, w))
    x2 = max(0, min(x2, w))
    y1 = max(0, min(y1, h))
    y2 = max(0, min(y2, h))
    if x2 <= x1:
        x1, x2 = 0, w
    if y2 <= y1:
        y1, y2 = 0, h
    return image[y1:y2, x1:x2]
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python3 -m pytest tests/test_face_pipeline.py -k crop_face -q`
Expected: PASS (2 passed).

- [ ] **Step 5: Commit**

```bash
git add face_pipeline.py tests/test_face_pipeline.py
git commit -m "feat: add crop_face helper"
```

---

## Task 3: pick_montage_faces

**Files:**
- Modify: `face_pipeline.py` (add helper before `run_label`)
- Modify: `tests/test_face_pipeline.py` (append)

- [ ] **Step 1: Write the failing test**

Append to `tests/test_face_pipeline.py`:

```python
def test_pick_montage_faces_top_k_by_score():
    faces = [
        {"image": "a.jpg", "face_id": 0, "bbox": [0, 0, 1, 1], "det_score": 0.5},
        {"image": "a.jpg", "face_id": 1, "bbox": [0, 0, 1, 1], "det_score": 0.9},
        {"image": "b.jpg", "face_id": 0, "bbox": [0, 0, 1, 1], "det_score": 0.7},
    ]
    out = fp.pick_montage_faces(faces, k=2)
    assert [f["det_score"] for f in out] == [0.9, 0.7]


def test_pick_montage_faces_returns_all_when_fewer_than_k():
    faces = [{"image": "a.jpg", "face_id": 0, "bbox": [0, 0, 1, 1], "det_score": 0.5}]
    out = fp.pick_montage_faces(faces, k=9)
    assert len(out) == 1
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python3 -m pytest tests/test_face_pipeline.py -k pick_montage -q`
Expected: FAIL — `AttributeError: ... pick_montage_faces`.

- [ ] **Step 3: Write minimal implementation**

Insert into `face_pipeline.py` before `def run_label`:

```python
def pick_montage_faces(faces: list[dict], k: int = 9) -> list[dict]:
    """The k highest-det_score faces, descending."""
    return sorted(faces, key=lambda f: f["det_score"], reverse=True)[:k]
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python3 -m pytest tests/test_face_pipeline.py -k pick_montage -q`
Expected: PASS (2 passed).

- [ ] **Step 5: Commit**

```bash
git add face_pipeline.py tests/test_face_pipeline.py
git commit -m "feat: add pick_montage_faces helper"
```

---

## Task 4: build_montage

**Files:**
- Modify: `face_pipeline.py` (add helper before `run_label`)
- Modify: `tests/test_face_pipeline.py` (append)

- [ ] **Step 1: Write the failing test**

Append to `tests/test_face_pipeline.py`:

```python
def test_build_montage_full_grid_shape():
    crops = [np.full((20, 20, 3), 200, np.uint8) for _ in range(9)]
    m = fp.build_montage(crops, cols=3, cell=128, pad=4)
    # 3 cols x 3 rows of 128 cells with 4px padding between and around
    expected = 3 * 128 + 4 * (3 + 1)
    assert m.shape == (expected, expected, 3)


def test_build_montage_partial_last_row():
    crops = [np.full((20, 20, 3), 200, np.uint8) for _ in range(4)]
    m = fp.build_montage(crops, cols=3, cell=128, pad=4)
    # 4 crops -> 2 rows (3 + 1)
    rows = 2
    exp_h = rows * 128 + 4 * (rows + 1)
    exp_w = 3 * 128 + 4 * (3 + 1)
    assert m.shape == (exp_h, exp_w, 3)


def test_build_montage_zero_crops_placeholder():
    m = fp.build_montage([], cols=3, cell=128, pad=4)
    assert m.ndim == 3 and m.shape[2] == 3
    assert m.shape[0] > 0 and m.shape[1] > 0
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python3 -m pytest tests/test_face_pipeline.py -k build_montage -q`
Expected: FAIL — `AttributeError: ... build_montage`.

- [ ] **Step 3: Write minimal implementation**

Insert into `face_pipeline.py` before `def run_label`:

```python
def _fit_cell(crop: np.ndarray, cell: int) -> np.ndarray:
    """Resize crop to fit a cell x cell box, letterboxed on gray, BGR."""
    import cv2
    canvas = np.full((cell, cell, 3), 128, np.uint8)
    h, w = crop.shape[:2]
    if h == 0 or w == 0:
        return canvas
    s = min(cell / w, cell / h)
    nw, nh = max(1, int(w * s)), max(1, int(h * s))
    resized = cv2.resize(crop, (nw, nh), interpolation=cv2.INTER_AREA)
    y0, x0 = (cell - nh) // 2, (cell - nw) // 2
    canvas[y0:y0 + nh, x0:x0 + nw] = resized
    return canvas


def build_montage(crops: list, cols: int = 3, cell: int = 128,
                  pad: int = 4) -> np.ndarray:
    """Tile crops into one BGR image: cols per row, gray padding, partial last row.

    Zero crops -> a single gray placeholder cell.
    """
    if not crops:
        return np.full((cell + 2 * pad, cell + 2 * pad, 3), 128, np.uint8)
    rows = (len(crops) + cols - 1) // cols
    h = rows * cell + pad * (rows + 1)
    w = cols * cell + pad * (cols + 1)
    montage = np.full((h, w, 3), 128, np.uint8)
    for i, crop in enumerate(crops):
        r, c = divmod(i, cols)
        y = pad + r * (cell + pad)
        x = pad + c * (cell + pad)
        montage[y:y + cell, x:x + cell] = _fit_cell(crop, cell)
    return montage
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python3 -m pytest tests/test_face_pipeline.py -k build_montage -q`
Expected: PASS (3 passed).

- [ ] **Step 5: Commit**

```bash
git add face_pipeline.py tests/test_face_pipeline.py
git commit -m "feat: add build_montage helper"
```

---

## Task 5: previous_names

**Files:**
- Modify: `face_pipeline.py` (add helper before `run_label`)
- Modify: `tests/test_face_pipeline.py` (append)

- [ ] **Step 1: Write the failing test**

Append to `tests/test_face_pipeline.py`:

```python
def test_previous_names_sorted_unique_nonempty():
    labels = {"person_000": "Bob", "person_001": "", "person_002": "Alice",
              "person_003": "Bob"}
    assert fp.previous_names(labels) == ["Alice", "Bob"]


def test_previous_names_empty():
    assert fp.previous_names({"person_000": "", "person_001": ""}) == []
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python3 -m pytest tests/test_face_pipeline.py -k previous_names -q`
Expected: FAIL — `AttributeError: ... previous_names`.

- [ ] **Step 3: Write minimal implementation**

Insert into `face_pipeline.py` before `def run_label`:

```python
def previous_names(labels_map: dict) -> list[str]:
    """Sorted unique non-empty names already entered."""
    return sorted({v for v in labels_map.values() if v})
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python3 -m pytest tests/test_face_pipeline.py -k previous_names -q`
Expected: PASS (2 passed).

- [ ] **Step 5: Commit**

```bash
git add face_pipeline.py tests/test_face_pipeline.py
git commit -m "feat: add previous_names helper"
```

---

## Task 6: write_labels

**Files:**
- Modify: `face_pipeline.py` (add helper before `run_label`)
- Modify: `tests/test_face_pipeline.py` (append)

- [ ] **Step 1: Write the failing test**

Append to `tests/test_face_pipeline.py`:

```python
def test_write_labels_roundtrip(tmp_path):
    d = str(tmp_path)
    path = fp.write_labels(d, {"person_000": "Alice", "person_001": "Bob"})
    assert path == os.path.join(d, "labels.json")
    with open(path) as f:
        data = json.load(f)
    assert data == {"person_000": "Alice", "person_001": "Bob"}
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python3 -m pytest tests/test_face_pipeline.py -k write_labels -q`
Expected: FAIL — `AttributeError: ... write_labels`.

- [ ] **Step 3: Write minimal implementation**

Insert into `face_pipeline.py` before `def run_label`:

```python
def write_labels(images_dir: str, labels_map: dict) -> str:
    """Write labels.json (the after-each save). Returns the path."""
    path = os.path.join(images_dir, "labels.json")
    with open(path, "w") as f:
        json.dump(labels_map, f, indent=2)
    return path
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python3 -m pytest tests/test_face_pipeline.py -k write_labels -q`
Expected: PASS (1 passed). Then run the full file: `python3 -m pytest tests/test_face_pipeline.py -q` (should be all green).

- [ ] **Step 5: Commit**

```bash
git add face_pipeline.py tests/test_face_pipeline.py
git commit -m "feat: add write_labels helper"
```

---

## Task 7: LabelerApp (Tkinter UI) + run_label rewrite

**Files:**
- Modify: `face_pipeline.py` (add `LabelerApp` before `run_label`; replace body of `run_label` at lines 294-313)

This is the human-operated UI shell — NOT unit-tested (per repo convention for `Editor`-style HighGUI/Tk classes). Verified by a manual session. It uses the six pure helpers from Tasks 1-6 (all tested).

- [ ] **Step 1: Add the LabelerApp class**

Insert into `face_pipeline.py` immediately before `def run_label`:

```python
class LabelerApp:
    """Tkinter labeler: montage per cluster -> a name, saved after each step."""

    def __init__(self, images_dir, cluster_index, labels_map):
        import tkinter as tk
        self.tk = tk
        self.images_dir = images_dir
        self.cluster_ids = sorted(cluster_index)
        self.cluster_index = cluster_index
        self.labels_map = labels_map
        self.idx = 0
        self._img_cache = {}      # source image (BGR) per filename
        self._photo = None        # keep a ref so Tk doesn't GC the image

        self.root = tk.Tk()
        self.root.title("Face Labeler")
        self.title_var = tk.StringVar()
        tk.Label(self.root, textvariable=self.title_var,
                 font=("TkDefaultFont", 13, "bold")).pack(pady=(8, 4))
        self.canvas = tk.Label(self.root)
        self.canvas.pack(padx=8)

        self.name_var = tk.StringVar()
        self.entry = tk.Entry(self.root, textvariable=self.name_var, width=30)
        self.entry.pack(pady=6)
        self.entry.bind("<Return>", lambda e: self._next())

        tk.Label(self.root, text="Previous names (click to reuse):").pack()
        self.listbox = tk.Listbox(self.root, height=6)
        self.listbox.pack(fill="x", padx=8)
        self.listbox.bind("<<ListboxSelect>>", self._on_pick)

        bar = tk.Frame(self.root)
        bar.pack(pady=8)
        tk.Button(bar, text="Back", command=self._back).pack(side="left", padx=4)
        tk.Button(bar, text="Skip", command=self._skip).pack(side="left", padx=4)
        self.next_btn = tk.Button(bar, text="Next", command=self._next)
        self.next_btn.pack(side="left", padx=4)
        self.root.protocol("WM_DELETE_WINDOW", self._on_close)

    # ---- image helpers ----
    def _source(self, image_name):
        import cv2
        if image_name not in self._img_cache:
            self._img_cache[image_name] = cv2.imread(
                os.path.join(self.images_dir, image_name))
        return self._img_cache[image_name]

    def _montage_for(self, cluster_id):
        faces = pick_montage_faces(self.cluster_index[cluster_id], k=9)
        crops = []
        for f in faces:
            src = self._source(f["image"])
            if src is None:
                crops.append(np.full((64, 64, 3), 128, np.uint8))  # placeholder
            else:
                crops.append(crop_face(src, f["bbox"]))
        return build_montage(crops, cols=3)

    def _show(self):
        import cv2
        from PIL import Image, ImageTk
        cid = self.cluster_ids[self.idx]
        n = len(self.cluster_index[cid])
        self.title_var.set(f"Cluster {cid}  ({self.idx + 1} of "
                           f"{len(self.cluster_ids)}) — {n} faces")
        montage = self._montage_for(cid)
        rgb = cv2.cvtColor(montage, cv2.COLOR_BGR2RGB)
        self._photo = ImageTk.PhotoImage(Image.fromarray(rgb))
        self.canvas.configure(image=self._photo)
        self.name_var.set(self.labels_map.get(cid, ""))
        self.entry.focus_set()
        self._refresh_names()
        self.next_btn.configure(
            text="Done" if self.idx == len(self.cluster_ids) - 1 else "Next")

    def _refresh_names(self):
        self.listbox.delete(0, self.tk.END)
        for nm in previous_names(self.labels_map):
            self.listbox.insert(self.tk.END, nm)

    # ---- events ----
    def _on_pick(self, _evt):
        sel = self.listbox.curselection()
        if sel:
            self.name_var.set(self.listbox.get(sel[0]))

    def _commit_current(self):
        """Apply the entry to labels_map (name set, or removed if blank); save."""
        cid = self.cluster_ids[self.idx]
        name = self.name_var.get().strip()
        if name:
            self.labels_map[cid] = name
        else:
            self.labels_map.pop(cid, None)
        write_labels(self.images_dir, self.labels_map)

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

    def _skip(self):
        cid = self.cluster_ids[self.idx]
        self.labels_map.pop(cid, None)
        write_labels(self.images_dir, self.labels_map)
        if self.idx == len(self.cluster_ids) - 1:
            self.root.destroy()
            return
        self.idx += 1
        self._show()

    def _on_close(self):
        self._commit_current()
        self.root.destroy()

    def run(self):
        self._show()
        self.root.mainloop()
```

- [ ] **Step 2: Replace the body of run_label**

Replace the ENTIRE current `run_label` function (lines 294-313, from `def run_label(images_dir: str) -> int:` through its `return 0`) with:

```python
def run_label(images_dir: str) -> int:
    """Launch the interactive Tkinter labeler over the clustered faces."""
    try:
        import tkinter  # noqa: F401
    except ImportError as e:
        raise SystemExit(
            "tkinter not available. Install a Python build with Tk support "
            "(e.g. python.org installer, or `brew install python-tk`).") from e
    _, index = load_cache(images_dir)  # raises FileNotFoundError w/ guidance
    cluster_index = cluster_face_index(images_dir, index)
    if not cluster_index:
        print("No clusters found. Run 'cluster' first.")
        return 1
    labels_map = {}
    path = os.path.join(images_dir, "labels.json")
    if os.path.exists(path):  # resume: pre-fill existing names
        with open(path) as f:
            labels_map = json.load(f)
    app = LabelerApp(images_dir, cluster_index, labels_map)
    app.run()
    print(f"Saved labels -> {path}")
    return 0
```

- [ ] **Step 3: Confirm the unit suite still passes (no UI launched)**

Run: `python3 -m pytest tests/ -q`
Expected: all tests pass (the six helper tests + existing suite). This proves the helpers and the edited file import cleanly. The Tk loop is not exercised here.

- [ ] **Step 4: Manual UI smoke test**

This needs a human at the GUI (Claude cannot drive the Tk window). The cache and clusters already exist in `extracted/` (person_000 = 5 faces, person_001 = 30 faces). To start clean, first clear any existing labels file:

```bash
rm -f extracted/labels.json
python3 face_pipeline.py label --images extracted
```

Verify, at the window:
- Title shows `Cluster person_000 (1 of 2) — 5 faces`; a 3x3-style montage of face crops renders.
- Typing a name and clicking **Next** advances to `person_001`; `extracted/labels.json` now contains person_000's name (check in another terminal: `cat extracted/labels.json`).
- On person_001, the **previous-names list** shows the name entered for person_000; clicking it fills the entry.
- **Back** returns to person_000 with its saved name pre-filled.
- **Skip** on a cluster leaves it out of `labels.json`.
- The last cluster's button reads **Done**; clicking it (or closing the window) exits and the terminal prints `Saved labels -> extracted/labels.json`.

Report the final `cat extracted/labels.json` contents.

- [ ] **Step 5: Commit**

```bash
git add face_pipeline.py
git commit -m "feat: interactive Tkinter face labeler replaces headless run_label"
```

---

## Task 8: Update CLAUDE.md

**Files:**
- Modify: `CLAUDE.md`

- [ ] **Step 1: Update the label command line**

In `CLAUDE.md`, under the `## Face pipeline (face_pipeline.py)` section, replace the existing `label` bullet:

```
- `python3 face_pipeline.py label` - scaffold extracted/labels.json (cluster id -> name)
```

with:

```
- `python3 face_pipeline.py label` - interactive Tkinter labeler: shows a montage of each cluster's faces, type or reuse a name; writes extracted/labels.json after each step (needs a human at the GUI; Claude can't drive the Tk window)
```

- [ ] **Step 2: Commit**

```bash
git add CLAUDE.md
git commit -m "docs: document interactive labeler in CLAUDE.md"
```

---

## Self-Review Notes

- **Spec coverage:** cluster_face_index (T1), crop_face (T2), pick_montage_faces (T3), build_montage incl. partial-row + zero-crop placeholder (T4), previous_names (T5), write_labels (T6), LabelerApp window/montage/entry/listbox/Back-Skip-Next-Done/save-after-each/resume + run_label rewrite with tkinter guard, missing-cache guard, no-cluster guard (T7), docs (T8). All spec sections map to tasks.
- **Type consistency:** face dicts carry `{image, face_id, bbox, det_score}` uniformly across `cluster_face_index` (T1, producer), `pick_montage_faces` (T3), and `LabelerApp._montage_for` (T7, consumer). `crop_face(image, bbox)` and `build_montage(crops, cols, cell, pad)` signatures match their call sites in T7. `write_labels(images_dir, labels_map)` and `previous_names(labels_map)` match their LabelerApp uses. labels.json schema `{person_id: name}` unchanged from the existing pipeline.
- **Placeholders:** none — every code step shows complete code.
- **Note:** `scaffold_labels` and `_collect_persons_and_examples` remain in the file and keep their passing tests; they're simply no longer called by `run_label`. Intentional per spec (removal out of scope).
