# Per-Photo Match Review UI Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** After `match` scores faces, open an interactive per-photo review window that draws numbered bbox overlays on each photo with a numbered name input per face, persists name overrides to the sidecar `label`, and assigns/mints a cluster for a newly-named `unassigned` face (reusing a same-named person's cluster when one exists).

**Architecture:** All in `face_pipeline.py`. Five pure TDD-tested helpers (`existing_cluster_ids`, `next_cluster_id`, `resolve_or_create_cluster`, `prefill_name`, `apply_photo_edits`); a new `PhotoReviewApp` Tkinter class (manual-verification, like `LabelerApp`); `run_match` keeps its scoring + `match_report.json`, then launches the review UI by default (`--no-review` opts out). `faces.npy` is never rewritten — cluster assignment lives in sidecars.

**Tech Stack:** Python 3.13 venv (`.venv`, working Tk 9.0), numpy, OpenCV, Tkinter, PIL/Pillow, pytest. Run tests/GUI via `.venv/bin/python`.

---

## File Structure

- **Modify `face_pipeline.py`:**
  - Add 5 pure helpers near the other helpers (before `class LabelerApp` at line 391 is a good spot, or grouped before `run_match`).
  - Add `class PhotoReviewApp` (after `LabelerApp`, before `run_label`/`run_match`).
  - Modify `run_match` (`:629`) to collect per-face suggestions and launch the UI unless `--no-review`.
  - Modify `main` (`:684`) to add `--no-review` to the `match` subparser and pass it through.
- **Modify `tests/test_face_pipeline.py`:** add tests for the 5 helpers.
- **Modify `CLAUDE.md` + `README.md`:** note `match` now opens a per-photo review UI by default (`--no-review` for headless).

Face dict in sidecars: `{id, bbox, det_score, embedding_ref, cluster, label}`. Cluster ids are `person_NNN` or `"unassigned"`.

---

## Task 1: existing_cluster_ids + next_cluster_id

**Files:**
- Modify: `face_pipeline.py` (add both helpers before `class LabelerApp`, line 391)
- Modify: `tests/test_face_pipeline.py` (append)

- [ ] **Step 1: Write the failing test**

Append to `tests/test_face_pipeline.py`:

```python
def test_next_cluster_id_empty_and_gaps():
    assert fp.next_cluster_id(set()) == "person_000"
    assert fp.next_cluster_id({"person_000", "person_002"}) == "person_003"
    assert fp.next_cluster_id({"person_009"}) == "person_010"


def test_existing_cluster_ids_collects_person_ids(tmp_path):
    d = str(tmp_path)
    fp.write_faces_json(os.path.join(d, "a.jpg"), (100, 100), "buffalo_l", [
        {"id": 0, "bbox": [0, 0, 1, 1], "det_score": 0.9,
         "embedding_ref": 0, "cluster": "person_001", "label": ""},
        {"id": 1, "bbox": [0, 0, 1, 1], "det_score": 0.9,
         "embedding_ref": 1, "cluster": "unassigned", "label": ""},
    ])
    fp.write_faces_json(os.path.join(d, "b.jpg"), (100, 100), "buffalo_l", [
        {"id": 0, "bbox": [0, 0, 1, 1], "det_score": 0.9,
         "embedding_ref": 2, "cluster": "person_004", "label": ""},
    ])
    assert fp.existing_cluster_ids(d) == {"person_001", "person_004"}
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/python -m pytest tests/test_face_pipeline.py -k "cluster_id or existing_cluster" -q`
Expected: FAIL — `AttributeError`.

- [ ] **Step 3: Write minimal implementation**

Insert into `face_pipeline.py` immediately before `class LabelerApp:` (line 391). Add `import glob`/`re` only if not already imported — `glob` and `os` are already imported at top; add `import re` to the stdlib imports if absent:

```python
def existing_cluster_ids(images_dir: str) -> set:
    """All person_NNN cluster ids found across the sidecars."""
    ids = set()
    for path in glob.glob(os.path.join(images_dir, "*.faces.json")):
        with open(path) as f:
            data = json.load(f)
        for face in data.get("faces", []):
            c = face.get("cluster", "")
            if c and c != "unassigned":
                ids.add(c)
    return ids


def next_cluster_id(existing_ids) -> str:
    """Mint the next free person_NNN id (max numeric suffix + 1)."""
    nums = []
    for cid in existing_ids:
        if cid.startswith("person_"):
            try:
                nums.append(int(cid.split("_", 1)[1]))
            except ValueError:
                pass
    nxt = (max(nums) + 1) if nums else 0
    return f"person_{nxt:03d}"
```

(If `import re` was added but unused, remove it — the implementation above does
not need `re`. Do NOT add unused imports.)

- [ ] **Step 4: Run test to verify it passes**

Run: `.venv/bin/python -m pytest tests/test_face_pipeline.py -k "cluster_id or existing_cluster" -q`
Expected: PASS (2 passed).

- [ ] **Step 5: Commit**

```bash
git add face_pipeline.py tests/test_face_pipeline.py
git commit -m "feat: add existing_cluster_ids and next_cluster_id helpers"
```

---

## Task 2: resolve_or_create_cluster

**Files:**
- Modify: `face_pipeline.py` (add helper before `class LabelerApp`)
- Modify: `tests/test_face_pipeline.py` (append)

- [ ] **Step 1: Write the failing test**

Append to `tests/test_face_pipeline.py`:

```python
def test_resolve_or_create_cluster_reuses_existing_name():
    labels = {"person_000": "Alice", "person_001": "Bob"}
    cid, new_labels = fp.resolve_or_create_cluster(
        "Bob", labels, {"person_000", "person_001"})
    assert cid == "person_001"
    assert new_labels == labels  # unchanged


def test_resolve_or_create_cluster_mints_new_for_new_name():
    labels = {"person_000": "Alice"}
    cid, new_labels = fp.resolve_or_create_cluster(
        "Carol", labels, {"person_000"})
    assert cid == "person_001"
    assert new_labels["person_001"] == "Carol"
    assert new_labels["person_000"] == "Alice"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/python -m pytest tests/test_face_pipeline.py -k resolve_or_create -q`
Expected: FAIL — `AttributeError`.

- [ ] **Step 3: Write minimal implementation**

Insert into `face_pipeline.py` before `class LabelerApp`:

```python
def resolve_or_create_cluster(name, labels_map, existing_ids):
    """Return (cluster_id, labels_map) for a name.

    Reuse the cluster already mapped to this name; otherwise mint a new id and
    extend a COPY of labels_map. existing_ids should include any ids already in
    labels_map; the caller adds the returned id to its own existing_ids set.
    """
    for cid, nm in labels_map.items():
        if nm == name:
            return cid, labels_map
    new_id = next_cluster_id(set(existing_ids) | set(labels_map))
    new_labels = {**labels_map, new_id: name}
    return new_id, new_labels
```

- [ ] **Step 4: Run test to verify it passes**

Run: `.venv/bin/python -m pytest tests/test_face_pipeline.py -k resolve_or_create -q`
Expected: PASS (2 passed).

- [ ] **Step 5: Commit**

```bash
git add face_pipeline.py tests/test_face_pipeline.py
git commit -m "feat: add resolve_or_create_cluster helper"
```

---

## Task 3: prefill_name

**Files:**
- Modify: `face_pipeline.py` (add helper before `class LabelerApp`)
- Modify: `tests/test_face_pipeline.py` (append)

- [ ] **Step 1: Write the failing test**

Append to `tests/test_face_pipeline.py`:

```python
def test_prefill_name_precedence():
    labels = {"person_002": "Carol"}
    # existing label wins over everything
    assert fp.prefill_name(
        {"label": "Alice", "cluster": "person_002"}, "Bob", labels) == "Alice"
    # else suggestion
    assert fp.prefill_name(
        {"label": "", "cluster": "person_002"}, "Bob", labels) == "Bob"
    # else cluster's name in labels_map
    assert fp.prefill_name(
        {"label": "", "cluster": "person_002"}, "", labels) == "Carol"
    # else empty
    assert fp.prefill_name(
        {"label": "", "cluster": "unassigned"}, "", labels) == ""
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/python -m pytest tests/test_face_pipeline.py -k prefill_name -q`
Expected: FAIL — `AttributeError`.

- [ ] **Step 3: Write minimal implementation**

Insert into `face_pipeline.py` before `class LabelerApp`:

```python
def prefill_name(face, suggestion, labels_map):
    """Precedence: existing label -> match suggestion -> cluster's name -> ''."""
    if face.get("label"):
        return face["label"]
    if suggestion:
        return suggestion
    return labels_map.get(face.get("cluster", ""), "")
```

- [ ] **Step 4: Run test to verify it passes**

Run: `.venv/bin/python -m pytest tests/test_face_pipeline.py -k prefill_name -q`
Expected: PASS (1 passed).

- [ ] **Step 5: Commit**

```bash
git add face_pipeline.py tests/test_face_pipeline.py
git commit -m "feat: add prefill_name helper"
```

---

## Task 4: apply_photo_edits

**Files:**
- Modify: `face_pipeline.py` (add helper before `class LabelerApp`)
- Modify: `tests/test_face_pipeline.py` (append)

- [ ] **Step 1: Write the failing test**

Append to `tests/test_face_pipeline.py`:

```python
def test_apply_photo_edits_writes_label_only_for_clustered(tmp_path):
    d = str(tmp_path)
    img = os.path.join(d, "a.jpg")
    fp.write_faces_json(img, (100, 100), "buffalo_l", [
        {"id": 0, "bbox": [0, 0, 1, 1], "det_score": 0.9,
         "embedding_ref": 0, "cluster": "person_000", "label": ""},
    ])
    labels = {"person_000": "Alice"}
    out = fp.apply_photo_edits(d, "a.jpg", {0: "Renamed"}, labels, {"person_000"})
    data = fp.read_faces_json(img)
    assert data["faces"][0]["label"] == "Renamed"
    assert data["faces"][0]["cluster"] == "person_000"  # cluster untouched
    assert out == labels  # no new cluster minted


def test_apply_photo_edits_assigns_cluster_to_unassigned(tmp_path):
    d = str(tmp_path)
    img = os.path.join(d, "a.jpg")
    fp.write_faces_json(img, (100, 100), "buffalo_l", [
        {"id": 0, "bbox": [0, 0, 1, 1], "det_score": 0.9,
         "embedding_ref": 0, "cluster": "unassigned", "label": ""},
        {"id": 1, "bbox": [0, 0, 1, 1], "det_score": 0.9,
         "embedding_ref": 1, "cluster": "unassigned", "label": ""},
    ])
    labels = {"person_000": "Alice"}
    # face 0 reuses Alice's cluster; face 1 is a brand-new person -> new id
    out = fp.apply_photo_edits(d, "a.jpg", {0: "Alice", 1: "Dave"},
                               labels, {"person_000"})
    data = fp.read_faces_json(img)
    assert data["faces"][0]["cluster"] == "person_000"   # reused
    assert data["faces"][0]["label"] == "Alice"
    assert data["faces"][1]["cluster"] == "person_001"   # minted
    assert data["faces"][1]["label"] == "Dave"
    assert out["person_001"] == "Dave"


def test_apply_photo_edits_empty_name_clears_label(tmp_path):
    d = str(tmp_path)
    img = os.path.join(d, "a.jpg")
    fp.write_faces_json(img, (100, 100), "buffalo_l", [
        {"id": 0, "bbox": [0, 0, 1, 1], "det_score": 0.9,
         "embedding_ref": 0, "cluster": "person_000", "label": "Old"},
    ])
    fp.apply_photo_edits(d, "a.jpg", {0: ""}, {"person_000": "Alice"},
                         {"person_000"})
    assert fp.read_faces_json(img)["faces"][0]["label"] == ""
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/python -m pytest tests/test_face_pipeline.py -k apply_photo_edits -q`
Expected: FAIL — `AttributeError`.

- [ ] **Step 3: Write minimal implementation**

Insert into `face_pipeline.py` before `class LabelerApp`:

```python
def apply_photo_edits(images_dir, image, edits, labels_map, existing_ids):
    """Apply {face_id: name} edits to one photo's sidecar.

    Sets each face's label. For a non-empty name on an unassigned face, resolves
    or mints a cluster (reusing a same-named person's cluster) and sets the
    face's cluster. Returns the (possibly extended) labels_map. Mutates
    existing_ids in place as new ids are minted. Caller persists labels.json.
    """
    path = os.path.join(images_dir, image)
    data = read_faces_json(path)
    if data is None:
        return labels_map
    by_id = {f["id"]: f for f in data["faces"]}
    for face_id, name in edits.items():
        face = by_id.get(face_id)
        if face is None:
            continue
        face["label"] = name
        cur = face.get("cluster", "")
        if name and cur in ("", "unassigned"):
            cid, labels_map = resolve_or_create_cluster(
                name, labels_map, existing_ids)
            face["cluster"] = cid
            existing_ids.add(cid)
    write_faces_json(path, tuple(data["image_size"]), data["model"],
                     data["faces"])
    return labels_map
```

- [ ] **Step 4: Run test to verify it passes**

Run: `.venv/bin/python -m pytest tests/test_face_pipeline.py -k apply_photo_edits -q`
Expected: PASS (3 passed). Then full file: `.venv/bin/python -m pytest tests/test_face_pipeline.py -q` (all green).

- [ ] **Step 5: Commit**

```bash
git add face_pipeline.py tests/test_face_pipeline.py
git commit -m "feat: add apply_photo_edits helper"
```

---

## Task 5: PhotoReviewApp + run_match wiring + --no-review flag

**Files:**
- Modify: `face_pipeline.py` — add `class PhotoReviewApp` (after `LabelerApp`'s last method, before `def run_label`); modify `run_match` (`:629`); modify `main` (`:684`).

Human-operated UI — NOT unit-tested. The implementer verifies the unit suite still passes + module imports + `--no-review` path works headlessly; the implementer must NOT launch the Tk window. Uses tested helpers `existing_cluster_ids`, `prefill_name`, `apply_photo_edits`, plus `scale_to_fit`, `read_faces_json`, `write_labels`.

- [ ] **Step 1: Add the PhotoReviewApp class**

Insert into `face_pipeline.py` after the end of the `LabelerApp` class (right before `def run_label`):

```python
class PhotoReviewApp:
    """Per-photo review: numbered bbox overlays + a name input per face."""

    def __init__(self, images_dir, photos, suggestions, labels_map):
        import tkinter as tk
        self.tk = tk
        self.images_dir = images_dir
        self.photos = photos                  # list of image filenames, has faces
        self.suggestions = suggestions        # {(image, face_id): name}
        self.labels_map = labels_map
        self.existing_ids = existing_cluster_ids(images_dir) | set(labels_map)
        self.idx = 0
        self._img_cache = {}
        self._photo_img = None                 # keep PhotoImage ref alive
        self._entries = []                     # (face_id, StringVar) per row

        self.root = tk.Tk()
        self.root.title("Photo Review")
        self.title_var = tk.StringVar()
        tk.Label(self.root, textvariable=self.title_var,
                 font=("TkDefaultFont", 13, "bold")).pack(pady=(8, 4))

        body = tk.Frame(self.root)
        body.pack(fill="both", expand=True)
        self.canvas = tk.Label(body)
        self.canvas.pack(side="left", padx=8, pady=4)

        # scrollable inputs column on the right
        right = tk.Frame(body)
        right.pack(side="right", fill="y", padx=8)
        self.rows_canvas = tk.Canvas(right, width=240, height=600,
                                     highlightthickness=0)
        vbar = tk.Scrollbar(right, orient="vertical",
                            command=self.rows_canvas.yview)
        self.rows_canvas.configure(yscrollcommand=vbar.set)
        vbar.pack(side="right", fill="y")
        self.rows_canvas.pack(side="left", fill="y")
        self.rows_frame = tk.Frame(self.rows_canvas)
        self.rows_canvas.create_window((0, 0), window=self.rows_frame, anchor="nw")
        self.rows_frame.bind(
            "<Configure>",
            lambda e: self.rows_canvas.configure(
                scrollregion=self.rows_canvas.bbox("all")))

        bar = tk.Frame(self.root)
        bar.pack(pady=8)
        tk.Button(bar, text="Back", command=self._back).pack(side="left", padx=4)
        self.next_btn = tk.Button(bar, text="Next", command=self._next)
        self.next_btn.pack(side="left", padx=4)
        self.root.protocol("WM_DELETE_WINDOW", self._on_close)

    def _source(self, image_name):
        import cv2
        if image_name not in self._img_cache:
            self._img_cache[image_name] = cv2.imread(
                os.path.join(self.images_dir, image_name))
        return self._img_cache[image_name]

    def _faces(self, image_name):
        data = read_faces_json(os.path.join(self.images_dir, image_name))
        return data["faces"] if data else []

    def _show(self):
        import cv2
        from PIL import Image, ImageTk
        image = self.photos[self.idx]
        faces = self._faces(image)
        self.title_var.set(f"{image}  ({self.idx + 1} of {len(self.photos)}) "
                           f"— {len(faces)} faces")
        src = self._source(image)
        if src is None:
            # skip unreadable photo
            print(f"  ! cannot read {image}, skipping")
            self._advance()
            return
        h, w = src.shape[:2]
        nw, nh, s = scale_to_fit(w, h, 900)
        disp = cv2.resize(src, (nw, nh), interpolation=cv2.INTER_AREA)
        for n, face in enumerate(faces, 1):
            x1, y1, x2, y2 = [int(round(v * s)) for v in face["bbox"]]
            if x2 > x1 and y2 > y1:
                cv2.rectangle(disp, (x1, y1), (x2, y2), (0, 0, 255), 2)
            cv2.putText(disp, str(n), (x1 + 2, max(12, y1 + 16)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
        rgb = cv2.cvtColor(disp, cv2.COLOR_BGR2RGB)
        self._photo_img = ImageTk.PhotoImage(Image.fromarray(rgb))
        self.canvas.configure(image=self._photo_img)

        for child in self.rows_frame.winfo_children():
            child.destroy()
        self._entries = []
        for n, face in enumerate(faces, 1):
            row = self.tk.Frame(self.rows_frame)
            row.pack(fill="x", pady=2)
            self.tk.Label(row, text=f"{n}.", width=3).pack(side="left")
            var = self.tk.StringVar(
                value=prefill_name(face, self.suggestions.get(
                    (image, face["id"]), ""), self.labels_map))
            self.tk.Entry(row, textvariable=var, width=18).pack(side="left")
            tag = face.get("cluster", "") or "unassigned"
            self.tk.Label(row, text=tag, fg="#888").pack(side="left", padx=4)
            self._entries.append((face["id"], var))

        self.next_btn.configure(
            text="Done" if self.idx == len(self.photos) - 1 else "Next")

    def _commit_current(self):
        image = self.photos[self.idx]
        edits = {fid: var.get().strip() for fid, var in self._entries}
        self.labels_map = apply_photo_edits(
            self.images_dir, image, edits, self.labels_map, self.existing_ids)
        write_labels(self.images_dir, self.labels_map)

    def _advance(self):
        if self.idx == len(self.photos) - 1:
            self.root.destroy()
            return
        self.idx += 1
        self._show()

    def _next(self):
        self._commit_current()
        self._advance()

    def _back(self):
        self._commit_current()
        if self.idx > 0:
            self.idx -= 1
            self._show()

    def _on_close(self):
        self._commit_current()
        self.root.destroy()

    def run(self):
        self._show()
        self.root.mainloop()
```

- [ ] **Step 2: Wire run_match to launch the UI by default**

In `run_match`, change the signature to accept `review` and, after the existing
`match_report.json` write + per-face print loop (the block ending with
`print(f"Wrote {out_path}.")`), and BEFORE the `if apply:` block, insert the
review launch. Replace the `run_match` signature line:

```python
def run_match(images_dir: str, gallery: str, top: int = 3,
              threshold: float = 0.5, apply: bool = False) -> int:
```

with:

```python
def run_match(images_dir: str, gallery: str, top: int = 3,
              threshold: float = 0.5, apply: bool = False,
              review: bool = True) -> int:
```

Then, immediately AFTER the line `print(f"Wrote {out_path}.")` and BEFORE
`if apply:`, insert:

```python
    if review:
        try:
            import tkinter  # noqa: F401
        except ImportError as e:
            raise SystemExit(
                "tkinter not available for review UI. Use --no-review for "
                "headless, or run via a Python with Tk (see README).") from e
        suggestions = {
            (r["image"], r["face_id"]): (r["candidates"][0]["name"]
                                         if r["candidates"] else "")
            for r in report
        }
        photos = sorted({r["image"] for r in report})
        app = PhotoReviewApp(images_dir, photos, suggestions, labels_map)
        app.run()
        print("Review complete; labels saved.")
        return 0
```

(The `if apply:` block remains below and runs only in the non-review/headless
path, since review returns early. This is intentional: with the review UI on,
the human's edits ARE the apply mechanism, so `--apply` is only meaningful
alongside `--no-review`. Not a bug.)

- [ ] **Step 3: Add the --no-review flag in main**

In `main`, the `match` subparser block currently ends with:

```python
    m.add_argument("--apply", action="store_true")
```

Add directly after it:

```python
    m.add_argument("--no-review", dest="review", action="store_false")
```

And change the match dispatch call. Find:

```python
    if args.cmd == "match":
        return run_match(args.images, args.gallery, args.top,
                         args.threshold, args.apply)
```

Replace with:

```python
    if args.cmd == "match":
        return run_match(args.images, args.gallery, args.top,
                         args.threshold, args.apply, args.review)
```

- [ ] **Step 4: Verify (do NOT open a window)**

Run: `.venv/bin/python -m pytest tests/ -q`
Expected: all green (helper tests + existing suite).

Run: `.venv/bin/python -c "import face_pipeline as fp; assert hasattr(fp,'PhotoReviewApp'); print('import OK')"`
Expected: `import OK`, no window.

Headless path still works (no GUI): with the existing `extracted/` cache and a
labels.json that names at least one cluster, run
`.venv/bin/python face_pipeline.py match --images extracted --gallery extracted/labels.json --no-review`
Expected: prints per-face lines + `Wrote .../match_report.json.` and exits 0
WITHOUT opening a window. (If labels.json has no named clusters it prints
"No named clusters..." and returns 1 — that's fine; to make this check meaningful
the implementer may temporarily set one name in extracted/labels.json, run, then
restore it. Do NOT leave test labels behind.)

- [ ] **Step 5: Commit**

```bash
git add face_pipeline.py
git commit -m "feat: per-photo review UI launched after match (--no-review opts out)"
```

---

## Task 6: Human GUI smoke test (manual — performed by the user)

**Files:** none (verification only)

Cannot be done by an automated agent. The controller surfaces these to the human.

- [ ] **Step 1: Name a cluster so match has a gallery, then run review**

```bash
# ensure a labels.json with at least one name exists (use the label UI or edit it)
.venv/bin/python face_pipeline.py match --images extracted --gallery extracted/labels.json
```

- [ ] **Step 2: Verify**

- The window shows the first photo that has faces, downscaled, with each face
  boxed and numbered; the right column has a matching numbered input per face.
- Inputs prefill: an already-set label shows it; otherwise the match suggestion;
  otherwise the face's cluster name; otherwise blank. Each row shows the face's
  cluster tag (`person_NNN` / `unassigned`).
- Editing a name and clicking **Next** persists: that photo's sidecar `label`
  fields update (check `cat extracted/<photo>.faces.json` in another terminal).
- Typing a name on an **unassigned** face assigns a cluster: if the name matches
  an existing person, the same `person_NNN` is reused; if new, a fresh id is
  minted and added to `extracted/labels.json`. Confirm in the sidecar + labels.json.
- **Back** returns to the previous photo with its saved values; **Done** on the
  last photo (or closing the window) exits cleanly.

- [ ] **Step 3: Report** any glitches and confirm the new-cluster assignment landed.

---

## Task 7: Update docs

**Files:**
- Modify: `CLAUDE.md`, `README.md`

- [ ] **Step 1: Update CLAUDE.md match line**

In `CLAUDE.md`, replace the `match` bullet:

```
- `python3 face_pipeline.py match --gallery extracted/labels.json` - rank candidates vs labeled centroids
```

with:

```
- `python3 face_pipeline.py match --gallery extracted/labels.json` - rank candidates vs labeled centroids, then open a per-photo review UI (numbered bbox overlays + a name input per face; edits write face label; naming an unassigned face assigns/mints a cluster). `--no-review` for headless (report + optional `--apply`). Needs a human at the GUI.
```

- [ ] **Step 2: Update README.md match section**

In `README.md`, under the `### match` heading, append this paragraph after the
existing description:

```
After scoring, `match` opens an interactive per-photo review window: each photo
is shown with its detected faces boxed and numbered, and a matching numbered name
input per face (prefilled with the existing label, else the match suggestion).
Editing a name writes it to that face's sidecar `label`; naming a face that has
no cluster assigns it one (reusing a same-named person's cluster, or minting a
new `person_NNN`). Edits for a photo are saved when you move to the next one. Pass
`--no-review` to keep `match` headless (report only).
```

- [ ] **Step 3: Commit**

```bash
git add CLAUDE.md README.md
git commit -m "docs: document per-photo review UI on match"
```

---

## Self-Review Notes

- **Spec coverage:** numbered bbox overlays + numbered inputs from sidecar (T5 `_show`), override writes to sidecar label (T4 `apply_photo_edits` + T5 commit), new-cluster create/reuse for unassigned faces (T1 `next_cluster_id`/`existing_cluster_ids`, T2 `resolve_or_create_cluster`, applied in T4), prefill precedence (T3), per-photo save on navigation (T5 `_commit_current`), launch after match by default + `--no-review` (T5/T2-in-main), faces.npy untouched (no task writes it; documented), error handling (T5 unreadable-skip, tkinter guard), docs (T7), human smoke (T6). All spec sections map to tasks.
- **Type consistency:** suggestions keyed `(image, face_id)` produced in run_match (T5 step 2) and read in `_show` via `self.suggestions.get((image, face["id"]))` (T5 step 1). `apply_photo_edits(images_dir, image, edits={face_id:name}, labels_map, existing_ids)` matches its `_commit_current` call site. `prefill_name(face, suggestion, labels_map)` matches its `_show` call. `resolve_or_create_cluster(name, labels_map, existing_ids) -> (id, labels_map)` matches its use in `apply_photo_edits`. Face dict keys `id/bbox/cluster/label` consistent with sidecar schema. labels.json `{person_id: name}` unchanged.
- **Placeholders:** none — every code step shows complete code.
