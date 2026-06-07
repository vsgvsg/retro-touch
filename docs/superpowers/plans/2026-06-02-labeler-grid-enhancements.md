# Labeler Grid Enhancements Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Enhance `LabelerApp` so each cluster shows ALL its face crops in a scrollable grid, Ctrl-Click on a crop excludes that face from the cluster (persisted to its sidecar), and left-click opens the full uncropped photo with the face boxed.

**Architecture:** All in `face_pipeline.py`. Add three pure TDD-tested helpers (`grid_positions`, `scale_to_fit`, `exclude_face`); remove now-dead `build_montage`/`pick_montage_faces` (+ tests); rebuild `LabelerApp`'s view from a single montage `Label` into a scrollable `Canvas` of per-crop `Label` widgets with click bindings, plus a `Toplevel` preview window. UI widget code is manual-verification (human at the GUI), like `split_photos.py`'s `Editor`.

**Tech Stack:** Python 3.13 (venv `.venv` on Homebrew, working Tk 9.0), numpy, OpenCV, Tkinter, PIL/Pillow, pytest. Run tests/GUI via `.venv/bin/python` — system `/usr/bin/python3` has a broken Tk on this macOS.

---

## File Structure

- **Modify `face_pipeline.py`:**
  - Add `grid_positions`, `scale_to_fit`, `exclude_face` (pure helpers, near the other helpers before `LabelerApp`).
  - Remove `pick_montage_faces` (`:340`) and `build_montage` (`:360`). Replace `_fit_cell` (`:345`) with a UI method `_crop_to_photo` on `LabelerApp` (its letterbox logic, returning a `PhotoImage`).
  - Rebuild `LabelerApp.__init__` (montage `Label` → scrollable `Canvas`+`Frame`+`Scrollbar`), replace `_montage_for`/`_show` (build per-crop widgets), add `_make_crop_cell`, `_preview_full`, `_do_exclude`, `_close_preview`.
- **Modify `tests/test_face_pipeline.py`:** remove the 2 `pick_montage_faces` + 3 `build_montage` tests; add tests for `grid_positions`, `scale_to_fit`, `exclude_face`.
- **Modify `CLAUDE.md`:** note the labeler now shows all crops, click=preview, Ctrl-click=exclude, and that it needs the `.venv` Python.

---

## Task 1: grid_positions helper

**Files:**
- Modify: `face_pipeline.py` (add helper just before `class LabelerApp` at line 393)
- Modify: `tests/test_face_pipeline.py` (append)

- [ ] **Step 1: Write the failing test**

Append to `tests/test_face_pipeline.py`:

```python
def test_grid_positions_full_and_partial():
    assert fp.grid_positions(4, cols=3) == [(0, 0), (0, 1), (0, 2), (1, 0)]


def test_grid_positions_zero():
    assert fp.grid_positions(0, cols=3) == []
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/python -m pytest tests/test_face_pipeline.py -k grid_positions -q`
Expected: FAIL — `AttributeError: ... grid_positions`.

- [ ] **Step 3: Write minimal implementation**

Insert into `face_pipeline.py` immediately before `class LabelerApp:` (line 393):

```python
def grid_positions(n: int, cols: int = 3) -> list:
    """Row-major (row, col) position for each of n cells."""
    return [divmod(i, cols) for i in range(n)]
```

- [ ] **Step 4: Run test to verify it passes**

Run: `.venv/bin/python -m pytest tests/test_face_pipeline.py -k grid_positions -q`
Expected: PASS (2 passed).

- [ ] **Step 5: Commit**

```bash
git add face_pipeline.py tests/test_face_pipeline.py
git commit -m "feat: add grid_positions helper"
```

---

## Task 2: scale_to_fit helper

**Files:**
- Modify: `face_pipeline.py` (add helper before `class LabelerApp`)
- Modify: `tests/test_face_pipeline.py` (append)

- [ ] **Step 1: Write the failing test**

Append to `tests/test_face_pipeline.py`:

```python
def test_scale_to_fit_downscales_longer_side():
    w, h, s = fp.scale_to_fit(1800, 900, 900)
    assert (w, h) == (900, 450)
    assert abs(s - 0.5) < 1e-9


def test_scale_to_fit_no_upscale_when_small():
    w, h, s = fp.scale_to_fit(300, 200, 900)
    assert (w, h, s) == (300, 200, 1.0)


def test_scale_to_fit_square():
    w, h, s = fp.scale_to_fit(1000, 1000, 500)
    assert (w, h) == (500, 500)
    assert abs(s - 0.5) < 1e-9
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/python -m pytest tests/test_face_pipeline.py -k scale_to_fit -q`
Expected: FAIL — `AttributeError: ... scale_to_fit`.

- [ ] **Step 3: Write minimal implementation**

Insert into `face_pipeline.py` before `class LabelerApp`:

```python
def scale_to_fit(w: int, h: int, max_dim: int) -> tuple:
    """Uniform downscale so the longer side <= max_dim. Never upscales.

    Returns (new_w, new_h, scale) with scale <= 1.0.
    """
    longer = max(w, h)
    if longer <= max_dim:
        return w, h, 1.0
    s = max_dim / longer
    return max(1, int(round(w * s))), max(1, int(round(h * s))), s
```

- [ ] **Step 4: Run test to verify it passes**

Run: `.venv/bin/python -m pytest tests/test_face_pipeline.py -k scale_to_fit -q`
Expected: PASS (3 passed).

- [ ] **Step 5: Commit**

```bash
git add face_pipeline.py tests/test_face_pipeline.py
git commit -m "feat: add scale_to_fit helper"
```

---

## Task 3: exclude_face helper

**Files:**
- Modify: `face_pipeline.py` (add helper before `class LabelerApp`)
- Modify: `tests/test_face_pipeline.py` (append)

- [ ] **Step 1: Write the failing test**

Append to `tests/test_face_pipeline.py`:

```python
def test_exclude_face_sets_unassigned_and_persists(tmp_path):
    d = str(tmp_path)
    img = os.path.join(d, "a.jpg")
    fp.write_faces_json(img, (100, 100), "buffalo_l", [
        {"id": 0, "bbox": [0, 0, 10, 10], "det_score": 0.9,
         "embedding_ref": 0, "cluster": "person_000", "label": ""},
        {"id": 1, "bbox": [5, 5, 15, 15], "det_score": 0.8,
         "embedding_ref": 1, "cluster": "person_000", "label": ""},
    ])
    assert fp.exclude_face(d, "a.jpg", 0) is True
    data = fp.read_faces_json(img)
    assert data["faces"][0]["cluster"] == "unassigned"   # excluded
    assert data["faces"][1]["cluster"] == "person_000"   # untouched


def test_exclude_face_unknown_id_returns_false(tmp_path):
    d = str(tmp_path)
    img = os.path.join(d, "a.jpg")
    fp.write_faces_json(img, (100, 100), "buffalo_l", [
        {"id": 0, "bbox": [0, 0, 10, 10], "det_score": 0.9,
         "embedding_ref": 0, "cluster": "person_000", "label": ""},
    ])
    assert fp.exclude_face(d, "a.jpg", 99) is False
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/python -m pytest tests/test_face_pipeline.py -k exclude_face -q`
Expected: FAIL — `AttributeError: ... exclude_face`.

- [ ] **Step 3: Write minimal implementation**

Insert into `face_pipeline.py` before `class LabelerApp`:

```python
def exclude_face(images_dir: str, image: str, face_id: int) -> bool:
    """Set a face's cluster to 'unassigned' in its sidecar and save.

    Returns True if a matching face was found and updated, else False.
    """
    path = os.path.join(images_dir, image)
    data = read_faces_json(path)
    if data is None:
        return False
    found = False
    for face in data["faces"]:
        if face["id"] == face_id:
            face["cluster"] = "unassigned"
            found = True
            break
    if found:
        write_faces_json(path, tuple(data["image_size"]), data["model"],
                         data["faces"])
    return found
```

- [ ] **Step 4: Run test to verify it passes**

Run: `.venv/bin/python -m pytest tests/test_face_pipeline.py -k exclude_face -q`
Expected: PASS (2 passed).

- [ ] **Step 5: Commit**

```bash
git add face_pipeline.py tests/test_face_pipeline.py
git commit -m "feat: add exclude_face helper"
```

---

## Task 4: Remove dead code (build_montage, pick_montage_faces) + their tests

**Files:**
- Modify: `face_pipeline.py` (delete `pick_montage_faces` `:340`, `build_montage` `:360`; `_fit_cell` `:345` is handled in Task 5 — leave it for now since `_montage_for` still references things)
- Modify: `tests/test_face_pipeline.py` (delete 5 tests)

Note: `_montage_for` / `_show` still call `pick_montage_faces` and `build_montage` until Task 5 rewrites them. To keep the suite importable between tasks, this task removes ONLY the standalone module-level `pick_montage_faces` and `build_montage` **after** confirming nothing else references them — but `_montage_for` does. Therefore: do Task 5 FIRST mentally, but for clean commits, this task is reordered to run AFTER Task 5. **Skip ahead: implement Task 5, then return to Task 4.** (The controller will dispatch Task 5 before Task 4.)

- [ ] **Step 1: Confirm no remaining references**

Run: `grep -n "pick_montage_faces\|build_montage\|_fit_cell" face_pipeline.py`
Expected: after Task 5, only the definitions remain (no callers). If `_montage_for` still appears, STOP — Task 5 is not done.

- [ ] **Step 2: Delete the test functions**

In `tests/test_face_pipeline.py`, delete these five test functions entirely:
`test_pick_montage_faces_top_k_by_score`, `test_pick_montage_faces_returns_all_when_fewer_than_k`, `test_build_montage_full_grid_shape`, `test_build_montage_partial_last_row`, `test_build_montage_zero_crops_placeholder`.

- [ ] **Step 3: Delete the functions**

In `face_pipeline.py`, delete the `pick_montage_faces` function and the `build_montage` function. (`_fit_cell` was already removed/relocated in Task 5; if it somehow remains and is unreferenced, delete it too.)

- [ ] **Step 4: Run the full suite**

Run: `.venv/bin/python -m pytest tests/ -q`
Expected: all green, with 5 fewer tests than before this task.

- [ ] **Step 5: Commit**

```bash
git add face_pipeline.py tests/test_face_pipeline.py
git commit -m "refactor: remove dead montage helpers superseded by per-crop grid"
```

---

## Task 5: Rebuild LabelerApp into a scrollable per-crop grid with preview + exclude

**Files:**
- Modify: `face_pipeline.py` — `LabelerApp` (`:393`-`:520`)

This is the human-operated UI shell — NOT unit-tested. The implementer verifies the unit suite still passes and the module imports + `LabelerApp` is defined; the implementer must NOT launch the Tk window (it requires a human and hangs headless). The click-through smoke test is done by a human afterward.

It uses the tested helpers `grid_positions`, `scale_to_fit`, `exclude_face`, plus the existing `crop_face`, `cluster_face_index`, `previous_names`, `write_labels`.

- [ ] **Step 1: Replace `__init__`'s montage canvas with a scrollable canvas**

In `LabelerApp.__init__`, replace these two lines (currently `:412-413`):

```python
        self.canvas = tk.Label(self.root)
        self.canvas.pack(padx=8)
```

with:

```python
        # scrollable area holding one Label per crop
        wrap = tk.Frame(self.root)
        wrap.pack(padx=8, fill="both", expand=True)
        self.canvas = tk.Canvas(wrap, width=3 * 132 + 16, height=420,
                                highlightthickness=0)
        vbar = tk.Scrollbar(wrap, orient="vertical", command=self.canvas.yview)
        self.canvas.configure(yscrollcommand=vbar.set)
        vbar.pack(side="right", fill="y")
        self.canvas.pack(side="left", fill="both", expand=True)
        self.grid_frame = tk.Frame(self.canvas)
        self.canvas.create_window((0, 0), window=self.grid_frame, anchor="nw")
        self.grid_frame.bind(
            "<Configure>",
            lambda e: self.canvas.configure(scrollregion=self.canvas.bbox("all")))
        self.canvas.bind_all(
            "<MouseWheel>",
            lambda e: self.canvas.yview_scroll(int(-1 * (e.delta / 120)), "units"))
        self._cells = []     # keep PhotoImage refs alive
        self._preview = None  # single Toplevel preview window
```

- [ ] **Step 2: Add `_crop_to_photo` (replaces `_fit_cell`) and `_make_crop_cell`; delete `_montage_for`**

Delete the `_montage_for` method (`:441-450`). Add these methods to `LabelerApp` (e.g. right after `_source`):

```python
    def _crop_to_photo(self, crop, cell=128):
        """BGR crop -> letterboxed cell x cell Tk PhotoImage (gray padding)."""
        import cv2
        from PIL import Image, ImageTk
        canvas = np.full((cell, cell, 3), 128, np.uint8)
        h, w = crop.shape[:2]
        if h > 0 and w > 0:
            s = min(cell / w, cell / h)
            nw, nh = max(1, int(w * s)), max(1, int(h * s))
            resized = cv2.resize(crop, (nw, nh), interpolation=cv2.INTER_AREA)
            y0, x0 = (cell - nh) // 2, (cell - nw) // 2
            canvas[y0:y0 + nh, x0:x0 + nw] = resized
        rgb = cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB)
        return ImageTk.PhotoImage(Image.fromarray(rgb))

    def _make_crop_cell(self, face, row, col):
        """Create one crop Label at (row,col) bound to preview / exclude."""
        src = self._source(face["image"])
        readable = src is not None
        if readable:
            crop = crop_face(src, face["bbox"])
        else:
            crop = np.full((64, 64, 3), 128, np.uint8)  # placeholder
        photo = self._crop_to_photo(crop)
        self._cells.append(photo)  # keep ref
        lbl = self.tk.Label(self.grid_frame, image=photo,
                            borderwidth=1, relief="solid")
        lbl.grid(row=row, column=col, padx=2, pady=2)
        if readable:
            lbl.bind("<Button-1>", lambda e, f=face: self._preview_full(f))
            lbl.bind("<Control-Button-1>", lambda e, f=face: self._do_exclude(f))
            lbl.bind("<Button-2>", lambda e, f=face: self._do_exclude(f))  # mac fallback
```

- [ ] **Step 2 (cont.): Replace `_show` to build the grid**

Replace the `_show` method (`:452-467`) with:

```python
    def _show(self):
        cid = self.cluster_ids[self.idx]
        faces = self.cluster_index[cid]
        self.title_var.set(f"Cluster {cid}  ({self.idx + 1} of "
                           f"{len(self.cluster_ids)}) — {len(faces)} faces")
        for child in self.grid_frame.winfo_children():
            child.destroy()
        self._cells = []
        positions = grid_positions(len(faces), cols=3)
        for face, (r, c) in zip(faces, positions):
            self._make_crop_cell(face, r, c)
        self.canvas.yview_moveto(0)
        self.name_var.set(self.labels_map.get(cid, ""))
        self.entry.focus_set()
        self._refresh_names()
        self.next_btn.configure(
            text="Done" if self.idx == len(self.cluster_ids) - 1 else "Next")
```

- [ ] **Step 3: Add `_preview_full`, `_do_exclude`, `_close_preview`**

Add these methods to `LabelerApp` (e.g. after `_make_crop_cell`):

```python
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
        lbl = self.tk.Label(top, image=self._preview_photo)
        lbl.pack()
        lbl.bind("<Button-1>", lambda e: self._close_preview())
        top.bind("<Escape>", lambda e: self._close_preview())
        top.protocol("WM_DELETE_WINDOW", self._close_preview)

    def _close_preview(self):
        if self._preview is not None:
            try:
                self._preview.destroy()
            except self.tk.TclError:
                pass
            self._preview = None

    def _do_exclude(self, face):
        exclude_face(self.images_dir, face["image"], face["face_id"])
        cid = self.cluster_ids[self.idx]
        self.cluster_index[cid] = [
            f for f in self.cluster_index[cid]
            if not (f["image"] == face["image"] and f["face_id"] == face["face_id"])]
        self._show()
```

- [ ] **Step 4: Guard `_on_close` / window teardown for the preview**

Replace `_on_close` (`:514-516`) with:

```python
    def _on_close(self):
        self._close_preview()
        self._commit_current()
        self.root.destroy()
```

- [ ] **Step 5: Confirm unit suite passes and module imports (no window)**

Run: `.venv/bin/python -m pytest tests/ -q`
Expected: all green (helper tests + existing suite; the build_montage/pick tests are removed in Task 4 which runs next, so at THIS point they still exist and pass — the suite count is unchanged here).

Run: `.venv/bin/python -c "import face_pipeline as fp; assert hasattr(fp,'LabelerApp'); print('import OK')"`
Expected: prints `import OK`, no window.

Confirm `_montage_for` is gone and `_show` uses `grid_positions`:
Run: `grep -n "_montage_for\|grid_positions\|_make_crop_cell\|_preview_full\|_do_exclude" face_pipeline.py`
Expected: no `_montage_for`; `grid_positions`, `_make_crop_cell`, `_preview_full`, `_do_exclude` present.

- [ ] **Step 6: Commit**

```bash
git add face_pipeline.py
git commit -m "feat: scrollable per-crop grid with full-photo preview and Ctrl-click exclude"
```

> After this task, the controller dispatches **Task 4** (remove the now-unreferenced `pick_montage_faces`/`build_montage`/`_fit_cell` and their tests), then the human GUI smoke test below.

---

## Task 6: Human GUI smoke test (manual — performed by the user)

**Files:** none (verification only)

This cannot be done by an automated agent. The controller surfaces these steps to the human.

- [ ] **Step 1: Launch with a clean labels file**

```bash
rm -f extracted/labels.json
.venv/bin/python face_pipeline.py label --images extracted
```

- [ ] **Step 2: Verify**

- Cluster `person_001` shows ALL 30 crops in a 3-wide grid; the scrollbar and mouse-wheel scroll through them.
- **Left-click** a crop → a second window opens with the full source photo, a red rectangle around that face; clicking it / Escape / closing dismisses it; opening another preview replaces the first.
- **Ctrl-Click** a crop → it disappears from the grid and the title count drops by one. In another terminal, that face's sidecar shows `"cluster": "unassigned"` (`grep -l unassigned extracted/*.faces.json`). If Ctrl-Click does nothing but a two-finger/right click does, the `<Button-2>` fallback is what fires on this Mac — report which worked.
- Typing a name + **Next** still writes `labels.json`; **Back** still resumes; **Done** on the last cluster prints `Saved labels -> ...`.

- [ ] **Step 3: Report the outcome** (which exclude binding fired; any glitches).

---

## Task 7: Update CLAUDE.md

**Files:**
- Modify: `CLAUDE.md`

- [ ] **Step 1: Update the label bullet and add a venv note**

In `CLAUDE.md`, under `## Face pipeline (face_pipeline.py)`, replace the existing `label` bullet with:

```
- `python3 face_pipeline.py label` - interactive Tkinter labeler: scrollable grid of ALL of a cluster's face crops; left-click a crop = full-photo preview (face boxed), Ctrl-click a crop = exclude it (sets sidecar cluster to "unassigned"); type or reuse a name; writes extracted/labels.json after each step. Needs a human at the GUI.
```

Then add this line at the end of that section:

```
- The system /usr/bin/python3 ships a Tk that can't open a window on this macOS; run the GUI (and ideally the whole pipeline) via the venv: `.venv/bin/python face_pipeline.py ...` (Homebrew Python 3.13 + Tk 9.0). `.venv` is gitignored; recreate with `python3.13 -m venv .venv && .venv/bin/pip install -r requirements.txt`.
```

- [ ] **Step 2: Commit**

```bash
git add CLAUDE.md
git commit -m "docs: document grid labeler enhancements and venv requirement"
```

---

## Self-Review Notes

- **Spec coverage:** all crops in scrollable grid (T5 `_show`+`grid_positions`+Canvas), Ctrl-click exclude→sidecar unassigned persistent (T3 `exclude_face` + T5 `_do_exclude`/Button-2 fallback), full-photo preview with bbox + single-at-a-time + close behaviors (T5 `_preview_full`/`_close_preview`, T2 `scale_to_fit`), `_fit_cell` letterbox preserved as `_crop_to_photo` (T5), dead code removed (T4), placeholder/degenerate-bbox/unknown-id error handling (T3, T5), docs + venv note (T7), human smoke (T6). All spec sections map to tasks.
- **Task ordering:** Task 5 (UI rebuild) runs BEFORE Task 4 (dead-code removal), because removing `build_montage`/`pick_montage_faces` while `_montage_for` still calls them would break import. The controller dispatches in order 1,2,3,5,4,6,7. Task 4's text states this explicitly.
- **Type consistency:** the face dict `{image, face_id, bbox, det_score}` from `cluster_face_index` is what `_make_crop_cell`/`_preview_full`/`_do_exclude` read (`face["image"]`, `face["face_id"]`, `face["bbox"]`). `exclude_face(images_dir, image, face_id)` matches its `_do_exclude` call site. `scale_to_fit`'s returned `s` is applied to `bbox` in `_preview_full`. `grid_positions(n, cols)` zips with the faces list in `_show`. Consistent.
- **Placeholders:** none — every code step shows complete code.
