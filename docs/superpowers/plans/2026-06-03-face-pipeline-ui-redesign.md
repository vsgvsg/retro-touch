# Face Pipeline UI Redesign Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Restyle the two Tkinter GUIs in `face_pipeline.py` (`LabelerApp` and `PhotoReviewApp`) for visual polish, clearer workflow feedback, and better face visibility — without changing the pipeline, data formats, or any pure-helper logic.

**Architecture:** Presentation-only change. Add a `ttk.Style` setup applied once per app, draw crops/meters/cards on Tk `Canvas`, and recolor the existing OpenCV box render by a new pure `face_state()` classifier shared between the photo boxes and the row meters. All persistence helpers (`prefill_name`, `hint_for`, `apply_photo_edits`, `exclude_face`, `write_labels`, `previous_names`, `crop_face`, `grid_positions`, `scale_to_fit`) keep their current signatures and tests.

**Tech Stack:** Python 3.13 (via `.venv`), Tkinter + ttk + Tk Canvas, Pillow (`ImageTk`), OpenCV, NumPy. No new dependencies. Tests: pytest.

**Spec:** `docs/superpowers/specs/2026-06-03-face-pipeline-ui-redesign-design.md`

---

## File Structure

- **Modify** `face_pipeline.py`:
  - Add one pure helper `face_state(face, best_entry, threshold, labels_map) -> str` near the other review helpers (after `hint_for`, ~line 545).
  - Add a module-level `_install_theme(root)` helper that configures a shared `ttk.Style` (colors, button styles) — called by both apps.
  - Rewrite the *widget construction and rendering* of `LabelerApp` (~577–789) and `PhotoReviewApp` (~791–968). Keep all event-handler method names and persistence calls.
- **Modify** `tests/test_face_pipeline.py`: add `face_state` tests alongside the existing `prefill_name`/`hint_for` tests (~line 470).
- **No other files.** `split_photos.py` and `requirements.txt` are untouched.

**Run all GUI verification via `.venv/bin/python`** — the system Python's Tk can't open a window on this macOS (per CLAUDE.md). Run tests with `.venv/bin/python -m pytest`.

---

## Task 1: `face_state` classifier (the only new pure helper)

A single source of truth for the color/state of a face, used by both the photo box color and the row meter color so they always agree. States: `"confident"` (best candidate ≥ threshold, no conflicting existing label), `"matched"` (has an existing label, OR cluster-name prefill applies), `"unassigned"` (no label, no cluster name, candidate below threshold or absent). Mirrors `prefill_name` precedence so colors match what the box shows.

**Files:**
- Modify: `face_pipeline.py` (add `face_state` after `hint_for`, ~line 545)
- Test: `tests/test_face_pipeline.py` (~line 470, after the `hint_for` tests)

- [ ] **Step 1: Write the failing tests**

Add to `tests/test_face_pipeline.py`:

```python
def test_face_state_existing_label_is_matched():
    assert fp.face_state(
        {"label": "Alice", "cluster": "person_002"},
        ("Bob", 0.9), 0.5, {}) == "matched"


def test_face_state_candidate_above_threshold_is_confident():
    assert fp.face_state(
        {"label": "", "cluster": "unassigned"},
        ("Bob", 0.6), 0.5, {}) == "confident"


def test_face_state_cluster_name_prefill_is_matched():
    labels = {"person_002": "Carol"}
    assert fp.face_state(
        {"label": "", "cluster": "person_002"},
        ("Bob", 0.2), 0.5, labels) == "matched"


def test_face_state_weak_and_unassigned_is_unassigned():
    assert fp.face_state(
        {"label": "", "cluster": "unassigned"},
        ("Bob", 0.2), 0.5, {}) == "unassigned"
    assert fp.face_state(
        {"label": "", "cluster": "unassigned"},
        None, 0.5, {}) == "unassigned"
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `.venv/bin/python -m pytest tests/test_face_pipeline.py -k face_state -v`
Expected: FAIL with `AttributeError: module 'face_pipeline' has no attribute 'face_state'`

- [ ] **Step 3: Write minimal implementation**

Add after `hint_for` in `face_pipeline.py` (~line 545):

```python
def face_state(face, best_entry, threshold, labels_map):
    """Color/state classifier for a face, mirroring prefill_name precedence.

    Returns "matched" (has an existing label or a cluster-name prefill),
    "confident" (best candidate >= threshold), or "unassigned" (no label,
    no cluster name, candidate below threshold or absent). Used to color the
    photo box and the row's confidence meter consistently. best_entry is
    (name, score) or None.
    """
    if face.get("label"):
        return "matched"
    if best_entry is not None and best_entry[1] >= threshold:
        return "confident"
    if labels_map.get(face.get("cluster", ""), ""):
        return "matched"
    return "unassigned"
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `.venv/bin/python -m pytest tests/test_face_pipeline.py -k face_state -v`
Expected: PASS (4 tests)

- [ ] **Step 5: Run the full suite (no regressions)**

Run: `.venv/bin/python -m pytest tests/ -q`
Expected: all pass

- [ ] **Step 6: Commit**

```bash
git add face_pipeline.py tests/test_face_pipeline.py
git commit -m "feat: add face_state classifier shared by box + meter color"
```

---

## Task 2: Shared theme helper

A module-level function that configures one `ttk.Style` with the spec's color system, so both apps share a look. Defined once; each app calls it in `__init__` after creating its root.

**Files:**
- Modify: `face_pipeline.py` (add `_install_theme` near the top of the GUI section, just before `class LabelerApp`, ~line 576)

- [ ] **Step 1: Add the theme helper**

Insert before `class LabelerApp:`:

```python
# ---- shared GUI theme (ttk) ----
ACCENT = "#5a6cf0"
BG = "#fafaff"
CARD_BORDER = "#ececf2"
STATE_COLORS = {            # face_state -> (hex for box/badge/meter)
    "confident": "#2faf6a",
    "matched": "#5a6cf0",
    "unassigned": "#d8a23a",
}


def _install_theme(root):
    """Configure a shared ttk.Style; safe to call once per app root."""
    from tkinter import ttk
    style = ttk.Style(root)
    try:
        style.theme_use("clam")   # consistent, restyleable across platforms
    except Exception:
        pass
    root.configure(bg=BG)
    style.configure("TFrame", background=BG)
    style.configure("TLabel", background=BG, foreground="#1a1a2e")
    style.configure("Sub.TLabel", background=BG, foreground="#7a7a88")
    style.configure("Title.TLabel", background=BG, foreground="#1a1a2e",
                    font=("TkDefaultFont", 14, "bold"))
    style.configure("TButton", padding=(12, 6), relief="flat",
                    background="#ffffff", foreground="#1a1a2e")
    style.map("TButton", background=[("active", "#f0f0f8")])
    style.configure("Primary.TButton", padding=(14, 6), relief="flat",
                    background=ACCENT, foreground="#ffffff",
                    font=("TkDefaultFont", 11, "bold"))
    style.map("Primary.TButton", background=[("active", "#4a5ce0")])
    style.configure("TEntry", padding=4)
    return style
```

- [ ] **Step 2: Verify it imports without error**

Run: `.venv/bin/python -c "import face_pipeline as fp; import tkinter as tk; r=tk.Tk(); fp._install_theme(r); print('theme ok'); r.destroy()"`
Expected: prints `theme ok` (a window may flash briefly)

- [ ] **Step 3: Commit**

```bash
git add face_pipeline.py
git commit -m "feat: add shared ttk theme helper for the GUIs"
```

---

## Task 3: Rounded-crop Canvas helper on LabelerApp/PhotoReviewApp

Both apps render crops; add a shared rendering helper that produces a rounded-corner `ImageTk.PhotoImage` from a BGR crop, degrading to a square image on failure (spec error-handling). Place it as a module-level function so both classes use it.

**Files:**
- Modify: `face_pipeline.py` (add `crop_to_round_photo` near `_install_theme`)

- [ ] **Step 1: Add the helper**

```python
def crop_to_round_photo(crop, cell=128, radius=12):
    """BGR crop -> letterboxed cell x cell rounded-corner Tk PhotoImage.

    Degrades to a plain (square) letterboxed image if rounding fails, so the
    GUI never crashes on an odd crop. Returns an ImageTk.PhotoImage.
    """
    import cv2
    from PIL import Image, ImageDraw, ImageTk
    base = np.full((cell, cell, 3), 245, np.uint8)  # near-bg fill
    h, w = crop.shape[:2]
    if h > 0 and w > 0:
        s = min(cell / w, cell / h)
        nw, nh = max(1, int(w * s)), max(1, int(h * s))
        resized = cv2.resize(crop, (nw, nh), interpolation=cv2.INTER_AREA)
        y0, x0 = (cell - nh) // 2, (cell - nw) // 2
        base[y0:y0 + nh, x0:x0 + nw] = resized
    rgb = cv2.cvtColor(base, cv2.COLOR_BGR2RGB)
    img = Image.fromarray(rgb).convert("RGBA")
    try:
        mask = Image.new("L", (cell, cell), 0)
        ImageDraw.Draw(mask).rounded_rectangle(
            [0, 0, cell - 1, cell - 1], radius=radius, fill=255)
        img.putalpha(mask)
    except Exception:
        pass  # degrade to square
    return ImageTk.PhotoImage(img)
```

- [ ] **Step 2: Smoke-test the helper**

Run:
```bash
.venv/bin/python -c "
import numpy as np, tkinter as tk
import face_pipeline as fp
r = tk.Tk(); fp._install_theme(r)
p = fp.crop_to_round_photo(np.full((50,40,3),128,np.uint8))
print('rounded photo ok', p.width(), p.height()); r.destroy()"
```
Expected: prints `rounded photo ok 128 128`

- [ ] **Step 3: Commit**

```bash
git add face_pipeline.py
git commit -m "feat: add rounded-corner crop->PhotoImage helper"
```

---

## Task 4: Rebuild LabelerApp UI (header + progress + scrolling crops + hybrid reuse)

Rewrite `LabelerApp.__init__`, `_make_crop_cell`, `_show`, `_refresh_names`, and add chip/typeahead handlers. Keep `_source`, `_preview_full`, `_close_preview`, `_do_exclude`, `_commit_current`, `_next`, `_back`, `_skip`, `_on_close`, `_on_pick`(removed) — adjust only what the new widgets require. Crops stay in the existing scrollable Canvas (already supports overflow); they get larger + rounded. The `Listbox` is replaced by a chip row + typeahead filtering.

**This is a GUI task — verified manually, not unit-tested** (per CLAUDE.md convention). Make the change in one focused pass, then run the manual verification step.

**Files:**
- Modify: `face_pipeline.py` `class LabelerApp` (~577–789)

- [ ] **Step 1: Replace `__init__` widget construction**

Replace the body of `LabelerApp.__init__` (keep the data-field assignments: `self.tk`, `self.images_dir`, `self.cluster_ids`, `self.cluster_index`, `self.labels_map`, `self.idx`, `self._img_cache`) with ttk-based widgets:

```python
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
        self._excluded = set()       # (image, face_id) dimmed this session

        self.root = tk.Tk()
        self.root.title("Face Labeler")
        self.root.geometry("460x720")
        _install_theme(self.root)

        head = ttk.Frame(self.root)
        head.pack(fill="x", padx=16, pady=(12, 4))
        self.cluster_var = tk.StringVar()
        self.sub_var = tk.StringVar()
        ttk.Label(head, textvariable=self.cluster_var,
                  style="Title.TLabel").pack(anchor="w")
        ttk.Label(head, textvariable=self.sub_var,
                  style="Sub.TLabel").pack(anchor="w")
        self.progress = ttk.Progressbar(head, maximum=len(self.cluster_ids))
        self.progress.pack(fill="x", pady=(8, 0))

        # scrollable crop grid (existing pattern, fixed height)
        wrap = ttk.Frame(self.root)
        wrap.pack(padx=16, pady=8, fill="both", expand=True)
        self.canvas = tk.Canvas(wrap, height=300, highlightthickness=0,
                                bg=BG)
        vbar = ttk.Scrollbar(wrap, orient="vertical",
                             command=self.canvas.yview)
        self.canvas.configure(yscrollcommand=vbar.set)
        vbar.pack(side="right", fill="y")
        self.canvas.pack(side="left", fill="both", expand=True)
        self.grid_frame = ttk.Frame(self.canvas)
        self.canvas.create_window((0, 0), window=self.grid_frame, anchor="nw")
        self.grid_frame.bind(
            "<Configure>",
            lambda e: self.canvas.configure(
                scrollregion=self.canvas.bbox("all")))
        self.canvas.bind_all(
            "<MouseWheel>",
            lambda e: self.canvas.yview_scroll(
                int(-1 * (e.delta / 120)), "units"))
        self._cells = []
        self._preview = None
        self._preview_photo = None

        # name field (typeahead filters the chip row)
        field = ttk.Frame(self.root)
        field.pack(fill="x", padx=16)
        ttk.Label(field, text="WHO IS THIS?", style="Sub.TLabel").pack(anchor="w")
        self.name_var = tk.StringVar()
        self.name_var.trace_add("write", lambda *_: self._refresh_names())
        self.entry = ttk.Entry(field, textvariable=self.name_var)
        self.entry.pack(fill="x", pady=(2, 6))
        self.entry.bind("<Return>", lambda e: self._next())

        ttk.Label(self.root, text="REUSE:", style="Sub.TLabel").pack(
            anchor="w", padx=16)
        self.chips = ttk.Frame(self.root)
        self.chips.pack(fill="x", padx=16, pady=(2, 8))

        bar = ttk.Frame(self.root)
        bar.pack(fill="x", padx=16, pady=12)
        ttk.Button(bar, text="← Back", command=self._back).pack(side="left")
        ttk.Button(bar, text="Skip", command=self._skip).pack(
            side="left", padx=6)
        self.next_btn = ttk.Button(bar, text="Save & Next →",
                                   style="Primary.TButton", command=self._next)
        self.next_btn.pack(side="right")
        self.root.protocol("WM_DELETE_WINDOW", self._on_close)
```

- [ ] **Step 2: Update `_crop_to_photo` callers to use the rounded helper + bigger cells**

Replace `_make_crop_cell` so crops are 140px rounded and excluded ones dim. Replace the whole method:

```python
    def _make_crop_cell(self, face, row, col):
        src = self._source(face["image"])
        readable = src is not None
        if readable:
            crop = crop_face(src, face["bbox"])
        else:
            crop = np.full((64, 64, 3), 128, np.uint8)
        photo = crop_to_round_photo(crop, cell=140)
        self._cells.append(photo)
        excluded = (face["image"], face["face_id"]) in self._excluded
        lbl = self.tk.Label(self.grid_frame, image=photo, bg=BG,
                            borderwidth=0)
        if excluded:
            lbl.configure(text="excluded", compound="center",
                          fg="#ffffff")
        lbl.grid(row=row, column=col, padx=4, pady=4)
        if readable and not excluded:
            lbl.bind("<Button-1>", lambda e, f=face: self._preview_full(f))
            lbl.bind("<Control-Button-1>",
                     lambda e, f=face: self._do_exclude(f))
```

- [ ] **Step 3: Update `_do_exclude` to dim (keep in list) instead of removing**

Replace `_do_exclude`:

```python
    def _do_exclude(self, face):
        exclude_face(self.images_dir, face["image"], face["face_id"])
        self._excluded.add((face["image"], face["face_id"]))
        self._show()
```

(Persistence is unchanged — `exclude_face` still writes the sidecar `unassigned`. The face stays visible but dimmed for the rest of the session.)

- [ ] **Step 4: Update `_show` for the new vars + cols=3 grid (wider cells)**

Replace `_show`:

```python
    def _show(self):
        cid = self.cluster_ids[self.idx]
        faces = self.cluster_index[cid]
        self.cluster_var.set(f"Cluster {cid} · {len(faces)} faces")
        self.sub_var.set(
            f"Person {self.idx + 1} of {len(self.cluster_ids)} · "
            f"⌘-click a crop to exclude")
        self.progress.configure(value=self.idx)
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
            text="Done" if self.idx == len(self.cluster_ids) - 1
            else "Save & Next →")
```

- [ ] **Step 5: Replace `_refresh_names` (Listbox) with filtered chips; remove `_on_pick`**

Replace `_refresh_names` and delete `_on_pick`:

```python
    def _refresh_names(self):
        for child in self.chips.winfo_children():
            child.destroy()
        typed = self.name_var.get().strip().lower()
        names = previous_names(self.labels_map)
        if typed:
            matches = [n for n in names if typed in n.lower()]
        else:
            matches = names[:6]               # recent (hybrid: chips for top-6)
        for nm in matches[:8]:
            chip = self.tk.Label(self.chips, text=nm, bg="#ececfa",
                                 fg="#4a4ad0", padx=10, pady=3, cursor="hand2")
            chip.pack(side="left", padx=(0, 6), pady=2)
            chip.bind("<Button-1>", lambda e, n=nm: self.name_var.set(n))
        if typed and not any(n.lower() == typed for n in names):
            new = self.tk.Label(self.chips, text=f'+ Create "{typed}"',
                                bg=BG, fg="#4a4ad0", cursor="hand2")
            new.pack(side="left")
```

- [ ] **Step 6: Manual verification (GUI)**

Run: `.venv/bin/python face_pipeline.py label`
Confirm by eye:
- Header shows `Cluster N · K faces` and `Person i of N · ⌘-click…`; progress bar advances on Next/Back.
- Crops are larger, rounded; the grid scrolls when a cluster has many faces.
- ⌘-click a crop dims it with an "excluded" overlay (it stays visible); the face is set unassigned in its sidecar.
- Typing in the name field filters the chips; clicking a chip fills the field; `+ Create "..."` appears for a new name.
- Save & Next / Back / Skip still write `extracted/labels.json` (check the file mtime / content).

If the exclude-dims interaction feels wrong in practice, fall back: in `_do_exclude` also remove the face from `self.cluster_index[cid]` (the pre-redesign behavior) and skip the `_excluded` set — the rest of the UI is unaffected.

- [ ] **Step 7: Run the full suite (helpers unaffected)**

Run: `.venv/bin/python -m pytest tests/ -q`
Expected: all pass

- [ ] **Step 8: Commit**

```bash
git add face_pipeline.py
git commit -m "feat: redesign LabelerApp UI (progress, rounded crops, chip reuse)"
```

---

## Task 5: Recolor PhotoReviewApp boxes by face_state

Change only the OpenCV box/number color in `PhotoReviewApp._show` to use `face_state` + `STATE_COLORS` (BGR), so each box is colored by its state. Numbers keep their position. This is isolated from the row redesign in Task 6 so it can be verified on its own.

**Files:**
- Modify: `face_pipeline.py` `PhotoReviewApp._show` box-drawing loop (~895–901)

- [ ] **Step 1: Replace the box-drawing loop**

In `_show`, replace the `for n, face in enumerate(faces, 1):` block that draws rectangles/putText with:

```python
        def _bgr(hex_):
            h = hex_.lstrip("#")
            r, g, b = int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16)
            return (b, g, r)
        self._row_colors = []
        for n, face in enumerate(faces, 1):
            be = self.best.get((image, face["id"]))
            state = face_state(face, be, self.threshold, self.labels_map)
            color = _bgr(STATE_COLORS[state])
            self._row_colors.append(STATE_COLORS[state])
            x1, y1, x2, y2 = [int(round(v * s)) for v in face["bbox"]]
            if x2 > x1 and y2 > y1:
                cv2.rectangle(disp, (x1, y1), (x2, y2), color, 2)
            cv2.putText(disp, str(n), (x1 + 2, max(12, y1 + 16)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
```

(`self._row_colors` is consumed in Task 6 to color each row's number badge to match its box. `be`/`state` are recomputed in Task 6's row loop too — that's fine, it's cheap and keeps the loops independent.)

- [ ] **Step 2: Manual verification (boxes only)**

Run: `.venv/bin/python face_pipeline.py match --gallery extracted/labels.json`
Confirm: boxes are green (confident) / blue (labeled) / amber (unassigned) instead of all red; numbers match.

- [ ] **Step 3: Commit**

```bash
git add face_pipeline.py
git commit -m "feat: color review photo boxes by face_state"
```

---

## Task 6: Rebuild PhotoReviewApp rows (thumbnail + badge + meter + hint pill)

Rewrite `PhotoReviewApp.__init__` widget construction and the right-column row build in `_show`. Each face becomes a card: rounded thumbnail, color-matched number badge, name `ttk.Entry`, a Canvas confidence meter, and a clickable hint pill. Keep `_source`, `_faces`, `_commit_current`, `_advance`, `_next`, `_back`, `_on_close` and all persistence.

**GUI task — verified manually.**

**Files:**
- Modify: `face_pipeline.py` `class PhotoReviewApp` (~791–968)

- [ ] **Step 1: Update `__init__` to theme + ttk widgets**

In `PhotoReviewApp.__init__`, after `self.root = tk.Tk()` / title / geometry, add `from tkinter import ttk`, `self.ttk = ttk`, and `_install_theme(self.root)`. Replace the title `tk.Label` with `ttk.Label(..., style="Title.TLabel")`, the bottom button bar `tk.Button`s with `ttk.Button` (Back = default style; the next button uses `style="Primary.TButton"` and text `"Save & Next →"`), and set the right-column `tk.Canvas` `bg=BG`. Keep the fixed geometry, `photo_pane`, `self.canvas` (photo `tk.Label`), and the scrollable `rows_canvas`/`rows_frame` pattern exactly as-is.

```python
        from tkinter import ttk
        self.ttk = ttk
        # ... existing self.* data assignments ...
        self.root = tk.Tk()
        self.root.title("Photo Review")
        self.root.geometry("1120x780")
        self.root.resizable(False, False)
        _install_theme(self.root)
        self.title_var = tk.StringVar()
        ttk.Label(self.root, textvariable=self.title_var,
                  style="Title.TLabel").pack(side="top", pady=(10, 4))
        bar = ttk.Frame(self.root)
        bar.pack(side="bottom", pady=10)
        ttk.Button(bar, text="← Back", command=self._back).pack(
            side="left", padx=4)
        self.next_btn = ttk.Button(bar, text="Save & Next →",
                                   style="Primary.TButton", command=self._next)
        self.next_btn.pack(side="left", padx=4)
        # body / photo_pane / right column: keep existing pattern, but
        # set tk.Canvas backgrounds to BG and use ttk.Frame for body/right.
```

(Keep the existing `body`, `photo_pane`, `self.canvas`, `right`, `self.rows_canvas`, `vbar`, `self.rows_frame`, scrollregion bind, and mousewheel bind — only swap `tk.Frame`→`ttk.Frame` for `body`/`right` and add `bg=BG` to the two `tk.Canvas` widgets.)

- [ ] **Step 2: Add a meter-drawing helper method**

Add to `PhotoReviewApp`:

```python
    def _meter(self, parent, score, hex_color):
        """Thin Canvas confidence bar; score in [0,1]."""
        c = self.tk.Canvas(parent, width=120, height=6, highlightthickness=0,
                           bg="#e8e8f0")
        w = max(0, min(1.0, float(score))) * 120
        if w > 0:
            c.create_rectangle(0, 0, w, 6, fill=hex_color, width=0)
        return c
```

- [ ] **Step 3: Rebuild the right-column row loop in `_show`**

Replace the `for child in self.rows_frame.winfo_children(): destroy` + row-building block (~905–931) with the card layout:

```python
        for child in self.rows_frame.winfo_children():
            child.destroy()
        self._entries = []
        for n, face in enumerate(faces, 1):
            be = self.best.get((image, face["id"]))
            pre = prefill_name(face, be, self.threshold, self.labels_map)
            state = face_state(face, be, self.threshold, self.labels_map)
            color = STATE_COLORS[state]

            card = self.tk.Frame(self.rows_frame, bg="#ffffff",
                                 highlightthickness=1,
                                 highlightbackground=CARD_BORDER)
            card.pack(fill="x", padx=4, pady=4)

            # thumbnail
            crop = crop_face(src, face["bbox"])
            thumb = crop_to_round_photo(crop, cell=44, radius=7)
            self._cells = getattr(self, "_cells", [])
            self._cells.append(thumb)       # keep ref alive
            tlbl = self.tk.Label(card, image=thumb, bg="#ffffff")
            tlbl.pack(side="left", padx=6, pady=6)

            colf = self.tk.Frame(card, bg="#ffffff")
            colf.pack(side="left", fill="x", expand=True, padx=(0, 6), pady=6)

            top = self.tk.Frame(colf, bg="#ffffff")
            top.pack(fill="x")
            badge = self.tk.Label(top, text=str(n), bg=color, fg="#ffffff",
                                  width=2, font=("TkDefaultFont", 9, "bold"))
            badge.pack(side="left", padx=(0, 6))
            var = self.tk.StringVar(value=pre)
            self.ttk.Entry(top, textvariable=var, width=20).pack(
                side="left", fill="x", expand=True)

            meta = self.tk.Frame(colf, bg="#ffffff")
            meta.pack(fill="x", pady=(4, 0))
            score = be[1] if be else 0.0
            self._meter(meta, score, color).pack(side="left")
            tag = face.get("cluster", "") or "unassigned"
            self.tk.Label(meta, text=f"  {score:.2f} · {tag}",
                          bg="#ffffff", fg="#999",
                          font=("TkDefaultFont", 9)).pack(side="left")

            h = hint_for(face, be, self.threshold, pre)
            if h is not None:
                pill = self.tk.Label(
                    colf, text=f"→ {h[0]}? ({h[1]:.2f}) — click to use",
                    bg="#eafaf1", fg="#2a8a5a", cursor="hand2",
                    font=("TkDefaultFont", 9))
                pill.pack(anchor="w", pady=(4, 0))
                pill.bind("<Button-1>", lambda e, v=var, nm=h[0]: v.set(nm))
            self._entries.append((face["id"], var))
```

(Reset `self._cells = []` at the top of `_show` near where `self._photo_img` is set, so thumbnails from the previous photo are released — add `self._cells = []` right before the box-drawing loop.)

- [ ] **Step 4: Manual verification (GUI)**

Run: `.venv/bin/python face_pipeline.py match --gallery extracted/labels.json`
Confirm by eye:
- Each row is a card with a rounded face thumbnail, a number badge whose color matches that face's box in the photo, a name entry, a confidence meter + `score · cluster`, and (when applicable) a green clickable hint pill.
- Clicking a hint pill fills that entry without committing.
- Entries are prefilled per the unchanged precedence (existing label → candidate ≥ threshold → cluster name → blank).
- Save & Next writes `extracted/labels.json` and the per-photo sidecar (check mtime/content); Back re-shows the previous photo; unreadable photos are skipped.
- Rows scroll when a photo has many faces.

- [ ] **Step 5: Run the full suite**

Run: `.venv/bin/python -m pytest tests/ -q`
Expected: all pass

- [ ] **Step 6: Commit**

```bash
git add face_pipeline.py
git commit -m "feat: redesign PhotoReviewApp rows (thumbnail, badge, meter, hint pill)"
```

---

## Task 7: Update CLAUDE.md notes

The pipeline notes in CLAUDE.md describe the labeler and review UIs; update the wording so it matches the redesigned UI (progress bar, chip reuse, colored boxes, per-row thumbnail/meter). Behavior described (precedence, hint semantics, exclude) is unchanged except exclude now dims rather than hides.

**Files:**
- Modify: `CLAUDE.md` (Face pipeline section)

- [ ] **Step 1: Edit the `label` and `match`/review bullets**

Update the `label` bullet to mention: progress bar, larger rounded crops, ⌘-click dims (excluded overlay) rather than removes, and the typeahead+chips reuse control (replacing the listbox). Update the review bullets to mention: boxes are colored by face state (green confident / blue labeled / amber unassigned), and each row shows a face thumbnail + a confidence meter + the clickable hint pill. Keep all the precedence/centroid/incremental notes intact.

- [ ] **Step 2: Commit**

```bash
git add CLAUDE.md
git commit -m "docs: update CLAUDE.md for redesigned labeler + review UIs"
```

---

## Self-Review Notes

- **Spec coverage:** shared theme (Task 2) ✓; rounded crops + scrolling grid + progress + hybrid reuse for labeler (Tasks 3–4) ✓; recolored boxes via `face_state` (Tasks 1, 5) ✓; review thumbnails + badges + meters + hint pill (Tasks 3, 6) ✓; no-new-dependency / ttk+Canvas constraint ✓; `face_state` TDD-tested as the one new pure helper ✓; exclude-dims behavior with documented fallback ✓; CLAUDE.md update ✓.
- **Out-of-scope items** (click-box-to-focus-row, Canvas-overlay boxes, split_photos changes) correctly absent.
- **Type consistency:** `face_state` signature identical across Tasks 1/5/6; `STATE_COLORS` keys (`confident`/`matched`/`unassigned`) match `face_state` return values; `crop_to_round_photo(crop, cell, radius)` called consistently (cell=140 labeler, cell=44 review); `_row_colors`/`STATE_COLORS[state]` (hex) used for badges, `_bgr(...)` only for the cv2 draw.
- **GUI tasks are manual-verify** per CLAUDE.md — only `face_state` carries pytest steps; all other tasks end with the full suite run to confirm no helper regressions.
