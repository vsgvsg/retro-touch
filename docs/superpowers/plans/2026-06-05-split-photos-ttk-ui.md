# Split Photos Tkinter/ttk UI Port — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace `split_photos.py`'s OpenCV HighGUI editor with a Tkinter/ttk GUI (canvas + right sidebar) that matches `face_pipeline.py`'s look and interaction patterns.

**Architecture:** Keep the entire pure layer (`Box`, `detect_photos`, `crop_box`, sidecar I/O, geometry helpers) untouched. Add a new pure helper `box_state`. Copy face_pipeline's theme helpers (`_install_theme`, `crop_to_round_photo`, color constants) into `split_photos.py` (no cross-import). Replace `Editor`/`render`/`scale_base`/OpenCV `main()` chrome with `SplitterApp` (root + header + action bar) and `CanvasEditor` (a `tk.Canvas` drawing the scan + native box items), plus a ttk sidebar (PHOTOS list + ACTIVE BOX panel with inline rounded-crop preview).

**Tech Stack:** Python 3.13 (Homebrew, via `.venv`), Tkinter/ttk (clam theme), Pillow (`ImageTk`), OpenCV (image I/O + existing crop/detect), NumPy, pytest.

Spec: `docs/superpowers/specs/2026-06-05-split-photos-ttk-ui-design.md`

---

## File Structure

- **Modify `split_photos.py`** — the only source file. Sections, top to bottom:
  - *(unchanged)* `Box`, detector, `crop_box`, sidecar I/O, geometry helpers.
  - *(new, pure)* `box_state(box, scan_shape, active=False)`.
  - *(new, GUI)* theme block (`ACCENT`/`BG`/`STATE_COLORS`/`_install_theme`/`crop_to_round_photo`).
  - *(new, GUI)* `SplitterApp` and `CanvasEditor` classes.
  - *(rewritten)* `main()` — builds `SplitterApp`, Tk-less guard, updated help text.
  - *(deleted)* `Editor`, `render`, `scale_base`, `WINDOW`/`BANNER_H`/`HANDLE_R` constants used only by the OpenCV editor.
- **Modify `tests/test_split_photos.py`** — add `box_state` tests. Existing pure tests stay as the regression guard.
- **Modify `CLAUDE.md`** — Photo Scan Splitter section + Commands.

> **Run everything via `.venv/bin/python`** (system Python's Tk can't open a window on this macOS). Tests run on either, but GUI smoke tests need the venv.

---

### Task 1: `box_state` pure helper (TDD)

**Files:**
- Modify: `split_photos.py` (add helper after the geometry helpers, before the deleted Editor section — e.g. near line 286)
- Test: `tests/test_split_photos.py`

- [ ] **Step 1: Write the failing tests**

Add to `tests/test_split_photos.py`:

```python
# ---- box_state ----
def test_box_state_editing_when_active():
    b = sp.Box(center=[100, 100], size=[50, 50])
    assert sp.box_state(b, (800, 600), active=True) == "editing"

def test_box_state_attention_zero_size():
    b = sp.Box(center=[100, 100], size=[0, 50])
    assert sp.box_state(b, (800, 600)) == "attention"

def test_box_state_attention_off_canvas():
    # box centered near the right edge, wider than the remaining margin
    b = sp.Box(center=[590, 400], size=[300, 100])
    assert sp.box_state(b, (800, 600)) == "attention"  # scan_shape=(h,w)

def test_box_state_cropped_when_output_set():
    b = sp.Box(center=[300, 300], size=[100, 80], output="scan_01.jpg")
    assert sp.box_state(b, (800, 600)) == "cropped"

def test_box_state_neutral_default():
    b = sp.Box(center=[300, 300], size=[100, 80])
    assert sp.box_state(b, (800, 600)) == "neutral"

def test_box_state_active_beats_cropped():
    b = sp.Box(center=[300, 300], size=[100, 80], output="x.jpg")
    assert sp.box_state(b, (800, 600), active=True) == "editing"
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python3 -m pytest tests/test_split_photos.py -k box_state -q`
Expected: FAIL — `AttributeError: module 'split_photos' has no attribute 'box_state'`

- [ ] **Step 3: Implement `box_state`**

Add to `split_photos.py` after `scale_base` is removed / near the other geometry helpers (place it before the GUI section). Uses `cv2.boxPoints` (already imported) to get rotated corners:

```python
def box_state(box: "Box", scan_shape: tuple[int, int], active: bool = False) -> str:
    """Classify a box for color-coding. scan_shape is (h, w).

    Precedence: editing (active) > attention (zero-size or off-canvas) >
    cropped (already exported) > neutral. Mirrors face_pipeline.face_state so
    the sidebar dot and any canvas tint always agree.
    """
    if active:
        return "editing"
    bw, bh = box.size
    if bw < 1 or bh < 1:
        return "attention"
    h, w = scan_shape
    rect = ((box.center[0], box.center[1]), (bw, bh), box.angle)
    pts = cv2.boxPoints(rect)
    if (pts[:, 0].min() < 0 or pts[:, 0].max() > w
            or pts[:, 1].min() < 0 or pts[:, 1].max() > h):
        return "attention"
    return "cropped" if box.output else "neutral"
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python3 -m pytest tests/test_split_photos.py -k box_state -q`
Expected: PASS (6 passed)

- [ ] **Step 5: Run the full existing suite (regression guard)**

Run: `python3 -m pytest tests/test_split_photos.py -q`
Expected: PASS — all prior tests + the 6 new ones.

- [ ] **Step 6: Commit**

```bash
git add split_photos.py tests/test_split_photos.py
git commit -m "feat: box_state pure helper for split_photos color-coding

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>"
```

---

### Task 2: Theme + STATE_COLORS block (copied, no cross-import)

**Files:**
- Modify: `split_photos.py` (add a GUI theme section; map `box_state` → color)

- [ ] **Step 1: Add the theme block**

Insert a new section in `split_photos.py` (after `box_state`, before the classes). This is copied from `face_pipeline.py` plus a `neutral` entry and a `state_color` mapper:

```python
# ---------------------------------------------------------------------------
# Shared GUI theme (ttk) — copied from face_pipeline.py (tools never cross-import)
# ---------------------------------------------------------------------------
ACCENT = "#5a6cf0"
BG = "#fafaff"
CARD_BORDER = "#ececf2"
STATE_COLORS = {            # box_state -> hex for the sidebar dot / canvas box
    "editing": ACCENT,
    "cropped": "#2faf6a",
    "attention": "#d8a23a",
    "neutral": "#9a9aa8",
}


def state_color(state: str) -> str:
    return STATE_COLORS.get(state, STATE_COLORS["neutral"])


def _install_theme(root):
    """Configure a shared ttk.Style; safe to call once per app root."""
    from tkinter import ttk
    style = ttk.Style(root)
    try:
        style.theme_use("clam")
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


def crop_to_round_photo(crop, cell=128, radius=12):
    """BGR crop -> letterboxed cell x cell rounded-corner Tk PhotoImage.

    Degrades to a plain square if rounding fails, so the GUI never crashes.
    """
    from PIL import Image, ImageDraw, ImageTk
    base = np.full((cell, cell, 3), 245, np.uint8)
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
        pass
    return ImageTk.PhotoImage(img)
```

- [ ] **Step 2: Verify it imports cleanly**

Run: `.venv/bin/python -c "import split_photos as sp; print(sp.state_color('attention'), sp.STATE_COLORS['editing'])"`
Expected: `#d8a23a #5a6cf0`

- [ ] **Step 3: Commit**

```bash
git add split_photos.py
git commit -m "feat: ttk theme + STATE_COLORS for split_photos GUI

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>"
```

---

### Task 3: `CanvasEditor` — scan canvas with native box items

**Files:**
- Modify: `split_photos.py` (new class; reuses `point_in_box`, `grab_handle`, `resize_box`, `_orient_arrow`, `disp_to_full`, `crop_box`)

This class owns a `tk.Canvas`, draws the scan as a background `PhotoImage`, and draws each box as native canvas items. It exposes callbacks so `SplitterApp` can react to selection/geometry changes (to refresh the sidebar).

- [ ] **Step 1: Add the `CanvasEditor` class**

```python
class CanvasEditor:
    """A tk.Canvas showing one scan; boxes drawn as native canvas items.

    Geometry stays in full-resolution scan coords; self.scale maps to display.
    Calls on_change() after any selection or geometry edit so the host can
    refresh the sidebar.
    """
    HANDLE_R = 6  # display-px radius of a corner handle oval

    def __init__(self, parent, tk, on_change):
        self.tk = tk
        self.on_change = on_change
        self.image = None          # full-res BGR numpy scan
        self.boxes = []
        self.active = -1
        self.scale = 1.0
        self._bg_photo = None      # keep a ref so Tk doesn't GC it
        self.drag = None           # None | 'move' | 'new' | ('resize', handle)
        self.drag_start = None     # full-coord
        self.canvas = tk.Canvas(parent, highlightthickness=0, bg=BG,
                                cursor="tcross")
        self.canvas.pack(fill="both", expand=True)
        self.canvas.bind("<Button-1>", self._on_press)
        self.canvas.bind("<B1-Motion>", self._on_motion)
        self.canvas.bind("<ButtonRelease-1>", self._on_release)

    # ---- public API ----
    def set_scan(self, image, boxes):
        self.image = image
        self.boxes = boxes
        self.active = 0 if boxes else -1
        h, w = image.shape[:2]
        self.scale = min(1000 / w, 1000 / h, 1.0)
        self._render_background()
        self.redraw()

    def active_box(self):
        if 0 <= self.active < len(self.boxes):
            return self.boxes[self.active]
        return None

    # ---- rendering ----
    def _render_background(self):
        from PIL import Image, ImageTk
        disp = cv2.resize(self.image, None, fx=self.scale, fy=self.scale,
                          interpolation=cv2.INTER_AREA)
        rgb = cv2.cvtColor(disp, cv2.COLOR_BGR2RGB)
        self._bg_photo = ImageTk.PhotoImage(Image.fromarray(rgb))
        dh, dw = disp.shape[:2]
        self.canvas.configure(width=dw, height=dh)

    def redraw(self):
        c = self.canvas
        c.delete("overlay")
        if self._bg_photo is not None:
            c.delete("bg")
            c.create_image(0, 0, anchor="nw", image=self._bg_photo, tags="bg")
        for i, b in enumerate(self.boxes):
            active = (i == self.active)
            state = box_state(b, self.image.shape[:2], active=active)
            color = state_color(state)
            rect = ((b.center[0] * self.scale, b.center[1] * self.scale),
                    (b.size[0] * self.scale, b.size[1] * self.scale), b.angle)
            pts = cv2.boxPoints(rect)
            flat = [coord for xy in pts for coord in xy]
            c.create_polygon(flat, outline=color, width=3 if active else 2,
                             fill="", tags="overlay")
            # number badge at the box center
            c.create_text(b.center[0] * self.scale, b.center[1] * self.scale,
                          text=str(i + 1), fill=color,
                          font=("TkDefaultFont", 11, "bold"), tags="overlay")
            if active:
                for (px, py) in pts:
                    c.create_oval(px - self.HANDLE_R, py - self.HANDLE_R,
                                  px + self.HANDLE_R, py + self.HANDLE_R,
                                  outline=color, fill="#ffffff", tags="overlay")
                (x0, y0), (x1, y1) = _orient_arrow(b)
                c.create_line(x0 * self.scale, y0 * self.scale,
                              x1 * self.scale, y1 * self.scale,
                              fill="#e0405a", width=2, arrow="last",
                              tags="overlay")

    # ---- coordinate helper ----
    def _full(self, x, y):
        return disp_to_full((x, y), self.scale)

    # ---- mouse ----
    def _on_press(self, event):
        if self.image is None:
            return
        fx, fy = self._full(event.x, event.y)
        b = self.active_box()
        if b is not None:
            handle = grab_handle((fx, fy), b, self.HANDLE_R / self.scale)
            if handle is not None:
                self.drag = ("resize", handle)
                self.drag_start = (fx, fy)
                return
        for i, bx in enumerate(self.boxes):
            if point_in_box((fx, fy), bx):
                self.active = i
                self.drag = "move"
                self.drag_start = (fx, fy)
                self.redraw()
                self.on_change()
                return
        self.drag = "new"
        self.drag_start = (fx, fy)

    def _on_motion(self, event):
        if not self.drag:
            return
        fx, fy = self._full(event.x, event.y)
        b = self.active_box()
        if self.drag == "move" and b is not None:
            b.center[0] += fx - self.drag_start[0]
            b.center[1] += fy - self.drag_start[1]
            self.drag_start = (fx, fy)
        elif isinstance(self.drag, tuple) and self.drag[0] == "resize" and b is not None:
            resize_box(b, self.drag[1], (fx, fy))
        self.redraw()

    def _on_release(self, event):
        fx, fy = self._full(event.x, event.y)
        if self.drag == "new":
            x0, y0 = self.drag_start
            w, h = abs(fx - x0), abs(fy - y0)
            if w > 10 and h > 10:
                self.boxes.append(
                    Box(center=[(fx + x0) / 2, (fy + y0) / 2], size=[w, h]))
                self._renumber()
                self.active = len(self.boxes) - 1
        self.drag = None
        self.drag_start = None
        self.redraw()
        self.on_change()

    def _renumber(self):
        for i, b in enumerate(self.boxes, 1):
            b.id = i
```

- [ ] **Step 2: Smoke-test the class builds + renders headless**

Make a tiny synthetic scan and drive it auto-closing:

```bash
.venv/bin/python -c "
import numpy as np, tkinter as tk, split_photos as sp
img = np.full((400,600,3),245,np.uint8); img[100:250,150:400]=60
root = tk.Tk(); sp._install_theme(root)
ce = sp.CanvasEditor(root, tk, on_change=lambda: None)
ce.set_scan(img, sp.detect_photos(img))
root.after(150, root.destroy); root.mainloop()
print('canvas ok')
"
```
Expected: prints `canvas ok` with no traceback (a window flashes briefly).

- [ ] **Step 3: Commit**

```bash
git add split_photos.py
git commit -m "feat: CanvasEditor — Tk canvas with native box items

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>"
```

---

### Task 4: `SplitterApp` — root, header, sidebar, action bar

**Files:**
- Modify: `split_photos.py` (new class; reuses `crop_box`, `save_metadata`, `load_metadata`, `detect_photos`, `crop_to_round_photo`, `box_state`)

`SplitterApp` owns the window, the scan list/index, the `CanvasEditor` (left), and the ttk sidebar (right): a PHOTOS list (rows colored by `box_state`, click-to-select) and an ACTIVE BOX panel (angle stepper, orientation stepper, inline preview, Delete), plus the bottom action bar.

- [ ] **Step 1: Add the `SplitterApp` class**

```python
class SplitterApp:
    """Tkinter app: scan canvas (left) + box list / active-box panel (right)."""

    def __init__(self, scans, out_dir):
        import tkinter as tk
        from tkinter import ttk
        self.tk = tk
        self.ttk = ttk
        self.scans = scans
        self.out_dir = out_dir
        self.idx = 0
        self.image = None
        self.boxes = []
        self.scan_path = None
        self._preview_photo = None

        self.root = tk.Tk()
        self.root.title("Photo Scan Splitter")
        self.root.geometry("1180x820")
        _install_theme(self.root)

        # header
        head = ttk.Frame(self.root)
        head.pack(fill="x", padx=16, pady=(12, 4))
        self.title_var = tk.StringVar()
        self.sub_var = tk.StringVar()
        ttk.Label(head, textvariable=self.title_var, style="Title.TLabel").pack(anchor="w")
        ttk.Label(head, textvariable=self.sub_var, style="Sub.TLabel").pack(anchor="w")
        self.progress = ttk.Progressbar(head, maximum=len(self.scans), cursor="hand2")
        self.progress.pack(fill="x", pady=(8, 0))
        self.progress.bind("<Button-1>", self._on_progress_click)

        # body: canvas (left) + sidebar (right)
        body = ttk.Frame(self.root)
        body.pack(fill="both", expand=True, padx=16, pady=8)
        left = ttk.Frame(body)
        left.pack(side="left", fill="both", expand=True)
        self.editor = CanvasEditor(left, tk, on_change=self._refresh_sidebar)

        side = ttk.Frame(body, width=300)
        side.pack(side="right", fill="y", padx=(12, 0))
        side.pack_propagate(False)

        ttk.Label(side, text="PHOTOS", style="Sub.TLabel").pack(anchor="w")
        self.rows = ttk.Frame(side)
        self.rows.pack(fill="x", pady=(2, 10))

        ttk.Label(side, text="ACTIVE BOX", style="Sub.TLabel").pack(anchor="w")
        ang = ttk.Frame(side); ang.pack(fill="x", pady=(2, 4))
        ttk.Label(ang, text="angle").pack(side="left")
        ttk.Button(ang, text="◀", width=3, command=lambda: self._nudge(-0.5)).pack(side="left")
        self.angle_var = tk.StringVar(value="0.0°")
        ttk.Label(ang, textvariable=self.angle_var, width=7,
                  anchor="center").pack(side="left")
        ttk.Button(ang, text="▶", width=3, command=lambda: self._nudge(0.5)).pack(side="left")

        ori = ttk.Frame(side); ori.pack(fill="x", pady=(0, 8))
        ttk.Label(ori, text="orient").pack(side="left")
        ttk.Button(ori, text="◀", width=3, command=lambda: self._orient(-90)).pack(side="left")
        self.orient_var = tk.StringVar(value="0°")
        ttk.Label(ori, textvariable=self.orient_var, width=7,
                  anchor="center").pack(side="left")
        ttk.Button(ori, text="▶", width=3, command=lambda: self._orient(90)).pack(side="left")

        self.preview_label = ttk.Label(side)
        self.preview_label.pack(pady=(4, 8))
        ttk.Button(side, text="Delete box", command=self._delete).pack(fill="x")

        # action bar
        bar = ttk.Frame(self.root)
        bar.pack(fill="x", padx=16, pady=(4, 12))
        ttk.Button(bar, text="+ Re-detect", command=self._redetect).pack(side="left")
        ttk.Button(bar, text="Crop all", command=self._crop_all).pack(side="left", padx=(8, 0))
        self.next_btn = ttk.Button(bar, text="Next →", style="Primary.TButton",
                                   command=self._next)
        self.next_btn.pack(side="right")
        ttk.Button(bar, text="← Prev", command=self._prev).pack(side="right", padx=(0, 8))

        # keyboard (mirrors the old editor)
        self.root.bind("<Key>", self._on_key)

    # ---- scan lifecycle ----
    def _load_scan(self):
        path = self.scans[self.idx]
        self.scan_path = path
        self.image = cv2.imread(path)
        boxes = load_metadata(path)
        if boxes is None:
            boxes = detect_photos(self.image)
        self.boxes = boxes
        self.editor.set_scan(self.image, self.boxes)
        self._refresh_header()
        self._refresh_sidebar()

    def _refresh_header(self):
        self.title_var.set(os.path.basename(self.scan_path))
        cropped = sum(1 for b in self.boxes if b.output)
        self.sub_var.set(f"{len(self.boxes)} photos · {cropped} cropped")
        self.progress.configure(value=self.idx + 1)

    def _refresh_sidebar(self):
        for w in self.rows.winfo_children():
            w.destroy()
        for i, b in enumerate(self.boxes):
            active = (i == self.editor.active)
            color = state_color(box_state(b, self.image.shape[:2], active=active))
            row = self.tk.Frame(self.rows, bg=ACCENT if active else BG)
            row.pack(fill="x", pady=1)
            dot = self.tk.Label(row, text="●", fg=color,
                                bg=ACCENT if active else BG)
            dot.pack(side="left", padx=(4, 6))
            txt = f"{i+1}   {int(b.size[0])}×{int(b.size[1])}"
            lbl = self.tk.Label(row, text=txt, anchor="w",
                                fg="#ffffff" if active else "#1a1a2e",
                                bg=ACCENT if active else BG)
            lbl.pack(side="left", fill="x", expand=True)
            for wdg in (row, dot, lbl):
                wdg.bind("<Button-1>", lambda e, idx=i: self._select(idx))
        self._refresh_active_panel()
        self._refresh_header()

    def _refresh_active_panel(self):
        b = self.editor.active_box()
        if b is None:
            self.angle_var.set("—"); self.orient_var.set("—")
            self.preview_label.configure(image="")
            self._preview_photo = None
            return
        self.angle_var.set(f"{b.angle:.1f}°")
        self.orient_var.set(f"{b.orientation}°")
        try:
            crop = crop_box(self.image, b)
            self._preview_photo = crop_to_round_photo(crop, cell=180)
        except ValueError:
            self._preview_photo = crop_to_round_photo(
                np.full((10, 10, 3), 200, np.uint8), cell=180)
        self.preview_label.configure(image=self._preview_photo)

    # ---- selection / edits ----
    def _select(self, idx):
        self.editor.active = idx
        self.editor.redraw()
        self._refresh_sidebar()

    def _nudge(self, delta):
        b = self.editor.active_box()
        if b is not None:
            b.angle = max(-45.0, min(45.0, b.angle + delta))
            self.editor.redraw(); self._refresh_sidebar()

    def _orient(self, delta):
        b = self.editor.active_box()
        if b is not None:
            b.orientation = (b.orientation + delta) % 360
            self.editor.redraw(); self._refresh_sidebar()

    def _delete(self):
        i = self.editor.active
        if 0 <= i < len(self.boxes):
            del self.boxes[i]
            self.editor._renumber()
            self.editor.active = min(i, len(self.boxes) - 1)
            self.editor.redraw(); self._refresh_sidebar()

    # ---- actions ----
    def _redetect(self):
        self.boxes = detect_photos(self.image)
        self.editor.set_scan(self.image, self.boxes)
        self._refresh_sidebar()

    def _save(self):
        h, w = self.image.shape[:2]
        save_metadata(self.scan_path, (w, h), self.boxes)

    def _crop_all(self):
        os.makedirs(self.out_dir, exist_ok=True)
        stem = os.path.splitext(os.path.basename(self.scan_path))[0]
        for i, b in enumerate(self.boxes, 1):
            try:
                out = crop_box(self.image, b)
            except ValueError:
                print(f"  ! skipping box {i}: zero size")
                continue
            name = f"{stem}_{i:02d}.jpg"
            b.output = name
            cv2.imwrite(os.path.join(self.out_dir, name), out)
        self._save()
        self._refresh_sidebar()

    def _next(self):
        self._save()
        if self.idx < len(self.scans) - 1:
            self.idx += 1
            self._load_scan()

    def _prev(self):
        self._save()
        if self.idx > 0:
            self.idx -= 1
            self._load_scan()

    def _on_progress_click(self, event):
        width = self.progress.winfo_width()
        if width <= 0:
            return
        frac = max(0.0, min(1.0, event.x / width))
        target = min(len(self.scans) - 1, int(frac * len(self.scans)))
        if target != self.idx:
            self._save()
            self.idx = target
            self._load_scan()

    # ---- keyboard (mirrors the old OpenCV editor) ----
    def _on_key(self, event):
        b = self.editor.active_box()
        k = event.keysym
        step = 20.0
        if k in ("n", "Tab") and self.boxes:
            self.editor.active = (self.editor.active + 1) % len(self.boxes)
            self.editor.redraw(); self._refresh_sidebar()
        elif k in ("Left", "h"): self._move(-step, 0)
        elif k in ("Right", "l"): self._move(step, 0)
        elif k in ("Up", "k"): self._move(0, -step)
        elif k in ("Down", "j"): self._move(0, step)
        elif k == "bracketleft": self._orient(-90)
        elif k == "bracketright": self._orient(90)
        elif k == "comma": self._nudge(-0.5)
        elif k == "period": self._nudge(0.5)
        elif k == "less": self._nudge(-5.0)
        elif k == "greater": self._nudge(5.0)
        elif k in ("x", "Delete", "BackSpace"): self._delete()
        elif k == "Return": self._crop_all(); self._next()

    def _move(self, dx, dy):
        b = self.editor.active_box()
        if b is not None:
            b.center[0] += dx; b.center[1] += dy
            self.editor.redraw(); self._refresh_sidebar()

    def _show(self):
        self._load_scan()

    def run(self):
        self._show()
        self.root.mainloop()
```

- [ ] **Step 2: Headless wiring smoke test**

Use the project's documented pattern (a real scan from `images/` if present, else synthetic written to a temp file so `imread`/`scan_path` work):

```bash
.venv/bin/python -c "
import os, glob, numpy as np, cv2, tempfile, split_photos as sp
scans = sp.list_scans('images')
if not scans:
    d = tempfile.mkdtemp(); p = os.path.join(d,'t.jpg')
    img = np.full((400,600,3),245,np.uint8); img[80:240,120:380]=60
    cv2.imwrite(p, img); scans = [p]
app = sp.SplitterApp(scans, out_dir=tempfile.mkdtemp())
app._show()
app.root.after(150, app.root.destroy)
app.root.mainloop()
print('app ok')
"
```
Expected: prints `app ok`, no traceback.

- [ ] **Step 3: Commit**

```bash
git add split_photos.py
git commit -m "feat: SplitterApp — header, sidebar, action bar, inline preview

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>"
```

---

### Task 5: Rewire `main()`, delete OpenCV editor, update help text

**Files:**
- Modify: `split_photos.py` (`main()`, delete `Editor`/`render`/`scale_base` + unused constants)

- [ ] **Step 1: Delete the OpenCV editor code**

Remove from `split_photos.py`:
- the `render()` function
- the `scale_base()` function
- the entire `Editor` class
- the module constants used only by it: `WINDOW`, `BANNER_H`, and the standalone `HANDLE_R` (CanvasEditor has its own `HANDLE_R` attribute).

Keep `_orient_arrow` (used by `CanvasEditor.redraw`).

- [ ] **Step 2: Rewrite `main()`**

```python
def main(argv: list[str]) -> int:
    import tkinter as tk
    images_dir = argv[1] if len(argv) > 1 else "images"
    out_dir = "extracted"
    if not os.path.isdir(images_dir):
        print(f"No such images directory: {images_dir}")
        return 1
    scans = list_scans(images_dir)
    if not scans:
        print(f"No scans found in {images_dir}")
        return 1

    print("Controls: drag inside=move | drag edge/corner=resize | "
          "drag empty=new box | click a sidebar row=select | "
          "arrows/hjkl=nudge | n/Tab=next box | [ ]=orient | "
          ", . =tilt 0.5deg | < > =tilt 5deg | x/Del=delete | "
          "Enter=crop all + next | Re-detect / Crop all / Prev / Next buttons")

    try:
        app = SplitterApp(scans, out_dir)
    except tk.TclError:
        print("Cannot open a GUI window. Run via the venv Python "
              "(Homebrew Tk): .venv/bin/python split_photos.py")
        return 1
    app.run()
    print("Done.")
    return 0
```

- [ ] **Step 3: Verify the module imports and pure tests still pass**

Run: `python3 -m pytest tests/test_split_photos.py -q`
Expected: PASS (all pure tests + box_state). No reference to deleted `Editor`/`render`.

Run: `.venv/bin/python -c "import split_photos as sp; assert not hasattr(sp,'Editor') and not hasattr(sp,'render'); print('cleanup ok')"`
Expected: `cleanup ok`

- [ ] **Step 4: Full headless app smoke test again (post-cleanup)**

Re-run the Task 4 Step 2 smoke command.
Expected: `app ok`, no traceback.

- [ ] **Step 5: Commit**

```bash
git add split_photos.py
git commit -m "feat: rewire main() to SplitterApp; drop OpenCV editor

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>"
```

---

### Task 6: Update CLAUDE.md docs

**Files:**
- Modify: `CLAUDE.md`

- [ ] **Step 1: Update the Photo Scan Splitter overview + Commands**

In the top description, change the split_photos sentence to note it's now a ttk GUI. In **## Commands**, replace the split launch line with:

```
- `.venv/bin/python split_photos.py` - launch the Tkinter editor (canvas + sidebar, ttk-themed like the face pipeline GUIs). Needs a human at the GUI; run via the venv (system Python's Tk can't open a window on this macOS). `python3 split_photos.py` works only where a GUI-capable Tk is available.
```

- [ ] **Step 2: Add a split_photos GUI notes subsection**

Add after the Commands block (mirroring the face pipeline notes):

```
## Photo Scan Splitter GUI (split_photos.py)
- ttk app (clam theme, shared look with face_pipeline): header (filename + "N photos · M cropped" + clickable progress bar to jump scans), a `CanvasEditor` (tk.Canvas: scan as a background PhotoImage, each box drawn as native canvas items — polygon + corner handles + orientation arrow), and a right sidebar (PHOTOS list colored by `box_state`, click a row to select; ACTIVE BOX panel: angle/orientation steppers + inline rounded-crop preview + Delete).
- Mouse: drag inside=move, drag edge/corner=resize, drag empty=new box. Keyboard mirrors the old editor (arrows/hjkl nudge, [ ]=orient, , . < >=tilt, x/Del=delete, n/Tab=next box, Enter=crop all + next).
- `box_state(box, scan_shape, active)` is the pure helper (TDD-tested) that color-codes both the sidebar dot and the canvas box: editing(accent) > attention(amber: zero-size/off-canvas) > cropped(green) > neutral(grey). Mirrors face_pipeline's `face_state`.
- Theme helpers (`_install_theme`, `crop_to_round_photo`, color constants) are COPIED from face_pipeline.py — the tools never cross-import; the `extracted/`/`*.photos.json` artifacts remain the only interface.
- Catch GUI wiring errors without a human (same pattern as the face pipeline): build `SplitterApp(scans, out_dir)`, call `app._show()`, then `app.root.after(150, app.root.destroy); app.root.mainloop()`.
```

- [ ] **Step 3: Remove the now-obsolete OpenCV HighGUI editor gotchas**

In `## Gotchas (OpenCV HighGUI on macOS)`, the entries about `waitKeyEx`, window focus, `getWindowProperty`/`destroyWindow` apply only to the deleted editor. Keep the detector/crop-geometry gotchas (`normalize_rect` angle, `crop_box` orientation, best-effort auto-detect). Delete the three editor-only bullets (waitKeyEx, key focus, getWindowProperty) and retitle the section `## Gotchas (detector & crop geometry)`.

- [ ] **Step 4: Commit**

```bash
git add CLAUDE.md
git commit -m "docs: CLAUDE.md for split_photos Tkinter GUI

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>"
```

---

### Task 7: Final verification

- [ ] **Step 1: Full test suite**

Run: `python3 -m pytest tests/ -q`
Expected: PASS (face pipeline + restore + split, including the new box_state tests).

- [ ] **Step 2: Headless GUI smoke (venv)**

Re-run the Task 4 Step 2 command with `.venv/bin/python`.
Expected: `app ok`.

- [ ] **Step 3: Human GUI check (manual, requires a person)**

Run: `.venv/bin/python split_photos.py`
Verify: scan renders; boxes draw with colored outlines + number badges; dragging moves/resizes/creates; sidebar rows highlight the active box and recolor by state; angle/orient steppers + inline preview update live; Re-detect / Crop all / Prev / Next / progress-click work; Crop all writes to `extracted/`.

- [ ] **Step 4: Finish the branch**

Use the `superpowers:finishing-a-development-branch` skill to choose merge/PR/cleanup.

---

## Self-Review notes

- **Spec coverage:** full Tk port (T3–T5), Layout A sidebar (T4), native canvas items (T3), inline preview (T4 `_refresh_active_panel`), `box_state` mirroring `face_state` (T1), preserved pure layer + sidecar contract (untouched; verified T1/T5), theme copied not imported (T2), Tk-less guard (T5 `main`), tests + headless smoke + docs (T1/T4/T6/T7). All spec sections mapped.
- **Placeholder scan:** every code/command step has concrete content; no TBD/TODO.
- **Type consistency:** `box_state(box, scan_shape, active)`, `state_color(state)`, `CanvasEditor(parent, tk, on_change)` with `set_scan`/`active_box`/`redraw`/`active`/`HANDLE_R`, and `SplitterApp(scans, out_dir)` with `_show`/`run`/`_refresh_sidebar` used consistently across tasks. `scan_shape` is `(h, w)` everywhere; `save_metadata` still gets `(w, h)`.
