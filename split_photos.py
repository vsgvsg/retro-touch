"""Photo Scan Splitter — detect, adjust, and crop photos from scanner images."""
from __future__ import annotations

import glob
import json
import math
import os
import sys
from dataclasses import dataclass

import cv2
import numpy as np


@dataclass
class Box:
    """A detected/edited photo region, in full-resolution scan coordinates."""
    center: list[float]          # [x, y]
    size: list[float]            # [w, h]
    angle: float = 0.0           # deskew degrees, -45..45
    orientation: int = 0         # 0/90/180/270 — which way is "top"
    id: int = 0
    output: str = ""

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "center": [round(self.center[0], 1), round(self.center[1], 1)],
            "size": [round(self.size[0], 1), round(self.size[1], 1)],
            "angle": round(self.angle, 2),
            "orientation": self.orientation,
            "output": self.output,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "Box":
        return cls(
            center=list(d["center"]),
            size=list(d["size"]),
            angle=float(d.get("angle", 0.0)),
            orientation=int(d.get("orientation", 0)),
            id=int(d.get("id", 0)),
            output=d.get("output", ""),
        )


# ---------------------------------------------------------------------------
# Detector
# ---------------------------------------------------------------------------
def _estimate_background(gray: np.ndarray) -> float:
    """Estimate the uniform scanner-bed gray level.

    The bed is the largest near-uniform region, so its level dominates the
    high end of the histogram. Use a high percentile rather than assuming
    pure white, so mid-gray beds (not just white) are handled.
    """
    return float(np.percentile(gray, 85))


def normalize_rect(bw: float, bh: float, angle: float) -> tuple[float, float, float]:
    """Normalize a minAreaRect to a small deskew angle in (-45, 45].

    cv2.minAreaRect reports the angle of an arbitrary edge, so a near-axis
    photo can come back as ~0 or ~-90 with w/h swapped. Fold the ±90 ambiguity
    away so every box shares a consistent upright baseline; deliberate quarter
    turns are expressed via Box.orientation, not the deskew angle.
    """
    while angle <= -45:
        angle += 90
        bw, bh = bh, bw
    while angle > 45:
        angle -= 90
        bw, bh = bh, bw
    return bw, bh, angle


def detect_photos(image: np.ndarray, min_area_frac: float = 0.01) -> list[Box]:
    """Detect photo regions on a (light or mid-gray) scanner background.

    Background is the large near-uniform bed; photos are everything that
    differs from it. Uses connected components (one per photo) and morphological
    opening to break thin bridges between touching photos. Returns rotated
    Boxes in full-resolution coordinates, sorted top-to-bottom.

    Auto-detection is best-effort: touching or full-bleed photos may merge or
    split imperfectly. The human refines results in the editor.
    """
    h, w = image.shape[:2]
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blur = cv2.medianBlur(gray, 9)

    bg = _estimate_background(blur)
    # Foreground = pixels meaningfully darker than the bed. Margin avoids
    # marking bed noise as photo; 40 levels works across white and gray beds.
    thresh = max(0.0, bg - 40)
    fg = (blur < thresh).astype(np.uint8) * 255

    k = max(3, (min(h, w) // 80) | 1)  # odd kernel scaled to image
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
    # Open to drop speckle and snap thin bridges between adjacent photos.
    fg = cv2.morphologyEx(fg, cv2.MORPH_OPEN, kernel, iterations=2)
    # Close to fill bright interiors within a single photo.
    fg = cv2.morphologyEx(fg, cv2.MORPH_CLOSE, kernel, iterations=1)

    n, labels, stats, _ = cv2.connectedComponentsWithStats(fg, 8)
    min_area = min_area_frac * h * w
    boxes: list[Box] = []
    for i in range(1, n):  # 0 is background
        area = stats[i, cv2.CC_STAT_AREA]
        if area < min_area or area > 0.95 * h * w:
            continue
        mask = (labels == i).astype(np.uint8)
        cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not cnts:
            continue
        (cx, cy), (bw, bh), ang = cv2.minAreaRect(max(cnts, key=cv2.contourArea))
        if bw < 1 or bh < 1:
            continue
        bw, bh, ang = normalize_rect(bw, bh, ang)
        # A minAreaRect over an irregular merged blob can exceed page bounds.
        # Clamp size so the box stays on-canvas and remains editable by hand.
        bw = min(bw, float(w))
        bh = min(bh, float(h))
        boxes.append(Box(center=[cx, cy], size=[bw, bh], angle=ang))

    boxes.sort(key=lambda b: (round(b.center[1] / 50), b.center[0]))
    for i, b in enumerate(boxes, 1):
        b.id = i
    return boxes


# ---------------------------------------------------------------------------
# Cropper
# ---------------------------------------------------------------------------
def crop_box(image: np.ndarray, box: Box) -> np.ndarray:
    """Deskew the box, crop the upright rectangle, then apply orientation."""
    cx, cy = box.center
    bw, bh = int(round(box.size[0])), int(round(box.size[1]))
    if bw < 1 or bh < 1:
        raise ValueError("box has non-positive size")

    h, w = image.shape[:2]
    M = cv2.getRotationMatrix2D((cx, cy), box.angle, 1.0)
    rotated = cv2.warpAffine(image, M, (w, h), flags=cv2.INTER_CUBIC,
                             borderMode=cv2.BORDER_REPLICATE)
    crop = cv2.getRectSubPix(rotated, (bw, bh), (cx, cy))

    # orientation names which edge is the photo's real top (matching the UI
    # arrow). Rotate so that edge ends up at the top of the output.
    #   90 = top is on the right  -> CCW brings right edge up
    #   270 = top is on the left  -> CW brings left edge up
    orient = box.orientation % 360
    if orient == 90:
        crop = cv2.rotate(crop, cv2.ROTATE_90_COUNTERCLOCKWISE)
    elif orient == 180:
        crop = cv2.rotate(crop, cv2.ROTATE_180)
    elif orient == 270:
        crop = cv2.rotate(crop, cv2.ROTATE_90_CLOCKWISE)
    return crop


# ---------------------------------------------------------------------------
# Metadata sidecar I/O
# ---------------------------------------------------------------------------
def sidecar_path(scan_path: str) -> str:
    stem, _ = os.path.splitext(scan_path)
    return stem + ".photos.json"


def save_metadata(scan_path: str, scan_size: tuple[int, int], boxes: list[Box]) -> str:
    """scan_size is (width, height)."""
    data = {
        "scan": os.path.basename(scan_path),
        "scan_size": [int(scan_size[0]), int(scan_size[1])],
        "boxes": [b.to_dict() for b in boxes],
    }
    path = sidecar_path(scan_path)
    with open(path, "w") as f:
        json.dump(data, f, indent=2)
    return path


def load_metadata(scan_path: str) -> list[Box] | None:
    path = sidecar_path(scan_path)
    if not os.path.exists(path):
        return None
    with open(path) as f:
        data = json.load(f)
    return [Box.from_dict(d) for d in data.get("boxes", [])]


# ---------------------------------------------------------------------------
# Display / geometry helpers (pure)
# ---------------------------------------------------------------------------
def disp_to_full(pt: tuple[float, float], scale: float) -> tuple[float, float]:
    return (pt[0] / scale, pt[1] / scale)


def full_to_disp(pt: tuple[float, float], scale: float) -> tuple[float, float]:
    return (pt[0] * scale, pt[1] * scale)


def point_in_box(pt: tuple[float, float], box: Box) -> bool:
    """Is full-coord point inside the rotated box?"""
    cx, cy = box.center
    bw, bh = box.size
    ang = math.radians(box.angle)
    dx, dy = pt[0] - cx, pt[1] - cy
    # rotate point into box-local frame
    lx = dx * math.cos(-ang) - dy * math.sin(-ang)
    ly = dx * math.sin(-ang) + dy * math.cos(-ang)
    return abs(lx) <= bw / 2 and abs(ly) <= bh / 2


def _to_local(pt: tuple[float, float], box: Box) -> tuple[float, float]:
    """Map a full-coord point into the box's center-origin, axis-aligned frame."""
    ang = math.radians(box.angle)
    dx, dy = pt[0] - box.center[0], pt[1] - box.center[1]
    lx = dx * math.cos(-ang) - dy * math.sin(-ang)
    ly = dx * math.sin(-ang) + dy * math.cos(-ang)
    return lx, ly


def grab_handle(pt: tuple[float, float], box: Box, tol: float) -> tuple[int, int] | None:
    """Which resize handle (if any) is near pt, within `tol` full-coord units.

    Returns (sx, sy) where each is -1/0/+1 naming the grabbed edge in the box's
    local frame: e.g. (+1, 0)=right edge, (0, -1)=top edge, (+1, +1)=BR corner.
    Returns None if pt isn't near the box border.
    """
    bw, bh = box.size
    lx, ly = _to_local(pt, box)
    hw, hh = bw / 2, bh / 2
    # outside the box (plus tolerance) entirely -> no handle
    if abs(lx) > hw + tol or abs(ly) > hh + tol:
        return None
    sx = 1 if abs(lx - hw) <= tol else (-1 if abs(lx + hw) <= tol else 0)
    sy = 1 if abs(ly - hh) <= tol else (-1 if abs(ly + hh) <= tol else 0)
    if sx == 0 and sy == 0:
        return None  # interior, not an edge -> caller treats as move
    return (sx, sy)


def resize_box(box: Box, handle: tuple[int, int], pt: tuple[float, float]) -> None:
    """Move the grabbed edge(s) to pt, anchoring the opposite edge(s).

    Mutates box.size and box.center in place. Works in the rotated local frame,
    so the anchored side stays put on screen even for skewed boxes.
    """
    sx, sy = handle
    bw, bh = box.size
    lx, ly = _to_local(pt, box)
    # local-frame center shift accumulated from each grabbed axis
    shift_lx = shift_ly = 0.0
    if sx != 0:
        anchor = -sx * bw / 2          # opposite edge, fixed
        new_half = abs(lx - anchor)
        bw = max(10.0, new_half)
        edge = anchor + sx * bw        # new grabbed-edge position
        shift_lx = (anchor + edge) / 2  # new local center x
    if sy != 0:
        anchor = -sy * bh / 2
        new_half = abs(ly - anchor)
        bh = max(10.0, new_half)
        edge = anchor + sy * bh
        shift_ly = (anchor + edge) / 2
    # rotate the local center shift back into full coords
    ang = math.radians(box.angle)
    box.center[0] += shift_lx * math.cos(ang) - shift_ly * math.sin(ang)
    box.center[1] += shift_lx * math.sin(ang) + shift_ly * math.cos(ang)
    box.size[0] = bw
    box.size[1] = bh


def _orient_arrow(box: Box) -> tuple[tuple[int, int], tuple[int, int]]:
    """Endpoints (full coords) of an arrow pointing to the box's 'top'."""
    cx, cy = box.center
    bw, bh = box.size
    # base 'up' is -y in box-local frame; orientation rotates which edge is top
    length = min(bw, bh) * 0.35
    base = {0: (0, -1), 90: (1, 0), 180: (0, 1), 270: (-1, 0)}[box.orientation % 360]
    ang = math.radians(box.angle)
    ux = base[0] * math.cos(ang) - base[1] * math.sin(ang)
    uy = base[0] * math.sin(ang) + base[1] * math.cos(ang)
    return (int(cx), int(cy)), (int(cx + ux * length), int(cy + uy * length))


def nudge_box(box: "Box", dx: float, dy: float) -> None:
    """Shift a box's center by (dx, dy) full-coord pixels, in place."""
    box.center[0] += dx
    box.center[1] += dy


def tilt_angle(angle: float, delta: float) -> float:
    """Adjust a deskew angle by delta, keeping in-range angles within [-45, 45].

    A normal box lives in [-45, 45]; we clamp to that band so tilting can't
    spin it. But a box LOADED from older metadata can sit outside the band
    (e.g. -85): hard-clamping such a value would snap it ~40 deg on the first
    keypress (the bug). So only clamp when the starting angle is already in
    range; an out-of-range angle just moves by delta (toward range or freely),
    and clamping resumes naturally once it crosses back into [-45, 45].
    """
    new = angle + delta
    if -45.0 <= angle <= 45.0:
        return max(-45.0, min(45.0, new))
    return new


def orientation_label(orientation: int) -> str:
    """Human-readable name for which scan edge becomes the output's top.

    Box.orientation names that edge (0=top, 90=right, 180=bottom, 270=left),
    matching the on-screen arrow and crop_box's rotation.
    """
    return {0: "top", 90: "right", 180: "bottom", 270: "left"}.get(
        orientation % 360, "top")


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


# Single source of truth for keyboard/mouse shortcuts: (keys, description).
# Used by the in-GUI "? Shortcuts" popover and the console help in main().
SHORTCUTS = [
    ("drag inside", "move the box under the cursor"),
    ("drag edge/corner", "resize the active box"),
    ("drag empty area", "draw a new box"),
    ("click sidebar row", "select that box"),
    ("← ↑ → ↓  /  h j k l", "nudge the active box"),
    ("n  /  Tab", "select the next box"),
    ("[  ]", "rotate orientation 90° (which edge is 'top')"),
    (",  .", "tilt ∓0.5° (fine deskew)"),
    ("<  >", "tilt ∓5° (coarse deskew)"),
    ("x  /  Delete", "delete the active box"),
    ("Enter", "crop all boxes, then go to the next scan"),
]


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


# ---------------------------------------------------------------------------
# Tkinter GUI — CanvasEditor (scan canvas with native box items)
# ---------------------------------------------------------------------------
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
        self.drag_current = None   # full-coord
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

        # Draw dynamic preview of the box currently being drawn
        if self.drag == "new" and self.drag_start is not None and self.drag_current is not None:
            x0, y0 = self.drag_start[0] * self.scale, self.drag_start[1] * self.scale
            x1, y1 = self.drag_current[0] * self.scale, self.drag_current[1] * self.scale
            c.create_rectangle(x0, y0, x1, y1, outline=state_color("editing"), width=2, tags="overlay")

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
        self.drag_current = (fx, fy)

    def _on_motion(self, event):
        if not self.drag:
            return
        fx, fy = self._full(event.x, event.y)
        b = self.active_box()
        if self.drag == "move" and b is not None:
            nudge_box(b, fx - self.drag_start[0], fy - self.drag_start[1])
            self.drag_start = (fx, fy)
        elif isinstance(self.drag, tuple) and self.drag[0] == "resize" and b is not None:
            resize_box(b, self.drag[1], (fx, fy))
        elif self.drag == "new":
            self.drag_current = (fx, fy)
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
        self.drag_current = None
        self.redraw()
        self.on_change()

    def _renumber(self):
        for i, b in enumerate(self.boxes, 1):
            b.id = i


# ---------------------------------------------------------------------------
# Tkinter GUI — SplitterApp (root, header, sidebar, action bar)
# ---------------------------------------------------------------------------
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
        self._shortcuts_win = None

        self.root = tk.Tk()
        self.root.title("Photo Scan Splitter")
        self.root.geometry("1180x820")
        self.root.minsize(900, 600)  # keep the bottom action bar reachable
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

        # action bar — packed against the bottom FIRST so it always reserves
        # its space and is never pushed off-screen by a tall sidebar/preview.
        bar = ttk.Frame(self.root)
        bar.pack(side="bottom", fill="x", padx=16, pady=(4, 12))
        ttk.Button(bar, text="+ Re-detect", command=self._redetect).pack(side="left")
        ttk.Button(bar, text="Crop all", command=self._crop_all).pack(side="left", padx=(8, 0))
        ttk.Button(bar, text="? Shortcuts", command=self._show_shortcuts).pack(
            side="left", padx=(8, 0))
        self.next_btn = ttk.Button(bar, text="Next →", style="Primary.TButton",
                                   command=self._next)
        self.next_btn.pack(side="right")
        ttk.Button(bar, text="← Prev", command=self._prev).pack(side="right", padx=(0, 8))

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
        info = ttk.Frame(side); info.pack(fill="x", pady=(2, 6))
        self.orient_var = tk.StringVar(value="top")
        ttk.Label(info, text="top edge:").pack(side="left")
        ttk.Label(info, textvariable=self.orient_var, width=8,
                  anchor="w").pack(side="left")

        self.preview_label = ttk.Label(side)
        self.preview_label.pack(pady=(4, 8))
        ttk.Button(side, text="↻ Rotate", command=lambda: self._orient(90)).pack(
            fill="x", pady=(0, 4))
        ttk.Button(side, text="Delete box", command=self._delete).pack(fill="x")

        # keyboard (mirrors the old editor). bind_all puts the handler in the
        # global "all" bindtag so shortcuts fire no matter which button/widget
        # currently holds focus (root-only bind dies the moment a button is
        # clicked — its class bindings shadow the key). Same approach the
        # face_pipeline GUIs use for app-global keys.
        self.root.bind_all("<Key>", self._on_key)
        # Tab is consumed by Tk's focus-traversal binding on the widget's CLASS
        # tag, which runs before the "all" tag — so bind_all("<Tab>") never
        # fires. Binding on the canvas INSTANCE tag (processed before the class
        # tag) wins; the canvas is the focus owner after each scan load.
        # ISO_Left_Tab is Shift+Tab on X11/some macOS layouts.
        self.editor.canvas.bind("<Tab>", self._on_tab)
        self.editor.canvas.bind("<ISO_Left_Tab>", self._on_tab)

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
        # ensure the window owns keyboard focus so shortcuts work before any
        # button is clicked (clicks would otherwise be the first focus event)
        self.editor.canvas.focus_set()

    def _refresh_header(self):
        self.title_var.set(os.path.basename(self.scan_path))
        cropped = sum(1 for b in self.boxes if b.output)
        self.sub_var.set(f"{len(self.boxes)} photos · {cropped} cropped")
        self.progress.configure(value=self.idx + 1)

    def _refresh_sidebar(self):
        for w in self.rows.winfo_children():
            w.destroy()
        self._row_thumbs = []  # keep PhotoImage refs alive (Tk GCs otherwise)
        for i, b in enumerate(self.boxes):
            active = (i == self.editor.active)
            color = state_color(box_state(b, self.image.shape[:2], active=active))
            rowbg = ACCENT if active else BG
            row = self.tk.Frame(self.rows, bg=rowbg)
            row.pack(fill="x", pady=1)
            dot = self.tk.Label(row, text="●", fg=color, bg=rowbg)
            dot.pack(side="left", padx=(4, 4))
            try:
                thumb = crop_to_round_photo(crop_box(self.image, b), cell=40,
                                            radius=6)
            except ValueError:
                thumb = crop_to_round_photo(
                    np.full((10, 10, 3), 200, np.uint8), cell=40, radius=6)
            self._row_thumbs.append(thumb)
            tlbl = self.tk.Label(row, image=thumb, bg=rowbg)
            tlbl.pack(side="left", padx=(0, 6))
            txt = f"{i+1}   {int(b.size[0])}×{int(b.size[1])}"
            lbl = self.tk.Label(row, text=txt, anchor="w",
                                fg="#ffffff" if active else "#1a1a2e", bg=rowbg)
            lbl.pack(side="left", fill="x", expand=True)
            for wdg in (row, dot, tlbl, lbl):
                wdg.bind("<Button-1>", lambda e, idx=i: self._select(idx))
        self._refresh_active_panel()
        self._refresh_header()

    def _refresh_active_panel(self):
        b = self.editor.active_box()
        if b is None:
            self.orient_var.set("—")
            self.preview_label.configure(image="")
            self._preview_photo = None
            return
        self.orient_var.set(orientation_label(b.orientation))
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
        # Skew adjust is held down / repeated — keep it cheap. Redraw ONLY the
        # canvas (the active frame); skip the sidebar rebuild, which re-crops a
        # thumbnail per box and re-renders the preview and made tilt lag. The
        # thumbnail/preview refresh on the next select/orient/delete/scan.
        b = self.editor.active_box()
        if b is not None:
            b.angle = tilt_angle(b.angle, delta)
            self.editor.redraw()

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
        k = event.keysym
        # bind_all sees keys destined for focused buttons too; let a focused
        # button handle its own activation keys (avoids double-firing Return).
        if k in ("Return", "space") and isinstance(event.widget, self.ttk.Button):
            return
        # Punctuation keys: match on the typed CHARACTER, not the keysym name.
        # This Tk reports event.keysym as "]" (not the X11 name "bracketright")
        # for those keys, so name matching silently never fired.
        ch = event.char
        step = 20.0
        if k == "n":
            self._next_box()
        elif k in ("Left", "h"): self._move(-step, 0)
        elif k in ("Right", "l"): self._move(step, 0)
        elif k in ("Up", "k"): self._move(0, -step)
        elif k in ("Down", "j"): self._move(0, step)
        elif ch == "[": self._orient(-90)
        elif ch == "]": self._orient(90)
        elif ch == ",": self._nudge(-0.5)
        elif ch == ".": self._nudge(0.5)
        elif ch == "<": self._nudge(-5.0)
        elif ch == ">": self._nudge(5.0)
        elif k in ("x", "Delete", "BackSpace"): self._delete()
        elif k == "Return": self._crop_all(); self._next()
        elif ch == "?" or k == "F1": self._show_shortcuts()

    def _next_box(self):
        if self.boxes:
            self.editor.active = (self.editor.active + 1) % len(self.boxes)
            self.editor.redraw()
            self._refresh_sidebar()

    def _on_tab(self, event):
        # Tab is consumed by Tk's focus-traversal class binding before the
        # bind_all("<Key>") handler runs, so it never reaches _on_key. Bind it
        # explicitly and return "break" to suppress traversal and cycle boxes.
        self._next_box()
        return "break"

    def _move(self, dx, dy):
        b = self.editor.active_box()
        if b is not None:
            nudge_box(b, dx, dy)
            self.editor.redraw(); self._refresh_sidebar()

    # ---- shortcuts popover ----
    def _show_shortcuts(self):
        if getattr(self, "_shortcuts_win", None) is not None:
            self._close_shortcuts()
            return
        tk, ttk = self.tk, self.ttk
        top = tk.Toplevel(self.root)
        top.title("Keyboard & mouse shortcuts")
        top.configure(bg=BG)
        top.transient(self.root)
        top.resizable(False, False)
        self._shortcuts_win = top
        top.protocol("WM_DELETE_WINDOW", self._close_shortcuts)
        top.bind("<Escape>", lambda e: self._close_shortcuts())

        ttk.Label(top, text="Shortcuts", style="Title.TLabel").grid(
            row=0, column=0, columnspan=2, sticky="w", padx=16, pady=(14, 8))
        for i, (keys, desc) in enumerate(SHORTCUTS, start=1):
            key_lbl = tk.Label(top, text=keys, bg=BG, fg=ACCENT,
                               font=("TkFixedFont", 10, "bold"), anchor="e")
            key_lbl.grid(row=i, column=0, sticky="e", padx=(16, 10), pady=2)
            tk.Label(top, text=desc, bg=BG, fg="#1a1a2e", anchor="w").grid(
                row=i, column=1, sticky="w", padx=(0, 16), pady=2)
        ttk.Button(top, text="Close", command=self._close_shortcuts).grid(
            row=len(SHORTCUTS) + 1, column=0, columnspan=2, pady=(10, 14))

    def _close_shortcuts(self):
        win = getattr(self, "_shortcuts_win", None)
        if win is not None:
            try:
                win.destroy()
            except self.tk.TclError:
                pass
            self._shortcuts_win = None

    def _show(self):
        self._load_scan()

    def run(self):
        self._show()
        self.root.mainloop()


# ---------------------------------------------------------------------------
# Main loop — iterate scans
# ---------------------------------------------------------------------------
def list_scans(images_dir: str) -> list[str]:
    files = []
    for ext in ("*.jpg", "*.jpeg", "*.png", "*.tif", "*.tiff"):
        files.extend(glob.glob(os.path.join(images_dir, "**", ext), recursive=True))
    return sorted(files)


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

    print("Controls (also in the GUI's '? Shortcuts' button):")
    for keys, desc in SHORTCUTS:
        print(f"  {keys:<22} {desc}")

    try:
        app = SplitterApp(scans, out_dir)
    except tk.TclError:
        print("Cannot open a GUI window. Run via the venv Python "
              "(Homebrew Tk): .venv/bin/python split_photos.py")
        return 1
    app.run()
    print("Done.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
