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


def orientation_name(orient: int) -> str:
    """Map orientation degrees to friendly direction names."""
    m = {0: "Top", 90: "Right", 180: "Bottom", 270: "Left"}
    return m.get(int(orient) % 360, f"{orient}°")


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


def scale_base(image: np.ndarray, scale: float) -> np.ndarray:
    """The expensive full-res resize, done once per scale change (not per frame)."""
    return cv2.resize(image, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)


def render(base: np.ndarray, boxes: list[Box], active_idx: int, scale: float) -> np.ndarray:
    """Draw box overlays onto a copy of the pre-scaled base image."""
    disp = base.copy()
    for i, b in enumerate(boxes):
        rect = ((b.center[0] * scale, b.center[1] * scale),
                (b.size[0] * scale, b.size[1] * scale), b.angle)
        pts = cv2.boxPoints(rect).astype(np.int32)
        color = (0, 255, 0) if i == active_idx else (0, 255, 255)
        cv2.polylines(disp, [pts], True, color, 2)
        if i == active_idx:
            p0, p1 = _orient_arrow(b)
            cv2.arrowedLine(disp, (int(p0[0] * scale), int(p0[1] * scale)),
                            (int(p1[0] * scale), int(p1[1] * scale)),
                            (0, 0, 255), 2, tipLength=0.3)
    if boxes and 0 <= active_idx < len(boxes):
        b = boxes[active_idx]
        txt = f"[{active_idx+1}/{len(boxes)}] angle={b.angle:.1f} orient={b.orientation}"
    else:
        txt = f"[0/{len(boxes)}] no active box"
    # Status banner above the image (separate strip, so it never hides content).
    banner = np.zeros((BANNER_H, disp.shape[1], 3), np.uint8)
    cv2.putText(banner, txt, (6, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1)
    return cv2.vconcat([banner, disp])


# ---------------------------------------------------------------------------
# Editor (interactive HighGUI)
# ---------------------------------------------------------------------------
# ---- shared GUI theme (ttk) ----
ACCENT = "#5a6cf0"
BG = "#fafaff"
CARD_BORDER = "#ececf2"
STATE_COLORS = {
    "active": "#2faf6a",      # active box color
    "inactive": "#5a6cf0",    # inactive box color
}

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

def crop_to_round_photo(crop, cell=64, radius=8):
    """BGR crop -> letterboxed cell x cell rounded-corner Tk PhotoImage."""
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


WINDOW = "Photo Scan Splitter"
HANDLE_R = 12  # full-coord radius for corner-handle hit test (scaled at use)
BANNER_H = 26  # status banner height in px, drawn above the image



class Editor:
    """Interactive HighGUI editor for one scan."""

    def __init__(self, image: np.ndarray, boxes: list[Box], scan_path: str, out_dir: str, scan_idx: int = 0, total_scans: int = 1):
        import tkinter as tk
        self.tk = tk
        self.image = image
        self.boxes = boxes
        self.scan_path = scan_path
        self.out_dir = out_dir
        self.scan_idx = scan_idx
        self.total_scans = total_scans
        self.active = 0 if boxes else -1
        
        # Fit scan image to fixed pane
        self.PHOTO_W, self.PHOTO_H = 760, 680
        h, w = image.shape[:2]
        self.scale = min(self.PHOTO_W / w, self.PHOTO_H / h, 1.0)
        
        self.drag = None          # None | 'move' | 'new' | ('resize', handle)
        self.drag_start = None    # full-coord
        self.next_request = None  # 'next' | 'prev' | 'quit'
        self._preview_open = False
        self._base = scale_base(image, self.scale)
        self._dirty = True
        self._cells = []          # crop thumbnails PhotoImage references
        self._resize_job = None
        
        # Build Window
        from tkinter import ttk
        self.ttk = ttk
        self.root = tk.Tk()
        self.root.title("Photo Splitter")
        self.root.geometry("1120x780")
        self.root.resizable(True, True)
        self.root.minsize(1120, 780)
        _install_theme(self.root)

        # Top Progress Header
        head = ttk.Frame(self.root)
        head.pack(side="top", fill="x", padx=16, pady=(10, 4))
        self.title_var = tk.StringVar()
        ttk.Label(head, textvariable=self.title_var, style="Title.TLabel").pack(anchor="w")
        
        # We'll set the progress value dynamically in _show()
        self.progress = ttk.Progressbar(head, cursor="hand2")
        self.progress.pack(fill="x", pady=(6, 0))
        self.progress.bind("<Button-1>", self._on_progress_click)

        # Bottom Navigation
        bar = ttk.Frame(self.root)
        bar.pack(side="bottom", fill="x", padx=16, pady=10)
        
        left_btns = ttk.Frame(bar)
        left_btns.pack(side="left")
        ttk.Button(left_btns, text="← Back", command=self._back).pack(side="left", padx=4)
        ttk.Button(left_btns, text="Save & Next →", style="Primary.TButton", command=self._next).pack(side="left", padx=4)
        
        ttk.Button(bar, text="Keyboard Shortcuts", command=self._show_shortcuts).pack(side="right", padx=4)

        # Main Split Body
        body = ttk.Frame(self.root)
        body.pack(side="top", fill="both", expand=True, padx=16, pady=4)

        # Left View Pane
        self.photo_pane = ttk.Frame(body)
        self.photo_pane.pack(side="left", fill="both", expand=True)
        self.photo_pane.pack_propagate(False)
        self.photo_pane.bind("<Configure>", self._on_pane_configure)
        self.canvas = tk.Label(self.photo_pane, bg=BG)
        self.canvas.pack(expand=True)
        
        # Bind mouse events to the image viewer
        self.canvas.bind("<Button-1>", self.on_mouse_down)
        self.canvas.bind("<B1-Motion>", self.on_mouse_drag)
        self.canvas.bind("<ButtonRelease-1>", self.on_mouse_up)

        # Right Scrollable Card Sidebar
        right = ttk.Frame(body, width=320)
        right.pack(side="right", fill="y", padx=(8, 0))
        right.pack_propagate(False)
        
        self.rows_canvas = tk.Canvas(right, highlightthickness=0, bg=BG)
        self._vbar = self.ttk.Scrollbar(right, orient="vertical", command=self.rows_canvas.yview)
        self.rows_canvas.configure(yscrollcommand=self._vbar.set)
        self.rows_canvas.pack(side="left", fill="both", expand=True)
        
        self.rows_frame = tk.Frame(self.rows_canvas, bg=BG)
        self._rows_window = self.rows_canvas.create_window((0, 0), window=self.rows_frame, anchor="nw")
        
        self.rows_frame.bind("<Configure>", lambda e: (
            self.rows_canvas.configure(scrollregion=self.rows_canvas.bbox("all")),
            self._sync_scrollbar()
        ))
        self.rows_canvas.bind("<Configure>", lambda e: (
            self.rows_canvas.itemconfigure(self._rows_window, width=e.width),
            self._sync_scrollbar()
        ))
        self.rows_canvas.bind_all("<MouseWheel>", lambda e: self.rows_canvas.yview_scroll(
            int(-1 * (e.delta / 120)), "units"
        ))

        # Bind Window Keys
        self.root.bind("<Tab>", lambda e: (self._cycle_active(), "break"))
        self.root.bind("<Key-n>", lambda e: self._cycle_active())
        self.root.bind("<Left>", lambda e: self._move_active_kbd(-20, 0))
        self.root.bind("<Right>", lambda e: self._move_active_kbd(20, 0))
        self.root.bind("<Up>", lambda e: self._move_active_kbd(0, -20))
        self.root.bind("<Down>", lambda e: self._move_active_kbd(0, 20))
        self.root.bind("<Key-h>", lambda e: self._move_active_kbd(-20, 0))
        self.root.bind("<Key-l>", lambda e: self._move_active_kbd(20, 0))
        self.root.bind("<Key-k>", lambda e: self._move_active_kbd(0, -20))
        self.root.bind("<Key-j>", lambda e: self._move_active_kbd(0, 20))
        self.root.bind("<Key-bracketleft>", lambda e: self._rotate_active(-90))
        self.root.bind("<Key-bracketright>", lambda e: self._rotate_active(90))
        self.root.bind("<Key-comma>", lambda e: self._nudge_active(-0.5))
        self.root.bind("<Key-period>", lambda e: self._nudge_active(0.5))
        self.root.bind("<Key-less>", lambda e: self._nudge_active(-5.0))
        self.root.bind("<Key-greater>", lambda e: self._nudge_active(5.0))
        self.root.bind("<Key-x>", lambda e: self._delete_active())
        self.root.bind("<BackSpace>", lambda e: self._delete_active())
        self.root.bind("<Delete>", lambda e: self._delete_active())
        self.root.bind("<Return>", lambda e: self._next())
        self.root.bind("<Key-equal>", lambda e: self._next_no_crop())
        self.root.bind("<Key-minus>", lambda e: self._prev_no_crop())
        self.root.bind("<Key-s>", lambda e: self.save())
        self.root.bind("<Key-c>", lambda e: self.crop_all())
        self.root.bind("<Key-q>", lambda e: self._quit())
        
        self.root.protocol("WM_DELETE_WINDOW", self._on_close)

    # ---- coordinate helpers ----
    def _full(self, x, y):
        # We don't have the BANNER_H status banner offset on the canvas image anymore!
        return disp_to_full((x, y), self.scale)

    def _full_tk(self, event):
        """Translate tk.Label local event coords to full scan coords."""
        return event.x / self.scale, event.y / self.scale

    def _active_box(self):
        if 0 <= self.active < len(self.boxes):
            return self.boxes[self.active]
        return None

    # ---- mouse ----
    def on_mouse_down(self, event):
        fx, fy = self._full_tk(event)
        self._dirty = True
        
        # 1) Edge or corner handle of active box?
        b = self._active_box()
        if b is not None:
            handle = grab_handle((fx, fy), b, HANDLE_R / self.scale)
            if handle is not None:
                self.drag = ("resize", handle)
                self.drag_start = (fx, fy)
                return
        
        # 2) Click inside an existing box -> select + move
        for i, bx in enumerate(self.boxes):
            if point_in_box((fx, fy), bx):
                self.active = i
                self.drag = "move"
                self.drag_start = (fx, fy)
                self._show()  # refresh cards to highlight selection
                return
        
        # 3) Empty area -> start new box
        self.drag = "new"
        self.drag_start = (fx, fy)

    def on_mouse_drag(self, event):
        if not self.drag:
            return
        fx, fy = self._full_tk(event)
        b = self._active_box()
        if self.drag == "move" and b is not None:
            dx = fx - self.drag_start[0]
            dy = fy - self.drag_start[1]
            b.center[0] += dx
            b.center[1] += dy
            self.drag_start = (fx, fy)
        elif isinstance(self.drag, tuple) and self.drag[0] == "resize" and b is not None:
            resize_box(b, self.drag[1], (fx, fy))
        self._dirty = True
        self.root.after_idle(self._draw_overlay)

    def on_mouse_up(self, event):
        if not self.drag:
            return
        fx, fy = self._full_tk(event)
        if self.drag == "new":
            x0, y0 = self.drag_start
            w = abs(fx - x0)
            h = abs(fy - y0)
            if w > 10 and h > 10:
                box = Box(center=[(fx + x0) / 2, (fy + y0) / 2], size=[w, h])
                self.boxes.append(box)
                self._renumber()
                self.active = len(self.boxes) - 1
        self.drag = None
        self.drag_start = None
        self._dirty = True
        self._show()  # rebuilds sidebar cards with new thumbnails

    def _renumber(self):
        for i, b in enumerate(self.boxes, 1):
            b.id = i

    def _cycle_active(self):
        if self.boxes:
            self.active = (self.active + 1) % len(self.boxes)
            self._dirty = True
            self._show()

    def _move_active_kbd(self, dx, dy):
        b = self._active_box()
        if b is not None:
            b.center[0] += dx
            b.center[1] += dy
            self._dirty = True
            self._show()

    def _rotate_active(self, delta_deg):
        b = self._active_box()
        if b is not None:
            b.orientation = (b.orientation + delta_deg) % 360
            self._dirty = True
            self._show()

    def _nudge_active(self, delta):
        b = self._active_box()
        if b is not None:
            self._nudge_angle(b, delta)
            self._dirty = True
            self._show()

    def _delete_active(self):
        if 0 <= self.active < len(self.boxes):
            del self.boxes[self.active]
            self._renumber()
            self.active = min(self.active, len(self.boxes) - 1)
            self._dirty = True
            self._show()

    def _sync_scrollbar(self):
        content = self.rows_frame.winfo_reqheight()
        visible = self.rows_canvas.winfo_height()
        if content > visible:
            if not self._vbar.winfo_ismapped():
                self._vbar.pack(side="right", fill="y", before=self.rows_canvas)
        else:
            if self._vbar.winfo_ismapped():
                self._vbar.pack_forget()
            self.rows_canvas.yview_moveto(0)

    def _build_sidebar_cards(self):
        for child in self.rows_frame.winfo_children():
            child.destroy()
        self._cells = []
        
        for idx, box in enumerate(self.boxes):
            is_active = (idx == self.active)
            bg_color = "#f0f0f8" if is_active else BG
            border_color = ACCENT if is_active else CARD_BORDER
            
            # Card Outer frame
            card = self.ttk.Frame(self.rows_frame, padding=8)
            card.pack(fill="x", padx=6, pady=4)
            
            # Rounded thumbnail
            try:
                crop = crop_box(self.image, box)
            except ValueError:
                crop = np.full((64, 64, 3), 200, dtype=np.uint8)
            
            thumb = crop_to_round_photo(crop, cell=64, radius=8)
            self._cells.append(thumb) # keep reference alive
            
            # Card Layout Grid
            lbl_thumb = self.tk.Label(card, image=thumb, bg=bg_color)
            lbl_thumb.grid(row=0, column=0, rowspan=2, padx=(0, 8))
            
            # Click card selection bind
            lbl_thumb.bind("<Button-1>", lambda e, i=idx: self._select_card(i))
            
            badge_color = STATE_COLORS["active"] if is_active else STATE_COLORS["inactive"]
            lbl_badge = self.tk.Label(card, text=f" #{box.id} ", bg=badge_color, fg="#ffffff", font=("TkDefaultFont", 10, "bold"))
            lbl_badge.grid(row=0, column=1, sticky="w")
            lbl_badge.bind("<Button-1>", lambda e, i=idx: self._select_card(i))
            
            desc = f"{int(round(box.size[0]))}x{int(round(box.size[1]))} px\nAngle: {box.angle:.1f}° | Orient: {orientation_name(box.orientation)}"
            lbl_desc = self.ttk.Label(card, text=desc, font=("TkDefaultFont", 9))
            lbl_desc.grid(row=1, column=1, sticky="w", pady=(2, 0))
            lbl_desc.bind("<Button-1>", lambda e, i=idx: self._select_card(i))
            
            # Card Action Buttons
            btn_frame = self.ttk.Frame(card)
            btn_frame.grid(row=2, column=0, columnspan=2, pady=(6, 0), sticky="e")
            
            btn_rot = self.ttk.Button(btn_frame, text="↺ Rotate", command=lambda i=idx: self._rotate_card(i))
            btn_rot.pack(side="left", padx=2)
            
            btn_del = self.ttk.Button(btn_frame, text="Delete", command=lambda i=idx: self._delete_card(i))
            btn_del.pack(side="left", padx=2)
            
            # Background visual overrides
            card.configure(style="Card.TFrame")
            # Let's bind main frame clicks
            card.bind("<Button-1>", lambda e, i=idx: self._select_card(i))

    def _select_card(self, idx):
        self.active = idx
        self._dirty = True
        self._show()

    def _rotate_card(self, idx):
        self.boxes[idx].orientation = (self.boxes[idx].orientation - 90) % 360
        self._dirty = True
        self._show()

    def _delete_card(self, idx):
        del self.boxes[idx]
        self._renumber()
        self.active = min(idx, len(self.boxes) - 1)
        self._dirty = True
        self._show()

    def _scroll_active_into_view(self):
        if not self.boxes or self.active < 0:
            return
        total = len(self.boxes)
        # Estimate vertical position faction
        frac = self.active / total
        self.rows_canvas.yview_moveto(max(0.0, frac - 0.2))


    def _draw_overlay(self):
        """Draw OpenCV bounding boxes on top of the pre-scaled scan base."""
        if not self._dirty:
            return
        import cv2
        from PIL import Image, ImageTk
        
        disp = self._base.copy()
        for i, b in enumerate(self.boxes):
            rect = ((b.center[0] * self.scale, b.center[1] * self.scale),
                    (b.size[0] * self.scale, b.size[1] * self.scale), b.angle)
            pts = cv2.boxPoints(rect).astype(np.int32)
            color = (47, 175, 106) if i == self.active else (240, 108, 90) # BGR: active=green, inactive=blueish
            cv2.polylines(disp, [pts], True, color, 2)
            
            if i == self.active:
                p0, p1 = _orient_arrow(b)
                cv2.arrowedLine(disp, (int(p0[0] * self.scale), int(p0[1] * self.scale)),
                                (int(p1[0] * self.scale), int(p1[1] * self.scale)),
                                (0, 0, 255), 2, tipLength=0.3)
                                
        rgb = cv2.cvtColor(disp, cv2.COLOR_BGR2RGB)
        self._photo_img = ImageTk.PhotoImage(Image.fromarray(rgb))
        self.canvas.configure(image=self._photo_img)
        self._dirty = False

    def _show(self):
        scan_name = os.path.basename(self.scan_path)
        self.title_var.set(f"Scan: {scan_name} ({self.scan_idx + 1} of {self.total_scans}) — {len(self.boxes)} photo(s) detected")
        self.progress.configure(maximum=self.total_scans, value=self.scan_idx)
        self._build_sidebar_cards()
        self._draw_overlay()
        self._scroll_active_into_view()

    def _cancel_resize_job(self):
        if self._resize_job is not None:
            try:
                self.root.after_cancel(self._resize_job)
            except Exception:
                pass
            self._resize_job = None

    def _on_progress_click(self, event):
        width = self.progress.winfo_width()
        if width <= 0 or self.total_scans == 0:
            return
        frac = max(0.0, min(1.0, event.x / width))
        target = min(self.total_scans - 1, int(frac * self.total_scans))
        if target == self.scan_idx:
            return
        self._cancel_resize_job()
        self.save()
        self.next_request = f"jump_{target}"
        self.root.destroy()

    def _next(self):
        self._cancel_resize_job()
        self.crop_all()
        self.next_request = "next"
        self.root.destroy()

    def _next_no_crop(self):
        self._cancel_resize_job()
        self.save()
        self.next_request = "next"
        self.root.destroy()

    def _prev_no_crop(self):
        self._cancel_resize_job()
        self.save()
        self.next_request = "prev"
        self.root.destroy()

    def _back(self):
        self._prev_no_crop()

    def _quit(self):
        self._cancel_resize_job()
        self.save()
        self.next_request = "quit"
        self.root.destroy()

    def _on_close(self):
        self._quit()

    def _close_preview(self):
        pass

    def crop_all(self):
        os.makedirs(self.out_dir, exist_ok=True)
        stem = os.path.splitext(os.path.basename(self.scan_path))[0]
        saved = 0
        for i, b in enumerate(self.boxes, 1):
            try:
                out = crop_box(self.image, b)
            except ValueError:
                print(f"  ! skipping box {i}: zero size")
                continue
            name = f"{stem}_{i:02d}.jpg"
            b.output = name
            cv2.imwrite(os.path.join(self.out_dir, name), out)
            saved += 1
        self.save()
        print(f"  cropped {saved} photo(s) to {self.out_dir}/")

    def _on_pane_configure(self, event):
        """Handle viewer frame resize dynamically updating image fit."""
        if event.widget != self.photo_pane:
            return
        w, h = event.width, event.height
        if w < 100 or h < 100:
            return
        # Leave a small margin for padding
        target_w = w - 8
        target_h = h - 8
        if abs(target_w - self.PHOTO_W) > 5 or abs(target_h - self.PHOTO_H) > 5:
            if self._resize_job is not None:
                try:
                    self.root.after_cancel(self._resize_job)
                except Exception:
                    pass
            self._resize_job = self.root.after(100, self._do_resize, w, h)

    def _do_resize(self, w, h):
        self._resize_job = None
        try:
            if not self.root.winfo_exists():
                return
        except Exception:
            return
        target_w = w - 8
        target_h = h - 8
        self.PHOTO_W = target_w
        self.PHOTO_H = target_h
        h_img, w_img = self.image.shape[:2]
        self.scale = min(self.PHOTO_W / w_img, self.PHOTO_H / h_img, 1.0)
        self._base = scale_base(self.image, self.scale)
        self._dirty = True
        self._draw_overlay()

    def _show_shortcuts(self):
        """Display a clean Toplevel modal window listing all keyboard shortcuts."""
        import tkinter as tk
        from tkinter import ttk
        top = tk.Toplevel(self.root)
        top.title("Keyboard Shortcuts")
        top.geometry("400x420")
        top.resizable(False, False)
        top.transient(self.root)
        top.grab_set()
        
        # Apply style
        _install_theme(top)
        
        frame = ttk.Frame(top, padding=16)
        frame.pack(fill="both", expand=True)
        
        ttk.Label(frame, text="Keyboard Shortcuts", style="Title.TLabel").pack(anchor="w", pady=(0, 12))
        
        # Shortcuts grid
        grid = ttk.Frame(frame)
        grid.pack(fill="both", expand=True)
        
        shortcuts = [
            ("Tab / n", "Cycle active box selection"),
            ("Arrows / hjkl", "Nudge active box position"),
            ("[ / ]", "Rotate box orientation (90° CCW / CW)"),
            (", / .", "Fine tilt angle (-0.5° / +0.5°)"),
            ("< / >", "Coarse tilt angle (-5.0° / +5.0°)"),
            ("x / Del / Backspace", "Delete active box"),
            ("s", "Save metadata sidecar"),
            ("c", "Crop all photos to disk"),
            ("Enter / Return", "Crop all, save, and go to next scan"),
            ("= / -", "Next / Previous scan (no crop)"),
            ("q", "Save metadata and quit"),
        ]
        
        for idx, (key, desc) in enumerate(shortcuts):
            k_lbl = ttk.Label(grid, text=key, font=("TkDefaultFont", 10, "bold"), foreground=ACCENT)
            k_lbl.grid(row=idx, column=0, sticky="w", pady=4, padx=(0, 16))
            d_lbl = ttk.Label(grid, text=desc, font=("TkDefaultFont", 10))
            d_lbl.grid(row=idx, column=1, sticky="w", pady=4)
            
        ttk.Button(frame, text="Close", command=top.destroy).pack(pady=(12, 0))

    def save(self):
        h, w = self.image.shape[:2]
        path = save_metadata(self.scan_path, (w, h), self.boxes)
        print(f"  saved metadata -> {path}")

    def run(self) -> str:
        """Show the Tkinter root loop."""
        self._show()
        self.root.mainloop()
        return self.next_request or "next"

    def _nudge_angle(self, box: Box, delta: float):
        """Adjust tilt by delta degrees, clamped to the deskew range."""
        box.angle = max(-45.0, min(45.0, box.angle + delta))


# ---------------------------------------------------------------------------
# Main loop — iterate scans
# ---------------------------------------------------------------------------
def list_scans(images_dir: str) -> list[str]:
    files = []
    for ext in ("*.jpg", "*.jpeg", "*.png", "*.tif", "*.tiff"):
        files.extend(glob.glob(os.path.join(images_dir, ext)))
    return sorted(files)


def main(argv: list[str]) -> int:
    images_dir = argv[1] if len(argv) > 1 else "images"
    out_dir = "extracted"
    if not os.path.isdir(images_dir):
        print(f"No such images directory: {images_dir}")
        return 1
    scans = list_scans(images_dir)
    if not scans:
        print(f"No scans found in {images_dir}")
        return 1

    idx = 0
    while 0 <= idx < len(scans):
        path = scans[idx]
        image = cv2.imread(path)
        if image is None:
            print(f"  ! cannot read {path}, skipping")
            idx += 1
            continue
        boxes = load_metadata(path)
        if boxes is None:
            boxes = detect_photos(image)
            print(f"{os.path.basename(path)}: auto-detected {len(boxes)} photo(s)")
        else:
            print(f"{os.path.basename(path)}: loaded {len(boxes)} box(es) from metadata")

        editor = Editor(image, boxes, path, out_dir, idx, len(scans))
        req = editor.run()
        if req == "quit":
            break
        elif req == "prev":
            idx = max(0, idx - 1)
        elif req.startswith("jump_"):
            idx = int(req.split("_")[1])
        else:  # "next"
            idx += 1

    print("Done.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
