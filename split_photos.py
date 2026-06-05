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

    def __init__(self, image: np.ndarray, boxes: list[Box], scan_path: str, out_dir: str):
        self.image = image
        self.boxes = boxes
        self.scan_path = scan_path
        self.out_dir = out_dir
        self.active = 0 if boxes else -1
        h, w = image.shape[:2]
        # fit longest side to ~1000 px by default
        self.scale = min(1000 / w, 1000 / h, 1.0)
        self.drag = None          # None | 'move' | 'new' | ('resize', handle)
        self.drag_start = None    # full-coord
        self.next_request = None  # set to 'next' or 'quit' to end loop
        self._preview_open = False
        self._base = scale_base(image, self.scale)  # cached scaled image
        self._dirty = True        # redraw only when state changed

    # ---- coordinate helpers ----
    def _full(self, x, y):
        # the status banner shifts the image down by BANNER_H px on screen
        return disp_to_full((x, y - BANNER_H), self.scale)

    def _active_box(self):
        if 0 <= self.active < len(self.boxes):
            return self.boxes[self.active]
        return None

    # ---- mouse ----
    def on_mouse(self, event, x, y, flags, param):
        fx, fy = self._full(x, y)
        if event == cv2.EVENT_LBUTTONDOWN:
            self._dirty = True
            # 1) edge/corner handle of active box? (resize)
            b = self._active_box()
            if b is not None:
                handle = grab_handle((fx, fy), b, HANDLE_R / self.scale)
                if handle is not None:
                    self.drag = ("resize", handle)
                    self.drag_start = (fx, fy)
                    return
            # 2) click inside an existing box -> select + move
            for i, bx in enumerate(self.boxes):
                if point_in_box((fx, fy), bx):
                    self.active = i
                    self.drag = "move"
                    self.drag_start = (fx, fy)
                    return
            # 3) empty area -> start new box
            self.drag = "new"
            self.drag_start = (fx, fy)
        elif event == cv2.EVENT_MOUSEMOVE and self.drag:
            self._handle_drag(fx, fy)
            self._dirty = True
        elif event == cv2.EVENT_LBUTTONUP and self.drag:
            self._finish_drag(fx, fy)
            self._dirty = True

    def _handle_drag(self, fx, fy):
        b = self._active_box()
        if self.drag == "move" and b is not None:
            dx = fx - self.drag_start[0]
            dy = fy - self.drag_start[1]
            b.center[0] += dx
            b.center[1] += dy
            self.drag_start = (fx, fy)
        elif isinstance(self.drag, tuple) and self.drag[0] == "resize" and b is not None:
            resize_box(b, self.drag[1], (fx, fy))

    def _finish_drag(self, fx, fy):
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

    def _renumber(self):
        for i, b in enumerate(self.boxes, 1):
            b.id = i

    # Arrow keycodes vary by OS/backend; cover the common ones plus an
    # ASCII fallback (h/j/k/l) that works everywhere.
    ARROW_LEFT = (81, 0x250000, 63234)
    ARROW_UP = (82, 0x260000, 63232)
    ARROW_RIGHT = (83, 0x270000, 63235)
    ARROW_DOWN = (84, 0x280000, 63233)

    def _move_active(self, dx: float, dy: float):
        b = self._active_box()
        if b is not None:
            b.center[0] += dx
            b.center[1] += dy

    # ---- keyboard ----
    def on_key(self, key):
        """key is a full waitKeyEx code (not masked), or -1 for no key."""
        b = self._active_box()
        step = 20.0  # full-coord pixels per arrow press
        if key in (ord("n"), ord("N"), 9):  # n or Tab
            if self.boxes:
                self.active = (self.active + 1) % len(self.boxes)
        elif key in self.ARROW_LEFT or key == ord("h"):
            self._move_active(-step, 0)
        elif key in self.ARROW_RIGHT or key == ord("l"):
            self._move_active(step, 0)
        elif key in self.ARROW_UP or key == ord("k"):
            self._move_active(0, -step)
        elif key in self.ARROW_DOWN or key == ord("j"):
            self._move_active(0, step)
        elif key == ord("[") and b is not None:
            b.orientation = (b.orientation - 90) % 360
        elif key == ord("]") and b is not None:
            b.orientation = (b.orientation + 90) % 360
        elif key == ord(",") and b is not None:   # fine tilt counter-clockwise
            self._nudge_angle(b, -0.5)
        elif key == ord(".") and b is not None:   # fine tilt clockwise
            self._nudge_angle(b, 0.5)
        elif key == ord("<") and b is not None:   # coarse tilt counter-clockwise
            self._nudge_angle(b, -5.0)
        elif key == ord(">") and b is not None:   # coarse tilt clockwise
            self._nudge_angle(b, 5.0)
        elif key in (ord("x"), 8, 127) and b is not None:  # x, Backspace, or Del
            del self.boxes[self.active]
            self._renumber()
            self.active = min(self.active, len(self.boxes) - 1)
        elif key == ord("p") and b is not None:  # toggle preview
            if self._preview_open:
                self._close_preview()
            else:
                self.preview(b)
        elif key == 27:  # Esc dismisses the preview
            self._close_preview()
        elif key == ord("c"):
            self.crop_all()
        elif key == ord("s"):
            self.save()
        elif key in (13, 10):  # Enter / Return: crop, save, then next scan
            self.crop_all()  # crop_all() also saves metadata
            self.next_request = "next"
        elif key == ord("="):  # next scan without cropping (saves edits)
            self.save()
            self.next_request = "next"
        elif key == ord("-"):  # previous scan without cropping (saves edits)
            self.save()
            self.next_request = "prev"
        elif key == ord("q"):
            self.save()  # persist edits so a restart won't re-detect over them
            self.next_request = "quit"
        self._dirty = True

    # ---- actions ----
    PREVIEW_WINDOW = "Preview (p or Esc to close)"

    def preview(self, box: Box):
        try:
            out = crop_box(self.image, box)
        except ValueError:
            print("  ! cannot preview: box has zero size")
            return
        pv = out
        ph, pw = pv.shape[:2]
        s = min(700 / pw, 700 / ph, 1.0)
        if s < 1.0:
            pv = cv2.resize(pv, None, fx=s, fy=s, interpolation=cv2.INTER_AREA)
        cv2.imshow(self.PREVIEW_WINDOW, pv)
        self._preview_open = True

    def _close_preview(self):
        if self._preview_open:
            try:
                cv2.destroyWindow(self.PREVIEW_WINDOW)
            except cv2.error:
                pass
            self._preview_open = False

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

    def save(self):
        h, w = self.image.shape[:2]
        path = save_metadata(self.scan_path, (w, h), self.boxes)
        print(f"  saved metadata -> {path}")

    # ---- main loop for this scan ----
    def run(self) -> str:
        cv2.namedWindow(WINDOW, cv2.WINDOW_AUTOSIZE)
        cv2.setMouseCallback(WINDOW, self.on_mouse)
        while True:
            if self._dirty:  # redraw only when state changed (cheap overlays)
                disp = render(self._base, self.boxes, self.active, self.scale)
                cv2.imshow(WINDOW, disp)
                self._dirty = False
            key = cv2.waitKeyEx(15)  # full keycode (arrows survive); -1 if none
            if key != -1:
                self.on_key(key)
            if self.next_request:
                break
        req = self.next_request
        self.next_request = None
        return req

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

    print("Controls: drag inside=move | drag edge/corner=resize | "
          "drag empty=new box | arrows/hjkl=nudge box | n/Tab=next box | "
          "[ ]=orient | , . =tilt 0.5deg | < > =tilt 5deg | x=delete | "
          "p=toggle preview (Esc closes) | c=crop all | s=save | "
          "Enter=crop all + next scan | = / - =next/prev scan (no crop) | "
          "q=save & quit")

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

        editor = Editor(image, boxes, path, out_dir)
        req = editor.run()
        editor._close_preview()  # tidy the preview window before next scan
        if req == "quit":
            break
        elif req == "prev":
            idx = max(0, idx - 1)
        else:  # "next"
            idx += 1

    cv2.destroyAllWindows()
    print("Done.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
