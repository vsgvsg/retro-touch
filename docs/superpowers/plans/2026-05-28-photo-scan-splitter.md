# Photo Scan Splitter Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build `split_photos.py`, an OpenCV-based tool that auto-detects photos in scanner images and lets a human adjust, orient, and crop each into its own file.

**Architecture:** One script with three units — a pure `detect_photos()` detector (Otsu threshold + contours + minAreaRect), a pure `crop_box()` cropper (deskew rotate → crop → orientation rotate), and an interactive HighGUI `Editor`. Pure units get TDD with synthetic images; the GUI is verified manually. All box geometry is stored in full-resolution coordinates; display scaling is applied only at render/mouse time.

**Tech Stack:** Python 3, opencv-python (cv2), numpy, pytest.

---

## File Structure

- Create: `split_photos.py` — the whole tool (Box dataclass, detector, cropper, editor, main loop)
- Create: `tests/test_split_photos.py` — TDD tests for detector + cropper
- Create: `requirements.txt` — opencv-python, numpy, pytest
- Runtime dirs (created on demand): `extracted/`, `<scan>.photos.json` sidecars

Single-file tool is appropriate: it's one-off, the three units are small, and keeping them together makes the manual-run workflow trivial (`python split_photos.py`).

---

## Task 1: Project scaffolding

**Files:**
- Create: `requirements.txt`
- Create: `split_photos.py` (initial stub with Box)
- Create: `tests/test_split_photos.py` (empty importable)

- [ ] **Step 1: Write requirements.txt**

```
opencv-python>=4.8
numpy>=1.24
pytest>=7.0
```

- [ ] **Step 2: Create the Box dataclass stub in split_photos.py**

```python
"""Photo Scan Splitter — detect, adjust, and crop photos from scanner images."""
from __future__ import annotations

import math
from dataclasses import dataclass, field

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
```

- [ ] **Step 3: Create importable test file**

```python
import numpy as np

import split_photos as sp


def test_box_roundtrip():
    b = sp.Box(center=[10, 20], size=[100, 50], angle=3.0, orientation=90, id=1, output="x.jpg")
    assert sp.Box.from_dict(b.to_dict()).to_dict() == b.to_dict()
```

- [ ] **Step 4: Run test to verify it passes**

Run: `cd /Users/svtorov/Projects/Claude-Demo/originals && python -m pytest tests/test_split_photos.py -v`
Expected: PASS (1 test)

---

## Task 2: Detector — `detect_photos()`

**Files:**
- Modify: `split_photos.py` (add `detect_photos`)
- Test: `tests/test_split_photos.py`

- [ ] **Step 1: Write a synthetic-image test fixture and failing test**

Add to `tests/test_split_photos.py`:

```python
def _make_scan_with_rects(rects, page=(800, 600)):
    """page=(h,w). rects: list of (cx, cy, w, h, angle_deg). Light bg, dark photos."""
    img = np.full((page[0], page[1], 3), 245, np.uint8)  # light scanner bed
    for (cx, cy, w, h, ang) in rects:
        rect = ((cx, cy), (w, h), ang)
        pts = cv2.boxPoints(rect).astype(np.int32)
        cv2.fillPoly(img, [pts], (60, 60, 60))  # dark photo region
    return img


def test_detect_finds_two_photos():
    img = _make_scan_with_rects([(200, 200, 220, 160, 0), (550, 400, 180, 240, 15)])
    boxes = sp.detect_photos(img)
    assert len(boxes) == 2


def test_detect_ignores_tiny_specks():
    img = _make_scan_with_rects([(300, 300, 300, 200, 0), (50, 50, 8, 8, 0)])
    boxes = sp.detect_photos(img)
    assert len(boxes) == 1


def test_detect_captures_angle():
    img = _make_scan_with_rects([(400, 300, 300, 200, 20)])
    boxes = sp.detect_photos(img)
    assert len(boxes) == 1
    # minAreaRect angle is ambiguous mod 90; just assert it's non-trivial
    assert abs(boxes[0].angle) > 1
```

- [ ] **Step 2: Run to verify it fails**

Run: `python -m pytest tests/test_split_photos.py -k detect -v`
Expected: FAIL with `AttributeError: module 'split_photos' has no attribute 'detect_photos'`

- [ ] **Step 3: Implement `detect_photos`**

Add to `split_photos.py`:

```python
def detect_photos(image: np.ndarray, min_area_frac: float = 0.02) -> list[Box]:
    """Detect photo regions on a light scanner background.

    Returns rotated Boxes in full-resolution coordinates, sorted top-to-bottom.
    """
    h, w = image.shape[:2]
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # Photos are darker than the light bed; THRESH_BINARY_INV + Otsu -> photos white
    _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)
    # Close to fill photo interiors (light areas inside photos)
    k = max(3, (min(h, w) // 100) | 1)  # odd kernel scaled to image
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
    closed = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel, iterations=2)

    contours, _ = cv2.findContours(closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    min_area = min_area_frac * h * w
    boxes: list[Box] = []
    for c in contours:
        if cv2.contourArea(c) < min_area:
            continue
        (cx, cy), (bw, bh), ang = cv2.minAreaRect(c)
        if bw < 1 or bh < 1:
            continue
        boxes.append(Box(center=[cx, cy], size=[bw, bh], angle=ang))

    boxes.sort(key=lambda b: (round(b.center[1] / 50), b.center[0]))
    for i, b in enumerate(boxes, 1):
        b.id = i
    return boxes
```

- [ ] **Step 4: Run to verify it passes**

Run: `python -m pytest tests/test_split_photos.py -k detect -v`
Expected: PASS (3 tests)

---

## Task 3: Cropper — `crop_box()`

**Files:**
- Modify: `split_photos.py` (add `crop_box`)
- Test: `tests/test_split_photos.py`

- [ ] **Step 1: Write failing tests for crop size + orientation**

Add to `tests/test_split_photos.py`:

```python
def test_crop_axis_aligned_size():
    img = _make_scan_with_rects([(400, 300, 300, 200, 0)])
    box = sp.Box(center=[400, 300], size=[300, 200], angle=0, orientation=0)
    out = sp.crop_box(img, box)
    assert abs(out.shape[1] - 300) <= 2  # width
    assert abs(out.shape[0] - 200) <= 2  # height


def test_crop_orientation_90_swaps_dims():
    img = _make_scan_with_rects([(400, 300, 300, 200, 0)])
    box = sp.Box(center=[400, 300], size=[300, 200], angle=0, orientation=90)
    out = sp.crop_box(img, box)
    # 90/270 rotation swaps width and height
    assert abs(out.shape[0] - 300) <= 2
    assert abs(out.shape[1] - 200) <= 2


def test_crop_deskew_recovers_content():
    img = _make_scan_with_rects([(400, 300, 300, 200, 20)])
    boxes = sp.detect_photos(img)
    out = sp.crop_box(img, boxes[0])
    # Cropped region should be mostly the dark photo, not light bg
    assert out.mean() < 150
```

- [ ] **Step 2: Run to verify it fails**

Run: `python -m pytest tests/test_split_photos.py -k crop -v`
Expected: FAIL with `AttributeError: ... 'crop_box'`

- [ ] **Step 3: Implement `crop_box`**

Add to `split_photos.py`:

```python
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

    orient = box.orientation % 360
    if orient == 90:
        crop = cv2.rotate(crop, cv2.ROTATE_90_CLOCKWISE)
    elif orient == 180:
        crop = cv2.rotate(crop, cv2.ROTATE_180)
    elif orient == 270:
        crop = cv2.rotate(crop, cv2.ROTATE_90_COUNTERCLOCKWISE)
    return crop
```

- [ ] **Step 4: Run to verify it passes**

Run: `python -m pytest tests/test_split_photos.py -k crop -v`
Expected: PASS (3 tests)

- [ ] **Step 5: Run full suite + commit-free checkpoint**

Run: `python -m pytest tests/test_split_photos.py -v`
Expected: PASS (7 tests total)

---

## Task 4: Metadata sidecar I/O

**Files:**
- Modify: `split_photos.py` (add `load_metadata`, `save_metadata`, `sidecar_path`)
- Test: `tests/test_split_photos.py`

- [ ] **Step 1: Write failing round-trip test**

Add to `tests/test_split_photos.py`:

```python
def test_metadata_roundtrip(tmp_path):
    scan = tmp_path / "original-005.jpg"
    scan.write_bytes(b"fake")  # path only needs to exist for naming
    boxes = [sp.Box(center=[640, 410], size=[1180, 760], angle=-2.3,
                    orientation=90, id=1, output="original-005_01.jpg")]
    sp.save_metadata(str(scan), (2550, 3507), boxes)
    side = sp.sidecar_path(str(scan))
    assert side.endswith("original-005.photos.json")
    loaded = sp.load_metadata(str(scan))
    assert len(loaded) == 1
    assert loaded[0].orientation == 90
    assert loaded[0].output == "original-005_01.jpg"


def test_load_metadata_missing_returns_none(tmp_path):
    scan = tmp_path / "nope.jpg"
    assert sp.load_metadata(str(scan)) is None
```

- [ ] **Step 2: Run to verify it fails**

Run: `python -m pytest tests/test_split_photos.py -k metadata -v`
Expected: FAIL with `AttributeError: ... 'save_metadata'`

- [ ] **Step 3: Implement metadata I/O**

Add to top of `split_photos.py` imports: `import json`, `import os`.

Add functions:

```python
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
```

- [ ] **Step 4: Run to verify it passes**

Run: `python -m pytest tests/test_split_photos.py -k metadata -v`
Expected: PASS (2 tests)

---

## Task 5: Editor — rendering and geometry helpers

**Files:**
- Modify: `split_photos.py` (add display/geometry helpers used by the GUI)
- Test: `tests/test_split_photos.py`

These helpers are pure and testable even though the GUI isn't.

- [ ] **Step 1: Write failing tests for scale conversion + hit-testing**

Add to `tests/test_split_photos.py`:

```python
def test_display_full_roundtrip():
    # display point -> full coords -> display, with scale 0.5
    assert sp.disp_to_full((100, 50), 0.5) == (200.0, 100.0)
    assert sp.full_to_disp((200, 100), 0.5) == (100.0, 50.0)


def test_point_in_box_center_hits():
    box = sp.Box(center=[100, 100], size=[80, 40], angle=0)
    assert sp.point_in_box((100, 100), box) is True
    assert sp.point_in_box((300, 300), box) is False


def test_point_in_box_respects_rotation():
    box = sp.Box(center=[100, 100], size=[80, 40], angle=90)
    # rotated 90deg, so a point 30px above center should now be inside
    assert sp.point_in_box((100, 70), box) is True
```

- [ ] **Step 2: Run to verify it fails**

Run: `python -m pytest tests/test_split_photos.py -k "disp or point_in" -v`
Expected: FAIL with `AttributeError`

- [ ] **Step 3: Implement helpers**

Add to `split_photos.py`:

```python
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
```

- [ ] **Step 4: Run to verify it passes**

Run: `python -m pytest tests/test_split_photos.py -k "disp or point_in" -v`
Expected: PASS (3 tests)

- [ ] **Step 5: Add the render function (no unit test — visual)**

Add to `split_photos.py`:

```python
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


def render(image: np.ndarray, boxes: list[Box], active_idx: int, scale: float) -> np.ndarray:
    disp = cv2.resize(image, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
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
    cv2.rectangle(disp, (0, 0), (disp.shape[1], 26), (0, 0, 0), -1)
    cv2.putText(disp, txt, (6, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1)
    return disp
```

- [ ] **Step 6: Run full suite**

Run: `python -m pytest tests/test_split_photos.py -v`
Expected: PASS (15 tests total)

---

## Task 6: Editor — interactive GUI class

**Files:**
- Modify: `split_photos.py` (add `Editor` class)

No unit tests — verified manually in Task 8.

- [ ] **Step 1: Implement the Editor class**

Add to `split_photos.py`:

```python
WINDOW = "Photo Scan Splitter"
HANDLE_R = 12  # full-coord radius for corner-handle hit test (scaled at use)


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
        self.drag = None          # None | 'move' | 'new' | ('corner', idx)
        self.drag_start = None    # full-coord
        self.next_request = None  # set to 'next' or 'quit' to end loop

    # ---- coordinate helpers ----
    def _full(self, x, y):
        return disp_to_full((x, y), self.scale)

    def _active_box(self):
        if 0 <= self.active < len(self.boxes):
            return self.boxes[self.active]
        return None

    # ---- mouse ----
    def on_mouse(self, event, x, y, flags, param):
        fx, fy = self._full(x, y)
        if event == cv2.EVENT_LBUTTONDOWN:
            # 1) corner handle of active box?
            b = self._active_box()
            if b is not None:
                rect = ((b.center[0], b.center[1]), (b.size[0], b.size[1]), b.angle)
                corners = cv2.boxPoints(rect)
                for ci, (cxp, cyp) in enumerate(corners):
                    if math.hypot(fx - cxp, fy - cyp) <= HANDLE_R / self.scale:
                        self.drag = ("corner", ci)
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
        elif event == cv2.EVENT_LBUTTONUP and self.drag:
            self._finish_drag(fx, fy)

    def _handle_drag(self, fx, fy):
        b = self._active_box()
        if self.drag == "move" and b is not None:
            dx = fx - self.drag_start[0]
            dy = fy - self.drag_start[1]
            b.center[0] += dx
            b.center[1] += dy
            self.drag_start = (fx, fy)
        elif isinstance(self.drag, tuple) and self.drag[0] == "corner" and b is not None:
            # resize: set size from distance of dragged corner to center (axis-aligned approx)
            dx = abs(fx - b.center[0]) * 2
            dy = abs(fy - b.center[1]) * 2
            b.size[0] = max(10, dx)
            b.size[1] = max(10, dy)

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

    # ---- keyboard ----
    def on_key(self, key):
        b = self._active_box()
        if key in (ord("n"), 9):  # n or Tab
            if self.boxes:
                self.active = (self.active + 1) % len(self.boxes)
        elif key == ord("[") and b is not None:
            b.orientation = (b.orientation - 90) % 360
        elif key == ord("]") and b is not None:
            b.orientation = (b.orientation + 90) % 360
        elif key in (ord("x"), 255) and b is not None:  # x or Del
            del self.boxes[self.active]
            self._renumber()
            self.active = min(self.active, len(self.boxes) - 1)
        elif key == ord("p") and b is not None:
            self.preview(b)
        elif key == ord("c"):
            self.crop_all()
        elif key == ord("s"):
            self.save()
        elif key == 13:  # Enter
            self.save()
            self.next_request = "next"
        elif key == ord("q"):
            self.next_request = "quit"

    # ---- actions ----
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
        cv2.imshow("Preview", pv)

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
        cv2.createTrackbar("Angle+45", WINDOW, 45, 90,
                           lambda v: self._set_angle(v - 45))
        cv2.createTrackbar("Scale%", WINDOW, int(self.scale * 100), 100,
                           lambda v: self._set_scale(max(0.1, v / 100)))
        while True:
            disp = render(self.image, self.boxes, self.active, self.scale)
            cv2.imshow(WINDOW, disp)
            key = cv2.waitKey(20) & 0xFF
            if key != 255:  # 255 == no key pressed this tick
                self.on_key(key)
            if self.next_request:
                break
        req = self.next_request
        self.next_request = None
        return req

    def _set_angle(self, val):
        b = self._active_box()
        if b is not None:
            b.angle = float(val)

    def _set_scale(self, val):
        self.scale = val
```

- [ ] **Step 2: Smoke-import check (no GUI launch)**

Run: `python -c "import split_photos; print('import OK')"`
Expected: prints `import OK`

---

## Task 7: Main loop — iterate scans

**Files:**
- Modify: `split_photos.py` (add `main()` + `__main__` guard)

- [ ] **Step 1: Implement main()**

Add to `split_photos.py`:

```python
import sys
import glob


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

    print("Controls: drag=draw/move/resize | n/Tab=next box | [ ]=orient | "
          "x=delete | p=preview | c=crop all | s=save | Enter=next scan | q=quit")

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
        cv2.destroyWindow("Preview") if cv2.getWindowProperty("Preview", 0) >= 0 else None
        if req == "quit":
            break
        idx += 1

    cv2.destroyAllWindows()
    print("Done.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
```

Note: move `import sys` / `import glob` to the top import block with the others (don't leave mid-file imports); shown here inline for locality.

- [ ] **Step 2: Smoke-import check**

Run: `python -c "import split_photos; print(split_photos.list_scans('images'))"`
Expected: prints the sorted list of 10 scan paths.

- [ ] **Step 3: Run full test suite**

Run: `python -m pytest tests/test_split_photos.py -v`
Expected: PASS (15 tests).

---

## Task 8: Manual verification (human-in-the-loop)

**Files:** none — interactive run.

- [ ] **Step 1: Install deps**

Run: `cd /Users/svtorov/Projects/Claude-Demo/originals && pip install -r requirements.txt`

- [ ] **Step 2: Launch on the real scans**

Run: `python split_photos.py`
Expected: window opens on `original-001.jpg` with auto-detected green/yellow boxes.

- [ ] **Step 3: Exercise each control and confirm**

Verify each: draw a new box on a missed photo; `n` cycles active; drag to move; corner-drag to resize; Angle trackbar deskews; `[`/`]` rotates the orientation arrow; `p` opens an upright preview; `c` writes files to `extracted/`; `s`/`Enter` writes `<scan>.photos.json`; re-launching reloads boxes from JSON; `q` quits.

- [ ] **Step 4: Confirm outputs**

Run: `ls extracted/ && ls images/*.photos.json`
Expected: cropped `_NN.jpg` files and per-scan JSON sidecars exist.
