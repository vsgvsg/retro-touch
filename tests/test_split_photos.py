import math
import os
import sys

import cv2
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import split_photos as sp


def _make_scan_with_rects(rects, page=(800, 600)):
    """page=(h,w). rects: list of (cx, cy, w, h, angle_deg). Light bg, dark photos."""
    img = np.full((page[0], page[1], 3), 245, np.uint8)  # light scanner bed
    for (cx, cy, w, h, ang) in rects:
        rect = ((cx, cy), (w, h), ang)
        pts = cv2.boxPoints(rect).astype(np.int32)
        cv2.fillPoly(img, [pts], (60, 60, 60))  # dark photo region
    return img


# ---- Box roundtrip ----
def test_box_roundtrip():
    b = sp.Box(center=[10, 20], size=[100, 50], angle=3.0, orientation=90, id=1, output="x.jpg")
    assert sp.Box.from_dict(b.to_dict()).to_dict() == b.to_dict()


# ---- Detector ----
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


def test_detect_angle_normalized_to_deskew_range():
    # minAreaRect can report ~-90 for a near-axis photo; the detector must
    # normalize that to a small deskew so the crop baseline is consistent.
    img = _make_scan_with_rects([(300, 400, 200, 320, 0)])
    boxes = sp.detect_photos(img)
    assert len(boxes) == 1
    assert -45 < boxes[0].angle <= 45
    # a tall photo (h>w) must stay tall after normalization
    assert boxes[0].size[1] > boxes[0].size[0]


def test_detect_never_returns_oversized_box():
    # Regression: a box must never be larger than the page (was the
    # page-sized-blob bug from merged contours + ballooned minAreaRect).
    img = _make_scan_with_rects([(150, 200, 200, 240, 0), (450, 200, 200, 240, 0),
                                 (300, 600, 220, 240, 0)])
    boxes = sp.detect_photos(img)
    h, w = img.shape[:2]
    assert len(boxes) >= 2  # separated photos detected individually
    for b in boxes:
        assert b.size[0] <= w * 1.05 and b.size[1] <= h * 1.05


# ---- Cropper ----
def test_crop_axis_aligned_size():
    img = _make_scan_with_rects([(400, 300, 300, 200, 0)])
    box = sp.Box(center=[400, 300], size=[300, 200], angle=0, orientation=0)
    out = sp.crop_box(img, box)
    assert abs(out.shape[1] - 300) <= 2  # width
    assert abs(out.shape[0] - 200) <= 2  # height


def test_crop_orientation_brings_marked_top_up():
    # Contract: box.orientation names which edge is the photo's real top
    # (matching the on-screen arrow). crop_box must rotate that edge to the
    # TOP of the output. Marker = bright stripe on the chosen edge.
    def img_with_top_marker():
        im = np.zeros((100, 60, 3), np.uint8)
        im[0:15, :] = 255  # bright stripe on the box's top edge
        return im

    def top_edge_is_brightest(out):
        e = {"top": out[0:15, :].mean(), "bottom": out[-15:, :].mean(),
             "left": out[:, 0:15].mean(), "right": out[:, -15:].mean()}
        return max(e, key=e.get) == "top"

    im = img_with_top_marker()
    # orientation=0: top edge is the real top -> stays up
    b = sp.Box(center=[30, 50], size=[60, 100], angle=0, orientation=0)
    assert top_edge_is_brightest(sp.crop_box(im, b))

    # If the real top is on the RIGHT (orientation=90), rotating to upright
    # must move that right edge to the top.
    im_r = np.zeros((100, 60, 3), np.uint8)
    im_r[:, -15:] = 255  # bright stripe on the right edge
    b = sp.Box(center=[30, 50], size=[60, 100], angle=0, orientation=90)
    assert top_edge_is_brightest(sp.crop_box(im_r, b))

    # Real top on the LEFT (orientation=270) -> left edge goes to top.
    im_l = np.zeros((100, 60, 3), np.uint8)
    im_l[:, 0:15] = 255
    b = sp.Box(center=[30, 50], size=[60, 100], angle=0, orientation=270)
    assert top_edge_is_brightest(sp.crop_box(im_l, b))

    # Real top on the BOTTOM (orientation=180) -> bottom edge goes to top.
    im_b = np.zeros((100, 60, 3), np.uint8)
    im_b[-15:, :] = 255
    b = sp.Box(center=[30, 50], size=[60, 100], angle=0, orientation=180)
    assert top_edge_is_brightest(sp.crop_box(im_b, b))


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


# ---- Metadata ----
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


# ---- Display / geometry helpers ----
def test_display_full_roundtrip():
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


# ---- Edge/corner resize handles ----
def test_grab_handle_right_edge():
    box = sp.Box(center=[100, 100], size=[80, 40], angle=0)
    # right edge is at local x = +40 -> full (140, 100)
    assert sp.grab_handle((140, 100), box, tol=5) == (1, 0)


def test_grab_handle_top_edge():
    box = sp.Box(center=[100, 100], size=[80, 40], angle=0)
    # top edge at local y = -20 -> full (100, 80)
    assert sp.grab_handle((100, 80), box, tol=5) == (0, -1)


def test_grab_handle_corner():
    box = sp.Box(center=[100, 100], size=[80, 40], angle=0)
    assert sp.grab_handle((140, 120), box, tol=5) == (1, 1)


def test_grab_handle_interior_is_none():
    box = sp.Box(center=[100, 100], size=[80, 40], angle=0)
    assert sp.grab_handle((100, 100), box, tol=5) is None


def test_resize_right_edge_anchors_left():
    box = sp.Box(center=[100, 100], size=[80, 40], angle=0)
    left_before = box.center[0] - box.size[0] / 2  # = 60
    sp.resize_box(box, (1, 0), (160, 100))  # drag right edge to x=160
    assert abs(box.size[0] - 100) < 1e-6     # 160 - 60
    assert abs((box.center[0] - box.size[0] / 2) - left_before) < 1e-6  # left fixed


def test_resize_top_edge_anchors_bottom():
    box = sp.Box(center=[100, 100], size=[80, 40], angle=0)
    bottom_before = box.center[1] + box.size[1] / 2  # = 120
    sp.resize_box(box, (0, -1), (100, 70))   # drag top edge up to y=70
    assert abs(box.size[1] - 50) < 1e-6      # 120 - 70
    assert abs((box.center[1] + box.size[1] / 2) - bottom_before) < 1e-6


# ---- normalize_rect (orientation-bug fix) ----
def test_normalize_rect_folds_minus_90():
    # minAreaRect ~-90 with swapped dims -> small angle, dims restored
    bw, bh, ang = sp.normalize_rect(200, 320, -90)
    assert -45 < ang <= 45
    assert (bw, bh) == (320, 200)


def test_normalize_rect_keeps_small_angle():
    bw, bh, ang = sp.normalize_rect(300, 200, -3)
    assert (bw, bh, ang) == (300, 200, -3)


# ---- Editor arrow-key movement ----
def test_editor_arrow_moves_active_box(tmp_path):
    img = np.full((200, 200, 3), 245, np.uint8)
    boxes = [sp.Box(center=[100, 100], size=[40, 40], angle=0, id=1)]
    ed = sp.Editor(img, boxes, str(tmp_path / "s.jpg"), str(tmp_path / "out"))
    cx0 = boxes[0].center[0]
    ed.on_key(sp.Editor.ARROW_RIGHT[0])  # primary right keycode
    assert boxes[0].center[0] > cx0
    cy0 = boxes[0].center[1]
    ed.on_key(ord("k"))  # ascii fallback: up
    assert boxes[0].center[1] < cy0


def test_resize_respects_rotation_anchor():
    # 90deg box: local +x axis points along world +y. The local-left edge
    # (anchor) sits at world y = 100 - 40 = 60. Dragging the local-right edge
    # to world y=180 should set width=120 and keep that anchor edge at y=60.
    box = sp.Box(center=[100, 100], size=[80, 40], angle=90)
    sp.resize_box(box, (1, 0), (100, 180))
    assert abs(box.size[0] - 120) < 1e-6
    # anchor edge is along local -x; in world that's center - (w/2) along +y axis
    ang = math.radians(box.angle)
    anchor_world_y = box.center[1] - (box.size[0] / 2) * math.sin(ang)
    assert abs(anchor_world_y - 60) < 1e-6
