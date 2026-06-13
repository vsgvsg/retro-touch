import math
import os
import sys
import tempfile

import cv2
import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import split_photos as sp


def _tk_usable():
    """True if this interpreter has a Tk that can actually open a window.

    The macOS system Python 3.9 ships Tk 8.5, which SIGABRTs on Tk() (an
    uncatchable C-level abort that also pops a 'Python quit unexpectedly'
    dialog) — so we must NOT call Tk() to detect it. Reading TkVersion opens
    no window. Tk >= 8.6 (the .venv's Tk 9.0) works; gate the GUI tests on
    that. Run the suite via .venv/bin/python so these tests actually execute.
    """
    try:
        import tkinter
        return tkinter.TkVersion >= 8.6
    except Exception:
        return False


_TK_OK = _tk_usable()
requires_tk = pytest.mark.skipif(not _TK_OK, reason="no usable Tk (need Tk>=8.6; use .venv)")


def _gui_app(root=None):
    """Build a SplitterApp on a tiny synthetic scan (caller is @requires_tk)."""
    d = tempfile.mkdtemp()
    p = os.path.join(d, "t.jpg")
    img = np.full((400, 600, 3), 245, np.uint8)
    img[80:240, 120:380] = 60
    cv2.imwrite(p, img)
    app = sp.SplitterApp([p], tempfile.mkdtemp(), root=root)
    app._show()
    app.root.update()
    return app


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


def test_list_scans_recursive(tmp_path):
    images_dir = tmp_path / "images"
    images_dir.mkdir()
    sub_dir = images_dir / "nested"
    sub_dir.mkdir()

    # Create image files
    img1 = images_dir / "photo1.jpg"
    img1.write_bytes(b"fake1")
    img2 = sub_dir / "photo2.png"
    img2.write_bytes(b"fake2")

    scans = sp.list_scans(str(images_dir))
    assert len(scans) == 2
    assert any(s.endswith("photo1.jpg") for s in scans)
    assert any(s.endswith("photo2.png") for s in scans)


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
def test_nudge_box_shifts_center():
    b = sp.Box(center=[100, 100], size=[40, 40], angle=0, id=1)
    sp.nudge_box(b, 20, 0)   # right
    assert b.center == [120, 100]
    sp.nudge_box(b, 0, -20)  # up
    assert b.center == [120, 80]


# ---- tilt_angle: skew nudge must not snap a far-out (legacy) angle ----
def test_tilt_angle_small_steps_in_range():
    assert sp.tilt_angle(0.0, 0.5) == 0.5
    assert sp.tilt_angle(0.5, -0.5) == 0.0


def test_tilt_angle_clamps_in_range_value_at_boundary():
    assert sp.tilt_angle(44.8, 0.5) == 45.0   # in-range stays bounded
    assert sp.tilt_angle(-44.8, -0.5) == -45.0


def test_tilt_angle_does_not_snap_out_of_range_angle():
    # regression: a box loaded at angle=-85.35 must not jump to -45 on first
    # tilt. delta applies relative; the value just moves by delta.
    assert sp.tilt_angle(-85.35, 0.5) == -84.85
    assert sp.tilt_angle(-85.35, -0.5) == -85.85


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


# ---- orientation_label ----
def test_orientation_label_all_quarters():
    assert sp.orientation_label(0) == "top"
    assert sp.orientation_label(90) == "right"
    assert sp.orientation_label(180) == "bottom"
    assert sp.orientation_label(270) == "left"


def test_orientation_label_wraps():
    assert sp.orientation_label(360) == "top"
    assert sp.orientation_label(-90) == "left"


# ---- GUI smoke: an arrow keypress drives _on_key and moves the active box ----
# NOTE: headless event_generate routes to the root regardless of focus, so this
# can't reproduce the real-WM "button steals focus" regression (the reason the
# app binds keys via bind_all, mirroring the face_pipeline GUIs). It does guard
# that the _on_key -> _move path stays wired.
@requires_tk
def test_arrow_key_moves_active_box(tk_root):
    app = _gui_app(root=tk_root)
    try:
        b = app.editor.active_box()
        cx0 = b.center[0]
        app.editor.canvas.event_generate("<Key>", keysym="Right")
        app.root.update()
        assert b.center[0] > cx0, "arrow shortcut did not move the active box"
    finally:
        for widget in app.root.winfo_children():
            widget.destroy()


@requires_tk
def test_tab_and_n_cycle_active_box(tk_root):
    # regression: Tab is eaten by Tk focus-traversal (class bindtag) before the
    # bind_all <Key> handler, so it must be bound on the canvas instance tag.
    # 'n' goes through _on_key. Two boxes so cycling is observable.
    d = tempfile.mkdtemp()
    p = os.path.join(d, "two.jpg")
    img = np.full((400, 600, 3), 245, np.uint8)
    img[40:180, 60:260] = 60
    img[40:180, 320:540] = 60
    cv2.imwrite(p, img)
    app = sp.SplitterApp([p], tempfile.mkdtemp(), root=tk_root)
    app._show()
    app.root.update()
    try:
        assert len(app.boxes) >= 2
        assert app.editor.active == 0
        app.editor.canvas.event_generate("<Tab>")
        app.root.update()
        assert app.editor.active == 1, "Tab did not advance the active box"
        app.editor.canvas.event_generate("<Key>", keysym="n")
        app.root.update()
        assert app.editor.active == 0, "n did not advance (wrap) the active box"
    finally:
        for widget in app.root.winfo_children():
            widget.destroy()


@requires_tk
def test_bracket_keys_rotate_orientation(tk_root):
    # regression: punctuation keys were matched on X11 keysym names
    # ("bracketright") but this Tk reports event.keysym/char as "]", so they
    # silently never fired. Match on event.char instead.
    app = _gui_app(root=tk_root)
    try:
        b = app.editor.active_box()
        assert b.orientation == 0
        app.editor.canvas.event_generate("<Key>", keysym="bracketright")  # ]
        app.root.update()
        assert b.orientation == 90, "] did not rotate"
        app.editor.canvas.event_generate("<Key>", keysym="bracketright")  # ]
        app.root.update()
        assert b.orientation == 180
    finally:
        for widget in app.root.winfo_children():
            widget.destroy()


@requires_tk
def test_sidebar_rows_have_thumbnails_and_word_orientation(tk_root):
    app = _gui_app(root=tk_root)
    try:
        # one thumbnail PhotoImage kept per box (prevents Tk GC)
        assert len(app._row_thumbs) == len(app.boxes) >= 1
        # active-box panel shows a word, not degrees
        assert app.orient_var.get() in ("top", "right", "bottom", "left")
        app._orient(90)  # Rotate
        app.root.update()
        assert app.orient_var.get() == "right"
    finally:
        for widget in app.root.winfo_children():
            widget.destroy()


@requires_tk
def test_shortcuts_popover_opens_and_closes(tk_root):
    app = _gui_app(root=tk_root)
    try:
        assert app._shortcuts_win is None
        app._show_shortcuts()
        app.root.update()
        assert app._shortcuts_win is not None
        # every shortcut row is rendered (key label + description = 2 widgets
        # each, plus title + close button)
        assert app._shortcuts_win.winfo_exists()
        app._close_shortcuts()
        app.root.update()
        assert app._shortcuts_win is None
    finally:
        for widget in app.root.winfo_children():
            widget.destroy()


@requires_tk
def test_drag_selection_draws_preview(tk_root):
    app = _gui_app(root=tk_root)
    try:
        # Initial drag state should be None
        assert app.editor.drag is None
        assert app.editor.drag_current is None
        
        # Simulate pressing at (100, 100)
        app.editor.canvas.event_generate("<Button-1>", x=100, y=100)
        app.root.update()
        assert app.editor.drag == "new"
        assert app.editor.drag_start == app.editor._full(100, 100)
        assert app.editor.drag_current == app.editor._full(100, 100)
        
        # Simulate dragging to (200, 200)
        app.editor.canvas.event_generate("<B1-Motion>", x=200, y=200)
        app.root.update()
        assert app.editor.drag == "new"
        assert app.editor.drag_current == app.editor._full(200, 200)
        
        # Simulate release
        app.editor.canvas.event_generate("<ButtonRelease-1>", x=200, y=200)
        app.root.update()
        assert app.editor.drag is None
        assert app.editor.drag_current is None
    finally:
        for widget in app.root.winfo_children():
            widget.destroy()
