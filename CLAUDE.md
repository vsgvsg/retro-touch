# Photo Scan Splitter

Single-tool repo: `split_photos.py` detects, lets a human adjust, and crops
multiple photos out of flatbed scan images in `images/` into `extracted/`.

## Commands
- `python3 split_photos.py` - launch the interactive editor (needs a human at the GUI; Claude can't drive the OpenCV window)
- `python3 -m pytest tests/test_split_photos.py -q` - run tests (cv2/numpy/pytest already installed system Python 3.9)

## Code conventions
- Keep it one file (`split_photos.py`); it's a one-off tool, not a package.
- Pure functions (detector, cropper, metadata I/O, geometry helpers) get TDD tests; the `Editor` HighGUI class is verified manually, not unit-tested.
- Box geometry is always stored in FULL-resolution scan coords; display scale is applied only at render/mouse time.

## Gotchas (OpenCV HighGUI on macOS)
- Use `cv2.waitKeyEx()` (not `waitKey() & 0xFF`) so arrow keys survive; no-key sentinel is `-1`.
- Keys only register when an OpenCV window has focus, not the terminal.
- `cv2.getWindowProperty` on a never-created window raises (doesn't return -1); guard `destroyWindow` with try/except `cv2.error`.
- Detected box `angle` is normalized to (-45, 45] via `normalize_rect`; deliberate quarter-turns go in `Box.orientation`, not the deskew angle.
- `crop_box` orientation = which edge is the photo's real top (matches the on-screen arrow): 90=right→CCW, 270=left→CW.
- Auto-detect is deliberately best-effort (non-white beds, touching photos mis-detect); the human fixes those in the editor. Don't over-tune the detector.

## Persistence
- Metadata saved per-scan as `images/<scan>.photos.json`; on restart, if it exists, boxes load from it and detection is skipped (never override manual edits).
