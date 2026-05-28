# Photo Scan Splitter — Design

**Date:** 2026-05-28
**Status:** Approved

## Problem

Photo scans in `images/` each contain multiple physical photos placed on a
light scanner bed at varied positions, angles, and orientations (some sideways
or upside down). Each photo must be cropped into its own file. Auto-detection is
imperfect, so a human must be able to review, adjust, add, and orient selections
before cropping.

## Scope

Single Python script `split_photos.py` using OpenCV HighGUI for the UI. One-off
tooling — not a packaged library. No tests required beyond manual verification
(interactive GUI, large binary scans).

## Architecture

Three logical units in one script:

1. **Detector** — `detect_photos(image) -> list[Box]`
   - Grayscale → Otsu threshold (inverted: photos darker than light bed)
   - Morphological close to fill photo interiors
   - External contours → filter by min area (>2% of page area)
   - `cv2.minAreaRect` per contour → rotated `Box` (center, size, angle)

2. **Editor** — OpenCV HighGUI window
   - Holds display-scaled scan + list of `Box` objects; one box is "active"
   - Renders all boxes as rotated rectangles; active in green with a "top"
     arrow, others in yellow; status text overlay
   - Mouse + trackbars + keys drive editing (see Controls)

3. **Cropper** — `crop_box(image, box) -> image`
   - Rotate full-res scan by `-angle` around box center (deskew)
   - Extract upright `size` rectangle
   - Rotate by `orientation` (0/90/180/270) so "top" is up
   - Save to `extracted/`

## Data Model

`Box` (all geometry in **full-resolution** scan coordinates):
- `center`: `[x, y]`
- `size`: `[w, h]`
- `angle`: float degrees (deskew, −45…+45)
- `orientation`: int, one of 0/90/180/270 (which way is "top")
- `id`: int (1-based)
- `output`: output filename

Display scale is applied only at render/mouse time, never stored in `Box`.

## Controls (HighGUI)

Window: scan scaled-to-fit; all boxes drawn. Active box green with top arrow,
others yellow. Status overlay: active index, angle, orientation, box count.

**Mouse (acts on active box):**
- Left-drag in empty area → draw new box (becomes active)
- Left-drag inside active box → move
- Left-drag on a corner handle → resize
- Click on a box → make it active

**Trackbars:** `Angle` (−45…+45 fine deskew of active box), `Display Scale`.

**Keys:**
- `Tab` / `n` → cycle active box
- `[` / `]` → rotate orientation top by −90 / +90
- `Del` / `x` → delete active box
- `p` → preview active crop (separate window: deskewed + oriented result)
- `c` → crop & save **all** boxes to `extracted/`
- `Enter` → save metadata + load **next** scan
- `s` → save metadata JSON now
- `q` → quit

## Metadata Sidecar

One `<scan-stem>.photos.json` per scan, written next to the scan (or in a
`metadata/` dir — next to scan chosen for simplicity):

```json
{
  "scan": "original-005.jpg",
  "scan_size": [2550, 3507],
  "boxes": [
    {"id": 1, "center": [640, 410], "size": [1180, 760],
     "angle": -2.3, "orientation": 90, "output": "original-005_01.jpg"}
  ]
}
```

On load, if a scan's JSON exists, reload its boxes instead of auto-detecting, so
human edits persist across sessions.

## Output

`extracted/<scan-stem>_<NN>.jpg`, NN zero-padded 1-based index.

## Dependencies

- `opencv-python` (cv2)
- `numpy`

## Error Handling

- Missing `images/` or no scans → print message, exit
- Unreadable scan → skip with warning, advance to next
- Crop with zero/negative size → skip that box with warning
- `extracted/` created on demand

## Out of Scope (YAGNI)

- Perspective/quadrilateral correction (rotation-only deskew is enough)
- Auto-orientation detection (human sets orientation)
- Undo history (delete + redraw is sufficient)
- Packaging, CLI args beyond an optional images-dir path
