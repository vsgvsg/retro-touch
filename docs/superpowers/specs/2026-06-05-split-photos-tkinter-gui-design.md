# Split Photos Tkinter GUI Redesign — Design

**Date:** 2026-06-05
**Status:** Approved

## Goal

Replace the OpenCV HighGUI interactive editor in `split_photos.py` with a modern, premium Tkinter/ttk desktop application. The new UI will match the visual style, theme, and behavior of the `face_pipeline`'s labeler, age labeler, and review screens.

Specifically:
- We will implement a side-by-side layout (the large scan image on the left, and a scrollable column of cropped photo cards on the right).
- We will reuse the same UI theme (`_install_theme`, `ACCENT` colors, and fonts) and custom scrollbar behavior from `face_pipeline.py`.
- Crop cards on the right will display interactive number badges, high-quality rounded thumbnails of the cropped boxes, dimensions, deskew angles, and action buttons for rotation and deletion.

## Non-goals

- We will **not** modify the underlying auto-detection algorithm (`detect_photos`) or deskew/cropping math (`crop_box`).
- We will **not** cross-import between `split_photos.py` and `face_pipeline.py`. All theme utilities and GUI classes will be self-contained or duplicated as necessary to respect the "no cross-import" convention of the tools.
- We will **not** change the metadata sidecar schema (`images/<scan>.photos.json`). All edits will write back to the existing schema.

## UI Design & Component Layout

The window geometry will be locked to `1120x780` pixels (non-resizable) to avoid size jitter between scans.

```
+-------------------------------------------------------------------------+
|  Photo Splitter                                                     _ X |
+-------------------------------------------------------------------------+
|  Scan: scan_001.jpg (1 of 5) — 4 photos detected                        |
|  [====================== Progress Bar ===============================] |
+-------------------------------------------------------------------------+
|  +-----------------------------------+  +----------------------------+  |
|  |                                   |  | Detected Photos (4 found)  |  |
|  |                                   |  |                            |  |
|  |                                   |  | +------------------------+ |  |
|  |                                   |  | | #1   [Img] Photo 1.jpg | |  |
|  |                                   |  | |      1024x768 px       | |  |
|  |                                   |  | |      Angle: 2.5 deg    | |  |
|  |            Scan Viewer            |  | |  [Rotate] [Delete]     | |  |
|  |          (with box edits)         |  | +------------------------+ |  |
|  |                                   |  |                            |  |
|  |                                   |  | +------------------------+ |  |
|  |                                   |  | | #2   [Img] Photo 2.jpg | |  |
|  |                                   |  | |      ...               | |  |
|  |                                   |  | +------------------------+ |  |
|  +-----------------------------------+  +----------------------------+  |
+-------------------------------------------------------------------------+
|         [ ← Back ]                       [ Save & Next → ]              |
|  Keys: Tab/n: Cycle box | x: Delete | []: Rotate | ,.: Tilt | s: Save   |
+-------------------------------------------------------------------------+
```

### 1. Header Frame
- **Status Title:** Left-aligned text displaying the scan name, current index, and total number of scans.
- **Progress Bar:** A clickable progress bar (`ttk.Progressbar`) that jumps directly to the scan corresponding to the clicked percentage.

### 2. Body Frame (Split)
- **Left Viewer Pane (`760x680`):**
  - Displays the scan image with overlaid boxes drawn via OpenCV (reusing the existing coordinate-scaling and arrow-rendering math).
  - Listens to mouse clicks and drags on the image to handle box selection, resizing (edge/corner handles), and new box creation.
- **Right Sidebar Frame (`300px` wide):**
  - A scrollable vertical column displaying a list of cropped photo cards.
  - Scrollbar is managed dynamically (`pack` or `pack_forget`) based on content height to prevent unnecessary visual noise.

### 3. Crop Cards
Each card will contain:
- A circular ID badge (green for active, yellow for inactive) matching the box color on the image.
- A `64x64` rounded crop thumbnail generated via `crop_to_round_photo`.
- Labels displaying the cropped photo's dimensions, deskew angle, and orientation.
- Interactive Buttons:
  - **Rotate Button:** Clicking it rotates the box orientation by 90 degrees CCW (updates arrow direction).
  - **Delete Button:** Deletes the box.
- Clicking any area of a card will activate the corresponding box in the scan viewer.

### 4. Bottom Controls Frame
- Navigation Buttons: `← Back` (Save & Previous) and `Save & Next →` (Crop all, Save, & Next).
- Text summary of keyboard shortcuts.

## Interaction & Event Handling

### Mouse Drag & Resize
All mouse interactions on the left-side image viewer will be captured via Tkinter event bindings (`<Button-1>`, `<B1-Motion>`, `<ButtonRelease-1>`):
- **Down (`<Button-1>`):**
  - Grab handle check: Checks if clicked near an edge/corner of the active box (`grab_handle`).
  - Move check: Checks if clicked inside an existing box (`point_in_box`).
  - Empty area check: Starts a drag-to-create state (`drag = "new"`).
- **Motion (`<B1-Motion>`):**
  - Updates selection coordinate nudges or resizes the active box geometry.
  - Re-renders the OpenCV overlay image in real-time.
- **Release (`<ButtonRelease-1>`):**
  - Finalizes the coordinates.
  - If a drag-to-create action finished with a width/height > 10 pixels, creates the box.
  - **Triggers thumbnail extraction/update for the right-hand sidebar cards.**

### Keyboard Bindings
- `n` / `Tab`: Cycle active box selection.
- `Arrow Keys` / `h, j, k, l`: Nudge active box center.
- `[` / `]`: Rotate active box orientation by 90 degrees CCW / CW.
- `,` / `.`: Adjust tilt angle by `-0.5°` / `+0.5°`.
- `<` / `>`: Adjust tilt angle by `-5.0°` / `+5.0°`.
- `Delete` / `Backspace` / `x`: Delete active box.
- `Enter`: Save scan metadata, crop all boxes, and advance.
- `=` / `-`: Save metadata and advance/return scan index without cropping.
- `q`: Save and quit.

## Error Handling & Robustness
- **Invalid Crop Dimensions:** If a box has a size of 0 or wraps outside image boundaries, the GUI handles it gracefully by clamp-scaling the crop to the nearest canvas pixel instead of raising a ValueError/crashing.
- **Missing Images:** If a scan cannot be loaded, the app skips to the next index, prints an error to console, and continues.
- **Non-Interactive Environments:** The GUI verifies that Tk is available and fails fast with a helpful CLI message if it cannot open a window (e.g. running on a headless CI).

## Testing Strategy
- Pure geometric and file I/O helpers remain unit-tested in `tests/test_split_photos.py`.
- We will add unit tests to verify:
  - Coordinate scaling and handle-grabbing math.
  - `Box` creation boundary limits.
- The Tkinter `SplitterApp` UI wiring and interactive drag-and-drop actions will be verified manually.
