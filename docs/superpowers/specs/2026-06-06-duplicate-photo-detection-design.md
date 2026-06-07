# Duplicate Photo Detection Spec

This document details the design for a duplicate photo detector (`detect_duplicates.py`) to find and clean up duplicate photos in `./extracted` based on image content.

## Goals
- Detect near-duplicate photos based on visual content (ignoring filename, formatting, or resolution differences).
- Preserve the highest resolution and best quality copy of each photo in `./extracted`.
- Move duplicates to `./extracted/duplicates/` (keeping sidecars if present).
- Run efficiently on hundreds of images.

## Design

### 1. Hashing and Geometry Collection
- Scan `extracted/` for JPEG images (matching `*.jpg`, `*.jpeg`, `*.png`).
- For each image, read it with OpenCV and collect:
  - Resolution: `width * height`
  - File size: in bytes
  - **dHash (Difference Hash)**:
    - Resize image to 9x8 and convert to grayscale.
    - Compute horizontal difference: `diff = pixels[:, 1:] > pixels[:, :-1]`.
    - Pack the resulting 64 boolean values into a 64-bit unsigned integer.

### 2. Similarity Grouping
- Compute the Hamming distance between hashes of all image pairs: `bin(h1 ^ h2).count('1')`.
- Build a graph where edges connect image pairs with a Hamming distance $\le 2$ bits (configurable via `--threshold`).
- Run Union-Find (Disjoint Set Union) to find connected components of duplicates.

### 3. Resolution & Tie-Breaking
For each duplicate group:
- Sort the images by:
  1. Resolution (`width * height`) descending.
  2. File size (bytes) descending.
  3. Filename alphabetically ascending (to ensure deterministic tie-breaking).
- The first image is the **primary copy** and is kept in `./extracted`.
- All subsequent images are **duplicate copies** and are moved to `./extracted/duplicates/`.
- If a duplicate copy has an associated `<filename>.faces.json` sidecar, that sidecar is moved to `./extracted/duplicates/` as well.

### 4. CLI Interface
The script `detect_duplicates.py` will support the following options:
- `--dir`: Directory containing images (default: `extracted`).
- `--threshold`: Hamming distance threshold for grouping (default: `2`).
- `--dry-run`: Log duplicates and actions without moving files.
