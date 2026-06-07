# Duplicate Photo Detection Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Implement a command-line tool `detect_duplicates.py` to identify visually identical photo crops in `./extracted`, retain the highest-resolution copy of each group, and move duplicates (along with their sidecars) to `./extracted/duplicates/`.

**Architecture:** Use Difference Hashing (dHash) to generate 64-bit image content signatures, compute Hamming distances between all image pairs, find connected components of near-duplicates using Union-Find, sort groups by resolution/size, and move duplicates.

**Tech Stack:** Python 3, OpenCV (`cv2`), NumPy, standard library `os`, `shutil`, `argparse`, and `pytest`.

---

### Task 1: Implement dHash and Union-Find Helpers

**Files:**
- Create: `detect_duplicates.py`
- Create: `tests/test_detect_duplicates.py`

- [ ] **Step 1: Write the failing tests for dHash and Union-Find**

Write the following code to `tests/test_detect_duplicates.py`:
```python
import os
import numpy as np
import pytest
from detect_duplicates import compute_dhash, UnionFind

def test_union_find():
    uf = UnionFind([1, 2, 3, 4])
    uf.union(1, 2)
    uf.union(2, 3)
    assert uf.find(1) == uf.find(3)
    assert uf.find(1) != uf.find(4)

def test_compute_dhash(tmp_path):
    import cv2
    # Create two different images
    img1_path = os.path.join(tmp_path, "img1.jpg")
    img2_path = os.path.join(tmp_path, "img2.jpg")
    
    # 100x100 white image
    img1 = np.ones((100, 100, 3), dtype=np.uint8) * 255
    # 100x100 white image with black stripe
    img2 = np.ones((100, 100, 3), dtype=np.uint8) * 255
    img2[:, :50] = 0
    
    cv2.imwrite(img1_path, img1)
    cv2.imwrite(img2_path, img2)
    
    h1 = compute_dhash(img1_path)
    h2 = compute_dhash(img2_path)
    
    assert h1 is not None
    assert h2 is not None
    assert h1 != h2
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/python -m pytest tests/test_detect_duplicates.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'detect_duplicates'`

- [ ] **Step 3: Write minimal implementation**

Write the following code to `detect_duplicates.py`:
```python
import os
import cv2
import numpy as np

def compute_dhash(image_path: str, hash_size: int = 8) -> int | None:
    img = cv2.imread(image_path)
    if img is None:
        return None
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    resized = cv2.resize(gray, (hash_size + 1, hash_size), interpolation=cv2.INTER_AREA)
    diff = resized[:, 1:] > resized[:, :-1]
    hash_val = 0
    for bit in diff.flatten():
        hash_val = (hash_val << 1) | int(bit)
    return hash_val

class UnionFind:
    def __init__(self, items):
        self.parent = {item: item for item in items}
    
    def find(self, item):
        path = []
        while self.parent[item] != item:
            path.append(item)
            item = self.parent[item]
        for node in path:
            self.parent[node] = item
        return item
    
    def union(self, item1, item2):
        root1 = self.find(item1)
        root2 = self.find(item2)
        if root1 != root2:
            self.parent[root1] = root2
```

- [ ] **Step 4: Run test to verify it passes**

Run: `.venv/bin/python -m pytest tests/test_detect_duplicates.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add detect_duplicates.py tests/test_detect_duplicates.py
git commit -m "feat: add dhash computation and union-find helper"
```

---

### Task 2: Implement Duplicate Grouping Logic

**Files:**
- Modify: `detect_duplicates.py`
- Modify: `tests/test_detect_duplicates.py`

- [ ] **Step 1: Write a failing test for duplicate grouping**

Add to `tests/test_detect_duplicates.py`:
```python
from detect_duplicates import find_duplicate_groups

def test_find_duplicate_groups(tmp_path):
    import cv2
    img1_path = os.path.join(tmp_path, "img1.jpg")
    img2_path = os.path.join(tmp_path, "img2.jpg") # duplicate of 1
    img3_path = os.path.join(tmp_path, "img3.jpg") # different
    
    img1 = np.ones((100, 100, 3), dtype=np.uint8) * 128
    img2 = np.ones((50, 50, 3), dtype=np.uint8) * 128 # resized version
    img3 = np.ones((100, 100, 3), dtype=np.uint8) * 255
    
    cv2.imwrite(img1_path, img1)
    cv2.imwrite(img2_path, img2)
    cv2.imwrite(img3_path, img3)
    
    groups, file_meta = find_duplicate_groups(str(tmp_path), threshold=2)
    
    # We expect img1 and img2 to be grouped as duplicates, and img3 to not be grouped (or in a group of size 1, which we filter out)
    assert len(groups) == 1
    dup_group = groups[0]
    assert img1_path in dup_group
    assert img2_path in dup_group
    assert img3_path not in dup_group
    
    assert img1_path in file_meta
    assert file_meta[img1_path]["width"] == 100
    assert file_meta[img2_path]["width"] == 50
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/python -m pytest tests/test_detect_duplicates.py -v`
Expected: FAIL with `ImportError: cannot import name 'find_duplicate_groups'`

- [ ] **Step 3: Write minimal implementation**

Append to `detect_duplicates.py`:
```python
import glob

def find_duplicate_groups(images_dir: str, threshold: int = 2) -> tuple[list[list[str]], dict[str, dict]]:
    file_patterns = [os.path.join(images_dir, ext) for ext in ("*.jpg", "*.jpeg", "*.png")]
    img_paths = []
    for pattern in file_patterns:
        img_paths.extend(glob.glob(pattern))
    img_paths = sorted(list(set(img_paths)))
    
    hashes = {}
    file_meta = {}
    
    for path in img_paths:
        h = compute_dhash(path)
        if h is None:
            continue
        img = cv2.imread(path)
        if img is None:
            continue
        h_dim, w_dim = img.shape[:2]
        hashes[path] = h
        file_meta[path] = {
            "width": w_dim,
            "height": h_dim,
            "size": os.path.getsize(path)
        }
        
    uf = UnionFind(hashes.keys())
    keys = list(hashes.keys())
    for i in range(len(keys)):
        for j in range(i + 1, len(keys)):
            dist = bin(hashes[keys[i]] ^ hashes[keys[j]]).count('1')
            if dist <= threshold:
                uf.union(keys[i], keys[j])
                
    groups_map = {}
    for path in hashes.keys():
        root = uf.find(path)
        groups_map.setdefault(root, []).append(path)
        
    duplicate_groups = [grp for grp in groups_map.values() if len(grp) > 1]
    return duplicate_groups, file_meta
```

- [ ] **Step 4: Run test to verify it passes**

Run: `.venv/bin/python -m pytest tests/test_detect_duplicates.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add detect_duplicates.py tests/test_detect_duplicates.py
git commit -m "feat: add duplicate grouping logic"
```

---

### Task 3: Implement Duplicate Resolution and Movement

**Files:**
- Modify: `detect_duplicates.py`
- Modify: `tests/test_detect_duplicates.py`

- [ ] **Step 1: Write a failing test for resolution and file movement**

Add to `tests/test_detect_duplicates.py`:
```python
from detect_duplicates import resolve_duplicates

def test_resolve_duplicates(tmp_path):
    # Set up files
    img1_path = os.path.join(tmp_path, "img1.jpg") # larger size/resolution
    img2_path = os.path.join(tmp_path, "img2.jpg") # smaller size/resolution
    
    with open(img1_path, "wb") as f:
        f.write(b"img1_data_large_fake")
    with open(img2_path, "wb") as f:
        f.write(b"img2_data")
        
    # Create fake sidecar for img2
    sidecar_path = os.path.join(tmp_path, "img2.faces.json")
    with open(sidecar_path, "w") as f:
        f.write('{"test": true}')
        
    groups = [[img1_path, img2_path]]
    file_meta = {
        img1_path: {"width": 100, "height": 100, "size": 20},
        img2_path: {"width": 50, "height": 50, "size": 9}
    }
    
    moved = resolve_duplicates(groups, file_meta, dry_run=False)
    
    # We expect img1 to be kept, img2 and its sidecar to be moved to duplicates/
    assert len(moved) == 1
    assert moved[0] == (img2_path, os.path.join(tmp_path, "duplicates", "img2.jpg"))
    
    assert os.path.exists(img1_path)
    assert not os.path.exists(img2_path)
    assert not os.path.exists(sidecar_path)
    assert os.path.exists(os.path.join(tmp_path, "duplicates", "img2.jpg"))
    assert os.path.exists(os.path.join(tmp_path, "duplicates", "img2.faces.json"))
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/python -m pytest tests/test_detect_duplicates.py -v`
Expected: FAIL with `ImportError: cannot import name 'resolve_duplicates'`

- [ ] **Step 3: Write minimal implementation**

Append to `detect_duplicates.py`:
```python
import shutil

def resolve_duplicates(duplicate_groups: list[list[str]], file_meta: dict[str, dict], dry_run: bool = False) -> list[tuple[str, str]]:
    moved_files = []
    for grp in duplicate_groups:
        # Sort group by: resolution desc, size desc, filename asc
        grp_sorted = sorted(grp, key=lambda p: (
            -(file_meta[p]["width"] * file_meta[p]["height"]),
            -file_meta[p]["size"],
            p
        ))
        
        primary = grp_sorted[0]
        duplicates = grp_sorted[1:]
        
        print(f"Keeping primary: {os.path.basename(primary)} ({file_meta[primary]['width']}x{file_meta[primary]['height']}, {file_meta[primary]['size']} bytes)")
        
        for dup in duplicates:
            dest_dir = os.path.join(os.path.dirname(dup), "duplicates")
            dest_path = os.path.join(dest_dir, os.path.basename(dup))
            print(f"  Duplicate to move: {os.path.basename(dup)} ({file_meta[dup]['width']}x{file_meta[dup]['height']}, {file_meta[dup]['size']} bytes) -> {dest_path}")
            
            # Check sidecar
            stem, _ = os.path.splitext(dup)
            sidecar = stem + ".faces.json"
            
            if not dry_run:
                os.makedirs(dest_dir, exist_ok=True)
                shutil.move(dup, dest_path)
                if os.path.exists(sidecar):
                    shutil.move(sidecar, os.path.join(dest_dir, os.path.basename(sidecar)))
            
            moved_files.append((dup, dest_path))
            
    return moved_files
```

- [ ] **Step 4: Run test to verify it passes**

Run: `.venv/bin/python -m pytest tests/test_detect_duplicates.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add detect_duplicates.py tests/test_detect_duplicates.py
git commit -m "feat: add resolution sorting and file moving logic"
```

---

### Task 4: Implement CLI interface and script execution

**Files:**
- Modify: `detect_duplicates.py`
- Modify: `tests/test_detect_duplicates.py`

- [ ] **Step 1: Write a failing test for argument parsing**

Add to `tests/test_detect_duplicates.py`:
```python
import subprocess

def test_cli_dry_run(tmp_path):
    import cv2
    img1_path = os.path.join(tmp_path, "img1.jpg")
    img2_path = os.path.join(tmp_path, "img2.jpg")
    
    img1 = np.ones((100, 100, 3), dtype=np.uint8) * 128
    cv2.imwrite(img1_path, img1)
    cv2.imwrite(img2_path, img1) # identical copy
    
    # Run script in dry-run mode
    cmd = [os.path.abspath(".venv/bin/python"), os.path.abspath("detect_duplicates.py"), "--dir", str(tmp_path), "--dry-run"]
    res = subprocess.run(cmd, capture_output=True, text=True, check=True)
    
    assert "Keeping primary" in res.stdout
    assert "Duplicate to move" in res.stdout
    assert os.path.exists(img1_path)
    assert os.path.exists(img2_path) # should not be moved in dry-run
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/python -m pytest tests/test_detect_duplicates.py -v`
Expected: FAIL (subprocess will run but it won't do anything because `__main__` is not implemented yet in `detect_duplicates.py`).

- [ ] **Step 3: Write minimal implementation**

Append to `detect_duplicates.py`:
```python
import argparse

def main():
    parser = argparse.ArgumentParser(description="Detect and clean up duplicate images.")
    parser.add_argument("--dir", default="extracted", help="Directory containing images to process (default: 'extracted')")
    parser.add_argument("--threshold", type=int, default=2, help="Hamming distance threshold for duplicates (default: 2)")
    parser.add_argument("--dry-run", action="store_true", help="Log duplicate files without moving them")
    
    args = parser.parse_args()
    
    if not os.path.exists(args.dir):
        print(f"Directory '{args.dir}' does not exist.")
        return 1
        
    print(f"Scanning for duplicates in '{args.dir}' with threshold {args.threshold}...")
    groups, file_meta = find_duplicate_groups(args.dir, threshold=args.threshold)
    
    if not groups:
        print("No duplicates found.")
        return 0
        
    print(f"Found {len(groups)} group(s) of duplicates:")
    moved = resolve_duplicates(groups, file_meta, dry_run=args.dry_run)
    
    if args.dry_run:
        print(f"[DRY RUN] Would have moved {len(moved)} duplicate file(s).")
    else:
        print(f"Successfully moved {len(moved)} duplicate file(s) to '{args.dir}/duplicates/'.")
    return 0

if __name__ == "__main__":
    import sys
    sys.exit(main())
```

- [ ] **Step 4: Run test to verify it passes**

Run: `.venv/bin/python -m pytest tests/test_detect_duplicates.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add detect_duplicates.py tests/test_detect_duplicates.py
git commit -m "feat: implement CLI interface and main entrypoint"
```
