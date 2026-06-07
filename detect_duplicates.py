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



