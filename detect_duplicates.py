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

