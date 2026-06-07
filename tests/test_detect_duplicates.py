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

def test_find_duplicate_groups(tmp_path):
    import cv2
    from detect_duplicates import find_duplicate_groups
    img1_path = os.path.join(tmp_path, "img1.jpg")
    img2_path = os.path.join(tmp_path, "img2.jpg") # duplicate of 1
    img3_path = os.path.join(tmp_path, "img3.jpg") # different
    
    img1 = np.ones((100, 100, 3), dtype=np.uint8) * 128
    img2 = np.ones((50, 50, 3), dtype=np.uint8) * 128 # resized version
    img3 = np.ones((100, 100, 3), dtype=np.uint8) * 128
    img3[:, :50] = 0 # add stripe to make hash different
    
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

