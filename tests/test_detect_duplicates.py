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
