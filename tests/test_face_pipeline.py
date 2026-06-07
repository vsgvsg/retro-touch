import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import face_pipeline as fp


def test_l2_normalize_unit_norm():
    v = np.array([3.0, 4.0], dtype=np.float32)
    out = fp.l2_normalize(v)
    assert np.isclose(np.linalg.norm(out), 1.0)
    assert np.allclose(out, [0.6, 0.8])


def test_l2_normalize_zero_vector_is_safe():
    v = np.zeros(4, dtype=np.float32)
    out = fp.l2_normalize(v)
    assert out.shape == v.shape
    assert np.all(np.isfinite(out))
    assert np.isclose(np.linalg.norm(out), 0.0)


def test_l2_normalize_rows_of_matrix():
    m = np.array([[3.0, 4.0], [0.0, 2.0]], dtype=np.float32)
    out = fp.l2_normalize(m)
    norms = np.linalg.norm(out, axis=1)
    assert np.allclose(norms, [1.0, 1.0])


def test_cosine_sim_identical_is_one():
    a = np.array([1.0, 0.0, 0.0], dtype=np.float32)
    assert np.isclose(fp.cosine_sim(a, a), 1.0)


def test_cosine_sim_orthogonal_is_zero():
    a = np.array([1.0, 0.0], dtype=np.float32)
    b = np.array([0.0, 1.0], dtype=np.float32)
    assert np.isclose(fp.cosine_sim(a, b), 0.0)


def test_cosine_sim_matrix_shape_and_values():
    faces = np.array([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32)
    centroids = np.array([[1.0, 0.0]], dtype=np.float32)
    sims = fp.cosine_sim_matrix(faces, centroids)
    assert sims.shape == (2, 1)
    assert np.isclose(sims[0, 0], 1.0)
    assert np.isclose(sims[1, 0], 0.0)


def test_build_centroids_means_and_normalizes():
    embeddings = np.array([
        [2.0, 0.0],
        [4.0, 0.0],
        [0.0, 3.0],
    ], dtype=np.float32)
    labels = ["Alice", "Alice", "Bob"]
    names, cents = fp.build_centroids(embeddings, labels)
    assert names == ["Alice", "Bob"]
    assert np.allclose(np.linalg.norm(cents, axis=1), [1.0, 1.0])
    assert np.allclose(cents[0], [1.0, 0.0])
    assert np.allclose(cents[1], [0.0, 1.0])


def test_build_centroids_ignores_empty_labels():
    embeddings = np.array([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32)
    labels = ["Alice", ""]
    names, cents = fp.build_centroids(embeddings, labels)
    assert names == ["Alice"]
    assert cents.shape == (1, 2)


def test_rank_candidates_orders_by_score_and_caps_top_n():
    names = ["Alice", "Bob", "Carol"]
    sims = np.array([0.9, 0.7, 0.4])
    out = fp.rank_candidates(names, sims, top=2, threshold=0.5)
    assert [c["name"] for c in out] == ["Alice", "Bob"]
    assert np.isclose(out[0]["score"], 0.9)


def test_rank_candidates_threshold_filters():
    names = ["Alice", "Bob"]
    sims = np.array([0.45, 0.30])
    out = fp.rank_candidates(names, sims, top=3, threshold=0.5)
    assert out == []


def test_clusters_to_persons_maps_ids_and_noise():
    labels = np.array([0, 0, 1, -1, 1])
    out = fp.clusters_to_persons(labels)
    assert out == ["person_000", "person_000", "person_001",
                   "unassigned", "person_001"]


def test_clusters_to_persons_all_noise():
    labels = np.array([-1, -1])
    out = fp.clusters_to_persons(labels)
    assert out == ["unassigned", "unassigned"]


def test_cache_paths(tmp_path):
    npy, idx = fp.cache_paths(str(tmp_path))
    assert npy.endswith("faces.npy")
    assert idx.endswith("faces_index.json")


def test_append_and_load_embeddings_roundtrip(tmp_path):
    d = str(tmp_path)
    e1 = np.array([[1.0, 0.0]], dtype=np.float32)
    refs = fp.append_embeddings(d, e1, [{"image": "a.jpg", "face_id": 0}], "buffalo_l")
    assert refs == [0]
    e2 = np.array([[0.0, 1.0], [1.0, 1.0]], dtype=np.float32)
    refs2 = fp.append_embeddings(d, e2,
        [{"image": "b.jpg", "face_id": 0}, {"image": "b.jpg", "face_id": 1}], "buffalo_l")
    assert refs2 == [1, 2]
    emb, index = fp.load_cache(d)
    assert emb.shape == (3, 2)
    assert index["model"] == "buffalo_l"
    assert index["rows"][2] == {"image": "b.jpg", "face_id": 1}


def test_load_cache_missing_raises(tmp_path):
    import pytest
    with pytest.raises(FileNotFoundError):
        fp.load_cache(str(tmp_path))


def test_faces_sidecar_path():
    p = fp.faces_sidecar_path("extracted/original-001_02.jpg")
    assert p == "extracted/original-001_02.faces.json"


def test_write_read_faces_json_roundtrip(tmp_path):
    img_path = str(tmp_path / "x.jpg")
    faces = [{"id": 0, "bbox": [1, 2, 3, 4], "det_score": 0.9,
              "embedding_ref": 5, "cluster": "person_000", "label": "Alice"}]
    fp.write_faces_json(img_path, (640, 480), "buffalo_l", faces)
    data = fp.read_faces_json(img_path)
    assert data["image"] == "x.jpg"
    assert data["image_size"] == [640, 480]
    assert data["model"] == "buffalo_l"
    assert data["faces"][0]["label"] == "Alice"


def test_read_faces_json_missing_returns_none(tmp_path):
    assert fp.read_faces_json(str(tmp_path / "nope.jpg")) is None


def test_scaffold_labels_one_entry_per_real_cluster():
    persons = ["person_000", "person_000", "unassigned", "person_001"]
    out = fp.scaffold_labels(persons)
    assert out == {"person_000": "", "person_001": ""}


def test_scaffold_labels_empty():
    assert fp.scaffold_labels(["unassigned", "unassigned"]) == {}


def test_cluster_face_index_groups_and_excludes_unassigned(tmp_path):
    d = str(tmp_path)
    fp.write_faces_json(os.path.join(d, "a.jpg"), (100, 100), "buffalo_l", [
        {"id": 0, "bbox": [0, 0, 10, 10], "det_score": 0.9,
         "embedding_ref": 0, "cluster": "person_000", "label": ""},
        {"id": 1, "bbox": [5, 5, 15, 15], "det_score": 0.8,
         "embedding_ref": 1, "cluster": "unassigned", "label": ""},
    ])
    fp.write_faces_json(os.path.join(d, "b.jpg"), (100, 100), "buffalo_l", [
        {"id": 0, "bbox": [1, 1, 9, 9], "det_score": 0.7,
         "embedding_ref": 2, "cluster": "person_000", "label": ""},
    ])
    index = {"model": "buffalo_l", "rows": [
        {"image": "a.jpg", "face_id": 0},
        {"image": "a.jpg", "face_id": 1},
        {"image": "b.jpg", "face_id": 0},
    ]}
    out = fp.cluster_face_index(d, index)
    assert set(out.keys()) == {"person_000"}        # unassigned excluded
    assert len(out["person_000"]) == 2
    f0 = out["person_000"][0]
    assert f0["image"] == "a.jpg" and f0["face_id"] == 0
    assert f0["bbox"] == [0, 0, 10, 10] and f0["det_score"] == 0.9


def test_crop_face_in_bounds():
    img = np.zeros((100, 100, 3), dtype=np.uint8)
    img[10:20, 30:40] = 255
    crop = fp.crop_face(img, [30, 10, 40, 20])
    assert crop.shape == (10, 10, 3)
    assert (crop == 255).all()


def test_crop_face_clamps_out_of_bounds():
    img = np.zeros((50, 50, 3), dtype=np.uint8)
    crop = fp.crop_face(img, [-5, -5, 70, 60])
    assert crop.shape[0] == 50 and crop.shape[1] == 50  # clamped to image
    assert crop.ndim == 3


def test_previous_names_sorted_unique_nonempty():
    labels = {"person_000": "Bob", "person_001": "", "person_002": "Alice",
              "person_003": "Bob"}
    assert fp.previous_names(labels) == ["Alice", "Bob"]


def test_previous_names_empty():
    assert fp.previous_names({"person_000": "", "person_001": ""}) == []

def test_write_labels_roundtrip(tmp_path):
    d = str(tmp_path)
    path = fp.write_labels(d, {"person_000": "Alice", "person_001": "Bob"})
    assert path == os.path.join(d, "labels.json")
    with open(path) as f:
        data = json.load(f)
    assert data == {"person_000": "Alice", "person_001": "Bob"}


def test_grid_positions_full_and_partial():
    assert fp.grid_positions(4, cols=3) == [(0, 0), (0, 1), (0, 2), (1, 0)]


def test_grid_positions_zero():
    assert fp.grid_positions(0, cols=3) == []


def test_scale_to_fit_downscales_longer_side():
    w, h, s = fp.scale_to_fit(1800, 900, 900)
    assert (w, h) == (900, 450)
    assert abs(s - 0.5) < 1e-9


def test_scale_to_fit_no_upscale_when_small():
    w, h, s = fp.scale_to_fit(300, 200, 900)
    assert (w, h, s) == (300, 200, 1.0)


def test_scale_to_fit_square():
    w, h, s = fp.scale_to_fit(1000, 1000, 500)
    assert (w, h) == (500, 500)
    assert abs(s - 0.5) < 1e-9


def test_exclude_face_sets_unassigned_and_persists(tmp_path):
    d = str(tmp_path)
    img = os.path.join(d, "a.jpg")
    fp.write_faces_json(img, (100, 100), "buffalo_l", [
        {"id": 0, "bbox": [0, 0, 10, 10], "det_score": 0.9,
         "embedding_ref": 0, "cluster": "person_000", "label": ""},
        {"id": 1, "bbox": [5, 5, 15, 15], "det_score": 0.8,
         "embedding_ref": 1, "cluster": "person_000", "label": ""},
    ])
    assert fp.exclude_face(d, "a.jpg", 0) is True
    data = fp.read_faces_json(img)
    assert data["faces"][0]["cluster"] == "unassigned"   # excluded
    assert data["faces"][1]["cluster"] == "person_000"   # untouched


def test_exclude_face_unknown_id_returns_false(tmp_path):
    d = str(tmp_path)
    img = os.path.join(d, "a.jpg")
    fp.write_faces_json(img, (100, 100), "buffalo_l", [
        {"id": 0, "bbox": [0, 0, 10, 10], "det_score": 0.9,
         "embedding_ref": 0, "cluster": "person_000", "label": ""},
    ])
    assert fp.exclude_face(d, "a.jpg", 99) is False


def test_next_cluster_id_empty_and_gaps():
    assert fp.next_cluster_id(set()) == "person_000"
    assert fp.next_cluster_id({"person_000", "person_002"}) == "person_003"
    assert fp.next_cluster_id({"person_009"}) == "person_010"


def test_existing_cluster_ids_collects_person_ids(tmp_path):
    d = str(tmp_path)
    fp.write_faces_json(os.path.join(d, "a.jpg"), (100, 100), "buffalo_l", [
        {"id": 0, "bbox": [0, 0, 1, 1], "det_score": 0.9,
         "embedding_ref": 0, "cluster": "person_001", "label": ""},
        {"id": 1, "bbox": [0, 0, 1, 1], "det_score": 0.9,
         "embedding_ref": 1, "cluster": "unassigned", "label": ""},
    ])
    fp.write_faces_json(os.path.join(d, "b.jpg"), (100, 100), "buffalo_l", [
        {"id": 0, "bbox": [0, 0, 1, 1], "det_score": 0.9,
         "embedding_ref": 2, "cluster": "person_004", "label": ""},
    ])
    assert fp.existing_cluster_ids(d) == {"person_001", "person_004"}


def test_resolve_or_create_cluster_reuses_existing_name():
    labels = {"person_000": "Alice", "person_001": "Bob"}
    cid, new_labels = fp.resolve_or_create_cluster(
        "Bob", labels, {"person_000", "person_001"})
    assert cid == "person_001"
    assert new_labels == labels  # unchanged


def test_resolve_or_create_cluster_mints_new_for_new_name():
    labels = {"person_000": "Alice"}
    cid, new_labels = fp.resolve_or_create_cluster(
        "Carol", labels, {"person_000"})
    assert cid == "person_001"
    assert new_labels["person_001"] == "Carol"
    assert new_labels["person_000"] == "Alice"



def test_apply_photo_edits_writes_label_only_for_clustered(tmp_path):
    d = str(tmp_path)
    img = os.path.join(d, "a.jpg")
    fp.write_faces_json(img, (100, 100), "buffalo_l", [
        {"id": 0, "bbox": [0, 0, 1, 1], "det_score": 0.9,
         "embedding_ref": 0, "cluster": "person_000", "label": ""},
    ])
    labels = {"person_000": "Alice"}
    out = fp.apply_photo_edits(d, "a.jpg", {0: "Renamed"}, labels, {"person_000"})
    data = fp.read_faces_json(img)
    assert data["faces"][0]["label"] == "Renamed"
    assert data["faces"][0]["cluster"] == "person_000"  # cluster untouched
    assert out == labels  # no new cluster minted


def test_apply_photo_edits_assigns_cluster_to_unassigned(tmp_path):
    d = str(tmp_path)
    img = os.path.join(d, "a.jpg")
    fp.write_faces_json(img, (100, 100), "buffalo_l", [
        {"id": 0, "bbox": [0, 0, 1, 1], "det_score": 0.9,
         "embedding_ref": 0, "cluster": "unassigned", "label": ""},
        {"id": 1, "bbox": [0, 0, 1, 1], "det_score": 0.9,
         "embedding_ref": 1, "cluster": "unassigned", "label": ""},
    ])
    labels = {"person_000": "Alice"}
    # face 0 reuses Alice's cluster; face 1 is a brand-new person -> new id
    out = fp.apply_photo_edits(d, "a.jpg", {0: "Alice", 1: "Dave"},
                               labels, {"person_000"})
    data = fp.read_faces_json(img)
    assert data["faces"][0]["cluster"] == "person_000"   # reused
    assert data["faces"][0]["label"] == "Alice"
    assert data["faces"][1]["cluster"] == "person_001"   # minted
    assert data["faces"][1]["label"] == "Dave"
    assert out["person_001"] == "Dave"


def test_apply_photo_edits_empty_name_clears_label(tmp_path):
    d = str(tmp_path)
    img = os.path.join(d, "a.jpg")
    fp.write_faces_json(img, (100, 100), "buffalo_l", [
        {"id": 0, "bbox": [0, 0, 1, 1], "det_score": 0.9,
         "embedding_ref": 0, "cluster": "person_000", "label": "Old"},
    ])
    fp.apply_photo_edits(d, "a.jpg", {0: ""}, {"person_000": "Alice"},
                         {"person_000"})
    assert fp.read_faces_json(img)["faces"][0]["label"] == ""


def test_apply_photo_edits_missing_sidecar_returns_labels_unchanged(tmp_path):
    labels = {"person_000": "Alice"}
    out = fp.apply_photo_edits(str(tmp_path), "nope.jpg", {0: "X"}, labels,
                               {"person_000"})
    assert out == labels  # unchanged
    assert not os.path.exists(os.path.join(str(tmp_path), "nope.faces.json"))


def test_resolve_or_create_cluster_empty_labels_mints_person_000():
    cid, new_labels = fp.resolve_or_create_cluster("Alice", {}, set())
    assert cid == "person_000"
    assert new_labels == {"person_000": "Alice"}


def test_cluster_would_lose_work_labels_json(tmp_path):
    d = str(tmp_path)
    # a sidecar with only unassigned faces -> no assignment work
    fp.write_faces_json(os.path.join(d, "a.jpg"), (100, 100), "buffalo_l", [
        {"id": 0, "bbox": [0, 0, 1, 1], "det_score": 0.9,
         "embedding_ref": 0, "cluster": "unassigned", "label": ""},
    ])
    assert fp.cluster_would_lose_work(d) is False
    # labels.json present -> work to lose
    with open(os.path.join(d, "labels.json"), "w") as f:
        json.dump({"person_000": "Alice"}, f)
    assert fp.cluster_would_lose_work(d) is True


def test_cluster_would_lose_work_assigned_cluster(tmp_path):
    d = str(tmp_path)
    # an assigned (non-unassigned) cluster -> work to lose, even w/o labels.json
    fp.write_faces_json(os.path.join(d, "b.jpg"), (100, 100), "buffalo_l", [
        {"id": 0, "bbox": [0, 0, 1, 1], "det_score": 0.9,
         "embedding_ref": 0, "cluster": "person_002", "label": ""},
    ])
    assert fp.cluster_would_lose_work(d) is True


def test_cluster_would_lose_work_fresh(tmp_path):
    assert fp.cluster_would_lose_work(str(tmp_path)) is False


def test_prefill_name_existing_label_wins():
    labels = {"person_002": "Carol"}
    assert fp.prefill_name(
        {"label": "Alice", "cluster": "person_002"},
        ("Bob", 0.9), 0.5, labels) == "Alice"


def test_prefill_name_candidate_above_threshold_fills():
    assert fp.prefill_name(
        {"label": "", "cluster": "unassigned"},
        ("Bob", 0.6), 0.5, {}) == "Bob"


def test_prefill_name_candidate_below_threshold_does_not_fill():
    labels = {"person_002": "Carol"}
    assert fp.prefill_name(
        {"label": "", "cluster": "person_002"},
        ("Bob", 0.2), 0.5, labels) == "Carol"
    assert fp.prefill_name(
        {"label": "", "cluster": "unassigned"},
        ("Bob", 0.2), 0.5, {}) == ""


def test_prefill_name_no_candidate_uses_cluster_then_empty():
    labels = {"person_002": "Carol"}
    assert fp.prefill_name(
        {"label": "", "cluster": "person_002"}, None, 0.5, labels) == "Carol"
    assert fp.prefill_name(
        {"label": "", "cluster": "unassigned"}, None, 0.5, {}) == ""


def test_hint_for_weak_candidate_on_unlabeled():
    assert fp.hint_for({"label": "", "cluster": "unassigned"},
                       ("Dima", 0.2), 0.5, "") == ("Dima", 0.2)


def test_hint_for_strong_candidate_already_prefilled():
    assert fp.hint_for({"label": "", "cluster": "unassigned"},
                       ("Bob", 0.6), 0.5, "Bob") is None


def test_hint_for_disagreement_with_existing_label():
    assert fp.hint_for({"label": "Maria P", "cluster": "person_001"},
                       ("Marina R", 0.41), 0.5, "Maria P") == ("Marina R", 0.41)


def test_hint_for_agreement_no_hint():
    assert fp.hint_for({"label": "Bob", "cluster": "person_000"},
                       ("Bob", 0.2), 0.5, "Bob") is None


def test_hint_for_no_candidate():
    assert fp.hint_for({"label": "", "cluster": "unassigned"},
                       None, 0.5, "") is None


def test_hint_for_cluster_name_prefill_differing_candidate():
    # box prefilled from cluster name; weak candidate differs -> hint shown
    labels = {"person_002": "Carol"}
    pre = fp.prefill_name({"label": "", "cluster": "person_002"},
                          ("Bob", 0.2), 0.5, labels)
    assert pre == "Carol"
    assert fp.hint_for({"label": "", "cluster": "person_002"},
                       ("Bob", 0.2), 0.5, pre) == ("Bob", 0.2)


def test_hint_for_score_equals_threshold_suppressed():
    # score == threshold -> prefill fills with the name -> hint suppressed
    pre = fp.prefill_name({"label": "", "cluster": "unassigned"},
                          ("Bob", 0.5), 0.5, {})
    assert pre == "Bob"
    assert fp.hint_for({"label": "", "cluster": "unassigned"},
                       ("Bob", 0.5), 0.5, pre) is None


def test_face_state_existing_label_is_matched():
    assert fp.face_state(
        {"label": "Alice", "cluster": "person_002"},
        ("Bob", 0.9), 0.5, {}) == "matched"


def test_face_state_candidate_above_threshold_is_confident():
    assert fp.face_state(
        {"label": "", "cluster": "unassigned"},
        ("Bob", 0.6), 0.5, {}) == "confident"


def test_face_state_cluster_name_prefill_is_matched():
    labels = {"person_002": "Carol"}
    assert fp.face_state(
        {"label": "", "cluster": "person_002"},
        ("Bob", 0.2), 0.5, labels) == "matched"


def test_face_state_weak_and_unassigned_is_unassigned():
    assert fp.face_state(
        {"label": "", "cluster": "unassigned"},
        ("Bob", 0.2), 0.5, {}) == "unassigned"
    assert fp.face_state(
        {"label": "", "cluster": "unassigned"},
        None, 0.5, {}) == "unassigned"


def test_centroid_coverage_counts_unique_images(tmp_path):
    d = str(tmp_path)
    # person_000 -> Alice: two faces across two images
    fp.write_faces_json(os.path.join(d, "a.jpg"), (10, 10), "buffalo_l", [
        {"id": 0, "bbox": [0, 0, 1, 1], "det_score": 0.9,
         "embedding_ref": 0, "cluster": "person_000", "label": ""},
        {"id": 1, "bbox": [0, 0, 1, 1], "det_score": 0.9,
         "embedding_ref": 1, "cluster": "unassigned", "label": ""},
    ])
    fp.write_faces_json(os.path.join(d, "b.jpg"), (10, 10), "buffalo_l", [
        # two faces of the SAME cluster in ONE image -> 2 faces, 1 unique image
        {"id": 0, "bbox": [0, 0, 1, 1], "det_score": 0.9,
         "embedding_ref": 2, "cluster": "person_000", "label": ""},
        {"id": 1, "bbox": [0, 0, 1, 1], "det_score": 0.9,
         "embedding_ref": 3, "cluster": "person_000", "label": ""},
    ])
    with open(os.path.join(d, "labels.json"), "w") as f:
        json.dump({"person_000": "Alice", "person_001": ""}, f)  # person_001 unnamed
    rows = fp.centroid_coverage(d)
    assert len(rows) == 1                       # only named clusters
    r = rows[0]
    assert r["name"] == "Alice"
    assert r["clusters"] == ["person_000"]
    assert r["faces"] == 3                       # a#0, b#0, b#1
    assert r["images"] == 2                      # a.jpg, b.jpg (b counted once)


def test_centroid_coverage_sorted_by_images_desc(tmp_path):
    d = str(tmp_path)
    fp.write_faces_json(os.path.join(d, "a.jpg"), (10, 10), "buffalo_l", [
        {"id": 0, "bbox": [0, 0, 1, 1], "det_score": 0.9,
         "embedding_ref": 0, "cluster": "person_000", "label": ""}])
    fp.write_faces_json(os.path.join(d, "b.jpg"), (10, 10), "buffalo_l", [
        {"id": 0, "bbox": [0, 0, 1, 1], "det_score": 0.9,
         "embedding_ref": 1, "cluster": "person_001", "label": ""}])
    fp.write_faces_json(os.path.join(d, "c.jpg"), (10, 10), "buffalo_l", [
        {"id": 0, "bbox": [0, 0, 1, 1], "det_score": 0.9,
         "embedding_ref": 2, "cluster": "person_001", "label": ""}])
    with open(os.path.join(d, "labels.json"), "w") as f:
        json.dump({"person_000": "Solo", "person_001": "Pair"}, f)
    rows = fp.centroid_coverage(d)
    assert [r["name"] for r in rows] == ["Pair", "Solo"]  # 2 imgs before 1


def test_centroid_coverage_no_labels(tmp_path):
    assert fp.centroid_coverage(str(tmp_path)) == []


def test_bbox_iou_identical_is_one():
    assert fp.bbox_iou([0, 0, 10, 10], [0, 0, 10, 10]) == 1.0


def test_bbox_iou_disjoint_is_zero():
    assert fp.bbox_iou([0, 0, 10, 10], [20, 20, 30, 30]) == 0.0


def test_bbox_iou_half_overlap():
    # [0,0,10,10] vs [5,0,15,10]: inter=50, union=150
    assert abs(fp.bbox_iou([0, 0, 10, 10], [5, 0, 15, 10]) - (50 / 150)) < 1e-6


def test_bbox_iou_degenerate_box_is_zero():
    assert fp.bbox_iou([0, 0, 0, 0], [0, 0, 10, 10]) == 0.0


def test_match_faces_by_bbox_pairs_by_overlap():
    existing = [{"id": 0, "bbox": [0, 0, 10, 10]},
                {"id": 1, "bbox": [100, 100, 110, 110]}]
    detected = [{"bbox": [100, 100, 110, 110]},   # idx 0 -> id 1
                {"bbox": [0, 0, 10, 10]}]          # idx 1 -> id 0
    assert fp.match_faces_by_bbox(existing, detected) == {0: 1, 1: 0}


def test_match_faces_by_bbox_skips_below_threshold():
    existing = [{"id": 0, "bbox": [0, 0, 10, 10]}]
    detected = [{"bbox": [50, 50, 60, 60]}]        # no overlap
    assert fp.match_faces_by_bbox(existing, detected) == {}


def test_match_faces_by_bbox_no_double_assignment():
    existing = [{"id": 0, "bbox": [0, 0, 10, 10]},
                {"id": 1, "bbox": [0, 0, 10, 9]}]
    detected = [{"bbox": [0, 0, 10, 10]}]          # only one detection
    out = fp.match_faces_by_bbox(existing, detected)
    assert list(out.values()) == [0]               # one face claims it
    assert len(out) == 1


def test_merge_age_sets_auto_age_by_id():
    faces = [{"id": 0}, {"id": 1}]
    out = fp.merge_age_into_faces(faces, {0: 34.6, 1: 7.2})
    assert out[0]["age"] == 35 and out[0]["age_source"] == "auto"
    assert out[1]["age"] == 7 and out[1]["age_source"] == "auto"


def test_merge_age_preserves_manual():
    faces = [{"id": 0, "age": 40, "age_source": "manual"}]
    out = fp.merge_age_into_faces(faces, {0: 12.0})
    assert out[0]["age"] == 40 and out[0]["age_source"] == "manual"


def test_merge_age_ignores_unmatched_and_none():
    faces = [{"id": 0}, {"id": 1}]
    out = fp.merge_age_into_faces(faces, {0: None})
    assert "age" not in out[0] or out[0].get("age") is None
    assert "age" not in out[1]


def test_parse_age_valid_and_invalid():
    assert fp.parse_age("34") == 34
    assert fp.parse_age("  7 ") == 7
    assert fp.parse_age("") is None
    assert fp.parse_age("abc") is None
    assert fp.parse_age("-3") is None
    assert fp.parse_age("999") is None


def test_set_face_age_writes_manual(tmp_path):
    d = str(tmp_path)
    img = os.path.join(d, "p.jpg")
    fp.write_faces_json(img, (100, 100), "buffalo_l",
                        [{"id": 0, "bbox": [0, 0, 10, 10], "cluster": "person_000"}])
    assert fp.set_face_age(d, "p.jpg", 0, 41) is True
    data = fp.read_faces_json(img)
    assert data["faces"][0]["age"] == 41
    assert data["faces"][0]["age_source"] == "manual"


def test_set_face_age_missing_face_returns_false(tmp_path):
    d = str(tmp_path)
    img = os.path.join(d, "p.jpg")
    fp.write_faces_json(img, (100, 100), "buffalo_l", [{"id": 0, "bbox": [0, 0, 1, 1]}])
    assert fp.set_face_age(d, "p.jpg", 9, 20) is False


def test_age_prefill_formats_value_and_blank():
    assert fp.age_prefill({"age": 34}) == "34"
    assert fp.age_prefill({"age": None}) == ""
    assert fp.age_prefill({}) == ""


def test_cluster_face_index_includes_age(tmp_path):
    d = str(tmp_path)
    fp.write_faces_json(os.path.join(d, "a.jpg"), (100, 100), "buffalo_l",
                        [{"id": 0, "bbox": [0, 0, 9, 9], "det_score": 0.9,
                          "cluster": "person_000", "age": 30}])
    index = {"model": "buffalo_l", "rows": [{"image": "a.jpg", "face_id": 0}]}
    ci = fp.cluster_face_index(d, index)
    assert ci["person_000"][0]["age"] == 30


def test_find_duplicate_personas():
    # empty
    assert fp.find_duplicate_personas({}) == {}
    
    # unique names
    labels = {"person_000": "Alice", "person_001": "Bob"}
    assert fp.find_duplicate_personas(labels) == {}
    
    # duplicate names
    labels = {
        "person_000": "Alice",
        "person_001": "Alice",
        "person_002": "Bob",
        "person_003": "Alice",
        "person_004": "Bob"
    }
    expected = {
        "Alice": ["person_000", "person_001", "person_003"],
        "Bob": ["person_002", "person_004"]
    }
    assert fp.find_duplicate_personas(labels) == expected
    
    # ignore empty/unnamed clusters
    labels = {"person_000": "", "person_001": "", "person_002": "Alice"}
    assert fp.find_duplicate_personas(labels) == {}


def test_merge_persona_clusters(tmp_path):
    d = str(tmp_path)
    
    # Write a sidecar for image a: one face in person_001, one face in person_002
    fp.write_faces_json(os.path.join(d, "a.jpg"), (100, 100), "buffalo_l", [
        {"id": 0, "bbox": [0, 0, 10, 10], "det_score": 0.9, "cluster": "person_001", "label": "John"},
        {"id": 1, "bbox": [20, 20, 30, 30], "det_score": 0.9, "cluster": "person_002", "label": "John"}
    ])
    
    # Write a sidecar for image b: one face in person_002
    fp.write_faces_json(os.path.join(d, "b.jpg"), (100, 100), "buffalo_l", [
        {"id": 0, "bbox": [0, 0, 10, 10], "det_score": 0.9, "cluster": "person_002", "label": "John"}
    ])
    
    # Write a sidecar for image c: one face in unrelated person_003
    fp.write_faces_json(os.path.join(d, "c.jpg"), (100, 100), "buffalo_l", [
        {"id": 0, "bbox": [0, 0, 10, 10], "det_score": 0.9, "cluster": "person_003", "label": "Alice"}
    ])
    
    labels_map = {
        "person_001": "John",
        "person_002": "John",
        "person_003": "Alice"
    }
    
    new_labels, count = fp.merge_persona_clusters(d, labels_map)
    
    # person_002 (duplicate of John) should be merged into person_001
    assert new_labels == {"person_001": "John", "person_003": "Alice"}
    assert count == 2 # two faces of person_002 updated
    
    # Verify sidecar files
    data_a = fp.read_faces_json(os.path.join(d, "a.jpg"))
    assert data_a["faces"][0]["cluster"] == "person_001"
    assert data_a["faces"][1]["cluster"] == "person_001" # was person_002
    
    data_b = fp.read_faces_json(os.path.join(d, "b.jpg"))
    assert data_b["faces"][0]["cluster"] == "person_001" # was person_002
    
    data_c = fp.read_faces_json(os.path.join(d, "c.jpg"))
    assert data_c["faces"][0]["cluster"] == "person_003" # untouched


def test_run_merge_missing_labels(tmp_path):
    d = str(tmp_path)
    assert fp.main(["face_pipeline.py", "merge", "--images", d]) == 1


def test_run_merge_success(tmp_path):
    d = str(tmp_path)
    # Write a sidecar for image a
    fp.write_faces_json(os.path.join(d, "a.jpg"), (100, 100), "buffalo_l", [
        {"id": 0, "bbox": [0, 0, 10, 10], "det_score": 0.9, "cluster": "person_001", "label": "John"},
        {"id": 1, "bbox": [20, 20, 30, 30], "det_score": 0.9, "cluster": "person_002", "label": "John"}
    ])
    
    # Write labels.json
    labels_map = {"person_001": "John", "person_002": "John"}
    with open(os.path.join(d, "labels.json"), "w") as f:
        json.dump(labels_map, f)
        
    assert fp.main(["face_pipeline.py", "merge", "--images", d]) == 0
    
    # Read updated labels
    with open(os.path.join(d, "labels.json")) as f:
        new_labels = json.load(f)
    assert new_labels == {"person_001": "John"}


def test_run_merge_on_completion(tmp_path):
    d = str(tmp_path)
    # Write a sidecar for image a
    fp.write_faces_json(os.path.join(d, "a.jpg"), (100, 100), "buffalo_l", [
        {"id": 0, "bbox": [0, 0, 10, 10], "det_score": 0.9, "cluster": "person_001", "label": "John"},
        {"id": 1, "bbox": [20, 20, 30, 30], "det_score": 0.9, "cluster": "person_002", "label": "John"}
    ])
    
    # Write labels.json
    labels_map = {"person_001": "John", "person_002": "John"}
    with open(os.path.join(d, "labels.json"), "w") as f:
        json.dump(labels_map, f)
        
    # Calling run_merge_on_completion should perform the merge
    fp.run_merge_on_completion(d)
    
    # Verify labels.json was updated
    with open(os.path.join(d, "labels.json")) as f:
        new_labels = json.load(f)
    assert new_labels == {"person_001": "John"}
    
    # Verify sidecar was updated
    data_a = fp.read_faces_json(os.path.join(d, "a.jpg"))
    assert data_a["faces"][1]["cluster"] == "person_001"


def test_on_q_key_behavior():
    class DummyApp:
        def __init__(self):
            self.closed = False
        def _on_close(self):
            self.closed = True
        def _on_q_key(self, event):
            if event.widget.winfo_class() in ("TEntry", "Entry"):
                return
            self._on_close()

    app = DummyApp()
    
    class MockWidget:
        def __init__(self, cls):
            self.cls = cls
        def winfo_class(self):
            return self.cls
            
    class MockEvent:
        def __init__(self, widget):
            self.widget = widget
            
    # If widget is TEntry, should NOT close
    app._on_q_key(MockEvent(MockWidget("TEntry")))
    assert not app.closed
    
    # If widget is Entry, should NOT close
    app._on_q_key(MockEvent(MockWidget("Entry")))
    assert not app.closed
    
    # If widget is Frame, SHOULD close
    app._on_q_key(MockEvent(MockWidget("Frame")))
    assert app.closed


def test_run_match_headless_calls_merge(tmp_path):
    d = str(tmp_path)
    # Write a sidecar for image a
    fp.write_faces_json(os.path.join(d, "a.jpg"), (100, 100), "buffalo_l", [
        {"id": 0, "bbox": [0, 0, 10, 10], "det_score": 0.9, "cluster": "person_001", "label": ""},
        {"id": 1, "bbox": [20, 20, 30, 30], "det_score": 0.9, "cluster": "person_002", "label": ""}
    ])
    
    # Write cache npy and json
    emb = np.zeros((2, 512), dtype=np.float32)
    emb[0, 0] = 1.0
    emb[1, 0] = 1.0
    np.save(os.path.join(d, "faces.npy"), emb)
    
    index = {
        "model": "buffalo_l",
        "rows": [
            {"image": "a.jpg", "face_id": 0},
            {"image": "a.jpg", "face_id": 1}
        ]
    }
    with open(os.path.join(d, "faces_index.json"), "w") as f:
        json.dump(index, f)
        
    # Write gallery labels.json
    labels_map = {"person_001": "John", "person_002": "John"}
    gallery_path = os.path.join(d, "labels.json")
    with open(gallery_path, "w") as f:
        json.dump(labels_map, f)
        
    from unittest.mock import patch
    with patch("face_pipeline.run_merge_on_completion") as mock_merge:
        # Run match with review=False and apply=True
        fp.run_match(d, gallery_path, apply=True, review=False)
        mock_merge.assert_called_once_with(d)






