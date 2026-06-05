import json
import os
import sys

import numpy as np
import cv2

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import restore_photos as rp


def _sidecar(d, name, faces, size=(100, 100)):
    path = os.path.join(d, name)
    data = {"image": name, "image_size": list(size), "model": "buffalo_l",
            "faces": faces}
    with open(os.path.splitext(path)[0] + ".faces.json", "w") as f:
        json.dump(data, f)


def test_read_faces_json_missing_returns_none(tmp_path):
    assert rp.read_faces_json(str(tmp_path / "nope.jpg")) is None


def test_load_labels_missing_is_empty(tmp_path):
    assert rp.load_labels(str(tmp_path)) == {}


def test_persona_faces_groups_by_name(tmp_path):
    d = str(tmp_path)
    with open(os.path.join(d, "labels.json"), "w") as f:
        json.dump({"person_000": "Alice"}, f)
    _sidecar(d, "a.jpg", [
        {"id": 0, "bbox": [0, 0, 10, 10], "det_score": 0.9,
         "cluster": "person_000", "age": 30},
        {"id": 1, "bbox": [0, 0, 5, 5], "det_score": 0.8,
         "cluster": "unassigned", "age": None}])
    out = rp.persona_faces(d, rp.load_labels(d))
    assert list(out) == ["Alice"]
    assert out["Alice"][0]["image"] == "a.jpg"
    assert out["Alice"][0]["age"] == 30


def test_crop_face_clamps_and_crops():
    img = np.zeros((20, 20, 3), np.uint8)
    img[5:15, 5:15] = 255
    crop = rp.crop_face(img, [5, 5, 15, 15])
    assert crop.shape == (10, 10, 3)
    assert int(crop.mean()) == 255


def test_sharpness_sharp_beats_blurred():
    sharp = np.zeros((40, 40), np.uint8)
    sharp[:, 20:] = 255                      # hard edge
    blurred = cv2.GaussianBlur(sharp, (9, 9), 5)
    assert rp.sharpness(sharp) > rp.sharpness(blurred)


def test_sharpness_empty_is_zero():
    assert rp.sharpness(np.zeros((0, 0, 3), np.uint8)) == 0.0


def test_reference_quality_monotonic():
    base = rp.reference_quality(100, 0.8, 50)
    assert rp.reference_quality(200, 0.8, 50) > base     # bigger area
    assert rp.reference_quality(100, 0.95, 50) > base    # higher det_score
    assert rp.reference_quality(100, 0.8, 80) > base      # sharper


def test_enrich_candidate_adds_quality_and_area():
    face = {"image": "a.jpg", "face_id": 0, "bbox": [0, 0, 10, 10],
            "det_score": 0.9, "age": 30}
    crop = np.zeros((10, 10, 3), np.uint8)
    crop[:, 5:] = 255
    out = rp.enrich_candidate(face, crop)
    assert out["area"] == 100
    assert out["sharpness"] >= 0.0
    assert out["quality"] >= 0.0
    assert out["age"] == 30 and out["face_id"] == 0


def _cand(image, fid, age, quality):
    return {"image": image, "face_id": fid, "age": age, "quality": quality}


def test_select_reference_prefers_in_window_by_quality():
    target = {"image": "t.jpg", "face_id": 0, "age": 30}
    cands = [_cand("a.jpg", 1, 28, 10.0), _cand("b.jpg", 2, 32, 50.0),
             _cand("c.jpg", 3, 5, 99.0)]            # out of window
    best, reason = rp.select_reference(target, cands, age_window=5)
    assert reason == "in_window" and best["image"] == "b.jpg"


def test_select_reference_age_fallback_when_none_in_window():
    target = {"image": "t.jpg", "face_id": 0, "age": 30}
    cands = [_cand("a.jpg", 1, 10, 10.0), _cand("b.jpg", 2, 50, 99.0)]
    best, reason = rp.select_reference(target, cands, age_window=5)
    assert reason == "age_fallback" and best["image"] == "b.jpg"  # |50-30|<|10-30|


def test_select_reference_no_age_uses_quality():
    target = {"image": "t.jpg", "face_id": 0, "age": None}
    cands = [_cand("a.jpg", 1, None, 10.0), _cand("b.jpg", 2, None, 40.0)]
    best, reason = rp.select_reference(target, cands, age_window=5)
    assert reason == "no_age" and best["image"] == "b.jpg"


def test_select_reference_excludes_target_and_empties():
    target = {"image": "t.jpg", "face_id": 0, "age": 30}
    cands = [_cand("t.jpg", 0, 30, 99.0)]           # only the target itself
    best, reason = rp.select_reference(target, cands, age_window=5)
    assert best is None and reason == "no_reference"


def test_needs_escalation_triggers_on_blur_or_small():
    assert rp.needs_escalation(sharp=10.0, area=40000,
                               sharp_thresh=100.0, min_area=6400) is True   # blurry
    assert rp.needs_escalation(sharp=500.0, area=100,
                               sharp_thresh=100.0, min_area=6400) is True   # small
    assert rp.needs_escalation(sharp=500.0, area=40000,
                               sharp_thresh=100.0, min_area=6400) is False  # fine


def test_composite_face_replaces_center_keeps_shape():
    base = np.zeros((100, 100, 3), np.uint8)
    restored = np.full((50, 50, 3), 255, np.uint8)
    out = rp.composite_face(base, restored, [25, 25, 75, 75], feather=0.2)
    assert out.shape == base.shape
    assert out[50, 50, 0] > 200                  # center fully replaced
    assert out[0, 0, 0] == 0                      # outside bbox untouched


def test_composite_face_offscreen_bbox_is_noop():
    base = np.zeros((100, 100, 3), np.uint8)
    restored = np.full((10, 10, 3), 255, np.uint8)
    out = rp.composite_face(base, restored, [200, 200, 210, 210])
    assert int(out.sum()) == 0


def test_face_provenance_stage2_is_ai_reconstructed():
    ref = {"image": "ref.jpg", "face_id": 2, "age": 31, "quality": 12.3456}
    p = rp.face_provenance(0, "Alice", ref, "stage2", "instant-id",
                           {"reason": "in_window"})
    assert p["ai_reconstructed"] is True
    assert p["reference"]["image"] == "ref.jpg"
    assert p["reference"]["quality"] == 12.3456
    assert p["params"]["reason"] == "in_window"


def test_face_provenance_stage1_has_no_reference_flag_false():
    p = rp.face_provenance(1, "", None, "stage1", "real-esrgan", {})
    assert p["ai_reconstructed"] is False and p["reference"] is None


def test_write_restore_json_roundtrip(tmp_path):
    out_img = str(tmp_path / "restored" / "a.jpg")
    os.makedirs(os.path.dirname(out_img))
    prov = {"source": "a.jpg", "faces": []}
    path = rp.write_restore_json(out_img, prov)
    assert path.endswith("a.restore.json")
    with open(path) as f:
        assert json.load(f)["source"] == "a.jpg"


def test_fake_provider_records_calls():
    p = rp.FakeProvider()
    img = np.zeros((4, 4, 3), np.uint8)
    assert p.enhance(img).shape == img.shape
    assert p.identity_restore(img, img).shape == img.shape
    assert [c[0] for c in p.calls] == ["enhance", "identity_restore"]


def _scene(tmp_path):
    """Two photos of 'Alice': a sharp reference and a blurry target."""
    d = str(tmp_path)
    with open(os.path.join(d, "labels.json"), "w") as f:
        json.dump({"person_000": "Alice"}, f)
    ref = np.zeros((120, 120, 3), np.uint8)
    ref[:, 60:] = 255
    cv2.imwrite(os.path.join(d, "ref.jpg"), ref)
    _sidecar(d, "ref.jpg", [{"id": 0, "bbox": [10, 10, 110, 110],
             "det_score": 0.95, "cluster": "person_000", "age": 30}],
             size=(120, 120))
    tgt = cv2.GaussianBlur(ref, (21, 21), 12)
    cv2.imwrite(os.path.join(d, "tgt.jpg"), tgt)
    _sidecar(d, "tgt.jpg", [{"id": 0, "bbox": [10, 10, 110, 110],
             "det_score": 0.9, "cluster": "person_000", "age": 31}],
             size=(120, 120))
    return d


def test_restore_photo_escalates_blurry_known_face(tmp_path):
    d = _scene(tmp_path)
    prov = rp.FakeProvider()
    out_img, provenance = rp.restore_photo(
        d, "tgt.jpg", "face", prov, rp.load_labels(d),
        sharp_thresh=10_000.0, min_area=1)        # force escalation
    assert ("identity_restore", (100, 100, 3)) in prov.calls
    assert provenance["faces"][0]["ai_reconstructed"] is True
    assert provenance["faces"][0]["reference"]["image"] == "ref.jpg"
    assert out_img.shape == (120, 120, 3)


def test_restore_photo_stage1_when_sharp(tmp_path):
    d = _scene(tmp_path)
    prov = rp.FakeProvider()
    _, provenance = rp.restore_photo(
        d, "ref.jpg", "face", prov, rp.load_labels(d),
        sharp_thresh=0.0, min_area=1)             # never escalate
    assert provenance["faces"][0]["stage"] == "stage1"
    assert all(c[0] == "enhance" for c in prov.calls)


def test_restore_photo_dry_run_makes_no_calls(tmp_path):
    d = _scene(tmp_path)
    prov = rp.FakeProvider()
    _, provenance = rp.restore_photo(
        d, "tgt.jpg", "face", prov, rp.load_labels(d),
        sharp_thresh=10_000.0, min_area=1, dry_run=True)
    assert prov.calls == []
    assert provenance["faces"][0]["stage"] in ("stage1", "stage2")


def test_main_dry_run_writes_nothing(tmp_path):
    d = _scene(tmp_path)
    rc = rp.main(["restore_photos.py", "face", "tgt.jpg", "--images", d,
                  "--provider", "fake", "--dry-run",
                  "--sharpness-thresh", "10000", "--min-area", "1"])
    assert rc == 0
    assert not os.path.exists(os.path.join(d, "restored", "tgt.jpg"))


def test_main_face_mode_writes_output_and_provenance(tmp_path):
    d = _scene(tmp_path)
    rc = rp.main(["restore_photos.py", "face", "tgt.jpg", "--images", d,
                  "--provider", "fake",
                  "--sharpness-thresh", "10000", "--min-area", "1"])
    assert rc == 0
    assert os.path.exists(os.path.join(d, "restored", "tgt.jpg"))
    with open(os.path.join(d, "restored", "tgt.restore.json")) as f:
        prov = json.load(f)
    assert prov["faces"][0]["ai_reconstructed"] is True
