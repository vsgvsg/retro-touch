"""Identity-grounded photo restoration — reads face_pipeline artifacts only."""
from __future__ import annotations

import argparse
import glob
import json
import os
import sys

import numpy as np


def faces_sidecar_path(image_path: str) -> str:
    stem, _ = os.path.splitext(image_path)
    return stem + ".faces.json"


def read_faces_json(image_path: str):
    path = faces_sidecar_path(image_path)
    if not os.path.exists(path):
        return None
    with open(path) as f:
        return json.load(f)


def load_labels(images_dir: str) -> dict:
    path = os.path.join(images_dir, "labels.json")
    if not os.path.exists(path):
        return {}
    with open(path) as f:
        return json.load(f)


def persona_faces(images_dir: str, labels_map: dict) -> dict:
    """{persona name: [{image, face_id, bbox, det_score, age}, ...]}."""
    out: dict = {}
    for path in sorted(glob.glob(os.path.join(images_dir, "*.faces.json"))):
        with open(path) as f:
            data = json.load(f)
        for face in data.get("faces", []):
            name = labels_map.get(face.get("cluster", ""), "")
            if not name:
                continue
            out.setdefault(name, []).append({
                "image": data["image"],
                "face_id": face["id"],
                "bbox": face["bbox"],
                "det_score": face.get("det_score", 0.0),
                "age": face.get("age"),
            })
    return out


def crop_face(image, bbox):
    """Crop to [x1,y1,x2,y2], clamped to bounds (mirrors face_pipeline)."""
    h, w = image.shape[:2]
    x1, y1, x2, y2 = [int(round(v)) for v in bbox]
    x1, x2 = max(0, min(x1, w)), max(0, min(x2, w))
    y1, y2 = max(0, min(y1, h)), max(0, min(y2, h))
    if x2 <= x1:
        x1, x2 = 0, w
    if y2 <= y1:
        y1, y2 = 0, h
    return image[y1:y2, x1:x2]


def sharpness(crop) -> float:
    """Variance of the Laplacian — higher is sharper. 0 for empty."""
    import cv2
    if crop is None or crop.size == 0:
        return 0.0
    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY) if crop.ndim == 3 else crop
    return float(cv2.Laplacian(gray, cv2.CV_64F).var())


def reference_quality(area, det_score, sharp) -> float:
    """Monotonic combined quality; sqrt(area) keeps size from dominating."""
    return float((max(0.0, area) ** 0.5) * max(0.0, det_score) *
                 (1.0 + max(0.0, sharp)))


def enrich_candidate(face, crop) -> dict:
    """face dict + computed sharpness, area, quality."""
    s = sharpness(crop)
    x1, y1, x2, y2 = face["bbox"]
    area = max(0, x2 - x1) * max(0, y2 - y1)
    return {**face, "sharpness": s, "area": area,
            "quality": reference_quality(area, face.get("det_score", 0.0), s)}


def select_reference(target, candidates, age_window):
    """Best same-person reference. Returns (candidate|None, reason).

    reason: 'in_window' | 'age_fallback' | 'no_age' | 'no_reference'.
    """
    pool = [c for c in candidates
            if not (c["image"] == target["image"]
                    and c["face_id"] == target["face_id"])]
    if not pool:
        return None, "no_reference"
    t_age = target.get("age")
    aged = [c for c in pool if c.get("age") is not None]
    if t_age is not None and aged:
        in_window = [c for c in aged if abs(c["age"] - t_age) <= age_window]
        if in_window:
            return max(in_window, key=lambda c: c["quality"]), "in_window"
        return min(aged, key=lambda c: (abs(c["age"] - t_age), -c["quality"])), \
            "age_fallback"
    return max(pool, key=lambda c: c["quality"]), "no_age"


def needs_escalation(sharp, area, sharp_thresh, min_area) -> bool:
    """A face needs identity-grounded Stage 2 when it is blurry or small."""
    return bool(sharp < sharp_thresh or area < min_area)


def composite_face(base, restored_face, bbox, feather: float = 0.2):
    """Blend restored_face into base at bbox with a feathered edge."""
    import cv2
    out = base.copy()
    h, w = out.shape[:2]
    x1, y1, x2, y2 = [int(round(v)) for v in bbox]
    x1, y1 = max(0, x1), max(0, y1)
    x2, y2 = min(w, x2), min(h, y2)
    bw, bh = x2 - x1, y2 - y1
    if bw <= 0 or bh <= 0:
        return out
    patch = cv2.resize(restored_face, (bw, bh), interpolation=cv2.INTER_AREA)
    mask = np.ones((bh, bw), np.float32)
    fpx = max(1, int(min(bw, bh) * feather))
    if 2 * fpx <= min(bw, bh):
        ramp = np.linspace(0.0, 1.0, fpx, dtype=np.float32)
        mask[:fpx, :] *= ramp[:, None]
        mask[-fpx:, :] *= ramp[::-1][:, None]
        mask[:, :fpx] *= ramp[None, :]
        mask[:, -fpx:] *= ramp[None, ::-1]
    m3 = mask[:, :, None]
    region = out[y1:y2, x1:x2].astype(np.float32)
    blended = patch.astype(np.float32) * m3 + region * (1.0 - m3)
    out[y1:y2, x1:x2] = blended.astype(np.uint8)
    return out


def face_provenance(face_id, persona, reference, stage, model, params) -> dict:
    """One face's restoration record. stage in {'stage1','stage2'}."""
    ref = None
    if reference is not None:
        ref = {"image": reference["image"], "face_id": reference["face_id"],
               "age": reference.get("age"),
               "quality": round(float(reference["quality"]), 4)}
    return {"face_id": face_id, "persona": persona, "reference": ref,
            "stage": stage, "model": model,
            "ai_reconstructed": stage == "stage2", "params": params}


def restore_sidecar_path(out_image_path: str) -> str:
    stem, _ = os.path.splitext(out_image_path)
    return stem + ".restore.json"


def write_restore_json(out_image_path: str, provenance: dict) -> str:
    path = restore_sidecar_path(out_image_path)
    with open(path, "w") as f:
        json.dump(provenance, f, indent=2)
    return path


class FakeProvider:
    """Test double: echoes the input image, records calls. No network."""
    name = "fake"

    def __init__(self):
        self.calls = []

    def enhance(self, bgr):
        self.calls.append(("enhance", bgr.shape))
        return bgr

    def identity_restore(self, degraded_bgr, reference_bgr):
        self.calls.append(("identity_restore", degraded_bgr.shape))
        return degraded_bgr


class ReplicateProvider:
    """Replicate-backed provider. Verified manually (network + paid)."""
    name = "replicate"

    # Pin specific model versions during manual verification; these slugs are
    # confirmed/updated against the live Replicate model pages when wiring the
    # account, since hosted model signatures change over time.
    ENHANCE_MODEL = "tencentarc/gfpgan"
    IDENTITY_MODEL = "zsxkib/instant-id"

    def __init__(self):
        if not os.environ.get("REPLICATE_API_TOKEN"):
            raise SystemExit(
                "REPLICATE_API_TOKEN not set. Export it or use --dry-run.")
        try:
            import replicate  # noqa: F401
        except ImportError as e:
            raise SystemExit(
                "replicate not installed. Run: pip install -r requirements.txt"
            ) from e

    def _run(self, model, inputs):
        import io
        import urllib.request
        import cv2
        import replicate
        from PIL import Image
        out = replicate.run(model, input=inputs)
        url = out[0] if isinstance(out, list) else out
        with urllib.request.urlopen(str(url)) as r:
            buf = r.read()
        rgb = np.array(Image.open(io.BytesIO(buf)).convert("RGB"))
        return cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)

    def _to_png(self, bgr):
        import io
        import cv2
        ok, buf = cv2.imencode(".png", bgr)
        return io.BytesIO(buf.tobytes())

    def enhance(self, bgr):
        return self._run(self.ENHANCE_MODEL, {"img": self._to_png(bgr)})

    def identity_restore(self, degraded_bgr, reference_bgr):
        return self._run(self.IDENTITY_MODEL, {
            "image": self._to_png(reference_bgr),
            "pose_image": self._to_png(degraded_bgr)})


def make_provider(name: str):
    """Provider factory used by the CLI."""
    if name == "fake":
        return FakeProvider()
    if name == "replicate":
        return ReplicateProvider()
    raise SystemExit(f"unknown provider: {name}")


def _enriched_personas(images_dir, by_persona):
    """persona -> [enriched candidate, ...] (loads each source once)."""
    import cv2
    src_cache: dict = {}
    out: dict = {}
    for name, cands in by_persona.items():
        lst = []
        for c in cands:
            if c["image"] not in src_cache:
                src_cache[c["image"]] = cv2.imread(
                    os.path.join(images_dir, c["image"]))
            src = src_cache[c["image"]]
            crop = (crop_face(src, c["bbox"]) if src is not None
                    else np.zeros((1, 1, 3), np.uint8))
            lst.append(enrich_candidate(c, crop))
        out[name] = lst
    return out


def restore_photo(images_dir, image, mode, provider, labels_map,
                  age_window: int = 5, sharp_thresh: float = 100.0,
                  min_area: int = 80 * 80, dry_run: bool = False,
                  log=print):
    """Restore one photo. Returns (out_image|None, provenance dict).

    mode: 'face' (composite restored faces into the original) or
    'photo' (enhance the whole image first, then identity-ground faces).
    """
    import cv2
    src = cv2.imread(os.path.join(images_dir, image))
    if src is None:
        log(f"  ! cannot read {image}")
        return None, {"source": image, "faces": []}
    data = read_faces_json(os.path.join(images_dir, image))
    faces = data["faces"] if data else []
    enriched = _enriched_personas(images_dir, persona_faces(images_dir, labels_map))

    if mode == "photo" and not dry_run:
        out_img = provider.enhance(src)
    else:
        out_img = src.copy()

    face_provs = []
    for face in faces:
        persona = labels_map.get(face.get("cluster", ""), "") or face.get("label", "")
        crop = crop_face(src, face["bbox"])
        s = sharpness(crop)
        x1, y1, x2, y2 = face["bbox"]
        area = max(0, x2 - x1) * max(0, y2 - y1)
        escalate = needs_escalation(s, area, sharp_thresh, min_area)

        reference, reason = None, "no_persona"
        if persona and persona in enriched:
            target = {"image": image, "face_id": face["id"], "age": face.get("age")}
            reference, reason = select_reference(target, enriched[persona], age_window)

        stage = "stage1"
        params = {"reason": reason, "sharpness": round(s, 2), "area": area,
                  "mode": mode}
        if escalate and reference is not None:
            stage = "stage2"
            if not dry_run:
                ref_src = cv2.imread(os.path.join(images_dir, reference["image"]))
                ref_crop = (crop_face(ref_src, reference["bbox"])
                            if ref_src is not None else crop)
                restored = provider.identity_restore(crop, ref_crop)
                out_img = composite_face(out_img, restored, face["bbox"])
        elif mode == "face" and not dry_run:
            enhanced = provider.enhance(crop)
            out_img = composite_face(out_img, enhanced, face["bbox"])

        model = (provider.name if stage == "stage2"
                 else (provider.name if mode == "face" else "whole-photo"))
        face_provs.append(face_provenance(
            face["id"], persona, reference, stage, model, params))
        log(f"  {image} #{face['id']}: {stage} "
            f"({persona or 'unknown'}, {reason})")

    return out_img, {"source": image, "faces": face_provs}


def run_restore(images_dir, image, mode, provider_name, age_window,
                sharp_thresh, min_area, out_dir, dry_run) -> int:
    import cv2
    labels_map = load_labels(images_dir)
    provider = FakeProvider() if dry_run else make_provider(provider_name)
    out_img, provenance = restore_photo(
        images_dir, image, mode, provider, labels_map,
        age_window=age_window, sharp_thresh=sharp_thresh,
        min_area=min_area, dry_run=dry_run)
    if dry_run:
        print("[dry-run] no images written.")
        return 0
    if out_img is None:
        return 1
    os.makedirs(os.path.join(images_dir, out_dir), exist_ok=True)
    out_path = os.path.join(images_dir, out_dir, image)
    cv2.imwrite(out_path, out_img)
    write_restore_json(out_path, provenance)
    print(f"Wrote {out_path} (+ provenance).")
    return 0


def main(argv: list[str]) -> int:
    p = argparse.ArgumentParser(description="Identity-grounded photo restoration")
    sub = p.add_subparsers(dest="cmd", required=True)
    for cmd, helptext in (("face", "restore faces only, composite back"),
                          ("photo", "restore the whole photo")):
        sp = sub.add_parser(cmd, help=helptext)
        sp.add_argument("image", help="filename within --images")
        sp.add_argument("--images", default="extracted")
        sp.add_argument("--provider", default="replicate",
                        choices=["replicate", "fake"])
        sp.add_argument("--age-window", type=int, default=5)
        sp.add_argument("--sharpness-thresh", type=float, default=100.0)
        sp.add_argument("--min-area", type=int, default=80 * 80)
        sp.add_argument("--out", default="restored")
        sp.add_argument("--dry-run", action="store_true")
    args = p.parse_args(argv[1:])
    return run_restore(args.images, args.image, args.cmd, args.provider,
                       args.age_window, args.sharpness_thresh, args.min_area,
                       args.out, args.dry_run)


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
