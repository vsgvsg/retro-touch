"""Face detection & tagging pipeline — detect, embed, cluster, match faces."""
from __future__ import annotations

import argparse
import glob
import json
import os
import sys

# Patch skimage.transform.SimilarityTransform.estimate to use from_estimate internally,
# avoiding the deprecation warning without globally suppressing warnings.
try:
    from skimage.transform import SimilarityTransform
    def _patched_estimate(self, src, dst):
        t = SimilarityTransform.from_estimate(src, dst)
        if t:
            self.params = t.params
            return True
        return False
    SimilarityTransform.estimate = _patched_estimate
except ImportError:
    pass

import numpy as np


def l2_normalize(x: np.ndarray) -> np.ndarray:
    """L2-normalize a vector, or each row of a 2-D array. Zero vectors stay zero."""
    x = np.asarray(x, dtype=np.float32)
    if x.ndim == 1:
        n = np.linalg.norm(x)
        return x / n if n > 0 else x
    norms = np.linalg.norm(x, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return x / norms


def cosine_sim(a: np.ndarray, b: np.ndarray) -> float:
    """Cosine similarity between two vectors."""
    a = l2_normalize(np.asarray(a, dtype=np.float32))
    b = l2_normalize(np.asarray(b, dtype=np.float32))
    return float(np.dot(a, b))


def cosine_sim_matrix(faces: np.ndarray, centroids: np.ndarray) -> np.ndarray:
    """(F,D) faces x (C,D) centroids -> (F,C) cosine similarities."""
    faces = l2_normalize(np.atleast_2d(np.asarray(faces, dtype=np.float32)))
    centroids = l2_normalize(np.atleast_2d(np.asarray(centroids, dtype=np.float32)))
    return faces @ centroids.T


def build_centroids(embeddings: np.ndarray, labels: list[str]):
    """Mean (then re-normalized) embedding per non-empty label.

    Returns (names_sorted, centroids) where centroids[i] corresponds to names[i].
    """
    embeddings = np.asarray(embeddings, dtype=np.float32)
    by_name: dict[str, list[int]] = {}
    for i, lab in enumerate(labels):
        if lab:
            by_name.setdefault(lab, []).append(i)
    names = sorted(by_name)
    if not names:
        return [], np.zeros((0, embeddings.shape[1]), dtype=np.float32)
    cents = np.stack([embeddings[by_name[n]].mean(axis=0) for n in names])
    return names, l2_normalize(cents)


def rank_candidates(names: list[str], sims: np.ndarray, top: int, threshold: float):
    """Top-N (name, score) dicts above threshold, highest first. Empty => unknown."""
    sims = np.asarray(sims, dtype=np.float32)
    order = np.argsort(-sims)
    out = []
    for j in order:
        if sims[j] < threshold:
            break
        out.append({"name": names[j], "score": round(float(sims[j]), 4)})
        if len(out) >= top:
            break
    return out


def clusters_to_persons(labels) -> list[str]:
    """HDBSCAN integer labels -> person_NNN strings; -1 (noise) -> 'unassigned'."""
    out = []
    for lab in labels:
        lab = int(lab)
        out.append("unassigned" if lab < 0 else f"person_{lab:03d}")
    return out


def cache_paths(images_dir: str) -> tuple[str, str]:
    return (os.path.join(images_dir, "faces.npy"),
            os.path.join(images_dir, "faces_index.json"))


def load_cache(images_dir: str) -> tuple[np.ndarray, dict]:
    """Load (embeddings, index). Raises FileNotFoundError if not built yet."""
    npy, idx = cache_paths(images_dir)
    if not (os.path.exists(npy) and os.path.exists(idx)):
        raise FileNotFoundError(
            f"No embedding cache in {images_dir}. Run 'detect' first.")
    emb = np.load(npy)
    with open(idx) as f:
        index = json.load(f)
    return emb, index


def append_embeddings(images_dir: str, new_emb: np.ndarray,
                      new_rows: list[dict], model: str) -> list[int]:
    """Append L2-normalized embeddings + index rows; return assigned row refs."""
    npy, idx = cache_paths(images_dir)
    new_emb = l2_normalize(np.atleast_2d(np.asarray(new_emb, dtype=np.float32)))
    if os.path.exists(npy) and os.path.exists(idx):
        emb = np.load(npy)
        with open(idx) as f:
            index = json.load(f)
    else:
        emb = np.zeros((0, new_emb.shape[1]), dtype=np.float32)
        index = {"model": model, "rows": []}
    start = emb.shape[0]
    emb = np.vstack([emb, new_emb]) if start else new_emb
    index["rows"].extend(new_rows)
    index["model"] = model
    np.save(npy, emb)
    with open(idx, "w") as f:
        json.dump(index, f, indent=2)
    return list(range(start, start + len(new_rows)))


def faces_sidecar_path(image_path: str) -> str:
    stem, _ = os.path.splitext(image_path)
    return stem + ".faces.json"


def write_faces_json(image_path: str, image_size: tuple[int, int],
                     model: str, faces: list[dict]) -> str:
    data = {
        "image": os.path.basename(image_path),
        "image_size": [int(image_size[0]), int(image_size[1])],
        "model": model,
        "faces": faces,
    }
    path = faces_sidecar_path(image_path)
    with open(path, "w") as f:
        json.dump(data, f, indent=2)
    return path


def read_faces_json(image_path: str) -> dict | None:
    path = faces_sidecar_path(image_path)
    if not os.path.exists(path):
        return None
    with open(path) as f:
        return json.load(f)


def scaffold_labels(persons: list[str]) -> dict:
    """Build a labels.json skeleton: every real cluster id -> empty name."""
    ids = sorted({p for p in persons if p != "unassigned"})
    return {pid: "" for pid in ids}


MODEL_NAME = "buffalo_l"


class FaceModel:
    """Thin InsightFace wrapper. Loads buffalo_l (RetinaFace + ArcFace r50)."""

    def __init__(self, name: str = MODEL_NAME, det_size: int = 640):
        try:
            from insightface.app import FaceAnalysis
        except ImportError as e:
            raise SystemExit(
                "insightface not installed. Run: pip install -r requirements.txt") from e
        self.name = name
        self.app = FaceAnalysis(name=name, providers=["CPUExecutionProvider"])
        self.app.prepare(ctx_id=-1, det_size=(det_size, det_size))

    def detect(self, image, det_thresh: float = 0.5) -> list[dict]:
        """Return [{bbox:[x1,y1,x2,y2], det_score, embedding(np.ndarray 512)}]."""
        faces = self.app.get(image)
        out = []
        for f in faces:
            if float(f.det_score) < det_thresh:
                continue
            x1, y1, x2, y2 = [int(round(v)) for v in f.bbox]
            age = getattr(f, "age", None)
            out.append({
                "bbox": [x1, y1, x2, y2],
                "det_score": float(f.det_score),
                "embedding": np.asarray(f.embedding, dtype=np.float32),
                "age": (None if age is None else float(age)),
            })
        return out


def list_images(images_dir: str) -> list[str]:
    files = []
    for ext in ("*.jpg", "*.jpeg", "*.png", "*.tif", "*.tiff"):
        files.extend(glob.glob(os.path.join(images_dir, ext)))
    return sorted(files)


def run_detect(images_dir: str, det_thresh: float = 0.5) -> int:
    import cv2
    images = list_images(images_dir)
    if not images:
        print(f"No images found in {images_dir}")
        return 1
    model = FaceModel()
    total = 0
    for path in images:
        existing = read_faces_json(path)
        if existing is not None and existing.get("model") == model.name:
            print(f"{os.path.basename(path)}: cached, skipping")
            continue
        img = cv2.imread(path)
        if img is None:
            print(f"  ! cannot read {path}, skipping")
            continue
        dets = model.detect(img, det_thresh=det_thresh)
        rows = [{"image": os.path.basename(path), "face_id": i}
                for i in range(len(dets))]
        refs = []
        if dets:
            emb = np.stack([d["embedding"] for d in dets])
            refs = append_embeddings(images_dir, emb, rows, model.name)
        faces = [{
            "id": i,
            "bbox": d["bbox"],
            "det_score": round(d["det_score"], 4),
            "embedding_ref": refs[i],
            "cluster": "",
            "label": "",
            "age": (None if d.get("age") is None else int(round(d["age"]))),
            "age_source": (None if d.get("age") is None else "auto"),
        } for i, d in enumerate(dets)]
        h, w = img.shape[:2]
        write_faces_json(path, (w, h), model.name, faces)
        print(f"{os.path.basename(path)}: {len(faces)} face(s)")
        total += len(faces)
    print(f"Detected {total} face(s) across {len(images)} image(s).")
    return 0


def run_backfill_age(images_dir: str, det_thresh: float = 0.5) -> int:
    """Merge auto ages into existing sidecars (preserves cluster/label/manual)."""
    import cv2
    images = list_images(images_dir)
    if not images:
        print(f"No images found in {images_dir}")
        return 1
    model = FaceModel()
    updated = 0
    for path in images:
        data = read_faces_json(path)
        if data is None:
            print(f"{os.path.basename(path)}: no sidecar, skipping")
            continue
        img = cv2.imread(path)
        if img is None:
            print(f"  ! cannot read {path}, skipping")
            continue
        dets = model.detect(img, det_thresh=det_thresh)
        id_to_j = match_faces_by_bbox(data["faces"], dets)
        age_by_id = {fid: dets[j].get("age") for fid, j in id_to_j.items()}
        merge_age_into_faces(data["faces"], age_by_id, source="auto")
        write_faces_json(path, tuple(data["image_size"]), data["model"],
                         data["faces"])
        n = sum(1 for v in age_by_id.values() if v is not None)
        updated += n
        print(f"{os.path.basename(path)}: aged {n}/{len(data['faces'])} face(s)")
    print(f"Backfilled age on {updated} face(s).")
    return 0


def _write_clusters_to_sidecars(images_dir: str, index: dict,
                                persons: list[str]) -> None:
    """Group person assignments by image and patch each sidecar's faces."""
    by_image: dict[str, dict[int, str]] = {}
    for row, person in zip(index["rows"], persons):
        by_image.setdefault(row["image"], {})[row["face_id"]] = person
    for image_name, face_map in by_image.items():
        path = os.path.join(images_dir, image_name)
        data = read_faces_json(path)
        if data is None:
            continue
        for face in data["faces"]:
            if face["id"] in face_map:
                face["cluster"] = face_map[face["id"]]
        write_faces_json(path, tuple(data["image_size"]), data["model"],
                         data["faces"])


def cluster_would_lose_work(images_dir: str) -> bool:
    """True if re-clustering would overwrite manual work.

    Manual work exists when a labels.json is present, or any sidecar already
    carries a real (non-unassigned) cluster assignment.
    """
    if os.path.exists(os.path.join(images_dir, "labels.json")):
        return True
    return bool(existing_cluster_ids(images_dir))


def run_cluster(images_dir: str, min_cluster_size: int = 3,
                min_samples: int | None = None, assume_yes: bool = False) -> int:
    try:
        import hdbscan
    except ImportError as e:
        raise SystemExit(
            "hdbscan not installed. Run: pip install -r requirements.txt") from e
    emb, index = load_cache(images_dir)  # raises FileNotFoundError w/ guidance
    if emb.shape[0] == 0:
        print("No embeddings to cluster. Run 'detect' first.")
        return 1

    # Re-clustering rebuilds person_NNN from scratch, overwriting any cluster
    # assignments made by hand (e.g. in photo review) and possibly desyncing
    # labels.json. Confirm before destroying that, unless --yes.
    if not assume_yes and cluster_would_lose_work(images_dir):
        labels_path = os.path.join(images_dir, "labels.json")
        named = []
        if os.path.exists(labels_path):
            with open(labels_path) as f:
                named = [v for v in json.load(f).values() if v]
        print("! WARNING: re-clustering overwrites ALL person_NNN cluster "
              "assignments in the sidecars (including any set by hand in "
              "review) and may desync labels.json. This cannot be undone.")
        if named:
            print(f"  labels.json names {len(named)} person(s): "
                  f"{', '.join(sorted(named))}")
        if not sys.stdin.isatty():
            print("Refusing to re-cluster non-interactively; pass --yes to "
                  "confirm.")
            return 1
        resp = input("Type 'yes' to re-cluster and discard manual edits: ")
        if resp.strip() != "yes":
            print("Aborted; clusters unchanged.")
            return 0

    clusterer = hdbscan.HDBSCAN(min_cluster_size=min_cluster_size,
                                min_samples=min_samples, metric="euclidean")
    labels = clusterer.fit_predict(emb)  # euclidean on L2-normed ~= cosine
    persons = clusters_to_persons(labels)
    _write_clusters_to_sidecars(images_dir, index, persons)

    if os.path.exists(os.path.join(images_dir, "labels.json")):
        print("! labels.json exists; cluster ids may have changed. "
              "Re-check labels against the new clusters below.")
    from collections import Counter
    counts = Counter(persons)
    for pid in sorted(counts):
        print(f"  {pid}: {counts[pid]} face(s)")
    n_real = len([p for p in set(persons) if p != "unassigned"])
    print(f"Clustered into {n_real} person(s); "
          f"{counts.get('unassigned', 0)} unassigned.")
    return 0


def _collect_persons_and_examples(images_dir: str, index: dict):
    """Return (persons_in_row_order, {person_id: example_image_name})."""
    persons, examples = [], {}
    for row in index["rows"]:
        data = read_faces_json(os.path.join(images_dir, row["image"]))
        person = ""
        if data is not None:
            for face in data["faces"]:
                if face["id"] == row["face_id"]:
                    person = face.get("cluster", "")
                    break
        persons.append(person)
        if person and person != "unassigned" and person not in examples:
            examples[person] = row["image"]
    return persons, examples


def cluster_face_index(images_dir: str, index: dict) -> dict:
    """Gather each real cluster's faces from the sidecars.

    Returns {person_id: [{"image", "face_id", "bbox", "det_score"}, ...]},
    in index-row order. 'unassigned' is excluded.
    """
    out: dict[str, list[dict]] = {}
    cache: dict[str, dict | None] = {}
    for row in index["rows"]:
        image = row["image"]
        if image not in cache:
            cache[image] = read_faces_json(os.path.join(images_dir, image))
        data = cache[image]
        if data is None:
            continue
        for face in data["faces"]:
            if face["id"] != row["face_id"]:
                continue
            cluster = face.get("cluster", "")
            if not cluster or cluster == "unassigned":
                break
            out.setdefault(cluster, []).append({
                "image": image,
                "face_id": face["id"],
                "bbox": face["bbox"],
                "det_score": face["det_score"],
                "age": face.get("age"),
            })
            break
    return out


def crop_face(image: np.ndarray, bbox: list) -> np.ndarray:
    """Crop image to bbox [x1,y1,x2,y2], clamped to image bounds."""
    h, w = image.shape[:2]
    x1, y1, x2, y2 = [int(round(v)) for v in bbox]
    x1 = max(0, min(x1, w))
    x2 = max(0, min(x2, w))
    y1 = max(0, min(y1, h))
    y2 = max(0, min(y2, h))
    if x2 <= x1:
        x1, x2 = 0, w
    if y2 <= y1:
        y1, y2 = 0, h
    return image[y1:y2, x1:x2]


def bbox_iou(a, b) -> float:
    """Intersection-over-union of two [x1,y1,x2,y2] boxes. 0 for degenerate."""
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    iw, ih = max(0, ix2 - ix1), max(0, iy2 - iy1)
    inter = iw * ih
    area_a = max(0, ax2 - ax1) * max(0, ay2 - ay1)
    area_b = max(0, bx2 - bx1) * max(0, by2 - by1)
    union = area_a + area_b - inter
    return float(inter / union) if union > 0 else 0.0


def match_faces_by_bbox(existing_faces, detected, iou_thresh: float = 0.5) -> dict:
    """{existing face id: detected index} by best IoU >= thresh; 1:1, greedy."""
    out: dict = {}
    used: set = set()
    for face in existing_faces:
        cands = [(bbox_iou(face["bbox"], d["bbox"]), j)
                 for j, d in enumerate(detected) if j not in used]
        cands = [(v, j) for v, j in cands if v >= iou_thresh]
        if cands:
            _, j = max(cands)
            out[face["id"]] = j
            used.add(j)
    return out


def merge_age_into_faces(faces, age_by_id, source: str = "auto"):
    """Set face['age']/['age_source'] from {id: age}, skipping manual faces."""
    for face in faces:
        if face.get("age_source") == "manual":
            continue
        age = age_by_id.get(face["id"])
        if age is not None:
            face["age"] = int(round(age))
            face["age_source"] = source
    return faces


def previous_names(labels_map: dict) -> list[str]:
    """Sorted unique non-empty names already entered."""
    return sorted({v for v in labels_map.values() if v})


def find_duplicate_personas(labels_map: dict[str, str]) -> dict[str, list[str]]:
    """Find names mapped to more than one cluster ID.

    Returns a dict mapping name -> list of cluster IDs sorted,
    where the list of cluster IDs has len > 1.
    """
    by_name: dict[str, list[str]] = {}
    for cid, name in labels_map.items():
        if name:
            by_name.setdefault(name, []).append(cid)
    return {name: sorted(cids) for name, cids in by_name.items() if len(cids) > 1}


def merge_persona_clusters(images_dir: str, labels_map: dict[str, str]) -> tuple[dict[str, str], int]:
    """Merge duplicate persona clusters in both labels_map and sidecar files.

    For each name with multiple cluster IDs:
      - Pick the first cluster ID as canonical.
      - Re-assign all faces in other duplicate clusters to the canonical ID.
      - Update labels_map to keep only the canonical ID.

    Returns (updated_labels_map, number_of_faces_reassigned).
    """
    duplicates = find_duplicate_personas(labels_map)
    if not duplicates:
        return labels_map, 0

    merge_map = {}
    for name, cids in duplicates.items():
        canonical = cids[0]
        for dup in cids[1:]:
            merge_map[dup] = canonical

    new_labels_map = {cid: name for cid, name in labels_map.items() if cid not in merge_map}

    faces_reassigned = 0
    for path in glob.glob(os.path.join(images_dir, "*.faces.json")):
        try:
            with open(path) as f:
                data = json.load(f)
        except Exception:
            continue
        modified = False
        for face in data.get("faces", []):
            cur_cluster = face.get("cluster", "")
            if cur_cluster in merge_map:
                face["cluster"] = merge_map[cur_cluster]
                modified = True
                faces_reassigned += 1
        if modified:
            with open(path, "w") as f:
                json.dump(data, f, indent=2)

    return new_labels_map, faces_reassigned





def age_prefill(face) -> str:
    """Age value as an editable string, '' when unset."""
    a = face.get("age")
    return "" if a is None else str(int(a))


def write_labels(images_dir: str, labels_map: dict) -> str:
    """Write labels.json (the after-each save). Returns the path."""
    path = os.path.join(images_dir, "labels.json")
    with open(path, "w") as f:
        json.dump(labels_map, f, indent=2)
    return path


def grid_positions(n: int, cols: int = 3) -> list:
    """Row-major (row, col) position for each of n cells."""
    return [divmod(i, cols) for i in range(n)]


def scale_to_fit(w: int, h: int, max_dim: int) -> tuple:
    """Uniform downscale so the longer side <= max_dim. Never upscales.

    Returns (new_w, new_h, scale) with scale <= 1.0.
    """
    longer = max(w, h)
    if longer <= max_dim:
        return w, h, 1.0
    s = max_dim / longer
    return max(1, int(round(w * s))), max(1, int(round(h * s))), s


def exclude_face(images_dir: str, image: str, face_id: int) -> bool:
    """Set a face's cluster to 'unassigned' in its sidecar and save.

    Returns True if a matching face was found and updated, else False.
    """
    path = os.path.join(images_dir, image)
    data = read_faces_json(path)
    if data is None:
        return False
    found = False
    for face in data["faces"]:
        if face["id"] == face_id:
            face["cluster"] = "unassigned"
            found = True
            break
    if found:
        write_faces_json(path, tuple(data["image_size"]), data["model"],
                         data["faces"])
    return found


def parse_age(text):
    """GUI text -> int age in [0,120], or None if blank/invalid."""
    text = (text or "").strip()
    if not text:
        return None
    try:
        v = int(text)
    except ValueError:
        return None
    return v if 0 <= v <= 120 else None


def set_face_age(images_dir, image, face_id, age) -> bool:
    """Set one face's manual age in its sidecar (age=None clears). Returns found."""
    path = os.path.join(images_dir, image)
    data = read_faces_json(path)
    if data is None:
        return False
    found = False
    for face in data["faces"]:
        if face["id"] == face_id:
            if age is None:
                face["age"] = None
                face["age_source"] = None
            else:
                face["age"] = int(age)
                face["age_source"] = "manual"
            found = True
            break
    if found:
        write_faces_json(path, tuple(data["image_size"]), data["model"],
                         data["faces"])
    return found


def existing_cluster_ids(images_dir: str) -> set:
    """All person_NNN cluster ids found across the sidecars."""
    ids = set()
    for path in glob.glob(os.path.join(images_dir, "*.faces.json")):
        with open(path) as f:
            data = json.load(f)
        for face in data.get("faces", []):
            c = face.get("cluster", "")
            if c and c != "unassigned":
                ids.add(c)
    return ids


def centroid_coverage(images_dir: str) -> list:
    """Per labeled person, how many faces and UNIQUE images feed its centroid.

    Reads labels.json ({cluster_id: name}) and the sidecars. Returns a list of
    {name, clusters, faces, images} for each non-empty name, sorted by unique
    image count descending then name. Only named clusters are included.
    """
    labels_path = os.path.join(images_dir, "labels.json")
    if not os.path.exists(labels_path):
        return []
    with open(labels_path) as f:
        labels = json.load(f)
    # cluster_id -> list of source images contributing a face
    imgs_by_cluster: dict[str, list[str]] = {}
    for path in glob.glob(os.path.join(images_dir, "*.faces.json")):
        with open(path) as f:
            data = json.load(f)
        for face in data.get("faces", []):
            c = face.get("cluster", "")
            if c and c != "unassigned":
                imgs_by_cluster.setdefault(c, []).append(data["image"])
    # aggregate per name (a name may map from >1 cluster id)
    by_name: dict[str, dict] = {}
    for cid, name in labels.items():
        if not name:
            continue
        entry = by_name.setdefault(
            name, {"name": name, "clusters": [], "faces": 0, "_imgs": set()})
        entry["clusters"].append(cid)
        imgs = imgs_by_cluster.get(cid, [])
        entry["faces"] += len(imgs)
        entry["_imgs"].update(imgs)
    rows = []
    for entry in by_name.values():
        rows.append({
            "name": entry["name"],
            "clusters": sorted(entry["clusters"]),
            "faces": entry["faces"],
            "images": len(entry["_imgs"]),
        })
    rows.sort(key=lambda r: (-r["images"], r["name"]))
    return rows


def next_cluster_id(existing_ids) -> str:
    """Mint the next free person_NNN id (max numeric suffix + 1)."""
    nums = []
    for cid in existing_ids:
        if cid.startswith("person_"):
            try:
                nums.append(int(cid.split("_", 1)[1]))
            except ValueError:
                pass
    nxt = (max(nums) + 1) if nums else 0
    return f"person_{nxt:03d}"


def resolve_or_create_cluster(name, labels_map, existing_ids):
    """Return (cluster_id, labels_map) for a name.

    Reuse the cluster already mapped to this name; otherwise mint a new id and
    extend a COPY of labels_map. existing_ids should include any ids already in
    labels_map; the caller adds the returned id to its own existing_ids set.
    """
    for cid, nm in labels_map.items():
        if nm == name:
            return cid, labels_map
    new_id = next_cluster_id(set(existing_ids) | set(labels_map))
    new_labels = {**labels_map, new_id: name}
    return new_id, new_labels


def prefill_name(face, best_entry, threshold, labels_map):
    """The name to put in the editable box.

    Precedence: existing label -> best candidate IF score >= threshold ->
    the face's cluster name in labels_map -> "". A below-threshold candidate
    does not fill the box (it surfaces as a hint instead; see hint_for).
    best_entry is (name, score) or None.
    """
    if face.get("label"):
        return face["label"]
    if best_entry is not None and best_entry[1] >= threshold:
        return best_entry[0]
    return labels_map.get(face.get("cluster", ""), "")


def hint_for(face, best_entry, threshold, prefilled):
    """The dim '(did you mean?)' candidate to show under the box, or None.

    Shows the best candidate when it exists AND either:
      - its score < threshold and the box is empty (weak match on an unlabeled
        face -- lets a thin persona still be suggested), or
      - the box has a value but the candidate name differs from it.
    Returns (name, score) or None. best_entry is (name, score) or None.
    """
    if best_entry is None:
        return None
    name, score = best_entry
    if name == prefilled:
        return None
    if not prefilled and score < threshold:
        return (name, score)
    if prefilled:
        return (name, score)
    return None


def face_state(face, best_entry, threshold, labels_map):
    """Color/state classifier for a face, mirroring prefill_name precedence.

    Returns "matched" (has an existing label or a cluster-name prefill),
    "confident" (best candidate >= threshold), or "unassigned" (no label,
    no cluster name, candidate below threshold or absent). Used to color the
    photo box and the row's confidence meter consistently. best_entry is
    (name, score) or None.
    """
    if face.get("label"):
        return "matched"
    if best_entry is not None and best_entry[1] >= threshold:
        return "confident"
    if labels_map.get(face.get("cluster", ""), ""):
        return "matched"
    return "unassigned"


def apply_photo_edits(images_dir, image, edits, labels_map, existing_ids):
    """Apply {face_id: name} edits to one photo's sidecar.

    Sets each face's label. For a non-empty name on an unassigned face, resolves
    or mints a cluster (reusing a same-named person's cluster) and sets the
    face's cluster. Returns the (possibly extended) labels_map. Mutates
    existing_ids in place as new ids are minted. Caller persists labels.json.
    """
    path = os.path.join(images_dir, image)
    data = read_faces_json(path)
    if data is None:
        return labels_map
    by_id = {f["id"]: f for f in data["faces"]}
    for face_id, name in edits.items():
        face = by_id.get(face_id)
        if face is None:
            continue
        face["label"] = name
        cur = face.get("cluster", "")
        if name and cur in ("", "unassigned"):
            cid, labels_map = resolve_or_create_cluster(
                name, labels_map, existing_ids)
            face["cluster"] = cid
            existing_ids.add(cid)
    write_faces_json(path, tuple(data["image_size"]), data["model"],
                     data["faces"])
    return labels_map


# ---- shared GUI theme (ttk) ----
ACCENT = "#5a6cf0"
BG = "#fafaff"
CARD_BORDER = "#ececf2"
STATE_COLORS = {            # face_state -> (hex for box/badge/meter)
    "confident": "#2faf6a",
    "matched": "#5a6cf0",
    "unassigned": "#d8a23a",
}


def _install_theme(root):
    """Configure a shared ttk.Style; safe to call once per app root."""
    from tkinter import ttk
    style = ttk.Style(root)
    try:
        style.theme_use("clam")   # consistent, restyleable across platforms
    except Exception:
        pass
    root.configure(bg=BG)
    style.configure("TFrame", background=BG)
    style.configure("TLabel", background=BG, foreground="#1a1a2e")
    style.configure("Sub.TLabel", background=BG, foreground="#7a7a88")
    style.configure("Title.TLabel", background=BG, foreground="#1a1a2e",
                    font=("TkDefaultFont", 14, "bold"))
    style.configure("TButton", padding=(12, 6), relief="flat",
                    background="#ffffff", foreground="#1a1a2e")
    style.map("TButton", background=[("active", "#f0f0f8")])
    style.configure("Primary.TButton", padding=(14, 6), relief="flat",
                    background=ACCENT, foreground="#ffffff",
                    font=("TkDefaultFont", 11, "bold"))
    style.map("Primary.TButton", background=[("active", "#4a5ce0")])
    style.configure("TEntry", padding=4)

    # Set window icon
    try:
        import os
        from PIL import Image, ImageTk
        script_dir = os.path.dirname(os.path.abspath(__file__))
        icon_path = os.path.join(script_dir, "docs", "icon.png")
        if os.path.exists(icon_path):
            img = Image.open(icon_path)
            icon_img = ImageTk.PhotoImage(img)
            root._icon_image = icon_img
            root.iconphoto(False, icon_img)
    except Exception:
        pass

    return style


def crop_to_round_photo(crop, cell=128, radius=12):
    """BGR crop -> letterboxed cell x cell rounded-corner Tk PhotoImage.

    Degrades to a plain (square) letterboxed image if rounding fails, so the
    GUI never crashes on an odd crop. Returns an ImageTk.PhotoImage.
    """
    import cv2
    from PIL import Image, ImageDraw, ImageTk
    base = np.full((cell, cell, 3), 245, np.uint8)  # near-bg fill
    h, w = crop.shape[:2]
    if h > 0 and w > 0:
        s = min(cell / w, cell / h)
        nw, nh = max(1, int(w * s)), max(1, int(h * s))
        resized = cv2.resize(crop, (nw, nh), interpolation=cv2.INTER_AREA)
        y0, x0 = (cell - nh) // 2, (cell - nw) // 2
        base[y0:y0 + nh, x0:x0 + nw] = resized
    rgb = cv2.cvtColor(base, cv2.COLOR_BGR2RGB)
    img = Image.fromarray(rgb).convert("RGBA")
    try:
        mask = Image.new("L", (cell, cell), 0)
        ImageDraw.Draw(mask).rounded_rectangle(
            [0, 0, cell - 1, cell - 1], radius=radius, fill=255)
        img.putalpha(mask)
    except Exception:
        pass  # degrade to square
    return ImageTk.PhotoImage(img)


class LabelerApp:
    """Tkinter labeler: scrollable crop grid per cluster -> a name, saved after each step."""

    def __init__(self, images_dir, cluster_index, labels_map):
        import tkinter as tk
        from tkinter import ttk
        self.tk = tk
        self.ttk = ttk
        self.images_dir = images_dir
        self.cluster_ids = sorted(cluster_index)
        self.cluster_index = cluster_index
        self.labels_map = labels_map
        self.idx = 0
        self._img_cache = {}
        self._excluded = set()       # (image, face_id) dimmed this session

        self.root = tk.Tk()
        self.root.title("Face Labeler")
        self.root.geometry("460x720")
        _install_theme(self.root)

        head = ttk.Frame(self.root)
        head.pack(fill="x", padx=16, pady=(12, 4))
        self.cluster_var = tk.StringVar()
        self.sub_var = tk.StringVar()
        ttk.Label(head, textvariable=self.cluster_var,
                  style="Title.TLabel").pack(anchor="w")
        ttk.Label(head, textvariable=self.sub_var,
                  style="Sub.TLabel").pack(anchor="w")
        self.progress = ttk.Progressbar(head, maximum=len(self.cluster_ids),
                                         cursor="hand2")
        self.progress.pack(fill="x", pady=(8, 0))
        # click the bar to jump to the cluster at the clicked position
        self.progress.bind("<Button-1>", self._on_progress_click)

        # scrollable crop grid (existing pattern, fixed height)
        wrap = ttk.Frame(self.root)
        wrap.pack(padx=16, pady=8, fill="both", expand=True)
        self.canvas = tk.Canvas(wrap, height=300, highlightthickness=0,
                                bg=BG)
        self._vbar = ttk.Scrollbar(wrap, orient="vertical",
                                   command=self.canvas.yview)
        self.canvas.configure(yscrollcommand=self._vbar.set)
        # scrollbar is packed/forgotten on demand by _sync_scrollbar
        self.canvas.pack(side="left", fill="both", expand=True)
        self.grid_frame = ttk.Frame(self.canvas)
        self.canvas.create_window((0, 0), window=self.grid_frame, anchor="nw")
        self.grid_frame.bind(
            "<Configure>",
            lambda e: (self.canvas.configure(
                scrollregion=self.canvas.bbox("all")),
                self._sync_scrollbar()))
        self.canvas.bind(
            "<Configure>", lambda e: self._sync_scrollbar())
        self.canvas.bind_all(
            "<MouseWheel>",
            lambda e: self.canvas.yview_scroll(
                int(-1 * (e.delta / 120)), "units"))
        self._cells = []
        self._preview = None
        self._preview_photo = None

        # name field (typeahead filters the chip row)
        field = ttk.Frame(self.root)
        field.pack(fill="x", padx=16)
        ttk.Label(field, text="WHO IS THIS?", style="Sub.TLabel").pack(anchor="w")
        self.name_var = tk.StringVar()
        self.name_var.trace_add("write", lambda *_: self._refresh_names())
        self.entry = ttk.Entry(field, textvariable=self.name_var)
        self.entry.pack(fill="x", pady=(2, 6))
        self.entry.bind("<Return>", lambda e: self._next())

        ttk.Label(self.root, text="REUSE:", style="Sub.TLabel").pack(
            anchor="w", padx=16)
        self.chips = ttk.Frame(self.root)
        self.chips.pack(fill="x", padx=16, pady=(2, 8))

        bar = ttk.Frame(self.root)
        bar.pack(fill="x", padx=16, pady=12)
        ttk.Button(bar, text="← Back", command=self._back).pack(side="left")
        ttk.Button(bar, text="Skip", command=self._skip).pack(
            side="left", padx=6)
        self.next_btn = ttk.Button(bar, text="Save & Next →",
                                   style="Primary.TButton", command=self._next)
        self.next_btn.pack(side="right")
        self.root.protocol("WM_DELETE_WINDOW", self._on_close)
        self.root.bind_all("q", self._on_q_key)


    # ---- image helpers ----
    def _source(self, image_name):
        import cv2
        if image_name not in self._img_cache:
            if len(self._img_cache) >= 50:
                self._img_cache.pop(next(iter(self._img_cache)))
            self._img_cache[image_name] = cv2.imread(
                os.path.join(self.images_dir, image_name))
        return self._img_cache[image_name]

    def _make_crop_cell(self, face, row, col):
        src = self._source(face["image"])
        readable = src is not None
        if readable:
            crop = crop_face(src, face["bbox"])
        else:
            crop = np.full((64, 64, 3), 128, np.uint8)
        photo = crop_to_round_photo(crop, cell=140)
        self._cells.append(photo)
        excluded = (face["image"], face["face_id"]) in self._excluded
        lbl = self.tk.Label(self.grid_frame, image=photo, bg=BG,
                            borderwidth=0)
        if excluded:
            lbl.configure(text="excluded", compound="center",
                          fg="red")
        lbl.grid(row=row, column=col, padx=4, pady=4)
        if readable and not excluded:
            lbl.bind("<Button-1>", lambda e, f=face: self._preview_full(f))
            lbl.bind("<Control-Button-1>",
                     lambda e, f=face: self._do_exclude(f))

    def _sync_scrollbar(self):
        """Show the vertical scrollbar only when the crop grid overflows."""
        content = self.grid_frame.winfo_reqheight()
        visible = self.canvas.winfo_height()
        if content > visible:
            if not self._vbar.winfo_ismapped():
                self._vbar.pack(side="right", fill="y", before=self.canvas)
        else:
            if self._vbar.winfo_ismapped():
                self._vbar.pack_forget()
            self.canvas.yview_moveto(0)

    def _on_progress_click(self, event):
        """Jump to the cluster at the clicked horizontal position on the bar."""
        width = self.progress.winfo_width()
        if width <= 0 or len(self.cluster_ids) == 0:
            return
        frac = max(0.0, min(1.0, event.x / width))
        target = min(len(self.cluster_ids) - 1, int(frac * len(self.cluster_ids)))
        if target == self.idx:
            return
        self._commit_current()
        self.idx = target
        self._show()

    def _preview_full(self, face):
        import cv2
        from PIL import Image, ImageTk
        src = self._source(face["image"])
        if src is None:
            return
        self._close_preview()
        h, w = src.shape[:2]
        nw, nh, s = scale_to_fit(w, h, 900)
        disp = cv2.resize(src, (nw, nh), interpolation=cv2.INTER_AREA)
        x1, y1, x2, y2 = [int(round(v * s)) for v in face["bbox"]]
        if x2 > x1 and y2 > y1:
            cv2.rectangle(disp, (x1, y1), (x2, y2), (0, 0, 255), 2)
        rgb = cv2.cvtColor(disp, cv2.COLOR_BGR2RGB)
        top = self.tk.Toplevel(self.root)
        top.title(face["image"])
        self._preview = top
        self._preview_photo = ImageTk.PhotoImage(Image.fromarray(rgb))
        lbl = self.tk.Label(top, image=self._preview_photo)
        lbl.pack()
        lbl.bind("<Button-1>", lambda e: self._close_preview())
        top.bind("<Escape>", lambda e: self._close_preview())
        top.protocol("WM_DELETE_WINDOW", self._close_preview)

    def _close_preview(self):
        if self._preview is not None:
            try:
                self._preview.destroy()
            except self.tk.TclError:
                pass
            self._preview = None
            self._preview_photo = None  # release the PhotoImage ref

    def _do_exclude(self, face):
        exclude_face(self.images_dir, face["image"], face["face_id"])
        self._excluded.add((face["image"], face["face_id"]))
        self._show(keep_scroll=True)

    def _show(self, keep_scroll=False):
        cid = self.cluster_ids[self.idx]
        faces = self.cluster_index[cid]
        self.cluster_var.set(f"Cluster {cid} · {len(faces)} faces")
        self.sub_var.set(
            f"Person {self.idx + 1} of {len(self.cluster_ids)} · "
            f"⌘-click a crop to exclude")
        self.progress.configure(value=self.idx)

        saved_y = 0.0
        has_focus = False
        cursor_pos = 0
        has_selection = False
        sel_start = 0
        sel_end = 0

        if keep_scroll:
            saved_y = self.canvas.yview()[0]
            # Save entry state: cursor position, selection, focus
            has_focus = (self.root.focus_get() == self.entry)
            try:
                cursor_pos = self.entry.index(self.tk.INSERT)
                has_selection = self.entry.select_present()
                if has_selection:
                    sel_start = self.entry.index(self.tk.SEL_FIRST)
                    sel_end = self.entry.index(self.tk.SEL_LAST)
            except (self.tk.TclError, AttributeError, ValueError):
                has_selection = False

        for child in self.grid_frame.winfo_children():
            child.destroy()
        self._cells = []
        positions = grid_positions(len(faces), cols=3)
        for face, (r, c) in zip(faces, positions):
            self._make_crop_cell(face, r, c)

        if keep_scroll:
            # Force layout updates so scroll region is recalculated
            self.root.update_idletasks()
            self.canvas.yview_moveto(saved_y)
            # Restore entry state if it had focus or selection
            if has_focus:
                self.entry.focus_set()
                try:
                    self.entry.icursor(cursor_pos)
                    if has_selection:
                        self.entry.select_range(sel_start, sel_end)
                except (self.tk.TclError, AttributeError, ValueError):
                    pass
        else:
            self.canvas.yview_moveto(0)
            self.name_var.set(self.labels_map.get(cid, ""))
            self.entry.focus_set()

        self._refresh_names()
        self.next_btn.configure(
            text="Done" if self.idx == len(self.cluster_ids) - 1
            else "Save & Next →")

    def _refresh_names(self):
        for child in self.chips.winfo_children():
            child.destroy()
        typed = self.name_var.get().strip().lower()
        names = previous_names(self.labels_map)
        if typed:
            matches = [n for n in names if typed in n.lower()]
        else:
            matches = names[:6]               # recent (hybrid: chips for top-6)
        for nm in matches[:8]:
            chip = self.tk.Label(self.chips, text=nm, bg="#ececfa",
                                 fg="#4a4ad0", padx=10, pady=3, cursor="hand2")
            chip.pack(side="left", padx=(0, 6), pady=2)
            chip.bind("<Button-1>", lambda e, n=nm: self.name_var.set(n))
        if typed and not any(n.lower() == typed for n in names):
            new = self.tk.Label(self.chips, text=f'+ Create "{typed}"',
                                bg=BG, fg="#4a4ad0", cursor="hand2")
            new.pack(side="left")

    def _commit_current(self):
        """Apply the entry to labels_map (name set, or removed if blank); save."""
        cid = self.cluster_ids[self.idx]
        name = self.name_var.get().strip()
        if name:
            self.labels_map[cid] = name
        else:
            self.labels_map.pop(cid, None)
        write_labels(self.images_dir, self.labels_map)

    def _next(self):
        self._commit_current()
        if self.idx == len(self.cluster_ids) - 1:
            self.root.destroy()
            return
        self.idx += 1
        self._show()

    def _back(self):
        self._commit_current()
        if self.idx > 0:
            self.idx -= 1
            self._show()

    def _skip(self):
        cid = self.cluster_ids[self.idx]
        self.labels_map.pop(cid, None)
        write_labels(self.images_dir, self.labels_map)
        if self.idx == len(self.cluster_ids) - 1:
            self.root.destroy()
            return
        self.idx += 1
        self._show()

    def _on_close(self):
        self._close_preview()
        self._commit_current()
        self.root.destroy()

    def _on_q_key(self, event):
        if event.widget.winfo_class() in ("TEntry", "Entry"):
            return
        self._on_close()


    def run(self):
        self._show()
        self.root.mainloop()


class AgeLabelerApp:
    """Per-persona crop grid; each crop has an age field (prefilled from auto)."""

    def __init__(self, images_dir, cluster_index, labels_map):
        import tkinter as tk
        from tkinter import ttk
        self.tk = tk
        self.ttk = ttk
        self.images_dir = images_dir
        self.cluster_ids = sorted(cluster_index)
        self.cluster_index = cluster_index
        self.labels_map = labels_map
        self.idx = 0
        self._img_cache = {}
        self._cells = []
        self._age_vars = []          # (image, face_id, StringVar)
        self._preview = None
        self._preview_photo = None

        self.root = tk.Tk()
        self.root.title("Face Ages")
        self.root.geometry("520x760")
        _install_theme(self.root)

        head = ttk.Frame(self.root)
        head.pack(fill="x", padx=16, pady=(12, 4))
        self.cluster_var = tk.StringVar()
        self.sub_var = tk.StringVar()
        ttk.Label(head, textvariable=self.cluster_var,
                  style="Title.TLabel").pack(anchor="w")
        ttk.Label(head, textvariable=self.sub_var,
                  style="Sub.TLabel").pack(anchor="w")
        self.progress = ttk.Progressbar(head, maximum=len(self.cluster_ids),
                                         cursor="hand2")
        self.progress.pack(fill="x", pady=(8, 0))
        self.progress.bind("<Button-1>", self._on_progress_click)

        wrap = ttk.Frame(self.root)
        wrap.pack(padx=16, pady=8, fill="both", expand=True)
        self.canvas = tk.Canvas(wrap, height=420, highlightthickness=0, bg=BG)
        self._vbar = ttk.Scrollbar(wrap, orient="vertical",
                                   command=self.canvas.yview)
        self.canvas.configure(yscrollcommand=self._vbar.set)
        self.canvas.pack(side="left", fill="both", expand=True)
        self.grid_frame = ttk.Frame(self.canvas)
        self.canvas.create_window((0, 0), window=self.grid_frame, anchor="nw")
        self.grid_frame.bind(
            "<Configure>",
            lambda e: (self.canvas.configure(
                scrollregion=self.canvas.bbox("all")), self._sync_scrollbar()))
        self.canvas.bind("<Configure>", lambda e: self._sync_scrollbar())
        self.canvas.bind_all(
            "<MouseWheel>",
            lambda e: self.canvas.yview_scroll(int(-1 * (e.delta / 120)), "units"))

        bar = ttk.Frame(self.root)
        bar.pack(fill="x", padx=16, pady=12)
        ttk.Button(bar, text="← Back", command=self._back).pack(side="left")
        self.next_btn = ttk.Button(bar, text="Save & Next →",
                                   style="Primary.TButton", command=self._next)
        self.next_btn.pack(side="right")
        self.root.protocol("WM_DELETE_WINDOW", self._on_close)
        self.root.bind_all("q", self._on_q_key)


    def _source(self, image_name):
        import cv2
        if image_name not in self._img_cache:
            if len(self._img_cache) >= 50:
                self._img_cache.pop(next(iter(self._img_cache)))
            self._img_cache[image_name] = cv2.imread(
                os.path.join(self.images_dir, image_name))
        return self._img_cache[image_name]

    def _sync_scrollbar(self):
        content = self.grid_frame.winfo_reqheight()
        visible = self.canvas.winfo_height()
        if content > visible:
            if not self._vbar.winfo_ismapped():
                self._vbar.pack(side="right", fill="y", before=self.canvas)
        else:
            if self._vbar.winfo_ismapped():
                self._vbar.pack_forget()
            self.canvas.yview_moveto(0)

    def _on_progress_click(self, event):
        width = self.progress.winfo_width()
        if width <= 0 or not self.cluster_ids:
            return
        frac = max(0.0, min(1.0, event.x / width))
        target = min(len(self.cluster_ids) - 1, int(frac * len(self.cluster_ids)))
        if target == self.idx:
            return
        self._commit_current()
        self.idx = target
        self._show()

    def _make_cell(self, face, row, col):
        src = self._source(face["image"])
        crop = crop_face(src, face["bbox"]) if src is not None else \
            np.full((64, 64, 3), 128, np.uint8)
        photo = crop_to_round_photo(crop, cell=120)
        self._cells.append(photo)
        cell = self.ttk.Frame(self.grid_frame)
        cell.grid(row=row, column=col, padx=6, pady=6)
        lbl = self.tk.Label(cell, image=photo, bg=BG, borderwidth=0)
        lbl.pack()
        if src is not None:
            lbl.bind("<Button-1>", lambda e, f=face: self._preview_full(f))
        var = self.tk.StringVar(value=age_prefill(face))
        ent = self.ttk.Entry(cell, textvariable=var, width=6, justify="center")
        ent.pack(pady=(4, 0))
        self._age_vars.append((face["image"], face["face_id"], var))

    def _preview_full(self, face):
        import cv2
        from PIL import Image, ImageTk
        src = self._source(face["image"])
        if src is None:
            return
        self._close_preview()
        h, w = src.shape[:2]
        nw, nh, s = scale_to_fit(w, h, 900)
        disp = cv2.resize(src, (nw, nh), interpolation=cv2.INTER_AREA)
        x1, y1, x2, y2 = [int(round(v * s)) for v in face["bbox"]]
        if x2 > x1 and y2 > y1:
            cv2.rectangle(disp, (x1, y1), (x2, y2), (0, 0, 255), 2)
        rgb = cv2.cvtColor(disp, cv2.COLOR_BGR2RGB)
        top = self.tk.Toplevel(self.root)
        top.title(face["image"])
        self._preview = top
        self._preview_photo = ImageTk.PhotoImage(Image.fromarray(rgb))
        plbl = self.tk.Label(top, image=self._preview_photo)
        plbl.pack()
        plbl.bind("<Button-1>", lambda e: self._close_preview())
        top.bind("<Escape>", lambda e: self._close_preview())
        top.protocol("WM_DELETE_WINDOW", self._close_preview)

    def _close_preview(self):
        if self._preview is not None:
            try:
                self._preview.destroy()
            except self.tk.TclError:
                pass
            self._preview = None
            self._preview_photo = None

    def _show(self):
        cid = self.cluster_ids[self.idx]
        faces = self.cluster_index[cid]
        name = self.labels_map.get(cid, "")
        self.cluster_var.set(f"{name or cid} · {len(faces)} faces")
        self.sub_var.set(f"Person {self.idx + 1} of {len(self.cluster_ids)} · "
                         f"enter age per crop · click a crop for full photo")
        self.progress.configure(value=self.idx)
        for child in self.grid_frame.winfo_children():
            child.destroy()
        self._cells = []
        self._age_vars = []
        positions = grid_positions(len(faces), cols=3)
        for face, (r, c) in zip(faces, positions):
            self._make_cell(face, r, c)
        self.canvas.yview_moveto(0)
        self.next_btn.configure(
            text="Done" if self.idx == len(self.cluster_ids) - 1
            else "Save & Next →")

    def _commit_current(self):
        for image, face_id, var in self._age_vars:
            set_face_age(self.images_dir, image, face_id, parse_age(var.get()))

    def _next(self):
        self._commit_current()
        if self.idx == len(self.cluster_ids) - 1:
            self.root.destroy()
            return
        self.idx += 1
        self._show()

    def _back(self):
        self._commit_current()
        if self.idx > 0:
            self.idx -= 1
            self._show()

    def _on_close(self):
        self._close_preview()
        self._commit_current()
        self.root.destroy()

    def _on_q_key(self, event):
        if event.widget.winfo_class() in ("TEntry", "Entry"):
            return
        self._on_close()


    def run(self):
        self._show()
        self.root.mainloop()


class PhotoReviewApp:
    """Per-photo review: numbered bbox overlays + a name input per face."""

    def __init__(self, images_dir, photos, best, labels_map, threshold):
        import tkinter as tk
        self.tk = tk
        self.images_dir = images_dir
        self.photos = photos                  # list of image filenames, has faces
        self.best = best          # {(image, face_id): (name, score)}
        self.threshold = threshold
        self.labels_map = labels_map
        self.existing_ids = existing_cluster_ids(images_dir) | set(labels_map)
        self.idx = 0
        self._img_cache = {}
        self._photo_img = None                 # keep PhotoImage ref alive
        self._entries = []                     # (face_id, StringVar) per row
        self._focused_entry_var = None         # keep track of focused entry for suggestions

        # Collect all existing labels from sidecars and labels.json once at startup
        self.existing_labels = set()
        labels_path = os.path.join(images_dir, "labels.json")
        if os.path.exists(labels_path):
            try:
                with open(labels_path) as f:
                    for v in json.load(f).values():
                        if v:
                            self.existing_labels.add(v)
            except Exception:
                pass
        for path in glob.glob(os.path.join(images_dir, "*.faces.json")):
            try:
                with open(path) as f:
                    for face in json.load(f).get("faces", []):
                        lbl = face.get("label", "")
                        if lbl:
                            self.existing_labels.add(lbl)
            except Exception:
                pass

        # Fixed window size so it never resizes per photo. Photo is fit into a
        # bounded pane (PHOTO_W x PHOTO_H) rather than driving the geometry.
        self.PHOTO_W, self.PHOTO_H = 760, 680
        from tkinter import ttk
        self.ttk = ttk
        self.root = tk.Tk()
        self.root.title("Photo Review")
        self.root.geometry("1120x780")
        self.root.resizable(False, False)
        _install_theme(self.root)

        head = ttk.Frame(self.root)
        head.pack(side="top", fill="x", padx=16, pady=(10, 4))
        self.title_var = tk.StringVar()
        ttk.Label(head, textvariable=self.title_var,
                  style="Title.TLabel").pack(anchor="w")
        self.progress = ttk.Progressbar(head, maximum=len(self.photos),
                                         cursor="hand2")
        self.progress.pack(fill="x", pady=(6, 0))
        # click the bar to jump to the photo at the clicked position
        self.progress.bind("<Button-1>", self._on_progress_click)

        # Button bar packed at the BOTTOM first, so it is always reserved and
        # never pushed off-screen by the photo.
        bar = ttk.Frame(self.root)
        bar.pack(side="bottom", pady=10)
        ttk.Button(bar, text="← Back", command=self._back).pack(
            side="left", padx=4)
        self.next_btn = ttk.Button(bar, text="Save & Next →",
                                   style="Primary.TButton", command=self._next)
        self.next_btn.pack(side="left", padx=4)

        body = ttk.Frame(self.root)
        body.pack(side="top", fill="both", expand=True)

        # Left: fixed-size photo pane.
        photo_pane = ttk.Frame(body, width=self.PHOTO_W + 16,
                               height=self.PHOTO_H + 8)
        photo_pane.pack(side="left", padx=8, pady=4)
        photo_pane.pack_propagate(False)  # keep the pane fixed-size
        self.canvas = tk.Label(photo_pane, bg=BG)
        self.canvas.pack(expand=True)

        # Right: fixed-width scrollable inputs column. Scrollbar packs first on
        # the right so it reserves its own strip; the canvas fills what's left
        # and the inner window's width is pinned to the canvas width so cards
        # never render underneath the scrollbar.
        right = ttk.Frame(body, width=300)
        right.pack(side="right", fill="y", padx=8)
        right.pack_propagate(False)

        self.suggestions_frame = ttk.Frame(right)
        self.suggestions_frame.pack(side="bottom", fill="x", pady=(10, 0), padx=4)

        self.rows_canvas = tk.Canvas(right, highlightthickness=0, bg=BG)
        self._vbar = self.ttk.Scrollbar(right, orient="vertical",
                                        command=self.rows_canvas.yview)
        self.rows_canvas.configure(yscrollcommand=self._vbar.set)
        # scrollbar is packed/forgotten on demand by _sync_scrollbar
        self.rows_canvas.pack(side="left", fill="both", expand=True)
        self.rows_frame = tk.Frame(self.rows_canvas, bg=BG)
        self._rows_window = self.rows_canvas.create_window(
            (0, 0), window=self.rows_frame, anchor="nw")
        self.rows_frame.bind(
            "<Configure>",
            lambda e: (self.rows_canvas.configure(
                scrollregion=self.rows_canvas.bbox("all")),
                self._sync_scrollbar()))
        self.rows_canvas.bind(
            "<Configure>",
            lambda e: (self.rows_canvas.itemconfigure(
                self._rows_window, width=e.width),
                self._sync_scrollbar()))
        self.rows_canvas.bind_all(
            "<MouseWheel>",
            lambda e: self.rows_canvas.yview_scroll(
                int(-1 * (e.delta / 120)), "units"))

        self.root.protocol("WM_DELETE_WINDOW", self._on_close)
        self.root.bind_all("q", self._on_q_key)


    def _source(self, image_name):
        import cv2
        if image_name not in self._img_cache:
            if len(self._img_cache) >= 50:
                self._img_cache.pop(next(iter(self._img_cache)))
            self._img_cache[image_name] = cv2.imread(
                os.path.join(self.images_dir, image_name))
        return self._img_cache[image_name]

    def _faces(self, image_name):
        data = read_faces_json(os.path.join(self.images_dir, image_name))
        return data["faces"] if data else []

    def _meter(self, parent, score, hex_color):
        """Thin Canvas confidence bar; score in [0,1]."""
        c = self.tk.Canvas(parent, width=120, height=6, highlightthickness=0,
                           bg="#e8e8f0")
        w = max(0, min(1.0, float(score))) * 120
        if w > 0:
            c.create_rectangle(0, 0, w, 6, fill=hex_color, width=0)
        return c

    def _sync_scrollbar(self):
        """Show the vertical scrollbar only when content overflows the canvas."""
        content = self.rows_frame.winfo_reqheight()
        visible = self.rows_canvas.winfo_height()
        if content > visible:
            if not self._vbar.winfo_ismapped():
                # repack before the canvas so it keeps its right-edge strip
                self._vbar.pack(side="right", fill="y", before=self.rows_canvas)
        else:
            if self._vbar.winfo_ismapped():
                self._vbar.pack_forget()
            self.rows_canvas.yview_moveto(0)

    def _on_progress_click(self, event):
        """Jump to the photo at the clicked horizontal position on the bar."""
        width = self.progress.winfo_width()
        if width <= 0 or len(self.photos) == 0:
            return
        frac = max(0.0, min(1.0, event.x / width))
        target = min(len(self.photos) - 1, int(frac * len(self.photos)))
        if target == self.idx:
            return
        self._commit_current()
        self.idx = target
        self._show()

    def _show(self):
        import cv2
        from PIL import Image, ImageTk

        self._focused_entry_var = None
        for child in self.suggestions_frame.winfo_children():
            child.destroy()

        # Skip past any unreadable photos without recursing (a long run of
        # missing source images must not blow the stack).
        while self.idx < len(self.photos):
            src = self._source(self.photos[self.idx])
            if src is not None:
                break
            print(f"  ! cannot read {self.photos[self.idx]}, skipping")
            self.idx += 1
        if self.idx >= len(self.photos):
            self.root.destroy()
            return
        image = self.photos[self.idx]
        faces = self._faces(image)
        self.title_var.set(f"{image}  ({self.idx + 1} of {len(self.photos)}) "
                           f"— {len(faces)} faces")
        self.progress.configure(value=self.idx)
        h, w = src.shape[:2]
        # Fit into the fixed photo pane so the window never resizes per photo.
        s = min(self.PHOTO_W / w, self.PHOTO_H / h, 1.0)
        nw, nh = max(1, int(round(w * s))), max(1, int(round(h * s)))
        disp = cv2.resize(src, (nw, nh), interpolation=cv2.INTER_AREA)
        def _bgr(hex_):
            h = hex_.lstrip("#")
            r, g, b = int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16)
            return (b, g, r)
        self._row_colors = []
        self._cells = []          # release previous photo's row thumbnails
        for n, face in enumerate(faces, 1):
            be = self.best.get((image, face["id"]))
            state = face_state(face, be, self.threshold, self.labels_map)
            color = _bgr(STATE_COLORS[state])
            self._row_colors.append(STATE_COLORS[state])
            x1, y1, x2, y2 = [int(round(v * s)) for v in face["bbox"]]
            if x2 > x1 and y2 > y1:
                cv2.rectangle(disp, (x1, y1), (x2, y2), color, 2)
            cv2.putText(disp, str(n), (x1 + 2, max(12, y1 + 16)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
        rgb = cv2.cvtColor(disp, cv2.COLOR_BGR2RGB)
        self._photo_img = ImageTk.PhotoImage(Image.fromarray(rgb))
        self.canvas.configure(image=self._photo_img)

        for child in self.rows_frame.winfo_children():
            child.destroy()
        self._entries = []
        for n, face in enumerate(faces, 1):
            be = self.best.get((image, face["id"]))
            pre = prefill_name(face, be, self.threshold, self.labels_map)
            state = face_state(face, be, self.threshold, self.labels_map)
            color = STATE_COLORS[state]

            card = self.tk.Frame(self.rows_frame, bg="#ffffff",
                                 highlightthickness=1,
                                 highlightbackground=CARD_BORDER)
            card.pack(fill="x", padx=4, pady=4)

            # thumbnail
            crop = crop_face(src, face["bbox"])
            thumb = crop_to_round_photo(crop, cell=44, radius=7)
            self._cells.append(thumb)       # keep ref alive
            tlbl = self.tk.Label(card, image=thumb, bg="#ffffff")
            tlbl.pack(side="left", padx=6, pady=6)

            colf = self.tk.Frame(card, bg="#ffffff")
            colf.pack(side="left", fill="x", expand=True, padx=(0, 6), pady=6)

            top = self.tk.Frame(colf, bg="#ffffff")
            top.pack(fill="x")
            badge = self.tk.Label(top, text=str(n), bg=color, fg="#ffffff",
                                  width=2, font=("TkDefaultFont", 9, "bold"))
            badge.pack(side="left", padx=(0, 6))
            var = self.tk.StringVar(value=pre)
            entry = self.ttk.Entry(top, textvariable=var, width=20)
            entry.pack(side="left", fill="x", expand=True)

            # Trace and focus bindings for autocomplete matches
            var.trace_add("write", lambda *_, v=var: self._on_entry_change(v))
            entry.bind("<FocusIn>", lambda e, v=var: self._on_entry_focus(v))

            if n == 1:
                first_entry = entry

            meta = self.tk.Frame(colf, bg="#ffffff")
            meta.pack(fill="x", pady=(4, 0))
            score = be[1] if be else 0.0
            self._meter(meta, score, color).pack(side="left")
            tag = face.get("cluster", "") or "unassigned"
            self.tk.Label(meta, text=f"  {score:.2f} · {tag}",
                          bg="#ffffff", fg="#999",
                          font=("TkDefaultFont", 9)).pack(side="left")

            h = hint_for(face, be, self.threshold, pre)
            if h is not None:
                pill = self.tk.Label(
                    colf, text=f"→ {h[0]}? ({h[1]:.2f}) — click to use",
                    bg="#eafaf1", fg="#2a8a5a", cursor="hand2",
                    font=("TkDefaultFont", 9))
                pill.pack(anchor="w", pady=(4, 0))
                pill.bind("<Button-1>", lambda e, v=var, nm=h[0]: v.set(nm))
            self._entries.append((face["id"], var))

        self.next_btn.configure(
            text="Done" if self.idx == len(self.photos) - 1 else "Next")

        if first_entry:
            first_entry.focus_set()

    def _commit_current(self):
        image = self.photos[self.idx]
        edits = {fid: var.get().strip() for fid, var in self._entries}
        for name in edits.values():
            if name:
                self.existing_labels.add(name)
        self.labels_map = apply_photo_edits(
            self.images_dir, image, edits, self.labels_map, self.existing_ids)
        write_labels(self.images_dir, self.labels_map)

    def _advance(self):
        if self.idx == len(self.photos) - 1:
            self.root.destroy()
            return
        self.idx += 1
        self._show()

    def _next(self):
        self._commit_current()
        self._advance()

    def _back(self):
        self._commit_current()
        if self.idx > 0:
            self.idx -= 1
            self._show()

    def _on_close(self):
        self._commit_current()
        self.root.destroy()

    def _on_q_key(self, event):
        if event.widget.winfo_class() in ("TEntry", "Entry"):
            return
        self._on_close()


    def _on_entry_focus(self, var):
        self._focused_entry_var = var
        self._refresh_suggestions()

    def _on_entry_change(self, var):
        self._focused_entry_var = var
        self._refresh_suggestions()

    def _get_all_names(self):
        names = set(previous_names(self.labels_map))
        for _, var in self._entries:
            val = var.get().strip()
            if val:
                names.add(val)
        for name, _ in self.best.values():
            if name:
                names.add(name)
        return sorted(names)

    def _refresh_suggestions(self):
        for child in self.suggestions_frame.winfo_children():
            child.destroy()
        if not self._focused_entry_var:
            return
        typed = self._focused_entry_var.get().strip().lower()
        names = self._get_all_names()
        if typed:
            matches = [n for n in names if typed in n.lower()]
        else:
            matches = names[:5]
        if not matches:
            return

        lbl = self.ttk.Label(self.suggestions_frame, text="SUGGESTIONS:", style="Sub.TLabel")
        lbl.pack(anchor="w", pady=(0, 2))

        chips_container = self.ttk.Frame(self.suggestions_frame)
        chips_container.pack(fill="x")

        for nm in matches[:5]:
            chip = self.tk.Label(chips_container, text=nm, bg="#ececfa",
                                 fg="#4a4ad0", padx=10, pady=4, cursor="hand2",
                                 anchor="w", font=("TkDefaultFont", 9))
            chip.pack(fill="x", pady=1)
            chip.bind("<Enter>", lambda e, c=chip: c.configure(bg="#e0e0f5"))
            chip.bind("<Leave>", lambda e, c=chip: c.configure(bg="#ececfa"))
            chip.bind("<Button-1>", lambda e, v=self._focused_entry_var, nm=nm: (v.set(nm), self._refresh_suggestions()))

    def run(self):
        self._show()
        self.root.mainloop()


def run_label(images_dir: str) -> int:
    """Launch the interactive Tkinter labeler over the clustered faces."""
    try:
        import tkinter  # noqa: F401
    except ImportError as e:
        raise SystemExit(
            "tkinter not available. Install a Python build with Tk support "
            "(e.g. python.org installer, or `brew install python-tk`).") from e
    _, index = load_cache(images_dir)  # raises FileNotFoundError w/ guidance
    cluster_index = cluster_face_index(images_dir, index)
    if not cluster_index:
        print("No clusters found. Run 'cluster' first.")
        return 1
    labels_map = {}
    path = os.path.join(images_dir, "labels.json")
    if os.path.exists(path):  # resume: pre-fill existing names
        with open(path) as f:
            labels_map = json.load(f)
    app = LabelerApp(images_dir, cluster_index, labels_map)
    app.run()
    print(f"Saved labels -> {path}")
    run_merge_on_completion(images_dir)
    return 0



def run_ages(images_dir: str) -> int:
    """Launch the manual age-entry GUI over clustered faces."""
    try:
        import tkinter  # noqa: F401
    except ImportError as e:
        raise SystemExit(
            "tkinter not available. Run via a Python with Tk support.") from e
    _, index = load_cache(images_dir)
    cluster_index = cluster_face_index(images_dir, index)
    if not cluster_index:
        print("No clusters found. Run 'cluster' first.")
        return 1
    labels_map = {}
    path = os.path.join(images_dir, "labels.json")
    if os.path.exists(path):
        with open(path) as f:
            labels_map = json.load(f)
    AgeLabelerApp(images_dir, cluster_index, labels_map).run()
    print("Ages saved to sidecars.")
    run_merge_on_completion(images_dir)
    return 0



def run_match(images_dir: str, gallery: str, top: int = 3,
              threshold: float = 0.5, apply: bool = False,
              review: bool = True) -> int:
    emb, index = load_cache(images_dir)
    if not os.path.exists(gallery):
        print(f"No gallery file {gallery}. Run 'label' and fill it in first.")
        return 1
    with open(gallery) as f:
        labels_map = json.load(f)  # {person_id: name}

    persons, _ = _collect_persons_and_examples(images_dir, index)
    gallery_labels = [labels_map.get(p, "") for p in persons]
    names, centroids = build_centroids(emb, gallery_labels)
    if not names:
        print(f"No named clusters in {gallery}. Add names and retry.")
        return 1

    sims = cosine_sim_matrix(emb, centroids)  # (F, C)
    report = []
    for ref, row in enumerate(index["rows"]):
        cands = rank_candidates(names, sims[ref], top=top, threshold=threshold)
        report.append({
            "image": row["image"],
            "face_id": row["face_id"],
            "candidates": cands,
        })

    out_path = os.path.join(images_dir, "match_report.json")
    with open(out_path, "w") as f:
        json.dump(report, f, indent=2)
    for r in report:
        top1 = r["candidates"][0] if r["candidates"] else {"name": "unknown", "score": 0}
        print(f"  {r['image']} #{r['face_id']}: "
              f"{top1['name']} ({top1['score']})")
    print(f"Wrote {out_path}.")

    if review:
        try:
            import tkinter  # noqa: F401
        except ImportError as e:
            raise SystemExit(
                "tkinter not available for review UI. Use --no-review for "
                "headless, or run via a Python with Tk (see README).") from e
        # Unconditional best candidate per face (top centroid at ANY score),
        # so the review UI can surface weak matches as dim hints.
        best = {}
        if len(names):
            for ref, row in enumerate(index["rows"]):
                j = int(np.argmax(sims[ref]))
                best[(row["image"], row["face_id"])] = (
                    names[j], float(sims[ref][j]))
        photos = sorted({r["image"] for r in report})
        if not photos:
            print("No faces in cache to review.")
            return 0
        app = PhotoReviewApp(images_dir, photos, best, labels_map, threshold)
        app.run()
        print("Review complete; labels saved.")
        run_merge_on_completion(images_dir)
        return 0

    if apply:
        by_image: dict[str, dict[int, str]] = {}
        for r in report:
            if r["candidates"]:
                by_image.setdefault(r["image"], {})[r["face_id"]] = \
                    r["candidates"][0]["name"]
        for image_name, face_map in by_image.items():
            path = os.path.join(images_dir, image_name)
            data = read_faces_json(path)
            if data is None:
                continue
            for face in data["faces"]:
                if face["id"] in face_map:
                    face["label"] = face_map[face["id"]]
            write_faces_json(path, tuple(data["image_size"]), data["model"],
                             data["faces"])
        print("Applied top-1 labels to sidecars.")
        run_merge_on_completion(images_dir)
    return 0


def run_report(images_dir: str) -> int:
    """Print per-person centroid coverage: faces and unique source images."""
    rows = centroid_coverage(images_dir)
    if not rows:
        print(f"No labeled clusters in {images_dir}. Run 'label' first.")
        return 1
    print(f"{'person':<16}{'clusters':<24}{'faces':>6}{'uniq_imgs':>11}")
    print("-" * 57)
    for r in rows:
        print(f"{r['name']:<16}{','.join(r['clusters']):<24}"
              f"{r['faces']:>6}{r['images']:>11}")
    print(f"\n{len(rows)} labeled person(s).")
    return 0


def run_merge(images_dir: str) -> int:
    """Consolidate duplicate persona clusters with the same name.

    Scans labels.json, groups duplicate clusters, updates all matching face sidecar files
    to use the canonical cluster ID, writes back the updated labels.json, and prints a report.
    """
    labels_path = os.path.join(images_dir, "labels.json")
    if not os.path.exists(labels_path):
        print(f"No labels.json found in {images_dir}. Nothing to merge.")
        return 1
    try:
        with open(labels_path) as f:
            labels_map = json.load(f)
    except Exception as e:
        print(f"Failed to read {labels_path}: {e}")
        return 1

    duplicates = find_duplicate_personas(labels_map)
    if not duplicates:
        print("No duplicate persona names found in labels.json.")
        return 0

    print("Found duplicate personas:")
    for name, cids in duplicates.items():
        print(f"  '{name}': {', '.join(cids)} -> canonical: {cids[0]}")

    new_labels, count = merge_persona_clusters(images_dir, labels_map)

    try:
        write_labels(images_dir, new_labels)
    except Exception as e:
        print(f"Failed to write updated labels to {labels_path}: {e}")
        return 1

    print(f"\nSuccessfully merged persona clusters. Updated {count} face sidecars.")
    return 0


def run_merge_on_completion(images_dir: str):
    """Consolidate duplicate persona clusters with the same name silently/report style on completion."""
    labels_path = os.path.join(images_dir, "labels.json")
    if not os.path.exists(labels_path):
        return
    try:
        with open(labels_path) as f:
            labels_map = json.load(f)
    except Exception:
        return
    duplicates = find_duplicate_personas(labels_map)
    if not duplicates:
        return
    print("\nMerging duplicate personas...")
    for name, cids in duplicates.items():
        print(f"  '{name}': {', '.join(cids)} -> canonical: {cids[0]}")
    new_labels, count = merge_persona_clusters(images_dir, labels_map)
    try:
        write_labels(images_dir, new_labels)
        print(f"Successfully merged persona clusters. Updated {count} face sidecars.")
    except Exception as e:
        print(f"Failed to write updated labels: {e}")



def main(argv: list[str]) -> int:
    p = argparse.ArgumentParser(description="Face detection & tagging pipeline")
    sub = p.add_subparsers(dest="cmd", required=True)

    d = sub.add_parser("detect", help="detect faces + embeddings")
    d.add_argument("--images", default="extracted")
    d.add_argument("--det-thresh", type=float, default=0.5)
    d.add_argument("--backfill-age", action="store_true",
                   help="re-run model to merge age into existing sidecars only")

    c = sub.add_parser("cluster", help="cluster embeddings into persons")
    c.add_argument("--images", default="extracted")
    c.add_argument("--min-cluster-size", type=int, default=3)
    c.add_argument("--min-samples", type=int, default=None)
    c.add_argument("-y", "--yes", dest="assume_yes", action="store_true",
                   help="skip the re-cluster confirmation prompt")

    l = sub.add_parser("label", help="scaffold labels.json from clusters")
    l.add_argument("--images", default="extracted")

    ag = sub.add_parser("ages", help="manually enter/correct per-face ages")
    ag.add_argument("--images", default="extracted")

    m = sub.add_parser("match", help="match faces against a labeled gallery")
    m.add_argument("--images", default="extracted")
    m.add_argument("--gallery", required=True)
    m.add_argument("--top", type=int, default=3)
    m.add_argument("--threshold", type=float, default=0.5)
    m.add_argument("--apply", action="store_true")
    m.add_argument("--no-review", dest="review", action="store_false")

    rp = sub.add_parser("report",
                        help="per-person centroid coverage (faces + unique images)")
    rp.add_argument("--images", default="extracted")

    me = sub.add_parser("merge",
                        help="merge duplicate persona clusters sharing the same name")
    me.add_argument("--images", default="extracted")

    args = p.parse_args(argv[1:])
    if args.cmd == "detect":
        if args.backfill_age:
            return run_backfill_age(args.images, args.det_thresh)
        return run_detect(args.images, args.det_thresh)
    if args.cmd == "cluster":
        return run_cluster(args.images, args.min_cluster_size, args.min_samples,
                           args.assume_yes)
    if args.cmd == "label":
        return run_label(args.images)
    if args.cmd == "ages":
        return run_ages(args.images)
    if args.cmd == "match":
        return run_match(args.images, args.gallery, args.top,
                         args.threshold, args.apply, args.review)
    if args.cmd == "report":
        return run_report(args.images)
    if args.cmd == "merge":
        return run_merge(args.images)
    return 1


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
