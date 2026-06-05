# Always-Show Top Match Candidate Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** In the per-photo review UI, always surface each face's single best-matching person as a dim, clickable hint (`→ Name? (0.27)`) even below threshold, so a thinly-seeded persona becomes visible and one-click acceptable; high-confidence matches still prefill the box.

**Architecture:** All in `face_pipeline.py`. Rewrite `prefill_name` (new signature: `face, best_entry, threshold, labels_map`); add `hint_for`; change `run_match` to pass an unconditional `best = {(image,face_id): (name, score)}` (computed from `sims`) plus `threshold` to `PhotoReviewApp`; update `PhotoReviewApp.__init__`/`_show` to set the box via `prefill_name` and render a clickable hint line via `hint_for`. Pure helpers TDD-tested; Tk verified manually.

**Tech Stack:** Python 3.13 venv (`.venv`, Tk 9.0), numpy, OpenCV, Tkinter, PIL, pytest. Run via `.venv/bin/python`.

---

## File Structure

- **Modify `face_pipeline.py`:**
  - Rewrite `prefill_name` (`:468`) to the 4-arg signature.
  - Add `hint_for` next to it.
  - In `run_match` (`:955`), replace the `suggestions` dict with an unconditional `best` dict computed from `sims`/`names`; pass `best` + `threshold` to `PhotoReviewApp` (`:964`).
  - In `PhotoReviewApp.__init__` (`:723`), accept/store `best` + `threshold` (rename `self.suggestions` → `self.best`).
  - In `PhotoReviewApp._show` (`:836-850`), set the box via the new `prefill_name` and render the hint line via `hint_for`.
- **Modify `tests/test_face_pipeline.py`:** rewrite `test_prefill_name_precedence` (`:307`) to the new signature; add `hint_for` tests.

Face dict keys unchanged (`id, bbox, det_score, embedding_ref, cluster, label`). `best_entry` is `(name, score)` or `None`.

---

## Task 1: Rewrite prefill_name to the new signature

**Files:**
- Modify: `face_pipeline.py:468-474`
- Modify: `tests/test_face_pipeline.py:307` (rewrite the existing test)

- [ ] **Step 1: Rewrite the failing test**

Replace the entire existing `test_prefill_name_precedence` function in `tests/test_face_pipeline.py` with:

```python
def test_prefill_name_existing_label_wins():
    labels = {"person_002": "Carol"}
    # existing label beats any candidate, even a strong one
    assert fp.prefill_name(
        {"label": "Alice", "cluster": "person_002"},
        ("Bob", 0.9), 0.5, labels) == "Alice"


def test_prefill_name_candidate_above_threshold_fills():
    assert fp.prefill_name(
        {"label": "", "cluster": "unassigned"},
        ("Bob", 0.6), 0.5, {}) == "Bob"


def test_prefill_name_candidate_below_threshold_does_not_fill():
    labels = {"person_002": "Carol"}
    # weak candidate must NOT fill; fall through to cluster's name
    assert fp.prefill_name(
        {"label": "", "cluster": "person_002"},
        ("Bob", 0.2), 0.5, labels) == "Carol"
    # no cluster name either -> ""
    assert fp.prefill_name(
        {"label": "", "cluster": "unassigned"},
        ("Bob", 0.2), 0.5, {}) == ""


def test_prefill_name_no_candidate_uses_cluster_then_empty():
    labels = {"person_002": "Carol"}
    assert fp.prefill_name(
        {"label": "", "cluster": "person_002"}, None, 0.5, labels) == "Carol"
    assert fp.prefill_name(
        {"label": "", "cluster": "unassigned"}, None, 0.5, {}) == ""
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/python -m pytest tests/test_face_pipeline.py -k prefill_name -q`
Expected: FAIL — old `prefill_name` takes 3 args; new tests pass 4 → `TypeError`.

- [ ] **Step 3: Rewrite the implementation**

Replace `prefill_name` (`face_pipeline.py:468-474`) with:

```python
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
```

- [ ] **Step 4: Run test to verify it passes**

Run: `.venv/bin/python -m pytest tests/test_face_pipeline.py -k prefill_name -q`
Expected: PASS (4 passed).

Note: `prefill_name`'s only caller is `PhotoReviewApp._show`, updated in Task 3. The full suite will not pass until then — that's expected; this task's `-k prefill_name` run is green.

- [ ] **Step 5: Commit**

```bash
git add face_pipeline.py tests/test_face_pipeline.py
git commit -m "feat: prefill_name takes (best_entry, threshold); weak match no longer fills box"
```

---

## Task 2: Add hint_for

**Files:**
- Modify: `face_pipeline.py` (add `hint_for` immediately after `prefill_name`)
- Modify: `tests/test_face_pipeline.py` (append)

- [ ] **Step 1: Write the failing test**

Append to `tests/test_face_pipeline.py`:

```python
def test_hint_for_weak_candidate_on_unlabeled():
    # below threshold, box empty -> show the hint (the core fix)
    assert fp.hint_for({"label": "", "cluster": "unassigned"},
                       ("Dima", 0.2), 0.5, "") == ("Dima", 0.2)


def test_hint_for_strong_candidate_already_prefilled():
    # at/above threshold the box was prefilled with the same name -> no hint
    assert fp.hint_for({"label": "", "cluster": "unassigned"},
                       ("Bob", 0.6), 0.5, "Bob") is None


def test_hint_for_disagreement_with_existing_label():
    # box shows an existing label, candidate differs -> "did you mean?" hint
    assert fp.hint_for({"label": "Maria P", "cluster": "person_001"},
                       ("Marina R", 0.41), 0.5, "Maria P") == ("Marina R", 0.41)


def test_hint_for_agreement_no_hint():
    assert fp.hint_for({"label": "Bob", "cluster": "person_000"},
                       ("Bob", 0.2), 0.5, "Bob") is None


def test_hint_for_no_candidate():
    assert fp.hint_for({"label": "", "cluster": "unassigned"},
                       None, 0.5, "") is None
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/python -m pytest tests/test_face_pipeline.py -k hint_for -q`
Expected: FAIL — `AttributeError: ... hint_for`.

- [ ] **Step 3: Write the implementation**

Add immediately after `prefill_name` in `face_pipeline.py`:

```python
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
```

- [ ] **Step 4: Run test to verify it passes**

Run: `.venv/bin/python -m pytest tests/test_face_pipeline.py -k hint_for -q`
Expected: PASS (5 passed).

- [ ] **Step 5: Commit**

```bash
git add face_pipeline.py tests/test_face_pipeline.py
git commit -m "feat: add hint_for (always-on top-candidate hint decision)"
```

---

## Task 3: Wire run_match + PhotoReviewApp to pass and render the candidate

**Files:**
- Modify: `face_pipeline.py` — `run_match` (`:955-964`), `PhotoReviewApp.__init__` (`:723-728`), `PhotoReviewApp._show` (`:836-850`).

UI shell change — NOT unit-tested. Verify: full suite passes + module imports + `--no-review` headless path runs without a window. Do NOT launch the Tk window (human smoke-tests it in Task 4). Run via `.venv/bin/python`.

- [ ] **Step 1: Build the unconditional `best` dict in run_match**

In `run_match`, `sims` (F×C cosine) and `names` (centroid names) are already
computed above the review block. Replace the `suggestions` dict + the
`PhotoReviewApp(...)` construction (currently `face_pipeline.py:955-964`):

```python
        suggestions = {
            (r["image"], r["face_id"]): (r["candidates"][0]["name"]
                                         if r["candidates"] else "")
            for r in report
        }
        photos = sorted({r["image"] for r in report})
        if not photos:
            print("No faces in cache to review.")
            return 0
        app = PhotoReviewApp(images_dir, photos, suggestions, labels_map)
```

with:

```python
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
```

- [ ] **Step 2: Update PhotoReviewApp.__init__ signature + stored fields**

Change the `__init__` signature (`face_pipeline.py:723`) and the
`self.suggestions` line (`:728`). Replace:

```python
    def __init__(self, images_dir, photos, suggestions, labels_map):
```
with:
```python
    def __init__(self, images_dir, photos, best, labels_map, threshold):
```

And replace:
```python
        self.suggestions = suggestions        # {(image, face_id): name}
```
with:
```python
        self.best = best          # {(image, face_id): (name, score)}
        self.threshold = threshold
```

- [ ] **Step 3: Render box + hint in _show**

Replace the per-face row block (`face_pipeline.py:836-850`) with:

```python
        for n, face in enumerate(faces, 1):
            row = self.tk.Frame(self.rows_frame)
            row.pack(fill="x", pady=(4, 2), anchor="w")
            top = self.tk.Frame(row)
            top.pack(fill="x", anchor="w")
            self.tk.Label(top, text=f"{n}.", width=3).pack(side="left")
            be = self.best.get((image, face["id"]))
            pre = prefill_name(face, be, self.threshold, self.labels_map)
            var = self.tk.StringVar(value=pre)
            self.tk.Entry(top, textvariable=var, width=20).pack(side="left")
            # cluster tag on its own line so long ids (person_010) never clip
            tag = face.get("cluster", "") or "unassigned"
            self.tk.Label(row, text=f"   {tag}", fg="#888").pack(
                side="left", anchor="w")
            # dim, clickable best-candidate hint (shown below threshold or when
            # it disagrees with the current box value); click fills the box
            h = hint_for(face, be, self.threshold, pre)
            if h is not None:
                hint = self.tk.Label(
                    row, text=f"   → {h[0]}? ({h[1]:.2f})", fg="#3a7",
                    cursor="hand2")
                hint.pack(side="left", anchor="w")
                hint.bind("<Button-1>", lambda e, v=var, nm=h[0]: v.set(nm))
            self._entries.append((face["id"], var))
```

- [ ] **Step 4: Verify (do NOT open a window)**

Run: `.venv/bin/python -m pytest tests/ -q`
Expected: all green (the Task 1/2 helper tests + existing suite).

Run: `.venv/bin/python -c "import face_pipeline as fp; assert hasattr(fp,'hint_for'); print('import OK')"`
Expected: `import OK`, no window.

Headless path (no window): back up labels.json, set one cluster name, run
`--no-review`, restore. Steps:
```bash
cp extracted/labels.json /tmp/lbl_bak.json 2>/dev/null || true
.venv/bin/python -c "import json,os; p='extracted/labels.json'; d=json.load(open(p)) if os.path.exists(p) else {}; \
import face_pipeline as fp; \
ids=sorted(fp.existing_cluster_ids('extracted')); \
d[ids[0]]=d.get(ids[0]) or 'SmokeTest'; json.dump(d,open(p,'w'),indent=2)" 2>/dev/null
.venv/bin/python face_pipeline.py match --images extracted --gallery extracted/labels.json --no-review 2>/dev/null | tail -2
cp /tmp/lbl_bak.json extracted/labels.json 2>/dev/null || true
```
Expected: per-face lines + `Wrote .../match_report.json.`, exit 0, NO window. (The `--no-review` path never constructs `best`/`PhotoReviewApp`, so it exercises that `run_match` still runs headlessly after the edit.)

- [ ] **Step 5: Commit**

```bash
git add face_pipeline.py
git commit -m "feat: review always shows best candidate as a dim clickable hint"
```

---

## Task 4: Human GUI smoke test (manual — performed by the user)

**Files:** none (verification only). Cannot be done by an automated agent.

- [ ] **Step 1: Run review with a low-ish threshold so hints are visible**

```bash
.venv/bin/python face_pipeline.py match --images extracted --gallery extracted/labels.json
```

- [ ] **Step 2: Verify**

- Faces with a below-threshold best match show a dim green `→ Name? (0.27)` line
  under the (blank) input box; clicking it fills the box with that name.
- A confident match (≥ threshold) prefills the box and shows no hint.
- A face with an existing label that disagrees with the best candidate shows the
  label in the box AND a `→ Other? (0.41)` hint.
- Clicking a hint never commits on its own; only Next/Back/Done/close save.
- Accepting a weak hint for a 1-face persona, then re-running `match`, the same
  person now scores higher on their other photos (centroid grew).

- [ ] **Step 3: Report** any layout issues or wrong hint behavior.

---

## Task 5: Update docs

**Files:**
- Modify: `CLAUDE.md`, `README.md`

- [ ] **Step 1: CLAUDE.md — add a line under the face pipeline match notes**

After the existing line about the match centroid (the one starting "A person's
match centroid = mean of embeddings…"), add:

```
- Review always shows each face's best candidate: at/above threshold it prefills the box; below threshold (or when it disagrees with an existing label) it appears as a dim clickable `→ Name? (score)` hint you can click to accept. Lets a 1-face persona (scores ~0.06) still be surfaced and grown; hints never auto-commit.
```

- [ ] **Step 2: README.md — extend the match centroid paragraph**

In `README.md`, in the `### match` section, after the "Centroids strengthen as
you confirm more faces" paragraph, add:

```
The review window always shows each face's single best-matching person. Above the
threshold it prefills the name box; below it (or when the best guess disagrees
with a name you already set) it appears as a dim, clickable `→ Name? (0.27)` hint
— click to accept it into the box. This is what lets a freshly-created persona
(whose one-face centroid scores far below threshold) still be suggested, so you
can confirm it and grow the cluster. Hints never auto-commit; only moving to the
next photo saves.
```

- [ ] **Step 3: Commit**

```bash
git add CLAUDE.md README.md
git commit -m "docs: document always-on best-candidate hint in review"
```

---

## Self-Review Notes

- **Spec coverage:** unconditional `best` from sims + threshold pass-through (T3 step 1/2), `prefill_name` new signature with below-threshold-doesn't-fill (T1), `hint_for` weak-on-unlabeled + did-you-mean + agreement-None + no-candidate (T2), `_show` renders box via prefill_name and clickable hint via hint_for (T3 step 3), commit semantics unchanged (hint only sets the StringVar; existing _commit_current untouched), headless/`--apply` untouched (only the review block changed), docs (T5), human smoke (T4). All spec sections map to tasks.
- **Type consistency:** `best` dict value is `(name, score)` everywhere — built in run_match (T3.1), stored as `self.best` (T3.2), read as `be` and passed to `prefill_name(face, be, threshold, labels_map)` and `hint_for(face, be, threshold, pre)` (T3.3); both helper signatures match T1/T2 defs. `threshold` threaded run_match → `__init__(..., threshold)` → `self.threshold` → both helper calls. `np.argmax`/`np` already imported in face_pipeline.py (used by detect/match).
- **Placeholders:** none — every code step shows complete code.
- **Note:** Task 1's full suite is intentionally red between T1 and T3 (caller not yet updated); T1 verifies via `-k prefill_name` only, and T3 step 4 restores full green. Stated in T1 step 4.
