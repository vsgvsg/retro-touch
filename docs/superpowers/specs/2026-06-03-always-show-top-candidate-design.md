# Always-Show Top Match Candidate in Review — Design

**Date:** 2026-06-03
**Status:** Approved (pending spec review)

## Goal

In the per-photo review UI, **always surface the single best-matching person for
each face as a low-confidence hint — even when its score is below the match
threshold** — so a thinly-seeded persona (e.g. one created from a single face,
which scores ~0.06) can still be suggested and accepted, letting the user grow
its cluster. High-confidence matches keep the current behavior (prefilled).

## Background (why)

Root-caused 2026-06-03 (see `face-match-recognition-issue` memory): the matching
code is correct, but on this dataset of old grayscale cross-decade scans,
same-person ArcFace cosine is only ~0.3–0.5, and a one-face persona scores ~0.06
against a genuine same-person photo. The compounding loop works (centroid from
1/2/4/7 faces → 0.06/0.27/0.52/0.63), but a one-face persona never clears the
0.5 threshold, so it's never suggested and can't be grown. Showing the top
candidate regardless of threshold breaks that deadlock.

## Non-goals

- No change to the default threshold, centroid weighting, multi-candidate lists,
  or min-face gating.
- No change to the headless report (`match_report.json`), `--no-review`, or
  `--apply` paths.
- A below-threshold hint never auto-commits — it only populates the editable box
  when the user clicks it.

## Architecture

All in `face_pipeline.py`.

### run_match hand-off change

Today `run_match` passes `suggestions = {(image, face_id): name}` built from
`report[*].candidates[0]` — which is empty below threshold, so it loses weak
matches. Change it to compute and pass the **unconditional best** per face:

```python
best = {(image, face_id): (name, score)}   # single top centroid, ANY score
```

built from `sims` directly (top-1 of `sims[ref]` over the centroid `names`),
present whenever there is at least one centroid; omitted only when there are no
named clusters at all. `run_match` also passes `threshold` to `PhotoReviewApp`.
The `report`/`match_report.json` and `--apply` logic are unchanged (they still
use `rank_candidates` at threshold).

### Two pure helpers (replacing the single prefill decision)

`prefill_name` changes signature (the old `suggestion: str` arg is removed):

```python
prefill_name(face, best_entry, threshold, labels_map) -> str
```
What goes in the editable box. Precedence:
1. existing `face["label"]` if non-empty;
2. else `best_entry`'s name **if** `best_entry` is not None and its score ≥
   threshold;
3. else the name of `face["cluster"]` in `labels_map`;
4. else `""`.

A below-threshold candidate does NOT fill the box.

```python
hint_for(face, best_entry, threshold, prefilled) -> tuple | None
```
What the dim hint shows (or `None`). `best_entry` is `(name, score)` or `None`;
`prefilled` is the string `prefill_name` returned. Returns `(name, score)` when a
candidate exists AND either:
- its score < threshold and `prefilled` is empty (the weak-match-on-unlabeled
  case — the core fix), OR
- `prefilled` is non-empty and the candidate `name` != `prefilled` (a "did you
  mean?" disagreement, including the score ≥ threshold case where an existing
  label differs from a confident candidate).

Returns `None` when: no candidate; or the candidate name equals `prefilled`
(agreement — nothing to second-guess).

`best_entry` for a face is `best.get((image, face_id))`.

### PhotoReviewApp

- `__init__` stores `self.best` (was `self.suggestions`) and `self.threshold`.
- `_show` per face: compute `pre = prefill_name(face, be, self.threshold,
  self.labels_map)`, set the box `StringVar` to `pre`; then
  `h = hint_for(face, be, self.threshold, pre)` and, if not None, render a second
  line under the box: a dim grey **clickable** label `→ {name}? ({score:.2f})`
  whose click handler sets that face's `StringVar` to `{name}` (does not commit).

## UI layout

Each face row in the right scroll column:

```
1.  [ Dima            ]   person_004     # number + editable box + cluster tag
      → Yura P? (0.27)                   # dim clickable hint, only when present
```

Three resulting cases:
1. One-face persona, weak match (0.06): box blank, hint `→ Dima? (0.06)` shown →
   clickable to accept. (The core fix.)
2. Strong match (0.62): box prefilled `Dima`, no hint.
3. Already labeled "Maria P", candidate "Marina R" (0.41): box shows `Maria P`,
   hint `→ Marina R? (0.41)` flags it; ignore or click to switch.

Commit semantics unchanged: Next/Back/Done/close gather box values →
`apply_photo_edits`. The hint only populates a box; it never auto-commits.

## Error handling

- No named clusters at all → `run_match` already prints "No named clusters…" and
  returns before review (unchanged); `best` is empty so no hints regardless.
- A face with no centroid score / `best_entry is None` → no hint, box behaves as
  today (label → cluster name → blank).
- Score formatting: 2 decimals; a score that rounds to 0.00 still shows the hint
  (it's the visibility that matters).

## Testing

Pure helpers get TDD tests (synthetic dicts; no model, no GUI):

`prefill_name(face, best_entry, threshold, labels_map)`:
- existing label wins regardless of candidate/score;
- no label, candidate score ≥ threshold → candidate name;
- no label, candidate score < threshold → falls through to cluster's name in
  labels_map, else "" (below-threshold candidate does NOT fill the box);
- no candidate (None) → cluster name → "".

`hint_for(face, best_entry, threshold, prefilled)`:
- candidate score < threshold and prefilled == "" → returns (name, score);
- candidate score ≥ threshold and name == prefilled → None;
- prefilled non-empty and candidate name differs → (name, score) ("did you
  mean?");
- candidate name == prefilled → None;
- best_entry None → None.

**Signature change flagged:** the existing `test_prefill_name_precedence` test is
rewritten to the new 4-arg signature — a deliberate update, not a silent break.

Manual-verified (Tk convention): the dim hint rendering, click-to-accept setting
the box, and the three visual cases. `run_match`'s new `best`+threshold hand-off
is exercised by the existing `--no-review` headless smoke plus the GUI check.

Run via `.venv/bin/python -m pytest tests/ -q` (system Tk can't open a window on
this macOS).

## Scope

`face_pipeline.py`: rewrite `prefill_name`, add `hint_for`, change `run_match`'s
hand-off (`suggestions` → `best` + `threshold`), update `PhotoReviewApp.__init__`
and `_show`. Update the affected tests; add a README/CLAUDE.md note that review
always shows the best candidate (dim if below threshold). One focused plan.

## Code conventions

- One file. Pure helpers (`prefill_name`, `hint_for`) TDD-tested;
  `PhotoReviewApp` Tk code verified manually.
- The threshold still governs prefill and the headless report; this change only
  adds an always-on hint in the GUI.
