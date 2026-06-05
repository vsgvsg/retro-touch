# Face Pipeline UI Redesign — Design

**Date:** 2026-06-03
**Scope:** Visual + interaction redesign of the two Tkinter GUIs in `face_pipeline.py` — `LabelerApp` (`label` subcommand) and `PhotoReviewApp` (`match` review). No change to the pipeline, data formats, or pure helper logic.

## Goal

The two GUIs are functional but look dated and are hard to scan: default gray Tk widgets, tiny letterboxed crops, a raw listbox for name reuse, plain-text cluster tags and score hints, and no visual link between a numbered face in the photo and its name row. Redesign both for visual polish, clearer workflow feedback, and better face visibility — while keeping every existing behavior, data path, and the local Tkinter-only constraint.

## Constraints (non-negotiable)

- **Tkinter only.** No web frameworks. Implementation uses **ttk themed widgets + Tk Canvas** drawing — both ship with Python, no new dependency in `requirements.txt`.
- **No logic changes.** `prefill_name`, `hint_for`, `apply_photo_edits`, `exclude_face`, `write_labels`, `previous_names`, `crop_face`, `grid_positions`, `scale_to_fit` keep their current contracts and tests. This is presentation only.
- **Box geometry stays in full-resolution scan coords**; display scaling applied only at render time (existing convention).
- GUIs remain manually verified (not unit-tested), per project convention; any *new pure helper* extracted during the work gets a TDD test.

## Decisions

- **Styling tech:** ttk widgets for buttons/entries/labels/progressbar; Tk `Canvas` for crops, the confidence meter, and rounded "card" backgrounds. Targets ~90% of the mockup look with zero new deps.
- **Review photo rendering:** keep the existing OpenCV draw-into-image path and its scaling math; only **recolor** boxes by face state (green/blue/amber) instead of all-red. Row thumbnails are separate small crops. (Canvas-overlay boxes were considered and rejected to avoid rewriting the working render path.)
- **Name reuse (labeler):** **hybrid** — a typeahead-filtered name entry *plus* clickable chips for the most-recent names.

## Visual System (shared)

A small module-level style applied once per app via a `ttk.Style`:

- Accent color `#5a6cf0` (primary buttons), background `#fafaff`, card border `#ececf2`.
- State colors: **confident** (score ≥ threshold) `#2faf6a` green · **matched/labeled** `#5a6cf0` blue · **unassigned/weak** `#d8a23a` amber.
- Primary action button ("Save & Next") filled accent; secondary buttons (Back/Skip) flat outline.
- Crops/thumbnails rendered with rounded corners by drawing the image onto a Canvas with a rounded mask (or a rounded rectangle background behind a square image if masking proves fiddly — acceptable degradation).

## Component 1 — LabelerApp redesign

Same workflow: one cluster at a time → see its crops → name it → Back/Skip/Save&Next, saving after each step. Layout top-to-bottom:

1. **Header**: `Cluster {cid} · {n} faces` (bold) + subtitle `Person {i} of {N} · ⌘-click a crop to exclude`.
2. **Progress bar**: `ttk.Progressbar`, value = `idx / total`. (Replaces the text-only "X of Y" as the sole progress cue; the text stays in the subtitle.)
3. **Scrollable crop grid**: existing Canvas+Scrollbar+inner-frame pattern, fixed max height (~170px in mock; tune to window). Crops larger than today and rounded. **Excluded state**: an excluded crop dims to ~30% with an "excluded" overlay rather than vanishing (currently ⌘-click removes it from the list). *Behavior note:* `_do_exclude` currently drops the face from `cluster_index[cid]` and re-renders; to show a dimmed state we keep the face in the render list but mark it excluded. This is the one interaction change — confirm acceptable during implementation; if not, fall back to current remove-on-exclude with the new styling.
4. **Name field**: single `ttk.Entry`, label "Who is this?". Typing filters the reuse candidates (see below).
5. **Reuse (hybrid)**: row of clickable **chips** for the most-recent names (top ~6 from `previous_names`); clicking a chip sets the entry. As the user types, the chip row filters to matching names; a `+ Create "<text>"` affordance appears when the typed text is new. Replaces the `Listbox` entirely.
6. **Footer**: `← Back`, `Skip`, primary `Save & Next →` (label becomes `Done` on the last cluster, as today).

Unchanged: `_commit_current`, `_next/_back/_skip`, save-after-each-step, full-photo preview Toplevel on left-click (its boxed-face render can keep current style or adopt the new box color — minor).

## Component 2 — PhotoReviewApp redesign

Same workflow: per photo, numbered faces on the left, one name row per face on the right; Back / Save & Next; commit on advance. Same prefill precedence and hint rules.

**Left — photo pane** (fixed `PHOTO_W × PHOTO_H` as today): OpenCV-drawn boxes + numbers, but **box/number color = the face's state** computed from the same values `prefill_name`/`hint_for` already use:
- has label or candidate ≥ threshold → green (confident) / blue (labeled-but-matched) ,
- unassigned / below threshold → amber.

**Right — scrollable rows** (existing Canvas+Scrollbar pattern), one **card** per face:
- Small **face thumbnail** (crop via existing `crop_face` on the already-loaded source) on the left of the card.
- **Color-matched number badge** (same color as its box in the photo) so face↔row is unambiguous.
- **Name `ttk.Entry`** prefilled by the unchanged `prefill_name`.
- **Confidence meter**: a thin Canvas bar, width = score, colored by state, with the numeric score + `cluster`/`matched`/`unassigned` label beside it (replaces the plain gray tag text).
- **Hint pill**: when `hint_for` returns a candidate, show it as a clickable green pill `→ {name}? ({score}) — click to use`; clicking sets the entry (no commit), exactly as today.

Unchanged: `_commit_current` → `apply_photo_edits` → `write_labels`; skip-unreadable loop; fixed window geometry; commit-on-advance.

## Data flow

Unchanged. Both apps read the same inputs (`cluster_index`/`labels_map` for the labeler; `photos`/`best`/`labels_map`/`threshold` for review), call the same helpers, and write `labels.json` / sidecars exactly as before. The redesign only changes how rows/crops/boxes are *drawn* and how the reuse control is presented.

## Error handling

- Unreadable source image: labeler keeps the existing 64×64 gray placeholder; review keeps the skip-unreadable loop.
- Rounded-corner masking failure: degrade to square crop on a rounded background (no crash).
- Empty reuse list / no candidates: chip row simply renders empty; typeahead still allows free text.

## Testing

- Pure helpers extracted (if any — e.g. a `face_state(face, best, threshold, labels_map) -> "confident"|"matched"|"unassigned"` classifier shared by box color + meter color): TDD-tested. This classifier is the most likely new pure function and should be tested since both panes depend on it agreeing.
- GUI rendering: manual verification via `.venv/bin/python face_pipeline.py label` and `... match` (per CLAUDE.md, the venv Tk is required to open a window on this macOS). Verify: progress bar advances; chips filter on typing; exclude dims a crop; review boxes are color-coded and match row badge colors; hint pill click fills the entry without committing; save-after-step and commit-on-advance still write `labels.json`.
- Existing pure-helper tests must stay green (no signature changes).

## Out of scope (YAGNI)

- Click-a-box-in-photo-to-focus-its-row (would require the Canvas-overlay render path we rejected).
- Any change to `split_photos.py`.
- Keyboard-driven navigation beyond what exists (Return = next).
- Drag-to-exclude, multi-select, batch rename.
