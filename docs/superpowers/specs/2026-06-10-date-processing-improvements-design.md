# Design Spec: Default Date Processing Propagation

## Goal
Improve user speed when tagging large batches of photos by automatically using the last successfully saved year and month as the default values for the next untagged photo.

---

## 1. State Tracking
- Add two new attributes to the `TaggerApp` class inside `__init__`:
  - `self._last_saved_year = None` (stores `int`)
  - `self._last_saved_month = None` (stores `str` name like `"Jan"`)

---

## 2. Capture Last Saved Date
- Inside `_save_and_next()`, after verifying that the year is valid:
  - Set `self._last_saved_year = year`
  - Set `self._last_saved_month = month_str` (e.g., `"Jan"` or `""`)

---

## 3. Autofill Logic
- Inside `_autofill()`, check:
  1. If `self._sidecar` has `"taken"` date, use it.
  2. Else if `self._last_saved_year` is not `None`, pre-fill the year and month spinbox/combobox using the stored last manual inputs.
  3. Else, fall back to parsing the filename for the year.
- Note: Filename location hints (`hint`) should still be parsed and geocoded even if the date is carried over from the previous photo.

---

## 4. Verification and Testing
- Add a new unit test `test_tagger_app_date_propagation` that:
  - Instantiates `TaggerApp` with two photos.
  - Manually changes year/month on the first photo and calls `_save_and_next()`.
  - Verifies that the second photo automatically fills with the first photo's saved year and month.
- Run the full test suite (`python -m pytest tests/test_exif_pipeline.py`).
