# Translate Locations to English Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Create a standalone utility script `update_english_locations.py` to translate all locations in `.faces.json` sidecars, the centralized `locations.json` cache, and corresponding `.jpg` EXIF/XMP tags to English using Nominatim geocoding.

**Architecture:** A standalone Python CLI tool that parses sidecars/caches, coalesces coordinate queries to avoid duplicate Nominatim calls, uses Nominatim reverse geocoding with `accept-language=en`, and updates sidecars, caches, and image tags.

**Tech Stack:** Python 3.13, standard library (`urllib`, `json`, `argparse`, `pathlib`), and imports from `exif_pipeline.py`.

---

### Task 1: Test Suite Setup & Geocoding Core Logic
**Files:**
- Create: `tests/test_update_english_locations.py`
- Create: `update_english_locations.py`

- [ ] **Step 1: Write a failing test for Nominatim coordinates translation caching and coalescing**
  ```python
  import pytest
  from unittest.mock import MagicMock
  import update_english_locations as uel

  def test_translate_coordinates_coalesces_and_caches():
      # Mock NominatimClient
      mock_client = MagicMock()
      mock_client.reverse.side_effect = [
          {"address": {"city": "Penza", "state": "Penza Oblast", "country": "Russia"}},
          {"address": {"city": "Altenburg", "state": "Thuringia", "country": "Germany"}}
      ]

      # We have 3 coordinates. Two of them are within 1000m of each other (53.2000, 45.0000 and 53.2001, 45.0001).
      # The third is Altenburg (50.9852, 12.4340).
      coords = [
          (53.2000, 45.0000),
          (53.2001, 45.0001),  # Should coalesce with first
          (50.9852, 12.4340)
      ]

      cache = uel.TranslationCache(client=mock_client)
      
      res1 = cache.get_english_location(53.2000, 45.0000)
      res2 = cache.get_english_location(53.2001, 45.0001)
      res3 = cache.get_english_location(50.9852, 12.4340)

      assert mock_client.reverse.call_count == 2
      assert res1["city"] == "Penza"
      assert res2["city"] == "Penza"
      assert res3["city"] == "Altenburg"
  ```

- [ ] **Step 2: Run test to verify it fails**
  Run: `~/.venv/bin/python -m pytest tests/test_update_english_locations.py -v`
  Expected: FAIL (ModuleNotFoundError or AttributeError)

- [ ] **Step 3: Implement TranslationCache class in `update_english_locations.py`**
  ```python
  import sys
  import pathlib
  from exif_pipeline import NominatimClient, coalesce_location, parse_nominatim_address

  class TranslationCache:
      def __init__(self, client=None):
          self.client = client or NominatimClient()
          self._cache = []  # list of dicts with lat, lng, city, state, country, display_name

      def get_english_location(self, lat: float, lng: float) -> dict | None:
          # Check if we have an entry close to this lat/lng (within 1000m)
          existing = coalesce_location(lat, lng, self._cache, tolerance_m=1000)
          if existing:
              return existing

          # Call the API
          res = self.client.reverse(lat, lng)
          if not res:
              return None

          city, state, country, display_name = parse_nominatim_address(res)
          entry = {
              "lat": lat,
              "lng": lng,
              "city": city,
              "state": state,
              "country": country,
              "display_name": display_name
          }
          self._cache.append(entry)
          return entry
  ```

- [ ] **Step 4: Run test to verify it passes**
  Run: `~/.venv/bin/python -m pytest tests/test_update_english_locations.py -v`
  Expected: PASS

- [ ] **Step 5: Commit**
  ```bash
  git add update_english_locations.py tests/test_update_english_locations.py
  git commit -m "feat: add TranslationCache and initial unit tests"
  ```

---

### Task 2: Implement Sidecar & Central Cache Translation Logic
**Files:**
- Modify: `tests/test_update_english_locations.py`
- Modify: `update_english_locations.py`

- [ ] **Step 1: Write a failing test for translating sidecars and updating locations cache**
  ```python
  import json
  import update_english_locations as uel

  def test_translate_sidecars_and_cache(tmp_path):
      # Setup mock sidecar file
      sidecar_data = {
          "image": "test.jpg",
          "image_size": [100, 100],
          "faces": [{"label": "Sergey"}],
          "taken": {"year": 1970},
          "location": {
              "lat": 53.2,
              "lng": 45.0,
              "display_name": "Пенза, Россия",
              "city": "Пенза",
              "state": "Пензенская область",
              "country": "Россия"
          },
          "exif_written": True
      }
      sc_file = tmp_path / "test.faces.json"
      with open(sc_file, "w", encoding="utf-8") as f:
          json.dump(sidecar_data, f)

      # Setup mock locations.json cache
      loc_cache_data = {
          "locations": [
              {
                  "lat": 53.2,
                  "lng": 45.0,
                  "display_name": "Пенза, Россия",
                  "city": "Пенза",
                  "state": "Пензенская область",
                  "country": "Россия",
                  "use_count": 5
              }
          ]
      }
      loc_file = tmp_path / "locations.json"
      with open(loc_file, "w", encoding="utf-8") as f:
          json.dump(loc_cache_data, f)

      # Mock translation cache
      mock_trans_cache = MagicMock()
      mock_trans_cache.get_english_location.return_value = {
          "lat": 53.2,
          "lng": 45.0,
          "display_name": "Penza, Russia",
          "city": "Penza",
          "state": "Penza Oblast",
          "country": "Russia"
      }

      # Mock exif_pipeline.write_exif_xmp to avoid opening real files
      mock_write_exif = MagicMock(return_value=True)

      # Run translation (non dry-run)
      updates = uel.run_translation(
          sidecars=[sc_file],
          locations_path=loc_file,
          trans_cache=mock_trans_cache,
          write_exif_fn=mock_write_exif,
          dry_run=False
      )

      # Verify sidecar was modified
      with open(sc_file, encoding="utf-8") as f:
          updated_sc = json.load(f)
      assert updated_sc["location"]["city"] == "Penza"
      assert updated_sc["location"]["display_name"] == "Penza, Russia"

      # Verify locations.json cache was updated
      with open(loc_file, encoding="utf-8") as f:
          updated_locs = json.load(f)
      assert updated_locs["locations"][0]["city"] == "Penza"
      assert updated_locs["locations"][0]["use_count"] == 5

      # Verify EXIF write was called
      mock_write_exif.assert_called_once()
  ```

- [ ] **Step 2: Run test to verify it fails**
  Run: `~/.venv/bin/python -m pytest tests/test_update_english_locations.py -v`
  Expected: FAIL (AttributeError for uel.run_translation)

- [ ] **Step 3: Implement `run_translation` in `update_english_locations.py`**
  ```python
  import json
  from exif_pipeline import load_sidecar, save_sidecar

  def run_translation(sidecars, locations_path, trans_cache, write_exif_fn, dry_run=False):
      updates_count = 0
      
      # 1. Load the centralized locations cache
      locations_data = []
      if locations_path.exists():
          try:
              with open(locations_path, encoding="utf-8") as f:
                  content = json.load(f)
                  if isinstance(content, dict):
                      locations_data = content.get("locations", [])
                      if not isinstance(locations_data, list):
                          locations_data = []
          except Exception:
              pass

      # 2. Iterate through sidecars and update them
      for sc_path_str in sidecars:
          sc_path = pathlib.Path(sc_path_str)
          data = load_sidecar(str(sc_path))
          if not data or "location" not in data:
              continue

          loc = data["location"]
          lat, lng = loc.get("lat"), loc.get("lng")
          if lat is None or lng is None:
              continue

          eng_loc = trans_cache.get_english_location(lat, lng)
          if not eng_loc:
              continue

          # Check if actually different
          is_different = (
              loc.get("city") != eng_loc["city"] or
              loc.get("state") != eng_loc["state"] or
              loc.get("country") != eng_loc["country"] or
              loc.get("display_name") != eng_loc["display_name"]
          )

          if is_different:
              print(f"Update sidecar {sc_path.name}:")
              print(f"  Before: {loc.get('display_name')}")
              print(f"  After:  {eng_loc['display_name']}")
              
              if not dry_run:
                  loc["city"] = eng_loc["city"]
                  loc["state"] = eng_loc["state"]
                  loc["country"] = eng_loc["country"]
                  loc["display_name"] = eng_loc["display_name"]
                  save_sidecar(str(sc_path), data)

                  # Rewrite EXIF/XMP tags if needed
                  if data.get("exif_written"):
                      jpg_path = sc_path.parent / data["image"]
                      if jpg_path.exists():
                          write_exif_fn(
                              str(jpg_path),
                              data.get("taken", {}),
                              loc,
                              data.get("faces", []),
                              data.get("image_size", [1, 1])
                          )
              updates_count += 1

      # 3. Update locations cache
      loc_cache_updates = 0
      for entry in locations_data:
          lat, lng = entry.get("lat"), entry.get("lng")
          if lat is None or lng is None:
              continue
          eng_loc = trans_cache.get_english_location(lat, lng)
          if not eng_loc:
              continue

          is_different = (
              entry.get("city") != eng_loc["city"] or
              entry.get("state") != eng_loc["state"] or
              entry.get("country") != eng_loc["country"] or
              entry.get("display_name") != eng_loc["display_name"]
          )

          if is_different:
              if not dry_run:
                  entry["city"] = eng_loc["city"]
                  entry["state"] = eng_loc["state"]
                  entry["country"] = eng_loc["country"]
                  entry["display_name"] = eng_loc["display_name"]
              loc_cache_updates += 1

      if not dry_run and loc_cache_updates > 0:
          tmp_path = locations_path.with_suffix(locations_path.suffix + ".tmp")
          with open(tmp_path, "w", encoding="utf-8") as f:
              json.dump({"locations": locations_data}, f, indent=2, ensure_ascii=False)
          os.replace(tmp_path, locations_path)

      return updates_count
  ```

- [ ] **Step 4: Run test to verify it passes**
  Run: `~/.venv/bin/python -m pytest tests/test_update_english_locations.py -v`
  Expected: PASS

- [ ] **Step 5: Commit**
  ```bash
  git add update_english_locations.py tests/test_update_english_locations.py
  git commit -m "feat: implement run_translation logic and tests"
  ```

---

### Task 3: Complete update_english_locations.py CLI wrapper
**Files:**
- Modify: `update_english_locations.py`
- Modify: `tests/test_update_english_locations.py`

- [ ] **Step 1: Write a test for CLI main entry point dry-run and run**
  ```python
  def test_cli_main_dry_run(monkeypatch):
      # Test that main runs and exits cleanly
      import update_english_locations as uel
      monkeypatch.setattr("sys.argv", ["update_english_locations.py", "--dry-run"])
      
      mock_run = MagicMock(return_value=0)
      monkeypatch.setattr(uel, "run_translation", mock_run)
      
      try:
          uel.main()
      except SystemExit as e:
          assert e.code == 0
  ```

- [ ] **Step 2: Run test to verify it fails**
  Run: `~/.venv/bin/python -m pytest tests/test_update_english_locations.py -v`
  Expected: FAIL (AttributeError: uel has no main)

- [ ] **Step 3: Add `main()` and script entry point to `update_english_locations.py`**
  ```python
  import argparse
  import os
  from exif_pipeline import write_exif_xmp

  def main():
      parser = argparse.ArgumentParser(description="Translate location names to English in sidecars and locations cache.")
      parser.add_argument("--dry-run", action="store_true", help="Preview updates without making changes.")
      parser.add_argument("--dir", default="extracted", help="Directory containing sidecar files.")
      args = parser.parse_args()

      dir_path = pathlib.Path(args.dir)
      if not dir_path.exists():
          print(f"Error: directory '{args.dir}' does not exist.")
          sys.exit(1)

      sidecars = list(dir_path.glob("*.faces.json"))
      locations_path = dir_path / "locations.json"

      print(f"Scanning {len(sidecars)} sidecar files in {args.dir}...")
      if args.dry_run:
          print("DRY RUN: No files will be modified.")

      trans_cache = TranslationCache()
      
      count = run_translation(
          sidecars=sidecars,
          locations_path=locations_path,
          trans_cache=trans_cache,
          write_exif_fn=write_exif_xmp,
          dry_run=args.dry_run
      )

      if args.dry_run:
          print(f"Dry run complete. Would update {count} sidecar location values.")
      else:
          print(f"Translation complete. Updated {count} sidecar location values.")
      sys.exit(0)

  if __name__ == "__main__":
      main()
  ```

- [ ] **Step 4: Run test to verify it passes**
  Run: `~/.venv/bin/python -m pytest tests/test_update_english_locations.py -v`
  Expected: PASS

- [ ] **Step 5: Commit**
  ```bash
  git add update_english_locations.py tests/test_update_english_locations.py
  git commit -m "feat: complete CLI entry point and tests"
  ```
