# EXIF Pipeline Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build `exif_pipeline.py` — a standalone Tkinter GUI that tags scanned photo crops with year/month and lat/lng location, writes data to `.faces.json` sidecars and EXIF/XMP tags in the `.jpg`, and maintains a `locations.json` location cache.

**Architecture:** Single file `exif_pipeline.py` following the exact pattern of `face_pipeline.py` and `split_photos.py` — ttk clam theme, shared helpers copied in (no cross-imports), pure helper functions TDD-tested in `tests/test_exif_pipeline.py`, GUI verified manually. Data flows through `extracted/*.faces.json` sidecars and a new `extracted/locations.json` cache.

**Tech Stack:** Python 3.13 + Tkinter/ttk (clam theme), `tkintermapview` (OSM map widget), `piexif` (EXIF/GPS tags), `python-xmp-toolkit` + `exempi` (XMP MWG face regions), `requests` (Nominatim geocoding), `Pillow` (image display + EXIF verification)

---

## File Structure

| File | Action | Responsibility |
|------|--------|----------------|
| `exif_pipeline.py` | Create | Full standalone tool: pure helpers + GUI classes |
| `tests/test_exif_pipeline.py` | Create | TDD tests for all pure helpers |
| `requirements.txt` | Modify | Add `tkintermapview`, `piexif`, `python-xmp-toolkit` |
| `CLAUDE.md` | Modify | Document new tool, sidecar fields, locations.json |
| `extracted/locations.json` | Created at runtime | Location cache (not checked in) |

---

## Task 1: Dependencies + project setup

**Files:**
- Modify: `requirements.txt`
- Modify: `CLAUDE.md`

- [ ] **Step 1: Add new dependencies to `requirements.txt`**

  Replace the current `requirements.txt` content with:

  ```
  opencv-python>=4.8
  numpy>=1.24
  pytest>=7.0
  insightface>=0.7
  onnxruntime>=1.16
  hdbscan>=0.8
  scikit-learn>=1.3
  replicate>=0.25
  tkintermapview>=1.29
  piexif>=1.1
  python-xmp-toolkit>=2.0
  requests>=2.31
  Pillow>=10.0
  ```

- [ ] **Step 2: Install new dependencies**

  ```bash
  brew install exempi
  .venv/bin/pip install tkintermapview piexif python-xmp-toolkit requests Pillow
  ```

  Expected: All packages install without error. `import tkintermapview`, `import piexif`, `import libxmp` all succeed in `.venv/bin/python -c "..."`.

- [ ] **Step 3: Verify installs**

  ```bash
  .venv/bin/python -c "import tkintermapview; import piexif; import libxmp; import requests; print('OK')"
  ```

  Expected: prints `OK`.

- [ ] **Step 4: Commit**

  ```bash
  git add requirements.txt
  git commit -m "deps: add tkintermapview, piexif, python-xmp-toolkit, requests, Pillow"
  ```

---

## Task 2: Pure helpers — filename parsing + haversine + location cache

**Files:**
- Create: `exif_pipeline.py` (skeleton + first helpers)
- Create: `tests/test_exif_pipeline.py`

- [ ] **Step 1: Create `tests/test_exif_pipeline.py` with failing tests for `parse_filename`**

  ```python
  import pytest
  import sys, os
  sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
  import exif_pipeline as ep

  # --- parse_filename ---

  def test_parse_filename_standard():
      year, hint = ep.parse_filename("1960-penza-00004_04.jpg")
      assert year == 1960
      assert hint == "penza"

  def test_parse_filename_no_location():
      year, hint = ep.parse_filename("1960-00001_01.jpg")
      assert year == 1960
      assert hint == ""

  def test_parse_filename_no_year():
      year, hint = ep.parse_filename("scan_00001_01.jpg")
      assert year is None
      assert hint == ""

  def test_parse_filename_multi_segment():
      # "1970-hanlar-00001_01.jpg" -> hint = "hanlar"
      year, hint = ep.parse_filename("1970-hanlar-00001_01.jpg")
      assert year == 1970
      assert hint == "hanlar"
  ```

- [ ] **Step 2: Run tests, confirm they fail**

  ```bash
  .venv/bin/python -m pytest tests/test_exif_pipeline.py -v 2>&1 | head -30
  ```

  Expected: `ImportError` or `AttributeError` — `exif_pipeline` doesn't exist yet.

- [ ] **Step 3: Create `exif_pipeline.py` skeleton with `parse_filename`**

  ```python
  #!/usr/bin/env python3
  """exif_pipeline.py — EXIF timestamp + location tagging for scanned photos.

  Commands:
      python exif_pipeline.py tag      # interactive tagging GUI
      python exif_pipeline.py report   # coverage stats (read-only)
  """
  import re

  # ---------------------------------------------------------------------------
  # Pure helpers (TDD-tested)
  # ---------------------------------------------------------------------------

  def parse_filename(name: str) -> tuple:
      """Extract (year, location_hint) from a photo filename.

      Rules:
      - year: first 4-digit segment at the start of the stem
      - hint: second dash-separated segment if it contains only letters
      - Returns (None, "") if no year found
      """
      stem = re.sub(r"\.(jpg|jpeg|png|tiff?)$", "", name, flags=re.IGNORECASE)
      parts = stem.split("-")
      year = None
      hint = ""
      if parts and re.fullmatch(r"\d{4}", parts[0]):
          year = int(parts[0])
          if len(parts) >= 2 and re.fullmatch(r"[a-zA-Z]+", parts[1]):
              hint = parts[1].lower()
      return year, hint
  ```

- [ ] **Step 4: Run parse_filename tests, confirm they pass**

  ```bash
  .venv/bin/python -m pytest tests/test_exif_pipeline.py -v 2>&1 | head -30
  ```

  Expected: 4 PASSED.

- [ ] **Step 5: Add failing tests for `haversine` and `coalesce_location`**

  Append to `tests/test_exif_pipeline.py`:

  ```python
  # --- haversine ---

  def test_haversine_same_point():
      assert ep.haversine(53.2007, 45.0046, 53.2007, 45.0046) == pytest.approx(0.0)

  def test_haversine_known_distance():
      # Moscow to Penza is ~622 km
      dist = ep.haversine(55.7558, 37.6173, 53.2007, 45.0046)
      assert 600_000 < dist < 650_000

  def test_haversine_short_distance():
      # Two points ~111m apart (1 arcsecond latitude ~= 31m, so ~3.5 arcseconds)
      dist = ep.haversine(53.2007, 45.0046, 53.2010, 45.0046)
      assert 300 < dist < 400

  # --- coalesce_location ---

  CACHE = [
      {"lat": 53.2007, "lng": 45.0046, "display_name": "Penza, Russia",
       "city": "Penza", "state": "Penza Oblast", "country": "Russia", "use_count": 5},
      {"lat": 51.7727, "lng": 55.0988, "display_name": "Orenburg, Russia",
       "city": "Orenburg", "state": "Orenburg Oblast", "country": "Russia", "use_count": 3},
  ]

  def test_coalesce_location_hit():
      # Point 500m from Penza -> should match
      entry = ep.coalesce_location(53.2052, 45.0046, CACHE, tolerance_m=1000)
      assert entry is not None
      assert entry["city"] == "Penza"

  def test_coalesce_location_miss():
      # Point >1000m from both -> no match
      entry = ep.coalesce_location(52.0000, 44.0000, CACHE, tolerance_m=1000)
      assert entry is None

  def test_coalesce_location_exact():
      entry = ep.coalesce_location(53.2007, 45.0046, CACHE, tolerance_m=1000)
      assert entry is not None
      assert entry["city"] == "Penza"

  def test_coalesce_location_empty_cache():
      entry = ep.coalesce_location(53.2007, 45.0046, [], tolerance_m=1000)
      assert entry is None
  ```

- [ ] **Step 6: Run new tests, confirm they fail**

  ```bash
  .venv/bin/python -m pytest tests/test_exif_pipeline.py -v -k "haversine or coalesce" 2>&1 | head -20
  ```

  Expected: `AttributeError: module 'exif_pipeline' has no attribute 'haversine'`.

- [ ] **Step 7: Implement `haversine` and `coalesce_location` in `exif_pipeline.py`**

  Append after `parse_filename`:

  ```python
  import math

  def haversine(lat1: float, lng1: float, lat2: float, lng2: float) -> float:
      """Return distance in meters between two lat/lng points."""
      R = 6_371_000  # Earth radius in metres
      phi1, phi2 = math.radians(lat1), math.radians(lat2)
      dphi = math.radians(lat2 - lat1)
      dlambda = math.radians(lng2 - lng1)
      a = math.sin(dphi / 2) ** 2 + math.cos(phi1) * math.cos(phi2) * math.sin(dlambda / 2) ** 2
      return R * 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))


  def coalesce_location(lat: float, lng: float, cache: list, tolerance_m: float = 1000) -> dict | None:
      """Return first cache entry within tolerance_m of (lat, lng), or None."""
      for entry in cache:
          if haversine(lat, lng, entry["lat"], entry["lng"]) <= tolerance_m:
              return entry
      return None
  ```

- [ ] **Step 8: Run all tests, confirm they pass**

  ```bash
  .venv/bin/python -m pytest tests/test_exif_pipeline.py -v
  ```

  Expected: All PASSED.

- [ ] **Step 9: Commit**

  ```bash
  git add exif_pipeline.py tests/test_exif_pipeline.py
  git commit -m "feat: exif_pipeline skeleton + parse_filename, haversine, coalesce_location"
  ```

---

## Task 3: Pure helpers — format_taken, parse_nominatim_address, decimal_to_dms, normalize_bbox

**Files:**
- Modify: `exif_pipeline.py`
- Modify: `tests/test_exif_pipeline.py`

- [ ] **Step 1: Add failing tests**

  Append to `tests/test_exif_pipeline.py`:

  ```python
  # --- format_taken ---

  def test_format_taken_year_only():
      d = ep.format_taken(1960)
      assert d == {"year": 1960, "source": "manual"}
      assert "month" not in d

  def test_format_taken_with_month():
      d = ep.format_taken(1960, month=4)
      assert d == {"year": 1960, "month": 4, "source": "manual"}

  def test_format_taken_month_none_omitted():
      d = ep.format_taken(1965, month=None)
      assert "month" not in d

  # --- parse_nominatim_address ---

  def test_parse_nominatim_city():
      resp = {"address": {"city": "Penza", "state": "Penza Oblast", "country": "Russia"}}
      city, state, country, display = ep.parse_nominatim_address(resp)
      assert city == "Penza"
      assert state == "Penza Oblast"
      assert country == "Russia"
      assert display == "Penza, Penza Oblast, Russia"

  def test_parse_nominatim_town_fallback():
      resp = {"address": {"town": "Kstovo", "state": "Nizhny Novgorod Oblast", "country": "Russia"}}
      city, state, country, display = ep.parse_nominatim_address(resp)
      assert city == "Kstovo"

  def test_parse_nominatim_village_fallback():
      resp = {"address": {"village": "Sosnovka", "country": "Russia"}}
      city, state, country, display = ep.parse_nominatim_address(resp)
      assert city == "Sosnovka"
      assert display == "Sosnovka, Russia"

  def test_parse_nominatim_empty():
      city, state, country, display = ep.parse_nominatim_address({"address": {}})
      assert city == state == country == display == ""

  # --- decimal_to_dms ---

  def test_decimal_to_dms_positive():
      deg, mins, secs = ep.decimal_to_dms(53.2007)
      # 53 deg, 12 min, 2.52 sec (approx)
      assert deg == (53, 1)
      assert mins == (12, 1)
      d_secs = secs[0] / secs[1]
      assert abs(d_secs - 2.52) < 0.1

  def test_decimal_to_dms_zero():
      deg, mins, secs = ep.decimal_to_dms(0.0)
      assert deg == (0, 1)
      assert mins == (0, 1)
      assert secs[0] == 0

  # --- normalize_bbox ---

  def test_normalize_bbox_center():
      cx, cy, bw, bh = ep.normalize_bbox([100, 200, 300, 400], 1000, 1000)
      assert cx == pytest.approx(0.2)
      assert cy == pytest.approx(0.3)
      assert bw == pytest.approx(0.2)
      assert bh == pytest.approx(0.2)

  def test_normalize_bbox_full_image():
      cx, cy, bw, bh = ep.normalize_bbox([0, 0, 100, 100], 100, 100)
      assert cx == pytest.approx(0.5)
      assert cy == pytest.approx(0.5)
      assert bw == pytest.approx(1.0)
      assert bh == pytest.approx(1.0)
  ```

- [ ] **Step 2: Run tests, confirm they fail**

  ```bash
  .venv/bin/python -m pytest tests/test_exif_pipeline.py -v -k "format_taken or nominatim or dms or bbox" 2>&1 | head -20
  ```

  Expected: `AttributeError` for each missing function.

- [ ] **Step 3: Implement all four helpers in `exif_pipeline.py`**

  Append after `coalesce_location`:

  ```python
  def format_taken(year: int, month: int | None = None) -> dict:
      """Build the 'taken' sidecar dict. month is omitted (not None) when unknown."""
      d = {"year": year, "source": "manual"}
      if month is not None:
          d["month"] = month
      return d


  def parse_nominatim_address(response: dict) -> tuple:
      """Parse a Nominatim geocode/reverse response into (city, state, country, display_name)."""
      addr = response.get("address", {})
      city = addr.get("city") or addr.get("town") or addr.get("village") or ""
      state = addr.get("state") or addr.get("province") or ""
      country = addr.get("country") or ""
      display = ", ".join(filter(None, [city, state, country]))
      return city, state, country, display


  def decimal_to_dms(deg: float) -> tuple:
      """Convert decimal degrees to (deg, min, sec) as piexif rational tuples [(num,den)...]."""
      deg = abs(deg)
      d = int(deg)
      m = int((deg - d) * 60)
      s = (deg - d - m / 60) * 3600
      # Encode seconds as rational with denominator 100 for 2 decimal places
      return (d, 1), (m, 1), (round(s * 100), 100)


  def normalize_bbox(bbox: list, img_w: int, img_h: int) -> tuple:
      """Convert pixel bbox [x1,y1,x2,y2] to MWG normalized (cx,cy,bw,bh) in [0..1]."""
      x1, y1, x2, y2 = bbox
      cx = (x1 + x2) / 2 / img_w
      cy = (y1 + y2) / 2 / img_h
      bw = (x2 - x1) / img_w
      bh = (y2 - y1) / img_h
      return cx, cy, bw, bh
  ```

- [ ] **Step 4: Run all tests**

  ```bash
  .venv/bin/python -m pytest tests/test_exif_pipeline.py -v
  ```

  Expected: All PASSED.

- [ ] **Step 5: Commit**

  ```bash
  git add exif_pipeline.py tests/test_exif_pipeline.py
  git commit -m "feat: add format_taken, parse_nominatim_address, decimal_to_dms, normalize_bbox"
  ```

---

## Task 4: NominatimClient + LocationCache

**Files:**
- Modify: `exif_pipeline.py`

These are not pure-function tested (they touch network/disk) but are kept small and focused for manual verification.

- [ ] **Step 1: Append `NominatimClient` class to `exif_pipeline.py`**

  ```python
  import time
  import urllib.request
  import urllib.parse
  import json as _json

  class NominatimClient:
      """Thin Nominatim geocoder — enforces 1.1 s between requests, caches by query."""

      BASE = "https://nominatim.openstreetmap.org"
      UA = "retro-touch/1.0 (photo-archival-tool)"

      def __init__(self):
          self._last = 0.0
          self._cache: dict[str, list] = {}

      def _get(self, url: str) -> dict | list:
          elapsed = time.monotonic() - self._last
          if elapsed < 1.1:
              time.sleep(1.1 - elapsed)
          req = urllib.request.Request(url, headers={"User-Agent": self.UA})
          with urllib.request.urlopen(req, timeout=8) as r:
              data = _json.loads(r.read())
          self._last = time.monotonic()
          return data

      def search(self, query: str) -> list:
          """Return list of Nominatim result dicts for query string."""
          if query in self._cache:
              return self._cache[query]
          params = urllib.parse.urlencode({"q": query, "format": "json",
                                           "addressdetails": 1, "limit": 5})
          results = self._get(f"{self.BASE}/search?{params}")
          self._cache[query] = results
          return results

      def reverse(self, lat: float, lng: float) -> dict | None:
          """Reverse-geocode (lat, lng); return result dict or None on failure."""
          params = urllib.parse.urlencode({"lat": lat, "lon": lng,
                                           "format": "json", "addressdetails": 1})
          try:
              return self._get(f"{self.BASE}/reverse?{params}")
          except Exception:
              return None
  ```

- [ ] **Step 2: Append `LocationCache` class to `exif_pipeline.py`**

  ```python
  import pathlib

  LOCATIONS_PATH = pathlib.Path("extracted/locations.json")

  class LocationCache:
      """Reads/writes extracted/locations.json; coalesces entries within 1000 m."""

      TOLERANCE_M = 1000

      def __init__(self, path: pathlib.Path = LOCATIONS_PATH):
          self.path = path
          self._data: list = []
          self._load()

      def _load(self):
          if self.path.exists():
              with open(self.path) as f:
                  self._data = _json.load(f).get("locations", [])

      def _save(self):
          self.path.parent.mkdir(parents=True, exist_ok=True)
          with open(self.path, "w") as f:
              _json.dump({"locations": self._data}, f, indent=2, ensure_ascii=False)

      def top(self, n: int = 8) -> list:
          """Return top-n entries by use_count."""
          return sorted(self._data, key=lambda e: e["use_count"], reverse=True)[:n]

      def all_entries(self) -> list:
          return list(self._data)

      def record(self, lat: float, lng: float, city: str, state: str,
                 country: str, display_name: str) -> dict:
          """Add or update a location entry; return the (possibly updated) entry."""
          existing = coalesce_location(lat, lng, self._data, self.TOLERANCE_M)
          if existing:
              existing["use_count"] += 1
              self._save()
              return existing
          entry = {"lat": lat, "lng": lng, "display_name": display_name,
                   "city": city, "state": state, "country": country, "use_count": 1}
          self._data.append(entry)
          self._save()
          return entry
  ```

- [ ] **Step 3: Verify classes are importable**

  ```bash
  .venv/bin/python -c "from exif_pipeline import NominatimClient, LocationCache; print('OK')"
  ```

  Expected: `OK`.

- [ ] **Step 4: Run full test suite to confirm nothing broken**

  ```bash
  .venv/bin/python -m pytest tests/test_exif_pipeline.py -v
  ```

  Expected: All previously passing tests still pass.

- [ ] **Step 5: Commit**

  ```bash
  git add exif_pipeline.py
  git commit -m "feat: add NominatimClient and LocationCache"
  ```

---

## Task 5: EXIF/XMP writer

**Files:**
- Modify: `exif_pipeline.py`

- [ ] **Step 1: Append `write_exif_xmp` function to `exif_pipeline.py`**

  ```python
  import piexif
  import piexif.helper
  from PIL import Image
  from libxmp import XMPFiles, XMPMeta, XMPError
  from libxmp.utils import file_to_dict

  # XMP namespaces
  NS_MWG_RS  = "http://www.metadataworkinggroup.com/schemas/regions/"
  NS_MWG_RS_TYPE = "http://www.metadataworkinggroup.com/schemas/regions/"
  NS_DC      = "http://purl.org/dc/elements/1.1/"
  NS_IPTC    = "http://iptc.org/std/Iptc4xmpCore/1.0/xmlns/"

  def write_exif_xmp(jpg_path: str, taken: dict, location: dict, faces: list,
                     image_size: list) -> bool:
      """Write EXIF/GPS/XMP/IPTC tags to jpg_path.

      Args:
          jpg_path:   absolute or relative path to the .jpg file
          taken:      dict with 'year' and optional 'month'
          location:   dict with 'lat', 'lng', 'display_name', 'city', 'state', 'country'
          faces:      list of face dicts from the sidecar (only labeled ones used)
          image_size: [width, height] of the image

      Returns:
          True on success, False if write failed.
      """
      tmp_path = jpg_path + ".tmp"
      try:
          # --- Build DateTimeOriginal string ---
          year  = taken["year"]
          month = taken.get("month", 1)
          dt_str = f"{year:04d}:{month:02d}:01 00:00:00"

          # --- Read existing EXIF (or start fresh) ---
          try:
              exif_dict = piexif.load(jpg_path)
          except Exception:
              exif_dict = {"0th": {}, "Exif": {}, "GPS": {}, "1st": {}, "thumbnail": None}

          # Date tags
          dt_bytes = dt_str.encode("ascii")
          exif_dict["Exif"][piexif.ExifIFD.DateTimeOriginal]  = dt_bytes
          exif_dict["Exif"][piexif.ExifIFD.DateTimeDigitized] = dt_bytes
          exif_dict["0th"][piexif.ImageIFD.DateTime]          = dt_bytes

          # GPS tags
          lat, lng = location["lat"], location["lng"]
          lat_dms = decimal_to_dms(lat)
          lng_dms = decimal_to_dms(lng)
          exif_dict["GPS"][piexif.GPSIFD.GPSLatitude]     = lat_dms
          exif_dict["GPS"][piexif.GPSIFD.GPSLongitude]    = lng_dms
          exif_dict["GPS"][piexif.GPSIFD.GPSLatitudeRef]  = b"N" if lat >= 0 else b"S"
          exif_dict["GPS"][piexif.GPSIFD.GPSLongitudeRef] = b"E" if lng >= 0 else b"W"

          # Write EXIF to tmp file (copy image bytes first)
          import shutil
          shutil.copy2(jpg_path, tmp_path)
          exif_bytes = piexif.dump(exif_dict)
          piexif.insert(exif_bytes, tmp_path)

          # --- XMP: description + MWG regions + IPTC keywords ---
          img_w, img_h = image_size[0], image_size[1]
          labeled = [f for f in faces if f.get("label")]

          xmpfile = XMPFiles(file_path=tmp_path, open_forupdate=True)
          xmp = xmpfile.get_xmp()
          if xmp is None:
              xmp = XMPMeta()

          # dc:description
          xmp.register_namespace(NS_DC, "dc")
          try:
              xmp.set_localized_text(NS_DC, "description", "x-default", "x-default",
                                     location["display_name"])
          except XMPError:
              pass

          # MWG Regions
          xmp.register_namespace(NS_MWG_RS, "mwg-rs")
          xmp.register_namespace(
              "http://www.metadataworkinggroup.com/schemas/regions/", "mwg-rs")
          # Clear existing regions then write fresh
          try:
              xmp.delete_property(NS_MWG_RS, "Regions")
          except XMPError:
              pass
          if labeled:
              for i, face in enumerate(labeled):
                  cx, cy, bw, bh = normalize_bbox(face["bbox"], img_w, img_h)
                  base = f"Regions/mwg-rs:RegionList[{i+1}]"
                  xmp.set_property(NS_MWG_RS, f"{base}/mwg-rs:Name", face["label"])
                  xmp.set_property(NS_MWG_RS, f"{base}/mwg-rs:Type", "Face")
                  xmp.set_property(NS_MWG_RS, f"{base}/mwg-rs:Area/stArea:unit", "normalized")
                  xmp.set_property_float(NS_MWG_RS, f"{base}/mwg-rs:Area/stArea:x", cx)
                  xmp.set_property_float(NS_MWG_RS, f"{base}/mwg-rs:Area/stArea:y", cy)
                  xmp.set_property_float(NS_MWG_RS, f"{base}/mwg-rs:Area/stArea:w", bw)
                  xmp.set_property_float(NS_MWG_RS, f"{base}/mwg-rs:Area/stArea:h", bh)

          # IPTC keywords (person names)
          xmp.register_namespace(NS_IPTC, "Iptc4xmpCore")
          names = list(dict.fromkeys(f["label"] for f in labeled))  # unique, ordered
          try:
              xmp.delete_property(NS_IPTC, "PersonInImage")
          except XMPError:
              pass
          for name in names:
              xmp.append_array_item(NS_IPTC, "PersonInImage",
                                    name, {"prop_array_is_ordered": False})

          xmpfile.put_xmp(xmp)
          xmpfile.close_file()

          # --- Verify tmp file opens cleanly ---
          with Image.open(tmp_path) as im:
              im.verify()

          # --- Atomic replace ---
          import os
          os.replace(tmp_path, jpg_path)
          return True

      except Exception as e:
          # Clean up tmp on failure; leave original untouched
          try:
              import os
              os.unlink(tmp_path)
          except OSError:
              pass
          print(f"[exif] write failed for {jpg_path}: {e}")
          return False
  ```

- [ ] **Step 2: Verify the function is importable**

  ```bash
  .venv/bin/python -c "from exif_pipeline import write_exif_xmp; print('OK')"
  ```

  Expected: `OK` (no import errors).

- [ ] **Step 3: Run full test suite**

  ```bash
  .venv/bin/python -m pytest tests/test_exif_pipeline.py -v
  ```

  Expected: All pass.

- [ ] **Step 4: Commit**

  ```bash
  git add exif_pipeline.py
  git commit -m "feat: add write_exif_xmp (EXIF GPS + XMP MWG regions + IPTC keywords)"
  ```

---

## Task 6: Sidecar read/write helpers

**Files:**
- Modify: `exif_pipeline.py`
- Modify: `tests/test_exif_pipeline.py`

- [ ] **Step 1: Add failing tests for sidecar helpers**

  Append to `tests/test_exif_pipeline.py`:

  ```python
  import tempfile, json, pathlib

  # --- sidecar helpers ---

  def test_load_sidecar_basic():
      data = {"image": "test.jpg", "faces": [], "image_size": [100, 100]}
      with tempfile.NamedTemporaryFile(suffix=".json", mode="w", delete=False) as f:
          json.dump(data, f)
          fname = f.name
      loaded = ep.load_sidecar(fname)
      assert loaded["image"] == "test.jpg"
      pathlib.Path(fname).unlink()

  def test_load_sidecar_missing():
      loaded = ep.load_sidecar("/nonexistent/path.json")
      assert loaded is None

  def test_save_sidecar_roundtrip():
      data = {"image": "test.jpg", "faces": [], "image_size": [100, 100],
              "taken": {"year": 1960, "source": "manual"}}
      with tempfile.TemporaryDirectory() as d:
          path = pathlib.Path(d) / "test.faces.json"
          ep.save_sidecar(str(path), data)
          loaded = ep.load_sidecar(str(path))
          assert loaded["taken"]["year"] == 1960

  def test_sidecar_tagged_true():
      data = {"taken": {"year": 1960}, "location": {"lat": 53.2, "lng": 45.0}}
      assert ep.sidecar_is_tagged(data) is True

  def test_sidecar_tagged_false_no_location():
      data = {"taken": {"year": 1960}}
      assert ep.sidecar_is_tagged(data) is False

  def test_sidecar_tagged_false_empty():
      assert ep.sidecar_is_tagged({}) is False
  ```

- [ ] **Step 2: Run tests, confirm they fail**

  ```bash
  .venv/bin/python -m pytest tests/test_exif_pipeline.py -v -k "sidecar" 2>&1 | head -20
  ```

  Expected: `AttributeError`.

- [ ] **Step 3: Implement sidecar helpers in `exif_pipeline.py`**

  Append after `write_exif_xmp`:

  ```python
  def load_sidecar(path: str) -> dict | None:
      """Load a .faces.json sidecar; return None if not found."""
      try:
          with open(path, encoding="utf-8") as f:
              return _json.load(f)
      except (FileNotFoundError, _json.JSONDecodeError):
          return None


  def save_sidecar(path: str, data: dict) -> None:
      """Write sidecar dict to path as formatted JSON."""
      with open(path, "w", encoding="utf-8") as f:
          _json.dump(data, f, indent=2, ensure_ascii=False)


  def sidecar_is_tagged(data: dict) -> bool:
      """Return True if the sidecar has both taken.year and location.lat."""
      return bool(data.get("taken", {}).get("year") and
                  data.get("location", {}).get("lat") is not None)
  ```

- [ ] **Step 4: Run all tests**

  ```bash
  .venv/bin/python -m pytest tests/test_exif_pipeline.py -v
  ```

  Expected: All pass.

- [ ] **Step 5: Commit**

  ```bash
  git add exif_pipeline.py tests/test_exif_pipeline.py
  git commit -m "feat: add load_sidecar, save_sidecar, sidecar_is_tagged helpers"
  ```

---

## Task 7: `report` subcommand

**Files:**
- Modify: `exif_pipeline.py`

- [ ] **Step 1: Append `cmd_report` function and `__main__` entry point to `exif_pipeline.py`**

  ```python
  import pathlib
  import sys


  def cmd_report(extracted_dir: str = "extracted") -> None:
      """Print EXIF tagging coverage stats for all photos in extracted_dir."""
      p = pathlib.Path(extracted_dir)
      sidecars = sorted(p.glob("*.faces.json"))
      total = len(sidecars)
      tagged = 0
      exif_written = 0
      for sc in sidecars:
          data = load_sidecar(str(sc))
          if data is None:
              continue
          if sidecar_is_tagged(data):
              tagged += 1
          if data.get("exif_written"):
              exif_written += 1
      sidecar_only = tagged - exif_written
      untagged = total - tagged
      print(f"  {total:4d} photos total")
      print(f"  {tagged:4d} tagged (date + location)   {tagged*100//max(total,1):3d}%")
      print(f"  {exif_written:4d} EXIF written to jpg        {exif_written*100//max(total,1):3d}%")
      print(f"  {sidecar_only:4d} tagged sidecar only        {sidecar_only*100//max(total,1):3d}%")
      print(f"  {untagged:4d} untagged                   {untagged*100//max(total,1):3d}%")


  if __name__ == "__main__":
      cmd = sys.argv[1] if len(sys.argv) > 1 else "tag"
      if cmd == "report":
          cmd_report()
      elif cmd == "tag":
          cmd_tag()
      else:
          print(f"Unknown command: {cmd}")
          print("Usage: exif_pipeline.py [tag|report]")
          sys.exit(1)
  ```

  Note: `cmd_tag()` is defined in Task 8. For now, add a stub:

  ```python
  def cmd_tag():
      raise NotImplementedError("GUI not yet implemented")
  ```

- [ ] **Step 2: Test report runs (will show 0s — no tagged photos yet)**

  ```bash
  .venv/bin/python exif_pipeline.py report
  ```

  Expected: Prints a table of counts (all 0 tagged until GUI is used).

- [ ] **Step 3: Run full test suite**

  ```bash
  .venv/bin/python -m pytest tests/test_exif_pipeline.py -v
  ```

  Expected: All pass.

- [ ] **Step 4: Commit**

  ```bash
  git add exif_pipeline.py
  git commit -m "feat: add report subcommand and __main__ entry point"
  ```

---

## Task 8: Theme helpers + photo loading (copied from face_pipeline)

**Files:**
- Modify: `exif_pipeline.py`

These helpers are copied from `face_pipeline.py` — do NOT import from it.

- [ ] **Step 1: Copy theme constants and helpers into `exif_pipeline.py`** (append before `cmd_tag` stub)

  ```python
  import tkinter as tk
  from tkinter import ttk
  import threading

  # ── Shared theme (copied from face_pipeline.py — no cross-import) ──────────
  ACCENT   = "#5e9cf5"
  BG       = "#1e1e2e"
  SURFACE  = "#2a2a3d"
  TEXT     = "#cdd6f4"
  MUTED    = "#6c7086"
  GREEN    = "#a6e3a1"
  AMBER    = "#f9e2af"
  RED      = "#f38ba8"
  STATE_COLORS = {"tagged": GREEN, "current": ACCENT, "skipped": MUTED, "default": SURFACE}


  def _install_theme(root: tk.Tk) -> None:
      """Configure ttk clam theme with RetroTouch color palette."""
      style = ttk.Style(root)
      style.theme_use("clam")
      style.configure(".", background=BG, foreground=TEXT, fieldbackground=SURFACE,
                      font=("Helvetica", 12))
      style.configure("TButton", background=SURFACE, foreground=TEXT, padding=6)
      style.map("TButton", background=[("active", ACCENT)])
      style.configure("Accent.TButton", background=ACCENT, foreground=BG)
      style.map("Accent.TButton", background=[("active", "#4a8ae8")])
      style.configure("TLabel", background=BG, foreground=TEXT)
      style.configure("TFrame", background=BG)
      style.configure("TEntry", fieldbackground=SURFACE, foreground=TEXT)
      style.configure("TCombobox", fieldbackground=SURFACE, foreground=TEXT)
      style.configure("TSpinbox", fieldbackground=SURFACE, foreground=TEXT)
      style.configure("Horizontal.TProgressbar", background=ACCENT, troughcolor=SURFACE)


  def load_photo_image(jpg_path: str, max_height: int = 700) -> tk.PhotoImage | None:
      """Load a JPG and return a tk.PhotoImage scaled to max_height. Returns None on error."""
      try:
          from PIL import Image, ImageTk
          img = Image.open(jpg_path)
          w, h = img.size
          if h > max_height:
              scale = max_height / h
              img = img.resize((int(w * scale), int(h * scale)), Image.LANCZOS)
          return ImageTk.PhotoImage(img)
      except Exception as e:
          print(f"[photo] load error {jpg_path}: {e}")
          return None
  ```

- [ ] **Step 2: Verify theme helpers are importable**

  ```bash
  .venv/bin/python -c "from exif_pipeline import _install_theme, load_photo_image; print('OK')"
  ```

  Expected: `OK`.

- [ ] **Step 3: Run test suite**

  ```bash
  .venv/bin/python -m pytest tests/test_exif_pipeline.py -v
  ```

  Expected: All pass.

- [ ] **Step 4: Commit**

  ```bash
  git add exif_pipeline.py
  git commit -m "feat: add theme helpers and photo image loader"
  ```

---

## Task 9: TaggerApp GUI — skeleton, photo panel, header

**Files:**
- Modify: `exif_pipeline.py`

- [ ] **Step 1: Replace the `cmd_tag` stub with the full `TaggerApp` class skeleton and `cmd_tag` function**

  Replace `def cmd_tag(): raise NotImplementedError(...)` with:

  ```python
  class TaggerApp:
      """Interactive EXIF tagger — date + location per photo."""

      def __init__(self, photos: list[str], extracted_dir: str = "extracted"):
          """
          Args:
              photos: sorted list of absolute .jpg paths
              extracted_dir: directory containing .faces.json sidecars and locations.json
          """
          self.photos = photos
          self.extracted_dir = pathlib.Path(extracted_dir)
          self.cache = LocationCache(self.extracted_dir / "locations.json")
          self.nominatim = NominatimClient()
          self.idx = 0               # current photo index
          self._photo_img = None     # keep reference to avoid GC
          self._pin = None           # current map marker

          self.root = tk.Tk()
          self.root.title("RetroTouch — EXIF Tagger")
          self.root.configure(bg=BG)
          _install_theme(self.root)
          self._build_ui()
          self._bind_keys()
          self._load_photo(0)

      # ── Build UI ──────────────────────────────────────────────────────────

      def _build_ui(self):
          self._build_header()
          pane = ttk.PanedWindow(self.root, orient=tk.HORIZONTAL)
          pane.pack(fill=tk.BOTH, expand=True, padx=8, pady=4)
          self._build_photo_panel(pane)
          self._build_sidebar(pane)

      def _build_header(self):
          hdr = ttk.Frame(self.root)
          hdr.pack(fill=tk.X, padx=8, pady=(8, 0))
          ttk.Button(hdr, text="← Prev", command=self._prev).pack(side=tk.LEFT)
          self._title_var = tk.StringVar()
          ttk.Label(hdr, textvariable=self._title_var, font=("Helvetica", 13, "bold")
                    ).pack(side=tk.LEFT, expand=True)
          ttk.Button(hdr, text="Next →", command=self._next).pack(side=tk.RIGHT)
          # Progress bar
          self._progress_var = tk.DoubleVar(value=0)
          pb = ttk.Progressbar(self.root, variable=self._progress_var,
                               maximum=len(self.photos),
                               style="Horizontal.TProgressbar")
          pb.pack(fill=tk.X, padx=8, pady=4)
          pb.bind("<Button-1>", self._on_progress_click)
          self._progress_bar = pb

      def _build_photo_panel(self, parent):
          frame = ttk.Frame(parent, width=560)
          parent.add(frame, weight=3)
          self._canvas = tk.Canvas(frame, bg=BG, highlightthickness=0)
          self._canvas.pack(fill=tk.BOTH, expand=True)

      def _build_sidebar(self, parent):
          import tkintermapview
          sb = ttk.Frame(parent, width=380)
          parent.add(sb, weight=2)

          # ── DATE section ──
          date_frame = ttk.LabelFrame(sb, text="Date", padding=8)
          date_frame.pack(fill=tk.X, padx=8, pady=(8, 4))

          row = ttk.Frame(date_frame)
          row.pack(fill=tk.X)
          ttk.Label(row, text="Year:").pack(side=tk.LEFT)
          self._year_var = tk.StringVar(value="")
          self._year_spin = ttk.Spinbox(row, from_=1850, to=2030,
                                        textvariable=self._year_var, width=6)
          self._year_spin.pack(side=tk.LEFT, padx=4)
          ttk.Label(row, text="Month:").pack(side=tk.LEFT, padx=(12, 0))
          months = ["", "Jan", "Feb", "Mar", "Apr", "May", "Jun",
                    "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
          self._month_var = tk.StringVar(value="")
          self._month_combo = ttk.Combobox(row, textvariable=self._month_var,
                                           values=months, width=5, state="readonly")
          self._month_combo.pack(side=tk.LEFT, padx=4)

          # ── LOCATION section ──
          loc_frame = ttk.LabelFrame(sb, text="Location", padding=8)
          loc_frame.pack(fill=tk.X, padx=8, pady=4)

          self._search_var = tk.StringVar()
          search_entry = ttk.Entry(loc_frame, textvariable=self._search_var)
          search_entry.pack(fill=tk.X)
          self._search_entry = search_entry
          search_entry.bind("<Return>", lambda e: self._do_search())
          self._search_var.trace_add("write", self._on_search_changed)

          # Frequent chips frame
          self._chips_frame = ttk.Frame(loc_frame)
          self._chips_frame.pack(fill=tk.X, pady=(4, 0))
          self._refresh_chips()

          # Map widget
          self._map = tkintermapview.TkinterMapView(loc_frame, width=360, height=260,
                                                    corner_radius=0)
          self._map.pack(fill=tk.BOTH, expand=True, pady=(4, 0))
          self._map.set_tile_server("https://tile.openstreetmap.org/{z}/{x}/{y}.png")
          self._map.set_position(53.2007, 45.0046)  # default: Penza
          self._map.set_zoom(6)
          self._map.add_left_click_map_command(self._on_map_click)

          # Location display label
          self._loc_display = tk.StringVar(value="")
          ttk.Label(loc_frame, textvariable=self._loc_display,
                    foreground=MUTED, wraplength=340).pack(fill=tk.X, pady=(2, 0))

          # Buttons
          btn_row = ttk.Frame(sb)
          btn_row.pack(fill=tk.X, padx=8, pady=8)
          ttk.Button(btn_row, text="✓ Save & Next", style="Accent.TButton",
                     command=self._save_and_next).pack(side=tk.LEFT, expand=True, fill=tk.X)
          ttk.Button(btn_row, text="Skip →",
                     command=self._next).pack(side=tk.RIGHT, padx=(4, 0))

          # Internal location state
          self._lat = None
          self._lng = None
          self._city = ""
          self._state = ""
          self._country = ""
          self._display_name = ""
          self._search_after_id = None

      # ── Public entry point for wiring-error testing ───────────────────────

      def _show(self):
          """Run one event loop iteration (for GUI wiring tests)."""
          self.root.update_idletasks()
          self.root.update()

      def run(self):
          self.root.mainloop()
  ```

- [ ] **Step 2: Verify the class is importable**

  ```bash
  .venv/bin/python -c "from exif_pipeline import TaggerApp; print('OK')"
  ```

  Expected: `OK`.

- [ ] **Step 3: Smoke-test GUI wiring (auto-closes)**

  ```bash
  .venv/bin/python -c "
  import exif_pipeline as ep, pathlib
  photos = sorted(str(p) for p in pathlib.Path('extracted').glob('*.jpg'))[:3]
  app = ep.TaggerApp(photos)
  app._show()
  app.root.after(150, app.root.destroy)
  app.root.mainloop()
  print('GUI wiring OK')
  " 2>&1 | grep -v "^$"
  ```

  Expected: `GUI wiring OK` with no Python exceptions.

- [ ] **Step 4: Commit**

  ```bash
  git add exif_pipeline.py
  git commit -m "feat: TaggerApp skeleton — header, photo panel, sidebar layout"
  ```

---

## Task 10: TaggerApp — photo loading, face overlays, auto-fill

**Files:**
- Modify: `exif_pipeline.py`

- [ ] **Step 1: Implement `_load_photo`, `_draw_faces`, `_autofill` methods inside `TaggerApp`**

  Append inside the `TaggerApp` class:

  ```python
      def _load_photo(self, idx: int):
          if not self.photos:
              return
          self.idx = idx % len(self.photos)
          jpg = self.photos[self.idx]
          stem = pathlib.Path(jpg).stem
          self._title_var.set(f"{pathlib.Path(jpg).name}  {self.idx+1}/{len(self.photos)}")
          self._progress_var.set(self.idx)

          # Load sidecar
          sc_path = str(self.extracted_dir / f"{stem}.faces.json")
          self._sidecar = load_sidecar(sc_path) or {"image": pathlib.Path(jpg).name,
                                                      "faces": [], "image_size": [1, 1]}

          # Load photo image
          self._photo_img = load_photo_image(jpg, max_height=680)
          self._canvas.delete("all")
          if self._photo_img:
              self._canvas.create_image(0, 0, anchor="nw", image=self._photo_img)
              self._canvas.configure(scrollregion=self._canvas.bbox("all"))
              self._draw_faces()

          # Auto-fill from sidecar or filename
          self._autofill()

      def _draw_faces(self):
          """Overlay labeled face boxes on the canvas."""
          if not self._photo_img:
              return
          img_w, img_h = self._sidecar.get("image_size", [1, 1])
          disp_w = self._photo_img.width()
          disp_h = self._photo_img.height()
          sx = disp_w / img_w
          sy = disp_h / img_h
          for face in self._sidecar.get("faces", []):
              label = face.get("label", "")
              if not label:
                  continue
              x1, y1, x2, y2 = face["bbox"]
              self._canvas.create_rectangle(
                  x1*sx, y1*sy, x2*sx, y2*sy,
                  outline=ACCENT, width=2, tags="face"
              )
              self._canvas.create_text(
                  x1*sx + 4, y1*sy - 2, text=label,
                  anchor="sw", fill=ACCENT, font=("Helvetica", 10, "bold"), tags="face"
              )

      def _autofill(self):
          """Pre-fill year/month and location from sidecar or filename."""
          sc = self._sidecar
          if sc.get("taken"):
              self._year_var.set(str(sc["taken"]["year"]))
              month = sc["taken"].get("month")
              months = ["", "Jan", "Feb", "Mar", "Apr", "May", "Jun",
                        "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
              self._month_var.set(months[month] if month else "")
          else:
              year, hint = parse_filename(pathlib.Path(self.photos[self.idx]).name)
              self._year_var.set(str(year) if year else "")
              self._month_var.set("")
              if hint:
                  self._search_var.set(hint)
                  threading.Thread(target=self._search_and_fly,
                                   args=(hint,), daemon=True).start()

          if sc.get("location"):
              loc = sc["location"]
              self._set_location(loc["lat"], loc["lng"], loc["city"],
                                 loc.get("state", ""), loc["country"],
                                 loc["display_name"], fly=True)

      def _on_progress_click(self, event):
          frac = event.x / self._progress_bar.winfo_width()
          idx = int(frac * len(self.photos))
          self._load_photo(max(0, min(idx, len(self.photos) - 1)))

      def _prev(self):
          self._load_photo(self.idx - 1)

      def _next(self):
          self._load_photo(self.idx + 1)
  ```

- [ ] **Step 2: Smoke-test with real photos**

  ```bash
  .venv/bin/python -c "
  import exif_pipeline as ep, pathlib
  photos = sorted(str(p) for p in pathlib.Path('extracted').glob('*.jpg'))[:3]
  app = ep.TaggerApp(photos)
  app._show()
  app.root.after(500, app.root.destroy)
  app.root.mainloop()
  print('OK')
  " 2>&1 | tail -3
  ```

  Expected: `OK`, no Python exceptions.

- [ ] **Step 3: Commit**

  ```bash
  git add exif_pipeline.py
  git commit -m "feat: TaggerApp photo loading, face overlays, autofill"
  ```

---

## Task 11: TaggerApp — search, map interaction, location state

**Files:**
- Modify: `exif_pipeline.py`

- [ ] **Step 1: Implement search and map methods inside `TaggerApp`**

  Append inside `TaggerApp`:

  ```python
      # ── Location / map ────────────────────────────────────────────────────

      def _refresh_chips(self):
          for w in self._chips_frame.winfo_children():
              w.destroy()
          for entry in self.cache.top(8):
              name = entry.get("city") or entry.get("display_name", "?")
              btn = ttk.Button(self._chips_frame, text=name,
                               command=lambda e=entry: self._apply_cache_entry(e))
              btn.pack(side=tk.LEFT, padx=2, pady=2)

      def _apply_cache_entry(self, entry: dict):
          self._set_location(entry["lat"], entry["lng"], entry["city"],
                             entry.get("state", ""), entry["country"],
                             entry["display_name"], fly=True)
          self._search_var.set(entry["display_name"])

      def _on_search_changed(self, *_):
          if self._search_after_id:
              self.root.after_cancel(self._search_after_id)
          self._search_after_id = self.root.after(400, self._trigger_search)

      def _trigger_search(self):
          query = self._search_var.get().strip()
          if len(query) >= 2:
              threading.Thread(target=self._search_and_fly,
                               args=(query,), daemon=True).start()

      def _search_and_fly(self, query: str):
          results = self.nominatim.search(query)
          if not results:
              return
          r = results[0]
          lat = float(r["lat"])
          lng = float(r["lon"])
          city, state, country, display = parse_nominatim_address(r)
          self.root.after(0, lambda: self._set_location(
              lat, lng, city, state, country, display, fly=True))

      def _on_map_click(self, coords):
          lat, lng = coords
          if self._pin:
              self._pin.delete()
          self._pin = self._map.set_marker(lat, lng)
          threading.Thread(target=self._reverse_geocode,
                           args=(lat, lng), daemon=True).start()

      def _reverse_geocode(self, lat: float, lng: float):
          result = self.nominatim.reverse(lat, lng)
          if result:
              city, state, country, display = parse_nominatim_address(result)
          else:
              city = state = country = display = ""
          self.root.after(0, lambda: self._set_location(
              lat, lng, city, state, country, display, fly=False))

      def _set_location(self, lat: float, lng: float, city: str, state: str,
                        country: str, display_name: str, fly: bool = False):
          self._lat, self._lng = lat, lng
          self._city, self._state, self._country = city, state, country
          self._display_name = display_name
          self._loc_display.set(display_name)
          if self._pin:
              self._pin.delete()
          self._pin = self._map.set_marker(lat, lng)
          if fly:
              self._map.set_position(lat, lng)
              self._map.set_zoom(10)
  ```

- [ ] **Step 2: Smoke-test**

  ```bash
  .venv/bin/python -c "
  import exif_pipeline as ep, pathlib
  photos = sorted(str(p) for p in pathlib.Path('extracted').glob('*.jpg'))[:2]
  app = ep.TaggerApp(photos)
  app._show()
  app.root.after(600, app.root.destroy)
  app.root.mainloop()
  print('OK')
  " 2>&1 | tail -3
  ```

  Expected: `OK`.

- [ ] **Step 3: Commit**

  ```bash
  git add exif_pipeline.py
  git commit -m "feat: TaggerApp map search, click-to-set-pin, reverse geocode"
  ```

---

## Task 12: TaggerApp — save logic, keyboard bindings

**Files:**
- Modify: `exif_pipeline.py`

- [ ] **Step 1: Implement `_save_and_next` and `_bind_keys` inside `TaggerApp`**

  Append inside `TaggerApp`:

  ```python
      # ── Save ──────────────────────────────────────────────────────────────

      def _save_and_next(self):
          year_str = self._year_var.get().strip()
          if not year_str.isdigit():
              return  # no year — don't save
          year = int(year_str)

          month_str = self._month_var.get()
          months = ["", "Jan", "Feb", "Mar", "Apr", "May", "Jun",
                    "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
          month = months.index(month_str) if month_str in months[1:] else None

          if self._lat is None:
              return  # no location — don't save

          # Update sidecar
          sc = self._sidecar
          sc["taken"] = format_taken(year, month)
          sc["location"] = {
              "lat": self._lat, "lng": self._lng,
              "display_name": self._display_name,
              "city": self._city, "state": self._state, "country": self._country,
              "source": "manual",
          }

          # Write EXIF to jpg
          jpg = self.photos[self.idx]
          success = write_exif_xmp(
              jpg, sc["taken"], sc["location"],
              sc.get("faces", []), sc.get("image_size", [1, 1])
          )
          sc["exif_written"] = success

          # Save sidecar
          stem = pathlib.Path(jpg).stem
          sc_path = str(self.extracted_dir / f"{stem}.faces.json")
          save_sidecar(sc_path, sc)

          # Update location cache
          self.cache.record(self._lat, self._lng, self._city,
                            self._state, self._country, self._display_name)
          self._refresh_chips()

          # Advance
          self._load_photo(self.idx + 1)

      # ── Keyboard bindings ─────────────────────────────────────────────────

      def _bind_keys(self):
          months = ["", "Jan", "Feb", "Mar", "Apr", "May", "Jun",
                    "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]

          def on_key(event):
              k = event.keysym
              c = event.char
              focused = self.root.focus_get()
              in_entry = isinstance(focused, (ttk.Entry, tk.Entry))

              if k == "Return" and not in_entry:
                  self._save_and_next()
              elif k == "Escape":
                  self._next()
              elif c == "[":
                  self._prev()
              elif c == "]":
                  self._next()
              elif c == "?" or k == "F1":
                  self._show_shortcuts()
              elif not in_entry:
                  if k == "Left":
                      yr = self._year_var.get()
                      if yr.isdigit():
                          self._year_var.set(str(int(yr) - 1))
                  elif k == "Right":
                      yr = self._year_var.get()
                      if yr.isdigit():
                          self._year_var.set(str(int(yr) + 1))
                  elif k == "Up":
                      cur = self._month_var.get()
                      idx = months.index(cur) if cur in months else 0
                      self._month_var.set(months[(idx + 1) % 13])
                  elif k == "Down":
                      cur = self._month_var.get()
                      idx = months.index(cur) if cur in months else 0
                      self._month_var.set(months[(idx - 1) % 13])
                  elif c == "m":
                      self._month_var.set("")
                  elif c == "t":
                      self._search_entry.focus_set()

          self.root.bind_all("<Key>", on_key)

          # Tab: jump to search (prevent default focus traversal)
          self._search_entry.bind("<Tab>", lambda e: (self._search_entry.focus_set(), "break"))

      def _show_shortcuts(self):
          win = tk.Toplevel(self.root, bg=BG)
          win.title("Keyboard Shortcuts")
          lines = [
              ("←  /  →",    "Nudge year  −1 / +1"),
              ("↑  /  ↓",    "Nudge month"),
              ("m",           "Clear month"),
              ("t",           "Focus location search"),
              ("Enter",       "Save & Next"),
              ("Esc",         "Skip (no save)"),
              ("[  /  ]",     "Prev / Next photo"),
              ("?  / F1",     "This help"),
          ]
          for key, desc in lines:
              row = ttk.Frame(win)
              row.pack(fill=tk.X, padx=16, pady=2)
              ttk.Label(row, text=key, width=14, anchor="e",
                        foreground=ACCENT).pack(side=tk.LEFT)
              ttk.Label(row, text=desc).pack(side=tk.LEFT, padx=8)
          win.bind("<Escape>", lambda e: win.destroy())
          win.bind("<Key-Return>", lambda e: win.destroy())
  ```

- [ ] **Step 2: Wire up `cmd_tag` function (replace the stub)**

  Replace `def cmd_tag(): raise NotImplementedError(...)` with:

  ```python
  def cmd_tag(extracted_dir: str = "extracted") -> None:
      photos = sorted(str(p) for p in pathlib.Path(extracted_dir).glob("*.jpg"))
      if not photos:
          print(f"No .jpg files found in {extracted_dir}/")
          return
      app = TaggerApp(photos, extracted_dir)
      app.run()
  ```

- [ ] **Step 3: Run full test suite**

  ```bash
  .venv/bin/python -m pytest tests/test_exif_pipeline.py -v
  ```

  Expected: All pass.

- [ ] **Step 4: Smoke-test full GUI with auto-close**

  ```bash
  .venv/bin/python -c "
  import exif_pipeline as ep, pathlib
  photos = sorted(str(p) for p in pathlib.Path('extracted').glob('*.jpg'))[:3]
  app = ep.TaggerApp(photos)
  app._show()
  app.root.after(800, app.root.destroy)
  app.root.mainloop()
  print('Full GUI OK')
  " 2>&1 | tail -3
  ```

  Expected: `Full GUI OK`.

- [ ] **Step 5: Commit**

  ```bash
  git add exif_pipeline.py
  git commit -m "feat: TaggerApp save logic + keyboard bindings + shortcuts popover"
  ```

---

## Task 13: Update CLAUDE.md and run full test suite

**Files:**
- Modify: `CLAUDE.md`

- [ ] **Step 1: Add EXIF pipeline section to `CLAUDE.md`**

  Add a new section after the `## Photo restoration` section:

  ```markdown
  ## EXIF Pipeline (exif_pipeline.py)
  - `python exif_pipeline.py tag` - interactive GUI to tag each extracted photo with year (required), month (optional), and location (lat/lng + city/state/country). Auto-fills year and location from filename (e.g. `1960-penza-00004_04.jpg` → year=1960, map flies to Penza).
  - `python exif_pipeline.py report` - print tagging coverage stats (read-only).
  - Extends the existing `*.faces.json` sidecar with top-level `taken` and `location` keys; also writes EXIF DateTimeOriginal, GPS tags, IPTC Keywords (face names), and XMP MWG Regions (face rectangles) to the `.jpg` file.
  - Maintains `extracted/locations.json` — a cache of lat/lng → human name mappings with use counts. Entries within 1000m are coalesced. Top 8 by use count appear as quick-select chips above the map.
  - Map: `tkintermapview` (OpenStreetMap tiles). Geocoding: Nominatim (free, no API key, 1 req/sec rate limit enforced). EXIF: `piexif`. XMP: `python-xmp-toolkit` (requires `brew install exempi`).
  - Safe write: EXIF/XMP written to `.jpg.tmp`, verified with Pillow, then `os.replace()` — original never corrupted.
  - Sidecar `taken` dict: `{"year": 1960, "month": 4, "source": "manual"}` — `month` key omitted entirely when unknown.
  - Sidecar `location` dict: `{"lat": 53.2, "lng": 45.0, "display_name": "...", "city": "...", "state": "...", "country": "...", "source": "manual"}`.
  - `exif_written: true` added to sidecar after successful EXIF write.
  - Pure helpers TDD-tested: `parse_filename`, `haversine`, `coalesce_location`, `format_taken`, `parse_nominatim_address`, `decimal_to_dms`, `normalize_bbox`, `load_sidecar`, `save_sidecar`, `sidecar_is_tagged`.
  - GUI verified manually. System dependency: `brew install exempi` (for `python-xmp-toolkit`).
  ```

- [ ] **Step 2: Run the complete test suite**

  ```bash
  .venv/bin/python -m pytest tests/ -q
  ```

  Expected: All tests pass. Note any failures and fix before proceeding.

- [ ] **Step 3: Commit**

  ```bash
  git add CLAUDE.md
  git commit -m "docs: add exif_pipeline section to CLAUDE.md"
  ```

---

## Task 14: Manual end-to-end verification

This task cannot be automated — it requires a human at the GUI.

- [ ] **Step 1: Launch the tagger**

  ```bash
  .venv/bin/python exif_pipeline.py tag
  ```

- [ ] **Step 2: Verify auto-fill**

  On the first photo (e.g. `1950-penza-00001_01.jpg`):
  - Year field shows `1950`
  - Map flies to Penza and drops a pin
  - Location display shows "Penza, Penza Oblast, Russia" (or similar)

- [ ] **Step 3: Verify keyboard controls**

  - Press `←` / `→` — year changes by 1
  - Press `↑` / `↓` — month cycles
  - Press `m` — month clears
  - Press `?` — shortcuts popover appears, closes on Esc

- [ ] **Step 4: Tag one photo and verify sidecar**

  - Set year, leave month blank, confirm map pin is set
  - Press `Enter` to save
  - Check the sidecar:

  ```bash
  cat extracted/1950-penza-00001_01.faces.json | python3 -m json.tool | grep -A 10 '"taken"'
  ```

  Expected: `"taken": {"year": 1950, "source": "manual"}` (no month key).

- [ ] **Step 5: Verify EXIF written to jpg**

  ```bash
  .venv/bin/python -c "
  import piexif
  d = piexif.load('extracted/1950-penza-00001_01.jpg')
  print('Date:', d['Exif'].get(piexif.ExifIFD.DateTimeOriginal))
  print('GPS lat:', d['GPS'].get(piexif.GPSIFD.GPSLatitude))
  "
  ```

  Expected: Date and GPS fields are populated.

- [ ] **Step 6: Verify locations.json created**

  ```bash
  cat extracted/locations.json | python3 -m json.tool | head -20
  ```

  Expected: One entry with `use_count: 1`, city/state/country, lat/lng.

- [ ] **Step 7: Run report**

  ```bash
  .venv/bin/python exif_pipeline.py report
  ```

  Expected: 1 tagged photo shown in stats.

- [ ] **Step 8: Final commit**

  ```bash
  git add -A
  git commit -m "feat: exif_pipeline.py complete — date/location tagging GUI with EXIF/XMP write"
  ```
