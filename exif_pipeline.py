#!/usr/bin/env python3
"""exif_pipeline.py — EXIF timestamp + location tagging for scanned photos.

Commands:
    python exif_pipeline.py tag      # interactive tagging GUI
    python exif_pipeline.py report   # coverage stats (read-only)
"""
import re
import math
import os
import time
import pathlib
import urllib.request
import urllib.parse
import json as _json


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
        if not isinstance(entry, dict) or "lat" not in entry or "lng" not in entry:
            continue
        if haversine(lat, lng, entry["lat"], entry["lng"]) <= tolerance_m:
            return entry
    return None


def format_taken(year: int, month: int | None = None) -> dict:
    """Build the 'taken' sidecar dict. month is omitted (not None) when unknown."""
    if year <= 0:
        raise ValueError("year must be a positive integer")
    if month is not None and not (1 <= month <= 12):
        raise ValueError("month must be between 1 and 12 (inclusive)")
    d = {"year": year, "source": "manual"}
    if month is not None:
        d["month"] = month
    return d


def parse_nominatim_address(response: dict | None) -> tuple:
    """Parse a Nominatim geocode/reverse response into (city, state, country, display_name)."""
    if not response or not isinstance(response, dict):
        return "", "", "", ""
    addr = response.get("address", {})
    if not isinstance(addr, dict):
        addr = {}
    city = addr.get("city") or addr.get("town") or addr.get("village") or addr.get("hamlet") or addr.get("suburb") or addr.get("municipality") or ""
    state = addr.get("state") or addr.get("province") or ""
    country = addr.get("country") or ""
    display = ", ".join(filter(None, [city, state, country]))
    return city, state, country, display


def decimal_to_dms(deg: float) -> tuple:
    """Convert decimal degrees to (deg, min, sec) as piexif rational tuples [(num,den)...]."""
    if not (0.0 <= abs(deg) <= 180.0):
        raise ValueError("degrees must be between -180.0 and 180.0 (inclusive)")
    deg = abs(deg)
    total_sec_hundredths = round(deg * 360000)
    d = total_sec_hundredths // 360000
    rem = total_sec_hundredths % 360000
    m = rem // 6000
    s_hundredths = rem % 6000
    return (d, 1), (m, 1), (s_hundredths, 100)


def normalize_bbox(bbox: list, img_w: int, img_h: int) -> tuple:
    """Convert pixel bbox [x1,y1,x2,y2] to MWG normalized (cx,cy,bw,bh) in [0..1]."""
    if img_w <= 0 or img_h <= 0:
        raise ValueError("Image dimensions must be positive integers")
    x1, y1, x2, y2 = bbox
    # Ensure coordinate order
    x1, x2 = min(x1, x2), max(x1, x2)
    y1, y2 = min(y1, y2), max(y1, y2)
    # Clamp pixel inputs to image bounds
    x1 = max(0, min(x1, img_w))
    x2 = max(0, min(x2, img_w))
    y1 = max(0, min(y1, img_h))
    y2 = max(0, min(y2, img_h))

    cx = (x1 + x2) / 2 / img_w
    cy = (y1 + y2) / 2 / img_h
    bw = (x2 - x1) / img_w
    bh = (y2 - y1) / img_h

    # Clamp outputs to [0.0, 1.0]
    cx = max(0.0, min(cx, 1.0))
    cy = max(0.0, min(cy, 1.0))
    bw = max(0.0, min(bw, 1.0))
    bh = max(0.0, min(bh, 1.0))

    return cx, cy, bw, bh


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
        try:
            req = urllib.request.Request(url, headers={"User-Agent": self.UA})
            with urllib.request.urlopen(req, timeout=8) as r:
                return _json.loads(r.read())
        finally:
            self._last = time.monotonic()

    def search(self, query: str) -> list:
        """Return list of Nominatim result dicts for query string."""
        if query in self._cache:
            return self._cache[query]
        params = urllib.parse.urlencode({"q": query, "format": "json",
                                         "addressdetails": 1, "limit": 5})
        try:
            results = self._get(f"{self.BASE}/search?{params}")
        except Exception:
            return []
        self._cache[query] = results
        return results

    def reverse(self, lat: float, lng: float) -> dict | None:
        """Reverse-geocode (lat, lng); return result dict or None on failure."""
        try:
            lat = float(lat)
            lng = float(lng)
        except (TypeError, ValueError):
            return None
        if not (-90.0 <= lat <= 90.0) or not (-180.0 <= lng <= 180.0):
            return None
        params = urllib.parse.urlencode({"lat": lat, "lon": lng,
                                         "format": "json", "addressdetails": 1})
        try:
            return self._get(f"{self.BASE}/reverse?{params}")
        except Exception:
            return None


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
            try:
                with open(self.path, encoding="utf-8") as f:
                    content = _json.load(f)
                    if isinstance(content, dict):
                        raw_locations = content.get("locations", [])
                        self._data = raw_locations if isinstance(raw_locations, list) else []
                    else:
                        self._data = []
            except (FileNotFoundError, _json.JSONDecodeError):
                self._data = []

    def _save(self):
        self.path.parent.mkdir(parents=True, exist_ok=True)
        tmp_path = self.path.with_suffix(self.path.suffix + ".tmp")
        with open(tmp_path, "w", encoding="utf-8") as f:
            _json.dump({"locations": self._data}, f, indent=2, ensure_ascii=False)
        os.replace(tmp_path, self.path)

    def top(self, n: int = 8) -> list:
        """Return top-n entries by use_count."""
        return sorted(self._data, key=lambda e: e.get("use_count", 1), reverse=True)[:n]

    def all_entries(self) -> list:
        return list(self._data)

    def record(self, lat: float, lng: float, city: str, state: str,
               country: str, display_name: str) -> dict:
        """Add or update a location entry; return the (possibly updated) entry."""
        existing = coalesce_location(lat, lng, self._data, self.TOLERANCE_M)
        if existing:
            existing["use_count"] = existing.get("use_count", 1) + 1
            self._save()
            return existing
        entry = {"lat": lat, "lng": lng, "display_name": display_name,
                 "city": city, "state": state, "country": country, "use_count": 1}
        self._data.append(entry)
        self._save()
        return entry
