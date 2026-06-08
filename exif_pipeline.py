#!/usr/bin/env python3
"""exif_pipeline.py — EXIF timestamp + location tagging for scanned photos.

Commands:
    python exif_pipeline.py tag      # interactive tagging GUI
    python exif_pipeline.py report   # coverage stats (read-only)
"""
import re
import math

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
        if haversine(lat, lng, entry["lat"], entry["lng"]) <= tolerance_m:
            return entry
    return None


def format_taken(year: int, month: int | None = None) -> dict:
    """Build the 'taken' sidecar dict. month is omitted (not None) when unknown."""
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
    city = addr.get("city") or addr.get("town") or addr.get("village") or ""
    state = addr.get("state") or addr.get("province") or ""
    country = addr.get("country") or ""
    display = ", ".join(filter(None, [city, state, country]))
    return city, state, country, display


def decimal_to_dms(deg: float) -> tuple:
    """Convert decimal degrees to (deg, min, sec) as piexif rational tuples [(num,den)...]."""
    deg = abs(deg)
    total_sec_hundredths = round(deg * 360000)
    d = total_sec_hundredths // 360000
    rem = total_sec_hundredths % 360000
    m = rem // 6000
    s_hundredths = rem % 6000
    return (d, 1), (m, 1), (s_hundredths, 100)


def normalize_bbox(bbox: list, img_w: int, img_h: int) -> tuple:
    """Convert pixel bbox [x1,y1,x2,y2] to MWG normalized (cx,cy,bw,bh) in [0..1]."""
    x1, y1, x2, y2 = bbox
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


