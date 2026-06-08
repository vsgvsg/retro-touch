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
