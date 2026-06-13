#!/usr/bin/env python3
"""exif_pipeline.py — EXIF timestamp + location tagging for scanned photos.

Commands:
    python exif_pipeline.py tag      # interactive tagging GUI
    python exif_pipeline.py report   # coverage stats (read-only)
"""
import sys
import re
import math
import os
import time
import pathlib
import urllib.request
import urllib.parse
import json as _json
import shutil
import piexif
import piexif.helper
from PIL import Image, ImageTk
from libxmp import XMPFiles, XMPMeta, XMPError
import tkinter as tk
from tkinter import ttk
import threading



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
                                         "addressdetails": 1, "limit": 10,
                                         "accept-language": "en"})
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
                                         "format": "json", "addressdetails": 1,
                                         "accept-language": "en"})
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

    def remove(self, entry: dict):
        """Remove a location entry from the cache by matching coordinates."""
        self._data = [
            e for e in self._data
            if not (isinstance(e, dict) and
                    e.get("lat") == entry.get("lat") and
                    e.get("lng") == entry.get("lng"))
        ]
        self._save()

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


# XMP namespaces
NS_MWG_RS  = "http://www.metadataworkinggroup.com/schemas/regions/"
NS_DC      = "http://purl.org/dc/elements/1.1/"
NS_IPTC_EXT = "http://iptc.org/std/Iptc4xmpExt/2008-02-29/"
NS_ST_DIM   = "http://ns.adobe.com/xap/1.0/sType/Dimensions#"

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
        # Copy image bytes first to tmp
        shutil.copy2(jpg_path, tmp_path)

        # --- XMP: description + MWG regions + IPTC keywords ---
        img_w, img_h = image_size[0], image_size[1]
        labeled = [f for f in faces if f.get("label")]

        xmpfile = XMPFiles(file_path=tmp_path, open_forupdate=True)
        try:
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
            xmp.register_namespace("http://ns.adobe.com/xmp/sType/Area#", "stArea")
            xmp.register_namespace(NS_ST_DIM, "stDim")
            # Clear existing regions then write fresh
            try:
                xmp.delete_property(NS_MWG_RS, "Regions")
            except XMPError:
                pass

            # Add mwg-rs:AppliedToDimensions to Regions
            xmp.set_property(NS_MWG_RS, "Regions/mwg-rs:AppliedToDimensions/stDim:unit", "pixel")
            xmp.set_property_int(NS_MWG_RS, "Regions/mwg-rs:AppliedToDimensions/stDim:w", img_w)
            xmp.set_property_int(NS_MWG_RS, "Regions/mwg-rs:AppliedToDimensions/stDim:h", img_h)

            if labeled:
                for i, face in enumerate(labeled):
                    xmp.append_array_item(NS_MWG_RS, "Regions/mwg-rs:RegionList", None, {"prop_array_is_unordered": True}, prop_value_is_struct=True)
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
            xmp.register_namespace(NS_IPTC_EXT, "Iptc4xmpExt")
            names = list(dict.fromkeys(f["label"] for f in labeled))  # unique, ordered
            try:
                xmp.delete_property(NS_IPTC_EXT, "PersonInImage")
            except XMPError:
                pass
            for name in names:
                xmp.append_array_item(NS_IPTC_EXT, "PersonInImage",
                                      name, {"prop_array_is_unordered": True})

            xmpfile.put_xmp(xmp)
        finally:
            try:
                xmpfile.close_file()
            except Exception:
                pass

        # --- Build DateTimeOriginal string ---
        year  = taken["year"]
        month = taken.get("month", 1)
        dt_str = f"{year:04d}:{month:02d}:01 00:00:00"

        # --- Read existing EXIF (or start fresh) ---
        try:
            exif_dict = piexif.load(tmp_path)
        except Exception:
            exif_dict = {"0th": {}, "Exif": {}, "GPS": {}, "1st": {}, "thumbnail": None}

        # Defensive EXIF initialization
        for key in ("0th", "Exif", "GPS"):
            if key not in exif_dict:
                exif_dict[key] = {}

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

        # Write EXIF to tmp file
        exif_bytes = piexif.dump(exif_dict)
        piexif.insert(exif_bytes, tmp_path)

        # --- Verify tmp file opens cleanly ---
        with Image.open(tmp_path) as im:
            im.verify()

        # --- Atomic replace ---
        os.replace(tmp_path, jpg_path)
        return True

    except Exception as e:
        # Clean up tmp on failure; leave original untouched
        try:
            os.unlink(tmp_path)
        except OSError:
            pass
        print(f"[exif] write failed for {jpg_path}: {e}")
        return False


def load_sidecar(path: str) -> dict | None:
    """Load a .faces.json sidecar; return None if not found or malformed."""
    try:
        with open(path, encoding="utf-8") as f:
            content = _json.load(f)
            return content if isinstance(content, dict) else None
    except (OSError, _json.JSONDecodeError):
        return None


def save_sidecar(path: str, data: dict) -> None:
    """Write sidecar dict to path as formatted JSON safely using atomic replacement."""
    tmp_path = path + ".tmp"
    try:
        with open(tmp_path, "w", encoding="utf-8") as f:
            _json.dump(data, f, indent=2, ensure_ascii=False)
        os.replace(tmp_path, path)
    except Exception as e:
        try:
            os.unlink(tmp_path)
        except OSError:
            pass
        raise e


def sidecar_is_tagged(data: dict) -> bool:
    """Return True if the sidecar has both taken.year and location.lat."""
    if not isinstance(data, dict):
        return False
    taken = data.get("taken")
    location = data.get("location")
    
    has_year = isinstance(taken, dict) and bool(taken.get("year"))
    has_lat = isinstance(location, dict) and location.get("lat") is not None
    return has_year and has_lat


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

# ── Shared theme (copied from face_pipeline.py — no cross-import) ──────────
ACCENT   = "#5a6cf0"
BG       = "#fafaff"
SURFACE  = "#ffffff"
TEXT     = "#1a1a2e"
MUTED    = "#7a7a88"
GREEN    = "#2faf6a"
AMBER    = "#d8a23a"
RED      = "#ff6b6b"
STATE_COLORS = {"tagged": GREEN, "current": ACCENT, "skipped": MUTED, "default": SURFACE}


def _install_theme(root: tk.Tk) -> None:
    """Configure ttk clam theme with RetroTouch color palette."""
    root.configure(bg=BG)
    style = ttk.Style(root)
    try:
        style.theme_use("clam")
    except tk.TclError:
        pass
    style.configure(".", background=BG, foreground=TEXT, fieldbackground=SURFACE,
                    font=("TkDefaultFont", 11))
    style.configure("TFrame", background=BG)
    style.configure("TLabel", background=BG, foreground=TEXT)
    style.configure("Sub.TLabel", background=BG, foreground=MUTED)
    style.configure("Title.TLabel", background=BG, foreground=TEXT,
                    font=("TkDefaultFont", 14, "bold"))
    style.configure("TButton", padding=(12, 6), relief="flat",
                    background=SURFACE, foreground=TEXT)
    style.map("TButton", background=[("active", "#f0f0f8")])
    style.configure("Accent.TButton", padding=(14, 6), relief="flat",
                    background=ACCENT, foreground="#ffffff",
                    font=("TkDefaultFont", 11, "bold"))
    style.map("Accent.TButton", background=[("active", "#4a5ce0")])
    style.configure("TEntry", fieldbackground=SURFACE, foreground=TEXT, padding=4)
    style.configure("TCombobox", fieldbackground=SURFACE, foreground=TEXT)
    style.configure("TSpinbox", fieldbackground=SURFACE, foreground=TEXT)
    style.configure("Horizontal.TProgressbar", background=ACCENT, troughcolor="#ececf2")
    style.configure("TLabelframe", background=BG)
    style.configure("TLabelframe.Label", background=BG, foreground=TEXT,
                    font=("TkDefaultFont", 11, "bold"))

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


def load_photo_image(jpg_path: str, max_height: int = 680, max_width: int = 560) -> tk.PhotoImage | None:
    """Load a JPG and return a tk.PhotoImage scaled to max_height and max_width. Returns None on error."""
    try:
        img = Image.open(jpg_path)
        w, h = img.size
        scale = 1.0
        if w > max_width or h > max_height:
            scale = min(max_width / w, max_height / h)
            img = img.resize((int(w * scale), int(h * scale)), Image.LANCZOS)
        return ImageTk.PhotoImage(img)
    except Exception as e:
        print(f"[photo] load error {jpg_path}: {e}")
        return None


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
        self._bg_threads: list = []  # track background threads for clean teardown
        self._last_saved_year = None
        self._last_saved_month = None

        self.root = tk.Tk()
        self.root.title("RetroTouch — EXIF Tagger")
        self.root.configure(bg=BG)
        _install_theme(self.root)
        self._destroyed = False
        self._build_ui()
        self._bind_keys()
        self.root.protocol("WM_DELETE_WINDOW", self.destroy)
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
        ttk.Label(hdr, textvariable=self._title_var, style="Title.TLabel"
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
        loc_frame.pack(fill=tk.BOTH, expand=True, padx=8, pady=4)

        self._search_var = tk.StringVar()
        search_entry = ttk.Entry(loc_frame, textvariable=self._search_var)
        search_entry.pack(fill=tk.X)
        self._search_entry = search_entry
        search_entry.bind("<Return>", lambda e: self._do_search())
        self._search_var.trace_add("write", self._on_search_changed)

        # Scrollable frequent chips container
        self._chips_container = ttk.Frame(loc_frame)
        self._chips_container.pack(fill=tk.X, pady=(4, 0))

        self._chips_canvas = tk.Canvas(self._chips_container, height=40, bg=BG, highlightthickness=0)
        self._chips_canvas.pack(fill=tk.X, side=tk.TOP, expand=True)

        self._chips_frame = ttk.Frame(self._chips_canvas)
        self._chips_canvas.create_window((0, 0), window=self._chips_frame, anchor="nw")

        def _on_chips_configure(event):
            self._chips_canvas.configure(scrollregion=self._chips_canvas.bbox("all"))
        self._chips_frame.bind("<Configure>", _on_chips_configure)

        def _on_chips_wheel(event):
            if sys.platform == "darwin":
                self._chips_canvas.xview_scroll(-1 * event.delta, "units")
            else:
                self._chips_canvas.xview_scroll(-1 * (event.delta // 120), "units")
        self._chips_canvas.bind("<MouseWheel>", _on_chips_wheel)
        self._chips_frame.bind("<MouseWheel>", _on_chips_wheel)

        self._refresh_chips()

        # Map widget
        self._map = tkintermapview.TkinterMapView(loc_frame, width=360, height=360,
                                                  corner_radius=0)
        self._map.pack(fill=tk.BOTH, expand=True, pady=(4, 0))
        self._map.set_tile_server("https://tile.openstreetmap.org/{z}/{x}/{y}.png")
        self._map.set_position(53.2007, 45.0046)  # default: Penza
        self._map.set_zoom(6)
        self._map.add_left_click_map_command(self._on_map_click)

        # Custom macOS scroll-zoom fix
        def custom_mouse_zoom(event):
            relative_mouse_x = event.x / self._map.width
            relative_mouse_y = event.y / self._map.height
            
            raw_delta = event.delta
            if raw_delta == 0:
                return
                
            if sys.platform == "darwin":
                if abs(raw_delta) >= 120:
                    step = raw_delta / 120.0
                else:
                    step = raw_delta * 0.1
            elif sys.platform.startswith("win"):
                step = raw_delta * 0.01
            elif event.num == 4:
                step = 1.0
            elif event.num == 5:
                step = -1.0
            else:
                step = raw_delta * 0.1
                
            new_zoom = self._map.zoom + step
            self._map.set_zoom(new_zoom, relative_pointer_x=relative_mouse_x, relative_pointer_y=relative_mouse_y)

        self._map._custom_mouse_zoom = custom_mouse_zoom
        self._map.canvas.bind("<MouseWheel>", custom_mouse_zoom)
        self._map.canvas.bind("<Button-4>", custom_mouse_zoom)
        self._map.canvas.bind("<Button-5>", custom_mouse_zoom)

        # Location display label
        self._loc_display = tk.StringVar(value="")
        ttk.Label(loc_frame, textvariable=self._loc_display,
                  foreground=MUTED, wraplength=340).pack(fill=tk.X, pady=(2, 0))

        # Buttons
        btn_row = ttk.Frame(sb)
        btn_row.pack(fill=tk.X, padx=8, pady=8)
        self._copy_prev_btn = ttk.Button(btn_row, text="Copy Prev",
                                         command=self._copy_previous)
        self._copy_prev_btn.pack(side=tk.LEFT, padx=(0, 4))
        self._shortcuts_btn = ttk.Button(btn_row, text="?", width=3,
                                         command=self._show_shortcuts)
        self._shortcuts_btn.pack(side=tk.RIGHT, padx=(4, 0))
        ttk.Button(btn_row, text="Skip →",
                   command=self._next).pack(side=tk.RIGHT, padx=(4, 0))
        ttk.Button(btn_row, text="✓ Save & Next", style="Accent.TButton",
                   command=self._save_and_next).pack(side=tk.LEFT, expand=True, fill=tk.X)

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

    def destroy(self):
        """Mark destroyed and tear down the Tk root safely."""
        self._destroyed = True
        # Wait for background threads to finish (max 2s each for rate-limit sleep)
        for t in self._bg_threads:
            t.join(timeout=2.0)
        self._bg_threads.clear()
        try:
            self.root.destroy()
        except tk.TclError:
            pass

    def _safe_after(self, ms, func):
        """Schedule func on the Tk event loop, silently ignoring if root is destroyed."""
        if self._destroyed:
            return
        try:
            self.root.after(ms, func)
        except (tk.TclError, RuntimeError):
            pass

    def _bg_run(self, target, *args):
        """Spawn a daemon thread, track it for clean teardown."""
        # Clean up finished threads
        self._bg_threads = [t for t in self._bg_threads if t.is_alive()]
        t = threading.Thread(target=target, args=args, daemon=True)
        self._bg_threads.append(t)
        t.start()

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
                elif c == "c":
                    if self.idx > 0:
                        self._copy_previous()

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
            ("c",           "Copy from previous photo"),
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

        # Enable/disable "Copy Prev" button based on previous photo's sidecar presence
        if self.idx <= 0:
            self._copy_prev_btn.configure(state=tk.DISABLED)
        else:
            prev_jpg = self.photos[self.idx - 1]
            prev_stem = pathlib.Path(prev_jpg).stem
            prev_sc_path = str(self.extracted_dir / f"{prev_stem}.faces.json")
            prev_sc = load_sidecar(prev_sc_path)
            if prev_sc and (prev_sc.get("taken") or prev_sc.get("location")):
                self._copy_prev_btn.configure(state=tk.NORMAL)
            else:
                self._copy_prev_btn.configure(state=tk.DISABLED)

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
        elif getattr(self, "_last_saved_year", None) is not None:
            self._year_var.set(str(self._last_saved_year))
            self._month_var.set(self._last_saved_month or "")
            year, hint = parse_filename(pathlib.Path(self.photos[self.idx]).name)
            if hint:
                self._search_var.set(hint)
                self._bg_run(self._search_and_fly, hint)
        else:
            year, hint = parse_filename(pathlib.Path(self.photos[self.idx]).name)
            self._year_var.set(str(year) if year else "")
            self._month_var.set("")
            if hint:
                self._search_var.set(hint)
                self._bg_run(self._search_and_fly, hint)

        if sc.get("location"):
            loc = sc["location"]
            self._set_location(loc["lat"], loc["lng"], loc["city"],
                               loc.get("state", ""), loc["country"],
                               loc["display_name"], fly=True)

    def _search_and_fly(self, query: str):
        if self._destroyed:
            return
        results = self.nominatim.search(query)
        if not results or self._destroyed:
            return
        r = results[0]
        lat = float(r["lat"])
        lng = float(r["lon"])
        city, state, country, display = parse_nominatim_address(r)
        self._safe_after(0, lambda: self._set_location(
            lat, lng, city, state, country, display, fly=True))

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

    def _refresh_chips(self):
        for w in self._chips_frame.winfo_children():
            w.destroy()
        
        # Display all entries sorted by frequency
        entries = sorted(self.cache.all_entries(), key=lambda e: e.get("use_count", 1), reverse=True)
        
        def _on_chips_wheel(event):
            if sys.platform == "darwin":
                self._chips_canvas.xview_scroll(-1 * event.delta, "units")
            else:
                self._chips_canvas.xview_scroll(-1 * (event.delta // 120), "units")

        for entry in entries:
            name = entry.get("city") or entry.get("display_name", "?")
            btn = ttk.Button(self._chips_frame, text=name,
                             command=lambda e=entry: self._apply_cache_entry(e))
            btn.pack(side=tk.LEFT, padx=2, pady=2)
            
            # Bind scrolling to the button
            btn.bind("<MouseWheel>", _on_chips_wheel)
            
            # Bind Cmd/Ctrl+Click to delete
            btn.bind("<Command-Button-1>", lambda e, entry=entry: self._remove_cache_entry(entry))
            btn.bind("<Control-Button-1>", lambda e, entry=entry: self._remove_cache_entry(entry))

    def _remove_cache_entry(self, entry: dict):
        self.cache.remove(entry)
        self._refresh_chips()
        return "break"

    def _apply_cache_entry(self, entry: dict):
        self._set_location(entry["lat"], entry["lng"], entry["city"],
                           entry.get("state", ""), entry["country"],
                           entry["display_name"], fly=True)
        self._search_var.set(entry["display_name"])

    def _do_search(self):
        query = self._search_var.get().strip()
        if len(query) >= 2:
            self._bg_run(self._search_and_fly, query)

    def _on_search_changed(self, *_):
        if self._search_after_id:
            self.root.after_cancel(self._search_after_id)
        self._search_after_id = self.root.after(400, self._trigger_search)

    def _trigger_search(self):
        query = self._search_var.get().strip()
        if len(query) >= 2:
            self._bg_run(self._search_and_fly, query)

    def _on_map_click(self, coords):
        lat, lng = coords
        if self._pin:
            self._pin.delete()
        self._pin = self._map.set_marker(lat, lng)
        self._bg_run(self._reverse_geocode, lat, lng)

    def _reverse_geocode(self, lat: float, lng: float):
        if self._destroyed:
            return
        result = self.nominatim.reverse(lat, lng)
        if result:
            city, state, country, display = parse_nominatim_address(result)
        else:
            city = state = country = display = ""
        self._safe_after(0, lambda: self._set_location(
            lat, lng, city, state, country, display, fly=False))

    def _save_and_next(self):
        year_str = self._year_var.get().strip()
        if not year_str.isdigit():
            return  # no year — don't save
        year = int(year_str)

        month_str = self._month_var.get()
        months = ["", "Jan", "Feb", "Mar", "Apr", "May", "Jun",
                  "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
        month = months.index(month_str) if month_str in months[1:] else None
        self._last_saved_year = year
        self._last_saved_month = month_str

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

    def _prev(self):
        self._load_photo(self.idx - 1)

    def _next(self):
        self._load_photo(self.idx + 1)

    def _copy_previous(self):
        if self.idx <= 0:
            return
        prev_jpg = self.photos[self.idx - 1]
        prev_stem = pathlib.Path(prev_jpg).stem
        prev_sc_path = str(self.extracted_dir / f"{prev_stem}.faces.json")
        prev_sc = load_sidecar(prev_sc_path)
        if not prev_sc:
            return

        # Overwrite taken info if present
        if prev_sc.get("taken"):
            taken = prev_sc["taken"]
            if taken.get("year"):
                self._year_var.set(str(taken["year"]))
            month = taken.get("month")
            months = ["", "Jan", "Feb", "Mar", "Apr", "May", "Jun",
                      "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
            if month is not None and (1 <= month <= 12):
                self._month_var.set(months[month])
            else:
                self._month_var.set("")

        # Overwrite location info if present
        if prev_sc.get("location"):
            loc = prev_sc["location"]
            self._set_location(
                loc["lat"], loc["lng"], loc["city"],
                loc.get("state", ""), loc["country"],
                loc["display_name"], fly=True
            )

    def _on_progress_click(self, event):
        frac = event.x / self._progress_bar.winfo_width()
        idx = int(frac * len(self.photos))
        self._load_photo(max(0, min(idx, len(self.photos) - 1)))



def cmd_tag(extracted_dir: str = "extracted") -> None:
    photos = sorted(str(p) for p in pathlib.Path(extracted_dir).glob("*.jpg"))
    if not photos:
        print(f"No .jpg files found in {extracted_dir}/")
        return
    app = TaggerApp(photos, extracted_dir)
    app.run()


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



