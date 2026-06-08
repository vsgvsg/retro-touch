# EXIF Pipeline Design Spec
**Date:** 2026-06-08
**Feature:** `exif_pipeline.py` — EXIF timestamp + location tagging GUI
**Status:** Approved, ready for implementation

---

## Overview

A new standalone tool (`exif_pipeline.py`) that lets a human annotate scanned photo crops with:
1. **Date taken** — year (required) + month (optional)
2. **Location** — lat/lng + human-readable City/State/Country

Data is written to two places:
- The existing `*.faces.json` sidecar (source of truth, merged in-place)
- EXIF/XMP/IPTC tags inside the `.jpg` file (for Photos.app, Lightroom, etc.)

The tool also writes face names as IPTC Keywords and XMP MWG Region metadata so that face assignments made in `face_pipeline.py` are embedded into the image file.

---

## Architecture

### File: `exif_pipeline.py`
Single standalone file, no cross-imports with other tools. Shares the same ttk "clam" theme + helpers (copied in, not imported). Follows the exact same pattern as `face_pipeline.py` and `split_photos.py`.

**Commands:**
```
.venv/bin/python exif_pipeline.py tag      # open the interactive tagging GUI
.venv/bin/python exif_pipeline.py report   # print coverage stats (read-only)
```

---

## Data Model

### Sidecar extension (`extracted/*.faces.json`)
New top-level keys added to the existing sidecar — no new file type:

```json
{
  "image": "1960-penza-00004_04.jpg",
  "image_size": [2250, 1670],
  "model": "buffalo_l",
  "taken": {
    "year": 1960,
    "month": 4,
    "source": "manual"
  },
  "location": {
    "lat": 53.2007,
    "lng": 45.0046,
    "display_name": "Penza, Penza Oblast, Russia",
    "city": "Penza",
    "state": "Penza Oblast",
    "country": "Russia",
    "source": "manual"
  },
  "exif_written": true,
  "faces": [ ... ]
}
```

- `taken.month` is omitted (not null) when month is unknown
- `taken.source` and `location.source` are always `"manual"` (user-entered)
- `exif_written` flag distinguishes "tagged sidecar only" from "tagged + EXIF written to jpg"

### Location cache (`extracted/locations.json`)
A flat list of all known lat/lng -> human name mappings, built incrementally:

```json
{
  "locations": [
    {
      "lat": 53.2007,
      "lng": 45.0046,
      "display_name": "Penza, Penza Oblast, Russia",
      "city": "Penza",
      "state": "Penza Oblast",
      "country": "Russia",
      "use_count": 47
    }
  ]
}
```

- New pins within **1000m** of an existing entry increment `use_count` and reuse the cached name
- Pins farther than 1000m from all existing entries create a new entry with `use_count: 1`
- `use_count` drives the "frequent locations" chip strip in the UI

---

## GUI Layout

Same ttk clam theme as `face_pipeline.py`, `split_photos.py`. Standard three-zone layout:

```
+---------------------------------------------------------------------+
|  HEADER: [<- Prev]  1960-penza-00004_04.jpg  47/312  [Next ->]     |
|          ####################..........  (clickable progress bar)   |
+-------------------------------+-------------------------------------+
|                               |  DATE                               |
|                               |  Year: [1960]  Month: [Apr v] (opt)|
|                               |  <- -> nudge year  up/dn nudge mo  |
|   PHOTO (fit-to-height)       +-------------------------------------+
|   (labeled faces boxed,       |  LOCATION                           |
|    name overlaid)             |  [Penza, Penza Oblast, Russia    x] |
|                               |  +- Frequent -----------------------+|
|                               |  | [Penza] [Orenburg] [Hanlar] ... ||
|                               |  +---------------------------------+|
|                               |  +- Map ---------------------------+|
|                               |  |   (tkintermapview widget)       ||
|                               |  |   click = set pin               ||
|                               |  |   scroll = zoom                 ||
|                               |  |   drag = pan                    ||
|                               |  +---------------------------------+|
|                               |  [Check Save & Next]   [Skip ->]    |
+-------------------------------+-------------------------------------+
```

**Photo panel:** Photo displayed fit-to-height. Labeled faces are boxed with name overlaid (same style as the match GUI). Unassigned/unlabeled faces are not shown in this view.

**Progress bar:** Clickable to jump to any photo (same `_on_progress_click` pattern). Color coding:
- Green = tagged (both `taken.year` and `location.lat` set)
- Grey = skipped
- Accent = current
- Default = unvisited

---

## Keyboard Shortcuts

| Key     | Action                                            |
|---------|---------------------------------------------------|
| <- / -> | Nudge year -1 / +1 (app-wide, any widget)         |
| up / dn | Nudge month +1 / -1                               |
| 0-9     | Type year directly in year Spinbox                |
| m       | Toggle month: clear to blank if set               |
| Enter   | Save & advance to next photo                      |
| Tab     | Jump focus to location search box                 |
| Esc     | Skip without saving, advance to next              |
| [ / ]   | Navigate to previous / next photo                 |
| ? / F1  | Show keyboard shortcuts popover                   |

Tk binding notes (same gotchas as `split_photos.py`):
- App-wide shortcuts use `root.bind_all("<Key>", ...)`, not `root.bind`
- Punctuation keys matched on `event.char`, named keys on `event.keysym`
- `Tab` bound on widget instance with `return "break"` to prevent focus-traversal capture

---

## Auto-Fill Behavior

On loading each photo:
1. **Parse filename** using `parse_filename()` — extract 4-digit year prefix and location token
   - e.g. `1960-penza-00004_04.jpg` -> year=`1960`, hint=`"penza"`
2. **Pre-fill year** field with parsed year
3. **Location lookup**: Nominatim search for hint -> fly map to first result, drop pin, populate fields
4. **If sidecar already has `taken`/`location`** -> load those values instead (re-edit mode)

---

## Map & Location

### Map widget
`tkintermapview` with OpenStreetMap tiles:
- `map_widget.set_address(query)` for flying to a location
- `map_widget.set_marker(lat, lng)` for pin
- `map_widget.add_left_click_map_command(callback)` for click-to-set-pin

### Search flow
1. User types in search box -> 400ms debounce -> Nominatim geocode query
2. Up to 5 results shown in dropdown
3. User picks result -> map flies there, pin set, city/state/country fields populated
4. Alternatively: click directly on map -> pin set -> background Nominatim reverse-geocode -> fields update

### Frequent locations strip
- Top 8 entries by `use_count` from `locations.json`
- Displayed as chip buttons above the map
- Click chip -> map flies there + pin set + fields filled (no network call)
- Strip refreshes after each save

### Nominatim address parsing
```python
city    = address.get("city") or address.get("town") or address.get("village") or ""
state   = address.get("state") or address.get("province") or ""
country = address.get("country") or ""
display = ", ".join(filter(None, [city, state, country]))
```

### Rate limiting
All Nominatim calls go through a `NominatimClient` that enforces 1.1s minimum between requests.
City-name lookups are cached in a session dict to avoid repeated identical queries.

---

## EXIF / XMP Writing

### Libraries
- **`piexif`** — EXIF and GPS tag read/write
- **`python-xmp-toolkit`** — XMP MWG Regions (requires `brew install exempi`)

### Tags written

| Tag                     | Value                                                      |
|-------------------------|------------------------------------------------------------|
| Exif.DateTimeOriginal   | "YYYY:MM:01 00:00:00" (day/time fixed when unknown)        |
| Exif.DateTimeDigitized  | Same as DateTimeOriginal                                   |
| GPS.GPSLatitude         | DMS rational from location.lat                             |
| GPS.GPSLongitude        | DMS rational from location.lng                             |
| GPS.GPSLatitudeRef      | "N" or "S"                                                 |
| GPS.GPSLongitudeRef     | "E" or "W"                                                 |
| IPTC Keywords           | List of unique non-empty face label values                 |
| XMP dc:description      | location.display_name                                      |
| XMP mwg-rs:RegionList   | One region per labeled face (name + normalized bbox)       |

### XMP MWG Region bbox normalization
```python
cx = (x1 + x2) / 2 / image_width
cy = (y1 + y2) / 2 / image_height
w  = (x2 - x1) / image_width
h  = (y2 - y1) / image_height
```

### Safe write sequence
1. Write updated EXIF+XMP to `<photo>.jpg.tmp`
2. Verify tmp file opens cleanly with Pillow
3. `os.replace(tmp, original)` — atomic, no corrupt originals
4. Update sidecar with `taken`, `location`, `exif_written: true`

---

## `report` Subcommand

```
exif_pipeline report
  312 photos total
  247 tagged (date + location)   79%
  198 EXIF written to jpg        64%
   65 tagged sidecar only        21%
   65 untagged                   21%
```

Read-only — no writes.

---

## Pure Helpers (TDD-tested)

| Function | Description |
|----------|-------------|
| `parse_filename(name) -> (year, hint)` | Extract year and location hint from filename |
| `coalesce_location(lat, lng, cache, tolerance_m=1000) -> entry or None` | Find existing cache entry within tolerance |
| `haversine(lat1, lng1, lat2, lng2) -> float` | Distance in meters between two points |
| `format_taken(year, month=None) -> dict` | Build the `taken` sidecar dict |
| `parse_nominatim_address(response) -> (city, state, country, display_name)` | Parse Nominatim address |
| `decimal_to_dms(deg) -> tuple` | Convert decimal degrees to DMS rationals for piexif |
| `normalize_bbox(bbox, w, h) -> (cx, cy, bw, bh)` | Normalize pixel bbox for MWG XMP |

---

## Dependencies Added to `requirements.txt`

```
tkintermapview>=1.29
piexif>=1.1
python-xmp-toolkit>=2.0
```

System dependency (documented in README):
```
brew install exempi
```

---

## CLAUDE.md Updates

Document the new tool following the same pattern as existing entries:
- Command syntax for `tag` and `report`
- Sidecar format additions (`taken`, `location`, `exif_written`)
- `locations.json` format and 1000m coalescing rule
- Tk binding gotchas (reuse existing note)
- Safe EXIF write pattern (write tmp -> verify -> atomic replace)
- Pure helper list and TDD scope
