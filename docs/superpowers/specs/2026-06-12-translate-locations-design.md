# Design Spec: Translate Locations to English

**Date:** 2026-06-12  
**Topic:** English translation/standardization of location data in JSON sidecars, EXIF tags, and the location cache.

## 1. Overview
The user has location metadata stored in `.faces.json` sidecars under `extracted/`, as well as a centralized `extracted/locations.json` cache. Some of these location names are in non-English languages (e.g. Russian, German, Greek, Azerbaijani, Ukrainian). We need a script to update all existing location names to their English counterparts.

## 2. Requirements & Scope
* **Re-geocode all locations:** Query Nominatim reverse geocoding for all coordinates found in the dataset with `accept-language=en` to ensure we retrieve the latest English location names.
* **Update JSON Sidecars:** Update `location.display_name`, `location.city`, `location.state`, and `location.country` in any sidecar containing location details.
* **Update EXIF/XMP Image Tags:** For any sidecar where `exif_written` is `true`, update the matching `.jpg` file using the existing `write_exif_xmp` routine.
* **Update Central Cache:** Update `extracted/locations.json` to reflect the updated English location names.
* **Dry Run Support:** Implement a `--dry-run` flag to preview proposed changes (before vs. after name comparisons) without modifying any files.
* **Respect Rate Limits:** Enforce Nominatim's rate limit policies (1.1s delay between queries) and use in-memory caching to avoid duplicate queries for identical/close coordinates.

## 3. Architecture & Code Integration
We will create a standalone script `update_english_locations.py` located in the root of the workspace.

Since we want to avoid duplicate work and keep formatting/writing safe, we will import the following helpers directly from `exif_pipeline.py`:
* `load_sidecar`
* `save_sidecar`
* `write_exif_xmp`
* `NominatimClient`
* `coalesce_location`
* `parse_nominatim_address`

### Core Algorithm
1. **Find all location data:**
   * Load `extracted/locations.json` and find all cached locations.
   * Scan `extracted/*.faces.json` sidecars to identify files that contain a `"location"` dictionary.
2. **Translate coordinates:**
   * Collect all distinct coordinates (lat, lng) from sidecars and the location cache.
   * For each unique coordinate (within a 1000m tolerance), request reverse geocoding via `NominatimClient` (which defaults to English).
   * Cache responses in memory to prevent duplicate requests.
3. **Dry-Run / Commit Phase:**
   * Display before/after diffs of location strings (e.g. `Рамзай, Пензенская область, Россия` -> `Ramzay, Penza Oblast, Russia`).
   * If not in `--dry-run` mode:
     * Save updated `*.faces.json` sidecars.
     * Re-write EXIF/XMP tags on the `.jpg` image files if `exif_written` is true.
     * Save updated `extracted/locations.json`.
