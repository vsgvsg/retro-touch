import sys
import pathlib
import json
import os
import argparse
from exif_pipeline import NominatimClient, coalesce_location, parse_nominatim_address, load_sidecar, save_sidecar, write_exif_xmp

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
