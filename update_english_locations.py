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
