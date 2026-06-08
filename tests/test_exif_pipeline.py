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

# --- haversine ---

def test_haversine_same_point():
    assert ep.haversine(53.2007, 45.0046, 53.2007, 45.0046) == pytest.approx(0.0)

def test_haversine_known_distance():
    # Moscow (55.7558, 37.6173) to Penza (53.2007, 45.0046) is ~555 km
    dist = ep.haversine(55.7558, 37.6173, 53.2007, 45.0046)
    assert 530_000 < dist < 580_000

def test_haversine_short_distance():
    # Two points ~333m apart (0.003 degrees latitude ~= 333m)
    dist = ep.haversine(53.2007, 45.0046, 53.2037, 45.0046)
    assert 300 < dist < 400

# --- coalesce_location ---

CACHE = [
    {"lat": 53.2007, "lng": 45.0046, "display_name": "Penza, Russia",
     "city": "Penza", "state": "Penza Oblast", "country": "Russia", "use_count": 5},
    {"lat": 51.7727, "lng": 55.0988, "display_name": "Orenburg, Russia",
     "city": "Orenburg", "state": "Orenburg Oblast", "country": "Russia", "use_count": 3},
]

def test_coalesce_location_hit():
    # Point ~500m from Penza -> should match within 1000m
    entry = ep.coalesce_location(53.2052, 45.0046, CACHE, tolerance_m=1000)
    assert entry is not None
    assert entry["city"] == "Penza"

def test_coalesce_location_non_first_hit():
    # Point ~500m from Orenburg (2nd entry in CACHE) -> should match within 1000m
    # Orenburg is at (51.7727, 55.0988)
    entry = ep.coalesce_location(51.7730, 55.0988, CACHE, tolerance_m=1000)
    assert entry is not None
    assert entry["city"] == "Orenburg"


def test_coalesce_location_miss():
    # Point far from both -> no match
    entry = ep.coalesce_location(52.0000, 44.0000, CACHE, tolerance_m=1000)
    assert entry is None

def test_coalesce_location_exact():
    entry = ep.coalesce_location(53.2007, 45.0046, CACHE, tolerance_m=1000)
    assert entry is not None
    assert entry["city"] == "Penza"

def test_coalesce_location_empty_cache():
    entry = ep.coalesce_location(53.2007, 45.0046, [], tolerance_m=1000)
    assert entry is None


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

def test_format_taken_invalid_month():
    with pytest.raises(ValueError):
        ep.format_taken(1960, month=13)
    with pytest.raises(ValueError):
        ep.format_taken(1960, month=0)
    with pytest.raises(ValueError):
        ep.format_taken(1960, month=-5)

def test_format_taken_invalid_year():
    with pytest.raises(ValueError):
        ep.format_taken(0)
    with pytest.raises(ValueError):
        ep.format_taken(-1960)


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

def test_parse_nominatim_none():
    city, state, country, display = ep.parse_nominatim_address(None)
    assert city == state == country == display == ""

def test_parse_nominatim_address_non_dict_address():
    city, state, country, display = ep.parse_nominatim_address({"address": "not-a-dict"})
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
    assert secs == (0, 100)

def test_decimal_to_dms_negative():
    deg, mins, secs = ep.decimal_to_dms(-53.2007)
    assert deg == (53, 1)
    assert mins == (12, 1)
    d_secs = secs[0] / secs[1]
    assert abs(d_secs - 2.52) < 0.1

def test_decimal_to_dms_out_of_range():
    with pytest.raises(ValueError):
        ep.decimal_to_dms(180.1)
    with pytest.raises(ValueError):
        ep.decimal_to_dms(-181.0)
    # 180.0 is boundary, should not raise
    deg, mins, secs = ep.decimal_to_dms(180.0)
    assert deg == (180, 1)


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

def test_normalize_bbox_clamping():
    cx, cy, bw, bh = ep.normalize_bbox([-50, -50, 150, 150], 100, 100)
    assert cx == pytest.approx(0.5)
    assert cy == pytest.approx(0.5)
    assert bw == pytest.approx(1.0)
    assert bh == pytest.approx(1.0)

def test_normalize_bbox_invalid_dimensions():
    with pytest.raises(ValueError):
        ep.normalize_bbox([0, 0, 10, 10], 0, 10)
    with pytest.raises(ValueError):
        ep.normalize_bbox([0, 0, 10, 10], 10, -5)

def test_normalize_bbox_coordinate_order():
    cx, cy, bw, bh = ep.normalize_bbox([100, 200, 0, 0], 100, 200)
    assert cx == pytest.approx(0.5)
    assert cy == pytest.approx(0.5)
    assert bw == pytest.approx(1.0)
    assert bh == pytest.approx(1.0)


# --- NominatimClient and LocationCache ---

def test_nominatim_client_search_cache(monkeypatch):
    # Test NominatimClient exists and uses cache
    from exif_pipeline import NominatimClient
    client = NominatimClient()
    
    call_count = 0
    def mock_get(url):
        nonlocal call_count
        call_count += 1
        return [{"lat": "53.2007", "lon": "45.0046"}]
    
    monkeypatch.setattr(client, "_get", mock_get)
    
    res1 = client.search("Penza")
    res2 = client.search("Penza")
    
    assert res1 == [{"lat": "53.2007", "lon": "45.0046"}]
    assert res2 == res1
    assert call_count == 1  # Cache was used

def test_nominatim_client_reverse(monkeypatch):
    from exif_pipeline import NominatimClient
    client = NominatimClient()
    
    def mock_get(url):
        return {"address": {"city": "Penza"}}
        
    monkeypatch.setattr(client, "_get", mock_get)
    
    res = client.reverse(53.2007, 45.0046)
    assert res == {"address": {"city": "Penza"}}

def test_location_cache_recording(tmp_path):
    from exif_pipeline import LocationCache
    db_file = tmp_path / "locations.json"
    cache = LocationCache(path=db_file)
    
    # Empty cache initially
    assert cache.all_entries() == []
    
    # Record first location
    entry1 = cache.record(53.2007, 45.0046, "Penza", "Penza Oblast", "Russia", "Penza, Russia")
    assert entry1["use_count"] == 1
    assert len(cache.all_entries()) == 1
    
    # Record same location (should coalesce within 1000m and increase use_count)
    # 53.2008, 45.0046 is extremely close to 53.2007, 45.0046
    entry2 = cache.record(53.2008, 45.0046, "Penza", "Penza Oblast", "Russia", "Penza, Russia")
    assert entry2["use_count"] == 2
    assert len(cache.all_entries()) == 1
    
    # Record far location
    entry3 = cache.record(55.7558, 37.6173, "Moscow", "Moscow", "Russia", "Moscow, Russia")
    assert entry3["use_count"] == 1
    assert len(cache.all_entries()) == 2
    
    # Test top() method
    top_entries = cache.top(1)
    assert len(top_entries) == 1
    assert top_entries[0]["city"] == "Penza"

    # Initialize second LocationCache and assert it correctly loads the newly recorded entries
    cache2 = LocationCache(path=db_file)
    entries2 = cache2.all_entries()
    assert len(entries2) == 2
    assert entries2[0]["city"] == "Penza"
    assert entries2[0]["use_count"] == 2
    assert entries2[1]["city"] == "Moscow"
    assert entries2[1]["use_count"] == 1

def test_location_cache_record_non_first_hit(tmp_path):
    from exif_pipeline import LocationCache
    db_file = tmp_path / "locations.json"
    cache = LocationCache(path=db_file)
    
    # Add first location
    cache.record(53.2007, 45.0046, "Penza", "Penza Oblast", "Russia", "Penza, Russia")
    # Add second location
    cache.record(51.7727, 55.0988, "Orenburg", "Orenburg Oblast", "Russia", "Orenburg, Russia")
    
    # Record near the second location
    entry = cache.record(51.7730, 55.0988, "Orenburg", "Orenburg Oblast", "Russia", "Orenburg, Russia")
    
    assert entry["city"] == "Orenburg"
    assert entry["use_count"] == 2
    
    # Check the actual list
    entries = cache.all_entries()
    assert len(entries) == 2
    assert entries[0]["city"] == "Penza"
    assert entries[0]["use_count"] == 1
    assert entries[1]["city"] == "Orenburg"
    assert entries[1]["use_count"] == 2

def test_nominatim_client_search_error_handling(monkeypatch):
    from exif_pipeline import NominatimClient
    client = NominatimClient()
    
    def mock_get_fail(url):
        raise OSError("Connection timed out")
        
    monkeypatch.setattr(client, "_get", mock_get_fail)
    
    # search() should swallow the error and return []
    res = client.search("Some Query")
    assert res == []

def test_location_cache_corrupted_json(tmp_path):
    from exif_pipeline import LocationCache
    db_file = tmp_path / "locations.json"
    
    # Write corrupted/invalid JSON
    with open(db_file, "w", encoding="utf-8") as f:
        f.write("{invalid json...")
        
    cache = LocationCache(path=db_file)
    # It should fall back to empty list on JSONDecodeError
    assert cache.all_entries() == []

class DummyResponse:
    def read(self):
        return b"[]"
    def __enter__(self):
        return self
    def __exit__(self, exc_type, exc_val, exc_tb):
        pass

def test_nominatim_client_rate_limiting(monkeypatch):
    import time
    import urllib.request
    from exif_pipeline import NominatimClient
    client = NominatimClient()
    
    # Mock urlopen
    def mock_urlopen(req, timeout=None):
        return DummyResponse()
    monkeypatch.setattr(urllib.request, "urlopen", mock_urlopen)
    
    # Control monotonic time
    t_values = [100.0, 100.5, 101.0, 101.0]
    def mock_monotonic():
        return t_values.pop(0) if t_values else 200.0
    monkeypatch.setattr(time, "monotonic", mock_monotonic)
    
    sleep_times = []
    monkeypatch.setattr(time, "sleep", sleep_times.append)
    
    # First request
    client._get("https://nominatim.openstreetmap.org/search?q=1")
    
    # Second request (elapsed is 0.5s, should sleep 0.6s)
    client._get("https://nominatim.openstreetmap.org/search?q=2")
    
    assert len(sleep_times) == 1
    assert sleep_times[0] == pytest.approx(0.6)

def test_location_cache_non_dict_json(tmp_path):
    from exif_pipeline import LocationCache
    db_file = tmp_path / "locations.json"
    
    # Write non-dict JSON (array)
    with open(db_file, "w", encoding="utf-8") as f:
        f.write("[]")
    cache = LocationCache(path=db_file)
    assert cache.all_entries() == []
    
    # Write non-dict JSON (string)
    with open(db_file, "w", encoding="utf-8") as f:
        f.write('"abc"')
    cache2 = LocationCache(path=db_file)
    assert cache2.all_entries() == []

def test_nominatim_client_reverse_coordinate_validation():
    from exif_pipeline import NominatimClient
    client = NominatimClient()
    
    # Invalid lat/lng should return None immediately without getting called
    assert client.reverse(90.1, 100.0) is None
    assert client.reverse(-90.1, 100.0) is None
    assert client.reverse(45.0, 180.1) is None
    assert client.reverse(45.0, -180.1) is None

def test_location_cache_invalid_locations_type(tmp_path):
    from exif_pipeline import LocationCache
    db_file = tmp_path / "locations.json"
    
    # Write JSON where "locations" is a string, not a list
    with open(db_file, "w", encoding="utf-8") as f:
        f.write('{"locations": "not-a-list"}')
    
    cache = LocationCache(path=db_file)
    assert cache.all_entries() == []

def test_coalesce_location_malformed_entries():
    # cache contains a non-dict and dicts missing "lat"/"lng"
    malformed_cache = [
        "not-a-dict",
        {"city": "Penza"}, # missing lat/lng
        {"lat": 53.2007, "city": "Penza"}, # missing lng
        {"lng": 45.0046, "city": "Penza"}, # missing lat
        {"lat": 53.2007, "lng": 45.0046, "city": "Penza"}, # valid
    ]
    entry = ep.coalesce_location(53.2007, 45.0046, malformed_cache, tolerance_m=1000)
    assert entry is not None
    assert entry["city"] == "Penza"

def test_nominatim_client_reverse_type_conversion(monkeypatch):
    from exif_pipeline import NominatimClient
    client = NominatimClient()
    
    def mock_get(url):
        return {"address": {"city": "Penza"}}
    monkeypatch.setattr(client, "_get", mock_get)
    
    # Strings representing valid coordinates should be parsed as float and succeed
    res = client.reverse("53.2007", "45.0046")
    assert res == {"address": {"city": "Penza"}}
    
    # Non-numeric string coordinates should return None
    assert client.reverse("abc", "45.0046") is None


# --- write_exif_xmp ---

def test_write_exif_xmp(tmp_path):
    from PIL import Image
    import piexif
    from libxmp import XMPFiles
    
    # 1. Create a dummy image
    img_path = str(tmp_path / "test.jpg")
    img = Image.new("RGB", (100, 100), color="blue")
    img.save(img_path, "JPEG")
    
    taken = {"year": 1995, "month": 8}
    location = {
        "lat": 53.2007,
        "lng": -45.0046,
        "display_name": "Test Location, World",
        "city": "Test City",
        "state": "Test State",
        "country": "Test Country"
    }
    faces = [
        {"bbox": [10, 20, 30, 40], "label": "Alice"},
        {"bbox": [50, 50, 70, 70], "label": "Bob"},
        {"bbox": [0, 0, 10, 10], "label": None}  # unlabeled, should be ignored
    ]
    image_size = [100, 100]
    
    # Call write_exif_xmp (should be defined in ep)
    success = ep.write_exif_xmp(img_path, taken, location, faces, image_size)
    assert success is True
    
    # 2. Verify EXIF metadata
    exif_dict = piexif.load(img_path)
    assert exif_dict["Exif"][piexif.ExifIFD.DateTimeOriginal] == b"1995:08:01 00:00:00"
    
    gps = exif_dict["GPS"]
    # Check GPS values and refs
    assert gps[piexif.GPSIFD.GPSLatitudeRef] == b"N"
    assert gps[piexif.GPSIFD.GPSLongitudeRef] == b"W"
    
    # Convert lat/lng back from DMS to verify
    def dms_to_decimal(dms, ref):
        # dms is ((d, 1), (m, 1), (s, 100))
        d = dms[0][0] / dms[0][1]
        m = dms[1][0] / dms[1][1]
        s = dms[2][0] / dms[2][1]
        val = d + m / 60.0 + s / 3600.0
        if ref in (b"S", b"W"):
            val = -val
        return val

    assert dms_to_decimal(gps[piexif.GPSIFD.GPSLatitude], gps[piexif.GPSIFD.GPSLatitudeRef]) == pytest.approx(53.2007)
    assert dms_to_decimal(gps[piexif.GPSIFD.GPSLongitude], gps[piexif.GPSIFD.GPSLongitudeRef]) == pytest.approx(-45.0046)
    
    # 3. Verify XMP metadata
    xmp_file = XMPFiles(file_path=img_path, open_forupdate=False)
    xmp = xmp_file.get_xmp()
    assert xmp is not None
    xmp_file.close_file()
    
    # Check description
    desc = xmp.get_localized_text("http://purl.org/dc/elements/1.1/", "description", "x-default", "x-default")
    assert desc == "Test Location, World"
    
    # Check PersonInImage
    # namespace NS_IPTC = "http://iptc.org/std/Iptc4xmpCore/1.0/xmlns/"
    ns_iptc = "http://iptc.org/std/Iptc4xmpCore/1.0/xmlns/"
    assert xmp.does_property_exist(ns_iptc, "PersonInImage")
    person_count = xmp.count_array_items(ns_iptc, "PersonInImage")
    assert person_count == 2
    persons = [xmp.get_array_item(ns_iptc, "PersonInImage", i+1) for i in range(2)]
    assert set(persons) == {"Alice", "Bob"}
    
    # Check MWG Regions
    ns_mwg_rs = "http://www.metadataworkinggroup.com/schemas/regions/"
    # Alice: cx, cy, bw, bh from [10, 20, 30, 40]
    # x1=10, y1=20, x2=30, y2=40
    # cx = 20/100 = 0.2, cy = 30/100 = 0.3
    # bw = 20/100 = 0.2, bh = 20/100 = 0.2
    
    # Bob: x1=50, y1=50, x2=70, y2=70
    # cx = 60/100 = 0.6, cy = 60/100 = 0.6
    # bw = 20/100 = 0.2, bh = 20/100 = 0.2
    
    # Verify both region entries
    region_names = []
    region_coords = {}
    for i in range(1, 3):
        base = f"Regions/mwg-rs:RegionList[{i}]"
        name = xmp.get_property(ns_mwg_rs, f"{base}/mwg-rs:Name")
        region_names.append(name)
        
        cx = xmp.get_property_float(ns_mwg_rs, f"{base}/mwg-rs:Area/stArea:x")
        cy = xmp.get_property_float(ns_mwg_rs, f"{base}/mwg-rs:Area/stArea:y")
        bw = xmp.get_property_float(ns_mwg_rs, f"{base}/mwg-rs:Area/stArea:w")
        bh = xmp.get_property_float(ns_mwg_rs, f"{base}/mwg-rs:Area/stArea:h")
        region_coords[name] = (cx, cy, bw, bh)
        
        reg_type = xmp.get_property(ns_mwg_rs, f"{base}/mwg-rs:Type")
        assert reg_type == "Face"
        
        unit = xmp.get_property(ns_mwg_rs, f"{base}/mwg-rs:Area/stArea:unit")
        assert unit == "normalized"
        
    assert set(region_names) == {"Alice", "Bob"}
    assert region_coords["Alice"] == (pytest.approx(0.2), pytest.approx(0.3), pytest.approx(0.2), pytest.approx(0.2))
    assert region_coords["Bob"] == (pytest.approx(0.6), pytest.approx(0.6), pytest.approx(0.2), pytest.approx(0.2))








