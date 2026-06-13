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
    
    requested_url = ""
    def mock_get(url):
        nonlocal requested_url
        requested_url = url
        return {"address": {"city": "Penza"}}
        
    monkeypatch.setattr(client, "_get", mock_get)
    
    res = client.reverse(53.2007, 45.0046)
    assert res == {"address": {"city": "Penza"}}
    assert "accept-language=en" in requested_url

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
    
    # Check PersonInImage (IPTC Extension namespace)
    ns_iptc_ext = "http://iptc.org/std/Iptc4xmpExt/2008-02-29/"
    assert xmp.does_property_exist(ns_iptc_ext, "PersonInImage")
    person_count = xmp.count_array_items(ns_iptc_ext, "PersonInImage")
    assert person_count == 2
    persons = [xmp.get_array_item(ns_iptc_ext, "PersonInImage", i+1) for i in range(2)]
    assert set(persons) == {"Alice", "Bob"}
    
    # Check MWG Regions
    ns_mwg_rs = "http://www.metadataworkinggroup.com/schemas/regions/"
    ns_st_dim = "http://ns.adobe.com/xap/1.0/sType/Dimensions#"
    
    # Check AppliedToDimensions
    assert xmp.get_property(ns_mwg_rs, "Regions/mwg-rs:AppliedToDimensions/stDim:unit") == "pixel"
    assert xmp.get_property_int(ns_mwg_rs, "Regions/mwg-rs:AppliedToDimensions/stDim:w") == 100
    assert xmp.get_property_int(ns_mwg_rs, "Regions/mwg-rs:AppliedToDimensions/stDim:h") == 100

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

def test_sidecar_tagged_none_values():
    data1 = {"taken": None, "location": {"lat": 53.2, "lng": 45.0}}
    data2 = {"taken": {"year": 1960}, "location": None}
    data3 = None
    assert ep.sidecar_is_tagged(data1) is False
    assert ep.sidecar_is_tagged(data2) is False
    assert ep.sidecar_is_tagged(data3) is False

def test_load_sidecar_non_dict():
    with tempfile.NamedTemporaryFile(suffix=".json", mode="w", delete=False) as f:
        json.dump([1, 2, 3], f)
        fname = f.name
    loaded = ep.load_sidecar(fname)
    assert loaded is None
    pathlib.Path(fname).unlink()


def test_cmd_report(capsys, tmp_path):
    # 1. Untagged sidecar
    (tmp_path / "1.faces.json").write_text(json.dumps({"taken": None, "location": None}), encoding="utf-8")
    
    # 2. Tagged but not exif_written
    (tmp_path / "2.faces.json").write_text(json.dumps({
        "taken": {"year": 1990},
        "location": {"lat": 1.0, "lng": 2.0}
    }), encoding="utf-8")
    
    # 3. Tagged and exif_written
    (tmp_path / "3.faces.json").write_text(json.dumps({
        "taken": {"year": 1995},
        "location": {"lat": 3.0, "lng": 4.0},
        "exif_written": True
    }), encoding="utf-8")
    
    # 4. Corrupted json
    (tmp_path / "4.faces.json").write_text("invalid json", encoding="utf-8")
    
    # 5. Non-faces.json file (should be ignored)
    (tmp_path / "other.json").write_text(json.dumps({}), encoding="utf-8")

    ep.cmd_report(str(tmp_path))
    
    captured = capsys.readouterr()
    out = captured.out
    
    # Expected output details:
    # 4 sidecar files total
    # 2 tagged (50%)
    # 1 exif_written (25%)
    # 1 sidecar_only (25%)
    # 2 untagged (50%)
    assert "4 photos total" in out
    assert "2 tagged (date + location)    50%" in out
    assert "1 EXIF written to jpg         25%" in out
    assert "1 tagged sidecar only         25%" in out
    assert "2 untagged                    50%" in out


def test_cmd_tag(capsys, tmp_path, monkeypatch):
    # Test case 1: empty directory
    ep.cmd_tag(str(tmp_path))
    captured = capsys.readouterr()
    assert "No .jpg files found in" in captured.out

    # Test case 2: directory with photos
    (tmp_path / "photo1.jpg").write_text("", encoding="utf-8")
    
    app_inited = False
    app_run_called = False
    
    class MockTaggerApp:
        def __init__(self, photos, extracted_dir):
            nonlocal app_inited
            app_inited = True
            assert len(photos) == 1
            assert photos[0] == str(tmp_path / "photo1.jpg")
            assert extracted_dir == str(tmp_path)
            
        def run(self):
            nonlocal app_run_called
            app_run_called = True
            
    monkeypatch.setattr(ep, "TaggerApp", MockTaggerApp)
    ep.cmd_tag(str(tmp_path))
    assert app_inited is True
    assert app_run_called is True


def test_theme_helpers_and_photo_loader(tmp_path):
    from exif_pipeline import (
        ACCENT, BG, SURFACE, TEXT, MUTED, GREEN, AMBER, RED, STATE_COLORS,
        _install_theme, load_photo_image
    )
    assert ACCENT == "#5a6cf0"
    assert BG == "#fafaff"
    assert STATE_COLORS["tagged"] == GREEN

    # Test _install_theme
    import tkinter as tk
    from tkinter import ttk
    root = tk.Tk()
    try:
        _install_theme(root)
        # Check that root bg is updated
        assert root.cget("bg") == BG
        # Verify no active state foreground mappings are configured (which was a dark-theme pattern)
        style = ttk.Style(root)
        assert not any(state == "active" or "active" in state for state, _ in style.map("TButton", "foreground"))
        assert not any(state == "active" or "active" in state for state, _ in style.map("Accent.TButton", "foreground"))
    finally:
        root.destroy()

    # Test load_photo_image with a non-existent file (returns None)
    assert load_photo_image("non_existent_file.jpg") is None

    # Test load_photo_image with a real image (constrained by height)
    from PIL import Image
    test_img = tmp_path / "test_img.jpg"
    img = Image.new("RGB", (100, 200), color="red")
    img.save(test_img, "JPEG")
    
    # We need tk root active to create PhotoImage
    root2 = tk.Tk()
    try:
        photo = load_photo_image(str(test_img), max_height=100, max_width=500)
        assert photo is not None
        # Width/height should be scaled (original height 200 -> max_height 100, so scale is 0.5, width becomes 50)
        assert photo.height() == 100
        assert photo.width() == 50

        # Test load_photo_image with a wide image (constrained by width)
        test_img_wide = tmp_path / "test_img_wide.jpg"
        img_wide = Image.new("RGB", (1000, 200), color="blue")
        img_wide.save(test_img_wide, "JPEG")
        # default max_width = 560, max_height = 680
        photo_wide = load_photo_image(str(test_img_wide))
        assert photo_wide is not None
        assert photo_wide.width() == 560
        assert photo_wide.height() == 112
    finally:
        root2.destroy()


# --- TaggerApp tests (Task 10) ---

def test_tagger_app_photo_loading_and_autofill(tmp_path, monkeypatch):
    from PIL import Image
    import json
    
    # 1. Create a dummy photo
    jpg_path = tmp_path / "1990-penza-00001.jpg"
    img = Image.new("RGB", (100, 100), color="red")
    img.save(jpg_path, "JPEG")
    
    # 2. Write locations.json and sidecar faces.json
    locs_file = tmp_path / "locations.json"
    locs_file.write_text(json.dumps({"locations": []}), encoding="utf-8")
    
    sc_file = tmp_path / "1990-penza-00001.faces.json"
    sc_data = {
        "image": "1990-penza-00001.jpg",
        "image_size": [100, 100],
        "faces": [
            {"bbox": [10, 10, 30, 30], "label": "Alice"}
        ],
        "taken": {"year": 1991, "month": 5},
        "location": {
            "lat": 53.2007,
            "lng": 45.0046,
            "city": "Penza",
            "state": "Penza Oblast",
            "country": "Russia",
            "display_name": "Penza, Russia"
        }
    }
    sc_file.write_text(json.dumps(sc_data), encoding="utf-8")
    
    # Mock search_and_fly and set_location to record calls
    set_location_calls = []
    def mock_set_location(lat, lng, city, state, country, display_name, fly=False):
        set_location_calls.append((lat, lng, city, state, country, display_name, fly))
        
    # Mock NominatimClient._get to avoid network calls and rate-limit sleep
    monkeypatch.setattr(ep.NominatimClient, "_get", lambda self, url: [])

    # Instantiate TaggerApp
    app = ep.TaggerApp([str(jpg_path)], extracted_dir=str(tmp_path))
    try:
        app._set_location = mock_set_location
        # Manually load the photo (since it was loaded during __init__ with empty stubs, or let's just call it now)
        app._load_photo(0)
        
        # Verify sidecar is loaded
        assert app._sidecar["taken"]["year"] == 1991
        
        # Verify text/title variables
        assert app._title_var.get() == "1990-penza-00001.jpg  1/1"
        assert app._progress_var.get() == 0.0
        
        # Verify autofill dates
        assert app._year_var.get() == "1991"
        assert app._month_var.get() == "May"
        
        # Verify set_location was called from sidecar
        assert len(set_location_calls) == 1
        assert set_location_calls[0] == (53.2007, 45.0046, "Penza", "Penza Oblast", "Russia", "Penza, Russia", True)
        
    finally:
        app.destroy()


def test_tagger_app_autofill_filename_parsing(tmp_path, monkeypatch):
    from PIL import Image
    import json
    
    # 1. Create a dummy photo
    jpg_path = tmp_path / "1985-tbilisi-00002.jpg"
    img = Image.new("RGB", (100, 100), color="blue")
    img.save(jpg_path, "JPEG")
    
    # 2. Locations.json (empty)
    locs_file = tmp_path / "locations.json"
    locs_file.write_text(json.dumps({"locations": []}), encoding="utf-8")
    
    # Mock NominatimClient._get to avoid network calls and rate-limit sleep
    monkeypatch.setattr(ep.NominatimClient, "_get", lambda self, url: [])
    
    # Track search_and_fly calls
    search_queries = []
    orig_search_and_fly = ep.TaggerApp._search_and_fly
    def tracking_search_and_fly(self, query):
        search_queries.append(query)
        return orig_search_and_fly(self, query)
    monkeypatch.setattr(ep.TaggerApp, "_search_and_fly", tracking_search_and_fly)

    app = ep.TaggerApp([str(jpg_path)], extracted_dir=str(tmp_path))
    try:
        # Verify default sidecar structure
        assert app._sidecar["image"] == "1985-tbilisi-00002.jpg"
        assert app._sidecar["faces"] == []
        
        # Verify autofill parsed from filename
        assert app._year_var.get() == "1985"
        assert app._month_var.get() == ""
        
        # Verify _search_and_fly was triggered asynchronously
        import time
        start_t = time.time()
        while not search_queries and time.time() - start_t < 1.0:
            time.sleep(0.01)
        assert "tbilisi" in search_queries
        
    finally:
        app.destroy()


def test_tagger_app_navigation(tmp_path, monkeypatch):
    from PIL import Image
    import json
    
    # Create 3 dummy photos
    jpgs = []
    for i in range(1, 4):
        jpg_path = tmp_path / f"1990-penza-0000{i}.jpg"
        img = Image.new("RGB", (100, 100), color="red")
        img.save(jpg_path, "JPEG")
        jpgs.append(str(jpg_path))
        
    locs_file = tmp_path / "locations.json"
    locs_file.write_text(json.dumps({"locations": []}), encoding="utf-8")
    
    # Mock NominatimClient._get to avoid network calls and rate-limit sleep
    monkeypatch.setattr(ep.NominatimClient, "_get", lambda self, url: [])

    app = ep.TaggerApp(jpgs, extracted_dir=str(tmp_path))
    try:
        app._load_photo(0)
        assert app.idx == 0
        
        app._next()
        assert app.idx == 1
        assert "00002" in app._title_var.get()
        
        app._next()
        assert app.idx == 2
        assert "00003" in app._title_var.get()
        
        app._next()
        assert app.idx == 0
        assert "00001" in app._title_var.get()
        
        app._prev()
        assert app.idx == 2
        assert "00003" in app._title_var.get()
        
        # Mock event for _on_progress_click
        class MockEvent:
            def __init__(self, x):
                self.x = x
        
        # If progress bar width is 100 pixels, and we click at x=70, frac = 0.7 -> idx = 2
        app._progress_bar.winfo_width = lambda: 100
        
        app._on_progress_click(MockEvent(70))
        assert app.idx == 2
        
        app._on_progress_click(MockEvent(10))
        assert app.idx == 0
    finally:
        app.destroy()


def test_nominatim_client_search_limit_10(monkeypatch):
    from exif_pipeline import NominatimClient
    client = NominatimClient()
    requested_url = ""
    def mock_get(url):
        nonlocal requested_url
        requested_url = url
        return []
    monkeypatch.setattr(client, "_get", mock_get)
    client.search("Rome")
    assert "limit=10" in requested_url
    assert "accept-language=en" in requested_url


def test_tagger_app_map_sizing_and_vertical_expansion(tmp_path, monkeypatch):
    from PIL import Image
    import json
    
    jpg_path = tmp_path / "1990-penza-00001.jpg"
    img = Image.new("RGB", (100, 100), color="red")
    img.save(jpg_path, "JPEG")
    
    locs_file = tmp_path / "locations.json"
    locs_file.write_text(json.dumps({"locations": []}), encoding="utf-8")
    
    sc_file = tmp_path / "1990-penza-00001.faces.json"
    sc_file.write_text(json.dumps({}), encoding="utf-8")
    
    monkeypatch.setattr(ep.NominatimClient, "_get", lambda self, url: [])
    
    app = ep.TaggerApp([str(jpg_path)], extracted_dir=str(tmp_path))
    try:
        # Verify map default height is 360
        assert app._map.height == 360
        
        # Verify loc_frame is packed with expansion (expand=True)
        # We can check pack_info of its master
        info = app._map.master.pack_info()
        assert bool(info.get("expand")) is True
        assert info.get("fill") == "both"
    finally:
        app.destroy()


def test_tagger_app_shortcuts_button(tmp_path, monkeypatch):
    from PIL import Image
    import json
    
    jpg_path = tmp_path / "1990-penza-00001.jpg"
    img = Image.new("RGB", (100, 100), color="red")
    img.save(jpg_path, "JPEG")
    
    locs_file = tmp_path / "locations.json"
    locs_file.write_text(json.dumps({"locations": []}), encoding="utf-8")
    
    sc_file = tmp_path / "1990-penza-00001.faces.json"
    sc_file.write_text(json.dumps({}), encoding="utf-8")
    
    called = False
    def mock_show_shortcuts(self):
        nonlocal called
        called = True
    monkeypatch.setattr(ep.TaggerApp, "_show_shortcuts", mock_show_shortcuts)
    monkeypatch.setattr(ep.NominatimClient, "_get", lambda self, url: [])
    
    app = ep.TaggerApp([str(jpg_path)], extracted_dir=str(tmp_path))
    try:
        # Verify the shortcuts button exists
        assert hasattr(app, "_shortcuts_btn")
        assert app._shortcuts_btn.cget("text") == "?"
        
        # Verify click invokes show_shortcuts
        app._shortcuts_btn.invoke()
        assert called is True
    finally:
        app.destroy()


def test_tagger_app_date_propagation(tmp_path, monkeypatch):
    from PIL import Image
    import json
    
    # Create two photos
    jpg1 = tmp_path / "img1.jpg"
    jpg2 = tmp_path / "img2.jpg"
    for p in (jpg1, jpg2):
        img = Image.new("RGB", (100, 100), color="blue")
        img.save(p, "JPEG")
        
    locs_file = tmp_path / "locations.json"
    locs_file.write_text(json.dumps({"locations": []}), encoding="utf-8")
    
    # Mock geocoder and reverse geocode
    monkeypatch.setattr(ep.NominatimClient, "_get", lambda self, url: [])
    
    app = ep.TaggerApp([str(jpg1), str(jpg2)], extracted_dir=str(tmp_path))
    try:
        # Set manual date
        app._year_var.set("1995")
        app._month_var.set("Jan")
        
        # Set location so it can be saved
        app._lat = 1.0
        app._lng = 2.0
        app._city = "City"
        app._state = "State"
        app._country = "Country"
        app._display_name = "City, State, Country"
        
        # Save & Next
        app._save_and_next()
        
        # Verify the app advanced to photo 2
        assert app.idx == 1
        
        # Verify photo 2 defaults to the previous manually entered year and month
        assert app._year_var.get() == "1995"
        assert app._month_var.get() == "Jan"
    finally:
        app.destroy()


def test_tagger_app_map_scroll_zoom(tmp_path, monkeypatch):
    from PIL import Image
    import json
    import sys
    
    jpg_path = tmp_path / "1990-penza-00001.jpg"
    img = Image.new("RGB", (100, 100), color="red")
    img.save(jpg_path, "JPEG")
    
    locs_file = tmp_path / "locations.json"
    locs_file.write_text(json.dumps({"locations": []}), encoding="utf-8")
    
    sc_file = tmp_path / "1990-penza-00001.faces.json"
    sc_file.write_text(json.dumps({}), encoding="utf-8")
    
    monkeypatch.setattr(ep.NominatimClient, "_get", lambda self, url: [])
    
    app = ep.TaggerApp([str(jpg_path)], extracted_dir=str(tmp_path))
    try:
        # Mock set_zoom to capture zoom steps
        zoom_calls = []
        def mock_set_zoom(new_zoom, relative_pointer_x=0.5, relative_pointer_y=0.5):
            zoom_calls.append(new_zoom)
        
        app._map.set_zoom = mock_set_zoom
        app._map.zoom = 10.0
        
        class MockEvent:
            def __init__(self, delta, x, y, num=0):
                self.delta = delta
                self.x = x
                self.y = y
                self.num = num
                
        # 1. Simulate standard scroll wheel event (delta=120)
        app._map._custom_mouse_zoom(MockEvent(120, 180, 180))
        
        # Verify standard mouse wheel zoom changes zoom by exactly 1 level
        if sys.platform == "darwin":
            assert len(zoom_calls) == 1
            assert abs(zoom_calls[0] - 11.0) < 0.001
            
        zoom_calls.clear()
        
        # 2. Simulate Magic Mouse scroll event (delta=1)
        app._map._custom_mouse_zoom(MockEvent(1, 180, 180))
        
        # Verify Magic Mouse uses default library behavior (zoom by 0.1 levels)
        if sys.platform == "darwin":
            assert len(zoom_calls) == 1
            assert abs(zoom_calls[0] - 10.1) < 0.001
    finally:
        app.destroy()


def test_tagger_app_copy_previous(tmp_path, monkeypatch):
    from PIL import Image
    import json
    import tkinter as tk
    
    # Create three photos
    jpg1 = tmp_path / "img1.jpg"
    jpg2 = tmp_path / "img2.jpg"
    jpg3 = tmp_path / "img3.jpg"
    for p in (jpg1, jpg2, jpg3):
        img = Image.new("RGB", (100, 100), color="blue")
        img.save(p, "JPEG")
        
    locs_file = tmp_path / "locations.json"
    locs_file.write_text(json.dumps({"locations": []}), encoding="utf-8")
    
    # Write metadata sidecar for photo 1
    sc1 = {
        "taken": {"year": 1955, "month": 6, "source": "manual"},
        "location": {
            "lat": 40.7128, "lng": -74.0060,
            "display_name": "New York, USA",
            "city": "New York", "state": "New York", "country": "USA",
            "source": "manual"
        }
    }
    sc1_path = tmp_path / "img1.faces.json"
    sc1_path.write_text(json.dumps(sc1), encoding="utf-8")
    
    # Write metadata sidecar for photo 2 (empty/none)
    sc2 = {}
    sc2_path = tmp_path / "img2.faces.json"
    sc2_path.write_text(json.dumps(sc2), encoding="utf-8")
    
    # Write metadata sidecar for photo 3 (partial, location only)
    sc3 = {
        "location": {
            "lat": 34.0522, "lng": -118.2437,
            "display_name": "Los Angeles, USA",
            "city": "Los Angeles", "state": "California", "country": "USA",
            "source": "manual"
        }
    }
    sc3_path = tmp_path / "img3.faces.json"
    sc3_path.write_text(json.dumps(sc3), encoding="utf-8")

    monkeypatch.setattr(ep.NominatimClient, "_get", lambda self, url: [])
    
    import tkintermapview
    class MockMarker:
        def delete(self):
            pass
    monkeypatch.setattr(tkintermapview.TkinterMapView, "set_marker", lambda self, lat, lng, **kwargs: MockMarker())
    
    app = ep.TaggerApp([str(jpg1), str(jpg2), str(jpg3)], extracted_dir=str(tmp_path))
    try:
        # 1. First photo (idx=0)
        assert app.idx == 0
        # Button should be disabled
        assert str(app._copy_prev_btn.cget("state")) == "disabled"
        
        # 2. Advance to photo 2 (idx=1)
        app._next()
        assert app.idx == 1
        # Button should be enabled since photo 1 has metadata
        assert str(app._copy_prev_btn.cget("state")) != "disabled"
        
        # Populate current fields with dummy data to verify overwrite
        app._year_var.set("2000")
        app._month_var.set("Oct")
        app._set_location(0.0, 0.0, "", "", "", "")
        
        # Click copy previous
        app._copy_previous()
        
        # Verify year and location were copied
        assert app._year_var.get() == "1955"
        assert app._month_var.get() == "Jun"
        assert app._lat == 40.7128
        assert app._lng == -74.0060
        assert app._city == "New York"
        
        # 3. Advance to photo 3 (idx=2)
        app._next()
        assert app.idx == 2
        # Button should be disabled since photo 2 sidecar is empty
        assert str(app._copy_prev_btn.cget("state")) == "disabled"
        
        # Fill in current fields
        app._year_var.set("2010")
        app._month_var.set("Dec")
        
        # Manually enable and call copy previous (simulating call from photo 3 where prev has only location)
        app.idx = 1
        app._copy_previous()
        assert app._year_var.get() == "1955"
        
        app.idx = 2
        app._year_var.set("2010")
        app._copy_previous()
        assert app._year_var.get() == "2010"  # Unchanged
        
    finally:
        app.destroy()


def test_location_cache_remove(tmp_path):
    import json
    locs_file = tmp_path / "locations.json"
    locs_file.write_text(json.dumps({
        "locations": [
            {"lat": 1.0, "lng": 2.0, "city": "A", "state": "", "country": "X", "use_count": 5},
            {"lat": 3.0, "lng": 4.0, "city": "B", "state": "", "country": "Y", "use_count": 2}
        ]
    }), encoding="utf-8")
    
    cache = ep.LocationCache(locs_file)
    assert len(cache.all_entries()) == 2
    
    # Remove entry A
    cache.remove({"lat": 1.0, "lng": 2.0})
    
    # Reload and verify
    cache2 = ep.LocationCache(locs_file)
    entries = cache2.all_entries()
    assert len(entries) == 1
    assert entries[0]["city"] == "B"


def test_tagger_app_remove_chip(tmp_path, monkeypatch):
    from PIL import Image
    import json
    
    jpg_path = tmp_path / "img1.jpg"
    img = Image.new("RGB", (100, 100), color="blue")
    img.save(jpg_path, "JPEG")
    
    locs_file = tmp_path / "locations.json"
    locs_file.write_text(json.dumps({
        "locations": [
            {"lat": 10.0, "lng": 20.0, "display_name": "Place X", "city": "Place X", "state": "", "country": "US", "use_count": 1}
        ]
    }), encoding="utf-8")
    
    monkeypatch.setattr(ep.NominatimClient, "_get", lambda self, url: [])
    
    import tkintermapview
    class MockMarker:
        def delete(self):
            pass
    monkeypatch.setattr(tkintermapview.TkinterMapView, "set_marker", lambda self, lat, lng, **kwargs: MockMarker())
    
    app = ep.TaggerApp([str(jpg_path)], extracted_dir=str(tmp_path))
    try:
        # Check initial chips count (should be 1)
        children = app._chips_frame.winfo_children()
        assert len(children) == 1
        assert children[0].cget("text") == "Place X"
        
        # Simulate Cmd+Click removal
        app._remove_cache_entry({"lat": 10.0, "lng": 20.0})
        
        # Check chips count again (should be 0)
        children = app._chips_frame.winfo_children()
        assert len(children) == 0
        
        # Verify locations.json has no entries
        assert len(app.cache.all_entries()) == 0
    finally:
        app.destroy()





