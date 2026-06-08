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
