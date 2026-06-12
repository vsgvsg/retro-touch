import pytest
from unittest.mock import MagicMock
import json
import update_english_locations as uel

def test_translate_coordinates_coalesces_and_caches():
    # Mock NominatimClient
    mock_client = MagicMock()
    mock_client.reverse.side_effect = [
        {"address": {"city": "Penza", "state": "Penza Oblast", "country": "Russia"}},
        {"address": {"city": "Altenburg", "state": "Thuringia", "country": "Germany"}}
    ]

    # We have 3 coordinates. Two of them are within 1000m of each other.
    coords = [
        (53.2000, 45.0000),
        (53.2001, 45.0001),  # Should coalesce with first
        (50.9852, 12.4340)
    ]

    cache = uel.TranslationCache(client=mock_client)
    
    res1 = cache.get_english_location(53.2000, 45.0000)
    res2 = cache.get_english_location(53.2001, 45.0001)
    res3 = cache.get_english_location(50.9852, 12.4340)

    assert mock_client.reverse.call_count == 2
    assert res1["city"] == "Penza"
    assert res2["city"] == "Penza"
    assert res3["city"] == "Altenburg"


def test_translate_sidecars_and_cache(tmp_path):
    # Setup mock sidecar file
    sidecar_data = {
        "image": "test.jpg",
        "image_size": [100, 100],
        "faces": [{"label": "Sergey"}],
        "taken": {"year": 1970},
        "location": {
            "lat": 53.2,
            "lng": 45.0,
            "display_name": "Пенза, Россия",
            "city": "Пенза",
            "state": "Пензенская область",
            "country": "Россия"
        },
        "exif_written": True
    }
    sc_file = tmp_path / "test.faces.json"
    with open(sc_file, "w", encoding="utf-8") as f:
        json.dump(sidecar_data, f)

    # Touch test.jpg so it exists
    (tmp_path / "test.jpg").touch()

    # Setup mock locations.json cache
    loc_cache_data = {
        "locations": [
            {
                "lat": 53.2,
                "lng": 45.0,
                "display_name": "Пенза, Россия",
                "city": "Пенза",
                "state": "Пензенская область",
                "country": "Россия",
                "use_count": 5
            }
          ]
    }
    loc_file = tmp_path / "locations.json"
    with open(loc_file, "w", encoding="utf-8") as f:
        json.dump(loc_cache_data, f)

    # Mock translation cache
    mock_trans_cache = MagicMock()
    mock_trans_cache.get_english_location.return_value = {
        "lat": 53.2,
        "lng": 45.0,
        "display_name": "Penza, Russia",
        "city": "Penza",
        "state": "Penza Oblast",
        "country": "Russia"
    }

    # Mock exif_pipeline.write_exif_xmp to avoid opening real files
    mock_write_exif = MagicMock(return_value=True)

    # Run translation (non dry-run)
    updates = uel.run_translation(
        sidecars=[sc_file],
        locations_path=loc_file,
        trans_cache=mock_trans_cache,
        write_exif_fn=mock_write_exif,
        dry_run=False
    )

    # Verify sidecar was modified
    with open(sc_file, encoding="utf-8") as f:
        updated_sc = json.load(f)
    assert updated_sc["location"]["city"] == "Penza"
    assert updated_sc["location"]["display_name"] == "Penza, Russia"

    # Verify locations.json cache was updated
    with open(loc_file, encoding="utf-8") as f:
        updated_locs = json.load(f)
    assert updated_locs["locations"][0]["city"] == "Penza"
    assert updated_locs["locations"][0]["use_count"] == 5

    # Verify EXIF write was called
    mock_write_exif.assert_called_once()


def test_cli_main_dry_run(monkeypatch):
    # Test that main runs and exits cleanly
    import update_english_locations as uel
    monkeypatch.setattr("sys.argv", ["update_english_locations.py", "--dry-run"])
    
    mock_run = MagicMock(return_value=0)
    monkeypatch.setattr(uel, "run_translation", mock_run)
    
    try:
        uel.main()
    except SystemExit as e:
        assert e.code == 0
