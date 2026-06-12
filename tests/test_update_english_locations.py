import pytest
from unittest.mock import MagicMock
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
