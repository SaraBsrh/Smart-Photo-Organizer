import json
import os
from pathlib import Path
from geopy.geocoders import Nominatim
from geopy.extra.rate_limiter import RateLimiter

# Paths
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
METADATA_FILE = DATA_DIR / "metadata.json"
OUTPUT_FILE = DATA_DIR / "metadata_with_location.json"
CACHE_FILE = DATA_DIR / "geocode_cache.json"

# Ensure data directory exists
DATA_DIR.mkdir(exist_ok=True)

def load_json(file_path):
    """Load JSON if file exists, else return empty dict."""
    if file_path.exists():
        with open(file_path, "r", encoding="utf-8") as f:
            return json.load(f)
    return {}

def save_json(file_path, data):
    """Save dictionary as JSON."""
    with open(file_path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=4, ensure_ascii=False)

def reverse_geocode(lat, lon, geolocator, cache):
    """Convert lat/lon to (city, country) with caching."""
    key = f"{lat}_{lon}"
    if key in cache:
        return cache[key]

    try:
        location = geolocator.reverse((lat, lon), language="en", exactly_one=True)
        if location and location.raw.get("address"):
            address = location.raw["address"]
            city = address.get("city") or address.get("town") or address.get("village")
            country = address.get("country")
            cache[key] = (city, country)
            return city, country
    except Exception as e:
        print(f"⚠️ Geocoding error for {lat}, {lon}: {e}")

    cache[key] = (None, None)
    return None, None

def main():
    # Load files
    metadata = load_json(METADATA_FILE)
    cache = load_json(CACHE_FILE)

    if not metadata:
        print(f"❌ No metadata found at {METADATA_FILE}")
        return

    # Setup geocoder
    geolocator = Nominatim(user_agent="smart-photo-organizer")
    geocode_rate_limited = RateLimiter(geolocator.reverse, min_delay_seconds=1)

    # Process each photo
    for filename, info in metadata.items():
        lat = info.get("lat")
        lon = info.get("lon")

        if lat is not None and lon is not None:
            city, country = reverse_geocode(lat, lon, geolocator, cache)
            metadata[filename]["city"] = city
            metadata[filename]["country"] = country
        else:
            metadata[filename]["city"] = None
            metadata[filename]["country"] = None

    # Save results
    save_json(OUTPUT_FILE, metadata)
    save_json(CACHE_FILE, cache)

    print(f"✅ Geocoding complete. Output saved to {OUTPUT_FILE}")
    print(f"📌 Cache saved to {CACHE_FILE}")

if __name__ == "__main__":
    main()