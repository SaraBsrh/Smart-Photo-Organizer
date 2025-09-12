import json
import os
from pathlib import Path
from geopy.geocoders import Nominatim
from geopy.extra.rate_limiter import RateLimiter
from typing import Dict, Any, Tuple

# existing constants (adjust if needed)
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
CACHE_FILE = DATA_DIR / "geocode_cache.json"
DATA_DIR.mkdir(exist_ok=True)

def load_json(file_path: Path):
    if file_path.exists():
        with open(file_path, "r", encoding="utf-8") as f:
            return json.load(f)
    return {}

def save_json(file_path: Path, data):
    file_path.parent.mkdir(parents=True, exist_ok=True)
    with open(file_path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=4, ensure_ascii=False)

def reverse_geocode(lat: float, lon: float, geolocator_or_fn, cache: dict) -> Tuple[str | None, str | None]:
    key = f"{lat:.6f}_{lon:.6f}"
    if key in cache:
        return cache[key]
    try:
        if hasattr(geolocator_or_fn, "reverse") and callable(getattr(geolocator_or_fn, "reverse")):
            location = geolocator_or_fn.reverse((lat, lon), language="en", exactly_one=True)
        else:
            # Assume it's a callable (RateLimiter or wrapper around geolocator.reverse)
            # RateLimiter wraps the reverse function, so call it directly
            location = geolocator_or_fn((lat, lon), language="en", exactly_one=True)
        if location and getattr(location, "raw", None) and isinstance(location.raw.get("address"), dict):
            address = location.raw["address"]
            city = address.get("city") or address.get("town") or address.get("village") or address.get("hamlet")
            country = address.get("country")
            cache[key] = (city, country)
            return city, country
    except Exception as e:
        # keep the error but continue - avoid raising inside batch jobs
        print(f"⚠️ Geocoding error for {lat},{lon}: {e}")

    cache[key] = (None, None)
    return None, None

def add_locations_to_metadata(metadata: Dict[str, Any], photo_dir: str | Path | None = None, use_cache: bool = True) -> Dict[str, Any]:
    """
    Add 'city' and 'country' keys to each metadata entry that has lat & lon.
    Signature: add_locations_to_metadata(metadata, photo_dir=None)
    Returns updated metadata dict.
    """
    # prepare geolocator and cache
    geolocator = Nominatim(user_agent="smart-photo-organizer")
    geocode_fn = RateLimiter(geolocator.reverse, min_delay_seconds=1, max_retries=2, error_wait_seconds=5)
    cache = load_json(CACHE_FILE) if use_cache else {}
    cache_path = DATA_DIR / "geocode_cache.json"
    if use_cache:
        cache = load_json(cache_path)

    # iterate metadata
    for fname, info in list(metadata.items()):
        # support both sanitized and raw metadata entries
        lat = None
        lon = None
        if isinstance(info, dict):
            lat = info.get("lat") or info.get("latitude")
            lon = info.get("lon") or info.get("longitude")
        # if lat/lon exist and are numeric
        if lat is not None and lon is not None:
            try:
                city, country = reverse_geocode(float(lat), float(lon), geocode_fn, cache)
            except Exception:
                # fallback to using geolocator directly (less safe)
                try:
                    city, country = reverse_geocode(float(lat), float(lon), geolocator, cache)
                except Exception:
                    city, country = None, None
            if isinstance(info, dict):
                info["city"] = city
                info["country"] = country
            else:
                metadata[fname] = {"datetime": info, "city": city, "country": country}
        else:
            # ensure keys exist
            if isinstance(info, dict):
                info.setdefault("city", None)
                info.setdefault("country", None)
            else:
                metadata[fname] = {"datetime": info, "city": None, "country": None}

    # save cache back to disk for faster subsequent runs
    if use_cache:
        try:
            save_json(cache_path, cache)
        except Exception:
            pass

    return metadata

# Keep your previous CLI main() if present (unchanged)
if __name__ == "__main__":
    # previous script behavior: read data/sample-data/metadata.json and write metadata_with_location.json
    METADATA_FILE = DATA_DIR / "metadata.json"
    OUTPUT_FILE = DATA_DIR / "metadata_with_location.json"
    if not METADATA_FILE.exists():
        print(f"No metadata file found at {METADATA_FILE}")
    else:
        meta = load_json(METADATA_FILE)
        updated = add_locations_to_metadata(meta)
        save_json(OUTPUT_FILE, updated)
        print(f"Saved {OUTPUT_FILE}")
