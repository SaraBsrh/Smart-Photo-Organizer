import json
import shutil
from pathlib import Path
from datetime import datetime

# ==== CONFIG ====
PHOTO_DIR = Path("/Users/sara/Desktop/backed-data/sample-data")
OUTPUT_DIR = Path("/Users/sara/Desktop/backed-data")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

DATE_FILE = PHOTO_DIR / "metadata.json"
LOCATION_FILE = PHOTO_DIR / "metadata_with_location.json"
TAGS_FILE = PHOTO_DIR / "metadata_with_tags.json"

DATE_FORMAT = "%Y-%m-%d"  # format in your metadata.json

# ==== LOAD DATA ====
with open(DATE_FILE) as f:
    date_data = json.load(f)

with open(LOCATION_FILE) as f:
    location_data = json.load(f)

with open(TAGS_FILE) as f:
    tags_data = json.load(f)

# ==== HELPER FUNCTIONS ====
import re

def _try_strptime(s: str):
    # list of formats to try (order matters)
    fmts = [
        "%Y:%m:%d %H:%M:%S",      # EXIF typical: "2024:06:03 21:46:06"
        "%Y-%m-%d %H:%M:%S",      # "2024-06-03 21:46:06"
        "%Y-%m-%d",               # "2024-06-03"
        "%Y:%m:%d",               # "2024:06:03"
        "%Y-%m-%dT%H:%M:%S",      # "2024-06-03T21:46:06"
        "%Y-%m-%dT%H:%M:%S%z",    # with timezone offset
        "%Y-%m-%d %H:%M:%S.%f",   # with fractional seconds
        "%Y:%m:%d %H:%M:%S.%f",   # EXIF + fractional
        "%d-%m-%Y",               # "03-06-2024"
        "%m/%d/%Y",               # "06/03/2024"
    ]
    for fmt in fmts:
        try:
            return datetime.strptime(s, fmt)
        except Exception:
            continue
    return None

def parse_date(date_input):
    """
    Accepts:
      - a string like "2024:06:03 21:46:06" or "2024-06-03"
      - a number (unix epoch seconds)
      - a dict (your metadata entry) - will search common keys
    Returns:
      - datetime object on success
      - None on failure
    """
    if not date_input:
        return None

    # If it's already a datetime
    if isinstance(date_input, datetime):
        return date_input

    # If numeric (epoch)
    if isinstance(date_input, (int, float)):
        try:
            return datetime.fromtimestamp(int(date_input))
        except Exception:
            return None

    # If dict: try common keys and nested exif
    if isinstance(date_input, dict):
        # try a few likely keys in order
        for k in ("datetime", "DateTimeOriginal", "DateTime", "DateTimeDigitized", "datetime_original"):
            v = date_input.get(k)
            if v:
                dt = parse_date(v)
                if dt:
                    return dt
        # try nested exif dict
        exif = date_input.get("exif") or date_input.get("EXIF")
        if isinstance(exif, dict):
            for k in ("DateTimeOriginal", "DateTime", "DateTimeDigitized"):
                if k in exif and exif[k]:
                    dt = parse_date(exif[k])
                    if dt:
                        return dt
        # sometimes your structure may store the datetime under top-level 'datetime' as string (already handled)
        return None

    # Otherwise assume string-like
    s = str(date_input).strip()
    if not s:
        return None

    # Normalize common nuisances
    # remove trailing 'Z' (UTC marker)
    if s.endswith("Z"):
        s = s[:-1]
    # replace 'T' with space
    s = s.replace("T", " ")
    # remove fractional seconds part like ".085" or ".123456"
    s = re.sub(r"\.\d+$", "", s)
    # sometimes timezone is given like +03:30, keep it (handled by formats)
    # try parsing with a set of formats
    dt = _try_strptime(s)
    if dt:
        return dt

    # Last-ditch: capture common EXIF "YYYY:MM:DD HH:MM:SS" with regex
    m = re.search(r"(\d{4}[:\-]\d{2}[:\-]\d{2})\s+(\d{2}:\d{2}:\d{2})", s)
    if m:
        candidate = f"{m.group(1)} {m.group(2)}"
        dt = _try_strptime(candidate)
        if dt:
            return dt

    # failed
    return None

def group_by_date(date_dict, days_threshold=4):
    """Group photos by date proximity."""
    parsed_items = [
        (photo, parsed_date) 
        for photo, date in date_dict.items() 
        if date and (parsed_date := parse_date(date)) is not None
    ]

    sorted_items = sorted(parsed_items, key=lambda x: x[1])

    groups = []
    current_group = []

    for i, (photo, date) in enumerate(sorted_items):
        if not current_group:
            current_group.append((photo, date))
            continue
        


        prev_date = current_group[-1][1]
        if (date - prev_date).days <= days_threshold:
            current_group.append((photo, date))
        else:
            groups.append(current_group)
            current_group = [(photo, date)]

    if current_group:
        groups.append(current_group)

    return groups

def most_common_tag(photos):
    """Find the most repeated tag in a group."""
    tag_count = {}
    for photo in photos:
        tags = tags_data.get(photo, [])
        for tag in tags:
            tag_count[tag] = tag_count.get(tag, 0) + 1
    if tag_count:
        return max(tag_count, key=tag_count.get)
    return None

# Extract string location from dicts
def extract_location(loc_val):
    if isinstance(loc_val, dict):
        # Choose what you want — e.g., city, or "city, country"
        city = loc_val.get("city", "")
        country = loc_val.get("country", "")
        return ", ".join([p for p in [city, country] if p])
    return str(loc_val) if loc_val else ""


# ==== GROUPING ====
groups = group_by_date(date_data)
# print(groups)
for group in groups:
    photos = [p for p, _ in group]
    group_dates = [d for _, d in group if d]
    group_date_str = group_dates[0].strftime(DATE_FORMAT) if group_dates else ""

    # Location → choose the most frequent one
    locations = [extract_location(location_data.get(p)) for p in photos if location_data.get(p)]
    locations = [loc for loc in locations if loc]  # remove empty strings
    location = max(set(locations), key=locations.count) if locations else ""

    # Tag → most common
    tag = most_common_tag(photos) or ""

    # Build folder name
    parts = [p for p in [tag, location, group_date_str] if p]
    folder_name = "_".join(parts)
    folder_path = OUTPUT_DIR / folder_name
    folder_path.mkdir(parents=True, exist_ok=True)

    # Move photos
    for photo in photos:
        src = PHOTO_DIR / photo
        if src.exists():
            shutil.copy(src, folder_path / photo)  # use .move() if you want to move instead
        else:
            print(f"⚠️ Missing file: {photo}")

print("✅ Grouping complete!")