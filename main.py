#!/usr/bin/env python3
"""
main.py

Integrates your existing metadata-based grouping with your existing object-detection
and face-detection scripts (which are executed if needed). Produces:
 - organized folders under OUTPUT_DIR (copies of photos)
 - OUTPUT_DIR/organized_metadata.json containing per-folder & per-photo metadata
Notes:
 - This script prefers to run the detection scripts via the same Python interpreter
   (sys.executable) so it uses the same venv.
 - If object/face detection outputs already exist, it will reuse them to avoid re-running.
"""

from pathlib import Path
from datetime import datetime
import subprocess
import sys
import json
import shutil
import re
from collections import defaultdict

# ---------------- CONFIG ----------------
PHOTO_DIR = Path("/Users/sara/Desktop/backed-data/sample-data")
OUTPUT_DIR = Path("/Users/sara/Desktop/backed-data/organized")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# metadata files (must exist in PHOTO_DIR)
DATE_FILE = PHOTO_DIR / "metadata.json"
LOCATION_FILE = PHOTO_DIR / "metadata_with_location.json"
TAGS_FILE = PHOTO_DIR / "metadata_with_tags.json"

# object & face detection script paths (adjust if your filenames/locations differ)
OBJECT_SCRIPT = PHOTO_DIR / "object-detection.py"   # script you provided (writes object_detection_results.json)
OBJECT_JSON = PHOTO_DIR / "object_detection_results.json"

FACE_SCRIPT = PHOTO_DIR / "face-detection.py"       # script you provided (saves crops to faces_detected/)
FACES_DETECTED_DIR = Path("/Users/sara/Desktop/backed-data/faces_detected")  # as in your face script

# grouping params
DATE_FORMAT = "%Y-%m-%d"
DAYS_THRESHOLD = 4  # photos within this many days are grouped together

# ---------------- Utilities / date parsing ----------------
def _try_strptime(s: str):
    fmts = [
        "%Y:%m:%d %H:%M:%S",
        "%Y-%m-%d %H:%M:%S",
        "%Y-%m-%d",
        "%Y:%m:%d",
        "%Y-%m-%dT%H:%M:%S",
        "%Y-%m-%dT%H:%M:%S%z",
        "%Y-%m-%d %H:%M:%S.%f",
        "%Y:%m:%d %H:%M:%S.%f",
        "%d-%m-%Y",
        "%m/%d/%Y",
    ]
    for fmt in fmts:
        try:
            return datetime.strptime(s, fmt)
        except Exception:
            continue
    return None

def parse_date(date_input):
    """Robust parse for many EXIF/JSON date variants."""
    if not date_input:
        return None
    if isinstance(date_input, datetime):
        return date_input
    if isinstance(date_input, (int, float)):
        try:
            return datetime.fromtimestamp(int(date_input))
        except Exception:
            return None
    if isinstance(date_input, dict):
        for k in ("datetime", "DateTimeOriginal", "DateTime", "DateTimeDigitized", "datetime_original"):
            v = date_input.get(k)
            if v:
                dt = parse_date(v)
                if dt:
                    return dt
        exif = date_input.get("exif") or date_input.get("EXIF")
        if isinstance(exif, dict):
            for k in ("DateTimeOriginal", "DateTime", "DateTimeDigitized"):
                if k in exif and exif[k]:
                    dt = parse_date(exif[k])
                    if dt:
                        return dt
        return None
    s = str(date_input).strip()
    if not s:
        return None
    if s.endswith("Z"):
        s = s[:-1]
    s = s.replace("T", " ")
    s = re.sub(r"\.\d+$", "", s)
    dt = _try_strptime(s)
    if dt:
        return dt
    m = re.search(r"(\d{4}[:\-]\d{2}[:\-]\d{2})\s+(\d{2}:\d{2}:\d{2})", s)
    if m:
        candidate = f"{m.group(1)} {m.group(2)}"
        dt = _try_strptime(candidate)
        if dt:
            return dt
    return None

def group_by_date(date_dict, days_threshold=DAYS_THRESHOLD):
    parsed_items = [
        (photo, parsed_date)
        for photo, date in date_dict.items()
        if date and (parsed_date := parse_date(date)) is not None
    ]
    sorted_items = sorted(parsed_items, key=lambda x: x[1])
    groups = []
    current_group = []
    for (photo, date) in sorted_items:
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

# ---------------- Helpers for location/tags ----------------
def extract_location(loc_val):
    if isinstance(loc_val, dict):
        city = loc_val.get("city", "") or loc_val.get("city_name", "")
        country = loc_val.get("country", "")
        return ", ".join([p for p in [city, country] if p])
    return str(loc_val) if loc_val else ""

def most_common_tag_for_group(photos, tags_data):
    tag_count = {}
    for photo in photos:
        tags = tags_data.get(photo, [])
        for tag in tags:
            tag_count[tag] = tag_count.get(tag, 0) + 1
    if tag_count:
        # return tag with highest count; tie-breaker by lexicographic for determinism
        best = max(sorted(tag_count.keys()), key=lambda k: tag_count[k])
        return best
    return None

# ---------------- Detection script runners ----------------
def run_object_detection_if_needed(object_script_path: Path, object_json_path: Path, force=False):
    """
    If object_json_path exists and not force, load and return mapping.
    Otherwise run the provided object detection script (via subprocess) and then load JSON.
    Returns dict mapping image filename -> list of objects (each is dict label/confidence/bbox)
    """
    if object_json_path.exists() and not force:
        try:
            with open(object_json_path, "r") as f:
                return json.load(f)
        except Exception:
            print("⚠️ Failed to load existing object JSON; will attempt to re-run detection.")

    if not object_script_path.exists():
        print(f"⚠️ Object detection script not found at {object_script_path}. Skipping object detection.")
        return {}

    print(f"Running object detection script: {object_script_path} ...")
    proc = subprocess.run([sys.executable, str(object_script_path)], capture_output=True, text=True)
    if proc.returncode != 0:
        print("❌ Object detection script failed:")
        print(proc.stdout)
        print(proc.stderr)
        return {}
    # After script runs, try to load JSON
    if object_json_path.exists():
        with open(object_json_path, "r") as f:
            return json.load(f)
    print("⚠️ Object detection did not produce JSON at expected path:", object_json_path)
    return {}

def run_face_detection_if_needed(face_script_path: Path, faces_dir: Path, force=False):
    """
    Runs your face-detection script if faces_dir is missing or empty (unless force=False).
    Returns a mapping photo filename -> list of detected face crop filenames (in faces_dir).
    """
    faces_dir.mkdir(parents=True, exist_ok=True)
    existing = list(faces_dir.glob("*"))
    if existing and not force:
        # build mapping from existing cropped faces
        mapping = defaultdict(list)
        for p in faces_dir.iterdir():
            if p.is_file():
                # try to infer original image stem from filename pattern like IMG_123_face0.jpg
                stem = p.stem
                # split at "_face" or similar
                orig = None
                if "_face" in stem:
                    orig = stem.split("_face")[0] + p.suffix
                else:
                    # fallback: use prefix up to first underscore + extension
                    parts = stem.split("_")
                    if len(parts) >= 2:
                        orig = parts[0] + "_" + parts[1] + p.suffix
                    else:
                        # unknown; put under unknown
                        orig = "__unknown__"
                mapping[orig].append(str(p))
        return mapping

    # If no crops or we force, run the script
    if not face_script_path.exists():
        print(f"⚠️ Face detection script not found at {face_script_path}. Skipping face detection.")
        return {}

    print(f"Running face detection script: {face_script_path} ...")
    proc = subprocess.run([sys.executable, str(face_script_path)], capture_output=True, text=True)
    if proc.returncode != 0:
        print("❌ Face detection script failed:")
        print(proc.stdout)
        print(proc.stderr)
        return {}

    # Build mapping similar to above after the script finishes
    mapping = defaultdict(list)
    for p in faces_dir.iterdir():
        if p.is_file():
            stem = p.stem
            if "_face" in stem:
                orig = stem.split("_face")[0] + p.suffix
            else:
                parts = stem.split("_")
                if len(parts) >= 2:
                    orig = parts[0] + "_" + parts[1] + p.suffix
                else:
                    orig = "__unknown__"
            mapping[orig].append(str(p))
    return mapping

# ---------------- Main organize function ----------------
def organize_photos(photo_dir=PHOTO_DIR, output_dir=OUTPUT_DIR, run_detection=True, force_detection=False):
    photo_dir = Path(photo_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load metadata files
    if not DATE_FILE.exists():
        raise FileNotFoundError(f"Date metadata file not found: {DATE_FILE}")
    if not LOCATION_FILE.exists():
        print(f"⚠️ Location file not found: {LOCATION_FILE} — location will be empty for photos.")
    if not TAGS_FILE.exists():
        print(f"⚠️ Tags file not found: {TAGS_FILE} — tags will be empty for photos.")

    with open(DATE_FILE, "r") as f:
        date_data = json.load(f)
    location_data = {}
    tags_data = {}
    if LOCATION_FILE.exists():
        with open(LOCATION_FILE, "r") as f:
            location_data = json.load(f)
    if TAGS_FILE.exists():
        with open(TAGS_FILE, "r") as f:
            tags_data = json.load(f)

    # Run detection scripts (if requested) and load outputs
    object_map = {}
    face_map = {}
    if run_detection:
        object_map = run_object_detection_if_needed(OBJECT_SCRIPT, OBJECT_JSON, force=force_detection)
        face_map = run_face_detection_if_needed(FACE_SCRIPT, FACES_DETECTED_DIR, force=force_detection)

    # Group by date
    groups = group_by_date(date_data, days_threshold=DAYS_THRESHOLD)

    organized = {}  # folder_name -> list of photo metadata dicts

    for group in groups:
        photos = [p for p, _ in group]
        group_dates = [d for _, d in group if d]
        group_date_str = group_dates[0].strftime(DATE_FORMAT) if group_dates else ""

        # Most frequent location for this group
        locations = [extract_location(location_data.get(p)) for p in photos if location_data.get(p)]
        locations = [loc for loc in locations if loc]
        location = max(set(locations), key=locations.count) if locations else ""

        # Most common tag (situation)
        tag = most_common_tag_for_group(photos, tags_data) or ""

        # Build folder name
        parts = [p for p in [tag, location, group_date_str] if p]
        folder_name = "_".join(parts) if parts else f"Group_{group_date_str or 'unknown'}"
        folder_path = output_dir / folder_name
        folder_path.mkdir(parents=True, exist_ok=True)

        per_photo_list = []
        for photo in photos:
            src = photo_dir / photo
            if not src.exists():
                print(f"⚠️ Missing file: {src}")
                continue

            # copy photo into folder
            dst = folder_path / photo
            try:
                shutil.copy(src, dst)
            except Exception as e:
                print(f"⚠️ Failed to copy {src} -> {dst}: {e}")
                continue

            # get object detections for this filename (object_map keys are image filenames)
            objects = object_map.get(photo, []) if isinstance(object_map, dict) else []

            # get face crops for this filename (face_map keys are filenames inferred earlier)
            faces = face_map.get(photo, []) if isinstance(face_map, dict) else []
            faces_count = len(faces)

            # build per-photo metadata entry
            entry = {
                "original_filename": str(src),
                "copied_path": str(dst),
                "date": (parse_date(date_data.get(photo)).strftime(DATE_FORMAT) if date_data.get(photo) else None),
                "location": extract_location(location_data.get(photo)),
                "tag": tags_data.get(photo, []),
                "objects": objects,
                "faces_count": faces_count,
                "face_crops": faces,
            }
            per_photo_list.append(entry)

        organized[folder_name] = per_photo_list

    # Save organized metadata to JSON
    out_meta = output_dir / "organized_metadata.json"
    with open(out_meta, "w") as f:
        json.dump(organized, f, indent=2)

    print(f"✅ Organization complete — folders created in: {output_dir}")
    print(f"✅ Metadata saved to: {out_meta}")
    return organized

# ---------------- CLI ----------------
if __name__ == "__main__":
    # Example usage from command line: python main.py
    organize_photos()