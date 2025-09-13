"""
Organize user-uploaded photos end-to-end.

Exposes:
    organize_photos(photo_dir, output_dir, run_detection=True, force_detection=False)
"""
from pathlib import Path
import json
import shutil
import sys
import subprocess
import os
import re
from datetime import datetime
from collections import defaultdict, Counter
from typing import Dict, Any, List, Tuple
from PIL import Image, ExifTags
import numbers
from fractions import Fraction

# Optional numpy support
try:
    import numpy as _np
except Exception:
    _np = None

# ---------------- Config ----------------
DEFAULT_OUTPUT_DIR = Path("organized_output")
DATE_FORMAT = "%Y-%m-%d"
DAYS_THRESHOLD = 4

# ---------------- Robust imports (optional helpers) ----------------
# Try to import user-provided helpers if available; fall back otherwise.
try:
    from app.utils.scene_tags import add_scene_tags as _add_scene_tags
except Exception:
    try:
        from utils.scene_tags import add_scene_tags as _add_scene_tags
    except Exception:
        _add_scene_tags = None  # fallback later

try:
    from app.utils.geocode import add_locations_to_metadata as _add_locations_to_metadata
except Exception:
    try:
        from utils.geocode import add_locations_to_metadata as _add_locations_to_metadata
    except Exception:
        _add_locations_to_metadata = None

# Try to import extract_metadata_from and make_serializable if provided by utils.metadata
_extract_meta = None
_make_serializable_external = None
try:
    from app.utils.metadata import extract_metadata_from as _ext_meta, make_serializable as _ms_ext
    _extract_meta = _ext_meta
    _make_serializable_external = _ms_ext
except Exception:
    try:
        from utils.metadata import extract_metadata_from as _ext_meta, make_serializable as _ms_ext
        _extract_meta = _ext_meta
        _make_serializable_external = _ms_ext
    except Exception:
        # will use fallback extractor and internal serializer
        _extract_meta = None
        _make_serializable_external = None

# ---------------- Helpers ----------------
def find_script_in_repo(name: str) -> Path | None:
    cwd = Path.cwd()
    for p in cwd.rglob(name):
        if ".venv" in p.parts or "site-packages" in str(p):
            continue
        return p
    return None

def run_script(script_path: Path, args: List[str]) -> Tuple[int, str, str]:
    cmd = [sys.executable, str(script_path)] + args
    proc = subprocess.run(cmd, capture_output=True, text=True)
    return proc.returncode, proc.stdout, proc.stderr

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

def parse_date(date_input) -> datetime | None:
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

def group_by_date(date_dict: Dict[str, Any], days_threshold: int = DAYS_THRESHOLD) -> List[List[Tuple[str, datetime]]]:
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

def sanitize_folder_name(s: str) -> str:
    s = s.strip()
    s = s.replace("/", "-").replace("\\", "-").replace(":", "-")
    s = re.sub(r"\s+", "_", s)
    return s[:200]

# ---------------- JSON sanitizer ----------------
def make_serializable(obj):
    """
    Convert common non-json types to JSON-friendly primitives.
    Handles IFDRational (Pillow), Fraction, numpy scalars/arrays, datetime, Path, tuples, sets, bytes.
    """
    # if external serializer exists prefer it
    if _make_serializable_external:
        try:
            return _make_serializable_external(obj)
        except Exception:
            pass

    # None / basic
    if obj is None:
        return None
    if isinstance(obj, (str, bool)):
        return obj
    if isinstance(obj, numbers.Number):
        # ints and floats
        return obj
    if _np is not None and isinstance(obj, _np.generic):
        return obj.item()
    if isinstance(obj, Fraction):
        return float(obj)
    # datetime
    if isinstance(obj, datetime):
        return obj.isoformat()
    # Path
    if isinstance(obj, Path):
        return str(obj)
    # dict
    if isinstance(obj, dict):
        out = {}
        for k, v in obj.items():
            try:
                ks = str(k)
            except Exception:
                ks = repr(k)
            out[ks] = make_serializable(v)
        return out
    # list/tuple/set
    if isinstance(obj, (list, tuple, set)):
        return [make_serializable(x) for x in obj]
    # numpy array
    if _np is not None and hasattr(_np, "ndarray") and isinstance(obj, _np.ndarray):
        return [make_serializable(x) for x in obj.tolist()]
    # bytes
    if isinstance(obj, (bytes, bytearray)):
        try:
            return obj.decode(errors="ignore")
        except Exception:
            return str(obj)
    # Pillow IFDRational / similar: try float conversion
    try:
        return float(obj)
    except Exception:
        pass
    # fallback to str
    try:
        return str(obj)
    except Exception:
        return None

# ---------------- Metadata extraction (fallback) ----------------
def extract_metadata_fallback(photo_dir: Path) -> Dict[str, Any]:
    def _get_exif(img: Image.Image):
        try:
            raw = img._getexif()
            if not raw:
                return {}
            ex = {}
            for tag, val in raw.items():
                name = ExifTags.TAGS.get(tag, tag)
                ex[name] = val
            return ex
        except Exception:
            return {}

    def _gps_to_decimal(gps):
        try:
            # gps might be ([(deg_num,deg_den),...], ref) or (deg, min, sec) objects
            coords, ref = gps
            d = coords[0]
            m = coords[1]
            s = coords[2]
            def to_float(x):
                if isinstance(x, tuple) and len(x) == 2:
                    return x[0] / x[1] if x[1] else 0.0
                return float(x)
            deg = to_float(d)
            minu = to_float(m)
            sec = to_float(s)
            dec = deg + (minu / 60.0) + (sec / 3600.0)
            if ref in ("S", "W"):
                dec = -dec
            return dec
        except Exception:
            return None

    out = {}
    for p in sorted(photo_dir.iterdir()):
        if not p.is_file() or p.suffix.lower() not in (".jpg", ".jpeg", ".png"):
            continue
        try:
            img = Image.open(p)
        except Exception:
            continue
        exif = _get_exif(img)
        date_val = exif.get("DateTimeOriginal") or exif.get("DateTime") or ""
        lat = None
        lon = None
        if "GPSInfo" in exif and isinstance(exif["GPSInfo"], dict):
            gps_raw = exif["GPSInfo"]
            lat_ref = gps_raw.get(1) or gps_raw.get("GPSLatitudeRef")
            lat_val = gps_raw.get(2) or gps_raw.get("GPSLatitude")
            lon_ref = gps_raw.get(3) or gps_raw.get("GPSLongitudeRef")
            lon_val = gps_raw.get(4) or gps_raw.get("GPSLongitude")
            if lat_val and lat_ref:
                try:
                    lat = _gps_to_decimal((lat_val, lat_ref))
                    lon = _gps_to_decimal((lon_val, lon_ref)) if lon_val and lon_ref else None
                except Exception:
                    lat = None
                    lon = None
        out[p.name] = {
            "datetime": date_val,
            "exif": exif,
            "lat": lat,
            "lon": lon,
        }
    # sanitize before returning
    return {k: make_serializable(v) for k, v in out.items()}

# ---------------- Main pipeline ----------------
def organize_photos(photo_dir, output_dir: Path | str = DEFAULT_OUTPUT_DIR, run_detection: bool = True, force_detection: bool = False):
    photo_dir = Path(photo_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # 1) Extract metadata
    metadata = None
    try:
        if _extract_meta:
            metadata = _extract_meta(str(photo_dir))
            # If external make_serializable exists, use it; otherwise use local sanitizer
            metadata = {k: (_make_serializable_external(v) if _make_serializable_external else make_serializable(v)) for k, v in metadata.items()}
        else:
            metadata = extract_metadata_fallback(photo_dir)
    except Exception:
        # fallback safe extractor
        metadata = extract_metadata_fallback(photo_dir)

    # Save basic date metadata (sanitized)
    date_file = output_dir / "metadata.json"
    with open(date_file, "w", encoding="utf-8") as f:
        json.dump(make_serializable(metadata), f, indent=2, ensure_ascii=False)

    # 2) Add locations
    if _add_locations_to_metadata:
        # try calling with (metadata, photo_dir) then fallback to (metadata,)
        try:
            metadata = _add_locations_to_metadata(metadata, photo_dir)
        except TypeError:
            try:
                metadata = _add_locations_to_metadata(metadata)
            except Exception:
                pass
        except Exception:
            pass

    location_file = output_dir / "metadata_with_location.json"
    with open(location_file, "w", encoding="utf-8") as f:
        json.dump(make_serializable(metadata), f, indent=2, ensure_ascii=False)

    # 3) Add scene tags
    if _add_scene_tags:
        try:
            metadata = _add_scene_tags(
                metadata,
                photo_dir,
                save_path=output_dir / "metadata_with_tags.json"
            )
        except Exception as e:
            print(f"⚠️ _add_scene_tags failed: {e}. Will try to load tags JSON if available.")

    else:
        # Ensure metadata entries are dicts and have 'tags' key
        for k, v in list(metadata.items()):
            if isinstance(v, dict):
                v.setdefault("tags", [])
            else:
                metadata[k] = {"datetime": v, "tags": []}

    # Path where we will write final tags JSON
    tags_file = output_dir / "metadata_with_tags.json"

    # Write a best-effort intermediate tags file (serializable)
    with open(tags_file, "w", encoding="utf-8") as f:
        json.dump(make_serializable(metadata), f, indent=2, ensure_ascii=False)

    # --- Merge external tags if scene_tags.py (or other process) wrote a tags file ---
    # Check both source folder and output folder for possible external tags JSON
    candidates = [photo_dir / "metadata_with_tags.json", tags_file]
    for candidate in candidates:
        if candidate.exists():
            try:
                with open(candidate, "r", encoding="utf-8") as tf:
                    ext_map = json.load(tf)
                # ext_map expected: { "IMG_001.jpg": ["beach","trip"], ... } OR maybe same shape as metadata
                # Merge: coerce tags to list, and place into metadata[filename]["tags"]
                for fname, tags in ext_map.items():
                    if tags is None:
                        continue
                    if isinstance(tags, dict):
                        # if ext_map uses full metadata entries, extract 'tags' key
                        tags = tags.get("tags", [])
                    if not isinstance(tags, list):
                        tags = [tags]
                    entry = metadata.get(fname)
                    if entry is None:
                        metadata[fname] = {"datetime": None, "tags": tags}
                    else:
                        if not isinstance(entry, dict):
                            metadata[fname] = {"datetime": entry, "tags": tags}
                        else:
                            # overwrite or set tags for this entry
                            entry["tags"] = tags
            except Exception as e:
                print(f"⚠️ Failed to merge tags from {candidate}: {e}")

    # Final sanity: ensure every metadata entry is a dict with a tags list
    for k, v in list(metadata.items()):
        if isinstance(v, dict):
            if "tags" not in v or v["tags"] is None:
                v["tags"] = []
        else:
            metadata[k] = {"datetime": v, "tags": []}

    # Write final merged tags file (single canonical place)
    with open(tags_file, "w", encoding="utf-8") as f:
        json.dump(make_serializable(metadata), f, indent=2, ensure_ascii=False)


    # 4) Run object detection if requested
    object_map = {}
    if run_detection:
        obj_script = find_script_in_repo("object-detection.py")
        if obj_script:
            obj_json_path = output_dir / "object_detection_results.json"
            rc, out, err = run_script(obj_script, [str(photo_dir), str(obj_json_path)])
            if rc != 0:
                # fallback try without args
                rc2, o2, e2 = run_script(obj_script, [])
            if obj_json_path.exists():
                try:
                    with open(obj_json_path, "r", encoding="utf-8") as f:
                        object_map = json.load(f)
                except Exception:
                    object_map = {}
        else:
            object_map = {}

    # 5) Run face detection if requested
    face_map = {}
    if run_detection:
        face_script = find_script_in_repo("face-detection.py")
        faces_out_dir = output_dir / "faces_detected"
        faces_out_dir.mkdir(exist_ok=True)
        if face_script:
            rc, out, err = run_script(face_script, [str(photo_dir), str(faces_out_dir)])
            if rc != 0:
                rc2, o2, e2 = run_script(face_script, [])
            # collect face crops
            fm = defaultdict(list)
            for p in faces_out_dir.iterdir():
                if p.is_file() and p.suffix.lower() in (".jpg", ".jpeg", ".png"):
                    stem = p.stem
                    if "_face" in stem:
                        orig = stem.split("_face")[0] + p.suffix
                    else:
                        orig = "__unknown__"
                    fm[orig].append(str(p))
            face_map = dict(fm)
        else:
            face_map = {}

    # 6) Group by date & build organized folders
    groups = group_by_date({fn: metadata.get(fn, {}).get("datetime") if isinstance(metadata.get(fn), dict) else metadata.get(fn) for fn in metadata})
    organized = {}

    for gi, group in enumerate(groups):
        photos = [p for p, _ in group]
        group_dates = [d for _, d in group if d]
        group_date_str = group_dates[0].strftime(DATE_FORMAT) if group_dates else ""

        # location -> most frequent
        locs = []
        for p in photos:
            info = metadata.get(p, {})
            if isinstance(info, dict):
                city = info.get("city") or info.get("City") or info.get("city_name") or ""
                country = info.get("country") or info.get("Country") or ""
                loc = ", ".join([x for x in [city, country] if x])
                if loc:
                    locs.append(loc)
        location = Counter(locs).most_common(1)[0][0] if locs else None

        # tags -> most common
        tag_candidates = []
        per_photo_tags = []
        for p in photos:
            info = metadata.get(p, {})
            if isinstance(info, dict):
                tags = info.get("tags") or info.get("tag") or []
                if isinstance(tags, str):
                    tags = [tags]
                    
                tag_candidates.extend(tags)
        tag = Counter(tag_candidates).most_common(1)[0][0] if tag_candidates else None

        parts = []
        if tag:
            parts.append(sanitize_folder_name(tag))
        if location:
            parts.append(sanitize_folder_name(location))
        if group_date_str:
            parts.append(group_date_str)
        folder_name = "_".join(parts) if parts else f"group_{gi}_{group_date_str or 'unknown'}"
        folder_path = output_dir / folder_name
        suffix = 1
        base_folder_path = folder_path
        while folder_path.exists():
            folder_path = Path(str(base_folder_path) + f"_{suffix}")
            suffix += 1
        folder_path.mkdir(parents=True, exist_ok=True)

        per_group_entries = []
        for photo in photos:
            src = photo_dir / photo
            if not src.exists():
                continue
            dst = folder_path / photo
            try:
                shutil.copy(src, dst)
            except Exception:
                dst = src

            objects = object_map.get(photo, [])
            faces = face_map.get(photo, [])
            faces_count = len(faces)

            info = metadata.get(photo, {})
            parsed_dt = parse_date(info.get("datetime") if isinstance(info, dict) else info)
            entry = {
                "original_filename": str(src),
                "copied_path": str(dst),
                "date": parsed_dt.strftime(DATE_FORMAT) if parsed_dt else None,
                "location": location or "",
                "group_tag": tag,                     # most common tag for the group
                "tags": info.get("tags", []),         # keep per-photo scene tags
                "objects": objects,
                "faces_count": faces_count,
                "face_crops": faces,
            }

            per_group_entries.append(make_serializable(entry))

        organized[folder_name] = per_group_entries

    organized_file = output_dir / "organized_metadata.json"
    metadataa = make_serializable(organized)
    with open(organized_file, "w", encoding="utf-8") as f:
        json.dump(metadataa, f, indent=2, ensure_ascii=False)

    return output_dir
    return metadataa

def load_organized_metadata(output_dir=DEFAULT_OUTPUT_DIR):
    organized_file = Path(output_dir) / "organized_metadata.json"
    if organized_file.exists():
        with open(organized_file, "r", encoding="utf-8") as f:
            return json.load(f)
    return {}

# ---------------- CLI ----------------
if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("photo_dir", nargs="?", default=".", help="Folder of photos")
    parser.add_argument("--out", "-o", default=str(DEFAULT_OUTPUT_DIR), help="Output folder")
    args = parser.parse_args()
    organize_photos(args.photo_dir, args.out)
