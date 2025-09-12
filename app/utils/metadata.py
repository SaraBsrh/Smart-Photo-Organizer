from PIL import Image
from PIL.ExifTags import TAGS, GPSTAGS
from pathlib import Path
from fractions import Fraction
from PIL.TiffImagePlugin import IFDRational
import os
import json

def get_exif_data(image_path):
    try:
        image = Image.open(image_path)
        exif_data_raw = image._getexif()
        exif_data = {}
        if exif_data_raw:
            for tag_id, value in exif_data_raw.items():
                tag = TAGS.get(tag_id, tag_id)
                if tag == "GPSInfo":
                    gps_data = {}
                    for key in value.keys():
                        name = GPSTAGS.get(key, key)
                        gps_data[name] = value[key]
                    exif_data["GPSInfo"] = gps_data
                else:
                    exif_data[tag] = value
        return exif_data
    except Exception as e:
        print(f"Error reading {image_path}: {e}")
        return {}

def _convert_to_degrees(value):
    """
    Convert GPS coordinates stored as:
    - [(num, den), (num, den), (num, den)] tuples
    - OR as IFDRational objects
    into decimal degrees.
    """
    def to_float(v):
        # Works for IFDRational or tuple
        if isinstance(v, tuple):
            return float(v[0]) / float(v[1])
        return float(v)  # IFDRational

    d = to_float(value[0])
    m = to_float(value[1])
    s = to_float(value[2])

    return d + (m / 60.0) + (s / 3600.0)

def get_lat_lon(gps_info):
    lat = lon = None
    if gps_info:
        try:
            lat = _convert_to_degrees(gps_info["GPSLatitude"])
            if gps_info.get("GPSLatitudeRef") != "N":
                lat = -lat
            lon = _convert_to_degrees(gps_info["GPSLongitude"])
            if gps_info.get("GPSLongitudeRef") != "E":
                lon = -lon
        except KeyError:
            pass
    return lat, lon

def extract_metadata_from(folder_path):
    metadata = {}
    for filename in os.listdir(folder_path):
        if filename.lower().endswith(('.jpg', '.jpeg', '.png')):
            full_path = os.path.join(folder_path, filename)
            exif = get_exif_data(full_path)
            gps_info = exif.get("GPSInfo", {})
            lat, lon = get_lat_lon(gps_info)
            metadata[filename] = {
                "datetime": exif.get("DateTimeOriginal"),
                "make": exif.get("Make"),
                "model": exif.get("Model"),
                "lat": lat,
                "lon": lon,
                "exif": exif  # keep raw data for debugging
            }
    return metadata

def make_serializable(obj):
    """Convert Pillow's EXIF objects to JSON-serializable types."""
    if isinstance(obj, IFDRational) or isinstance(obj, Fraction):
        return float(obj)  # convert to float
    elif isinstance(obj, bytes):
        try:
            return obj.decode(errors="ignore")
        except:
            return str(obj)
    elif isinstance(obj, tuple):
        return tuple(make_serializable(x) for x in obj)
    elif isinstance(obj, list):
        return [make_serializable(x) for x in obj]
    elif isinstance(obj, dict):
        return {k: make_serializable(v) for k, v in obj.items()}
    else:
        return obj

if __name__ == "__main__":
    folder = Path(os.getenv("PHOTO_DIR", "data/sample-data"))
    data = extract_metadata_from(folder)

    # Save to JSON for full detail
    clean_data = {fname: make_serializable(meta) for fname, meta in data.items()}

    with open("metadata.json", "w", encoding="utf-8") as f:
        json.dump(clean_data, f, indent=4, ensure_ascii=False)

    print(f"Extracted metadata for {len(data)} images → saved to metadata.json")