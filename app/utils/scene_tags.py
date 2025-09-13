"""
scene_tags.py

Provides:
 - add_scene_tags(metadata, photo_dir, topk=5, save_path=None)
 - CLI to run standalone and write organized_output/metadata_with_tags.json

Notes:
 - Heavy ML models are loaded lazily (only when first tagging happens),
   so importing this module is cheap and safe for main.py.
"""
from pathlib import Path
import json
from typing import Dict, Any, List, Tuple
import sys
import traceback
import torch

# --- Configuration (adjust paths as needed) ---
UPLOAD_DIR = Path("/Applications/portfolio-projects/smart-photo-organizer/uploaded_photos")
OUTPUT_DIR = Path("/Applications/portfolio-projects/smart-photo-organizer/organized_output")
DEFAULT_OUTPUT_FILE = OUTPUT_DIR / "metadata_with_tags.json"
METADATA_FILE_PREF = OUTPUT_DIR / "metadata.json"   # preferred base metadata produced by main.py

# ----------------- Lazy ML imports / initialization -----------------
# We perform heavy imports only when needed to avoid startup cost when main.py imports this module.
_models_initialized = False

# placeholders for models / transforms
_model_places = None
_places_classes = None
_places_transform = None

_model_clip = None
_clip_preprocess = None
_clip_device = None

# you can change this path to your Places365 model if needed
MODEL_PATH = "/Applications/portfolio-projects/smart-photo-organizer/models/wideresnet18_places365.pth.tar"
PLACES_CLASSES_PATH = "/Applications/portfolio-projects/smart-photo-organizer/app/utils/categories_places365.txt"

# We'll import ML packages only inside initializer
def _init_models():
    global _models_initialized
    global _model_places, _places_classes, _places_transform
    global _model_clip, _clip_preprocess, _clip_device

    if _models_initialized:
        return

    try:
        import torch
        import torch.nn.functional as F
        import torchvision.models as models
        from torchvision.transforms import Compose, Resize, CenterCrop, ToTensor, Normalize, ColorJitter
        from PIL import Image
        import clip
        # load Places365 classes file if present
        places365_classes = []
        try:
            with open(PLACES_CLASSES_PATH, "r", encoding="utf-8") as f:
                places365_classes = [line.strip().split(' ')[0][3:] for line in f.readlines()]
        except Exception:
            # fallback: empty list (we'll still run CLIP if available)
            places365_classes = []

        # Places model
        try:
            model_places = models.resnet18(num_classes=365)
            checkpoint = torch.load(MODEL_PATH, map_location='cpu')
            state_dict = {k.replace("module.", ""): v for k, v in checkpoint["state_dict"].items()}
            model_places.load_state_dict(state_dict)
            model_places.eval()
            # simple transform
            places_transform = Compose([
                Resize(256),
                CenterCrop(224),
                ColorJitter(brightness=0.2, contrast=0.2),
                ToTensor(),
                Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ])
        except Exception:
            model_places = None
            places_transform = None

        # CLIP
        try:
            device = "cuda" if torch.cuda.is_available() else "cpu"
            model_clip, preprocess_clip = clip.load("ViT-B/32", device=device)
        except Exception:
            model_clip = None
            preprocess_clip = None
            device = "cpu"

        # assign to module-level
        _model_places = model_places
        _places_classes = places365_classes
        _places_transform = places_transform

        _model_clip = model_clip
        _clip_preprocess = preprocess_clip
        _clip_device = device

    except Exception:
        # If any of these fail, we still keep module importable and will fallback to safe behavior.
        traceback.print_exc()
        _model_places = None
        _places_classes = []
        _places_transform = None
        _model_clip = None
        _clip_preprocess = None
        _clip_device = "cpu"
    finally:
        _models_initialized = True

# ----------------- Tagging logic (uses models lazily) -----------------
# The following functions largely mirror the behavior you had: Places365 + CLIP fusion.
# If models are missing, we fall back to simple tags to avoid crashes.

def _transform_image_places365(image_path):
    from PIL import Image
    if _places_transform is None:
        return None
    img = Image.open(image_path).convert("RGB")
    return _places_transform(img).unsqueeze(0)  # batch dim

def _predict_places365(image_tensor, topk=5):
    import torch.nn.functional as F
    with torch.no_grad():
        out = _model_places(image_tensor)
        probs = F.softmax(out, dim=1)
        top_probs, top_idxs = probs.topk(topk)
    return top_probs.squeeze(), top_idxs.squeeze()

def _transform_image_clip(image_path):
    from PIL import Image
    img = Image.open(image_path).convert("RGB")
    return _clip_preprocess(img).unsqueeze(0).to(_clip_device)

def _weighted_scene_fusion(places_tags, clip_tags, weight_clip=0.55):
    score_map = {}
    # higher earlier index -> higher score (reverse ordering)
    for i, tag in enumerate(places_tags):
        score_map[tag] = score_map.get(tag, 0) + (1 - weight_clip) * (len(places_tags) - i)
    for i, tag in enumerate(clip_tags):
        score_map[tag] = score_map.get(tag, 0) + weight_clip * (len(clip_tags) - i)
    # return tags sorted by score descending
    return sorted(score_map, key=score_map.get, reverse=True)

# A small custom categories list (you can expand)
_custom_categories = [
    "person", "people", "human", "portrait", "crowd", "selfie", "family", "friends",
    "birthday", "trip", "museum", "art-gallery", "vacation", "travel", "holiday",
    "hiking", "beach", "forest", "party", "wedding", "road trip", "camping", "mountains", "lake",
]

# We'll use fuzzy matching only if rapidfuzz is available
try:
    from rapidfuzz import process as _rpf_process
except Exception:
    _rpf_process = None

def _map_to_custom_category(tags: List[str]) -> str:
    # pick best match among tags to our custom list
    if not tags:
        return ""
    best_match = tags[0]
    best_score = 0
    if _rpf_process:
        for t in tags:
            match, score, _ = _rpf_process.extractOne(t, _custom_categories)
            if score > best_score:
                best_score = score
                best_match = match
        return best_match
    # fallback: return first tag
    return best_match

def combined_scene_tagging(image_path: str, topk: int = 5) -> List[str]:
    """
    Return a list of topk fused scene tags for the image.
    Uses Places365 + CLIP fusion if available; otherwise returns simple fallback tags.
    Models are initialized on first call.
    """
    # lazy init
    if not _models_initialized:
        _init_models()

    # If both models unavailable -> fallback
    if _model_clip is None and _model_places is None:
        # very small heuristic fallback tags
        name = Path(image_path).stem.lower()
        simple = []
        if any(x in name for x in ("selfie", "portrait")):
            simple.append("portrait")
        if any(x in name for x in ("beach", "sea", "sand")):
            simple.append("beach")
        if not simple:
            simple = ["photo"]
        return simple[:topk]

    places_tags = []
    try:
        if _model_places is not None and _places_transform is not None and _places_classes:
            t = _transform_image_places365(image_path)
            if t is not None:
                _, idxs = _predict_places365(t, topk=topk)
                # idxs may be a tensor
                places_tags = [_places_classes[int(i)] for i in idxs.tolist()]
    except Exception:
        places_tags = []

    clip_tags = []
    try:
        if _model_clip is not None and _clip_preprocess is not None:
            # build candidate texts from union of places_tags and custom categories
            all_tags = list(dict.fromkeys((places_tags or []) + _custom_categories))
            text_inputs = _model_clip.tokenize([f"a photo of {tag}" for tag in all_tags]).to(_clip_device)
            image_clip = _transform_image_clip(image_path)
            with __import__("torch").no_grad():
                image_features = _model_clip.encode_image(image_clip)
                text_features = _model_clip.encode_text(text_inputs)
            image_features /= image_features.norm(dim=-1, keepdim=True)
            text_features /= text_features.norm(dim=-1, keepdim=True)
            similarity = (image_features @ text_features.T).squeeze(0)
            top_sim_values, top_sim_idxs = similarity.topk(topk)
            clip_tags = [all_tags[int(i)] for i in top_sim_idxs.tolist()]
    except Exception:
        clip_tags = []

    # If we have no tags from either, fallback
    if not places_tags and not clip_tags:
        return ["photo"]

    fused = _weighted_scene_fusion(places_tags or [], clip_tags or [], weight_clip=0.55)
    # deduplicate preserve order
    seen = set()
    out = []
    for t in fused:
        if t not in seen:
            seen.add(t)
            out.append(t)
        if len(out) >= topk:
            break
    # Map to custom category for single-label decisions if needed (not used here)
    return out[:topk]

# ----------------- File helpers -----------------
def _load_existing_metadata(photo_dir: Path) -> Dict[str, Any]:
    """
    Prefer organized_output/metadata.json (produced by main.py).
    If missing, try uploaded_photos/metadata.json, otherwise build minimal metadata
    by listing image filenames.
    """
    # prefer METADATA_FILE_PREF if it exists
    if METADATA_FILE_PREF.exists():
        try:
            with open(METADATA_FILE_PREF, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            pass

    # fallback: check photo_dir / "metadata.json"
    m2 = photo_dir / "metadata.json"
    if m2.exists():
        try:
            with open(m2, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            pass

    # final fallback: build minimal metadata from files found in photo_dir
    meta = {}
    for p in sorted(photo_dir.iterdir()):
        if not p.is_file() or p.suffix.lower() not in (".jpg", ".jpeg", ".png"):
            continue
        meta[p.name] = {
            "datetime": None,
            "exif": {},
            "lat": None,
            "lon": None,
        }
    return meta

# ----------------- Core function exported to main.py -----------------
def add_scene_tags(metadata: Dict[str, Any], photo_dir: str | Path, topk: int = 5, save_path: str | Path = None) -> Dict[str, Any]:
    """
    Merge scene tags into an existing metadata dict.
    - metadata: dict keyed by filename (may be dict or str). This function will ensure each entry is a dict
                and will set/update entry['tags'] = [...tags...].
    - photo_dir: folder where images live (used to locate files)
    - topk: number of tags per image
    - save_path: if provided, write merged metadata JSON to this path. If None, defaults to organized_output/metadata_with_tags.json
    Returns the updated metadata dict.
    """
    photo_dir = Path(photo_dir)
    if save_path is None:
        save_path = DEFAULT_OUTPUT_FILE
    else:
        save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)

    # normalize metadata dict: ensure per-file dict
    for k, v in list(metadata.items()):
        if isinstance(v, dict):
            v.setdefault("tags", [])
        else:
            metadata[k] = {"datetime": v, "tags": []}

    # For filenames in metadata, compute tags
    for fname in sorted(metadata.keys()):
        try:
            img_path = photo_dir / fname
            if not img_path.exists():
                # try case-insensitive match in folder
                found = None
                for p in photo_dir.iterdir():
                    if p.is_file() and p.name.lower() == fname.lower():
                        found = p
                        break
                if found:
                    img_path = found
                else:
                    # file not found; leave tags as-is (empty or existing)
                    continue

            tags = combined_scene_tagging(str(img_path), topk=topk)
            # ensure serializable list
            tags = [str(t) for t in tags]
            entry = metadata.get(fname)
            if entry is None:
                metadata[fname] = {"datetime": None, "tags": tags}
            else:
                if not isinstance(entry, dict):
                    metadata[fname] = {"datetime": entry, "tags": tags}
                else:
                    entry["tags"] = tags

        except Exception:
            # don't fail on single-file errors; print and continue
            print(f"⚠️ Tagging failed for {fname}:", file=sys.stderr)
            traceback.print_exc()

    # persist merged metadata
    try:
        with open(save_path, "w", encoding="utf-8") as f:
            json.dump(metadata, f, indent=2, ensure_ascii=False)
    except Exception:
        print(f"⚠️ Failed to write tags file to {save_path}", file=sys.stderr)
        traceback.print_exc()

    return metadata

# ----------------- CLI / standalone behavior -----------------
def _cli_main():
    """
    Usage:
      python scene_tags.py [uploaded_photos_folder] [organized_output_folder]
    If folders are omitted it uses configured UPLOAD_DIR and OUTPUT_DIR.
    """
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("photo_dir", nargs="?", default=str(UPLOAD_DIR), help="Folder with uploaded photos")
    parser.add_argument("--out", "-o", default=str(OUTPUT_DIR), help="organized_output folder where metadata.json lives and metadata_with_tags.json will be written")
    parser.add_argument("--topk", type=int, default=5)
    args = parser.parse_args()

    photo_dir = Path(args.photo_dir)
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    save_path = out_dir / "metadata_with_tags.json"

    # Try to load base metadata produced by main.py; otherwise create minimal metadata
    metadata = _load_existing_metadata(photo_dir)
    print(f"Found {len(metadata)} entries in base metadata. Running tagging (topk={args.topk})...")

    metadata = add_scene_tags(metadata, photo_dir, topk=args.topk, save_path=save_path)
    print(f"\n✅ Tags saved to {save_path}")

if __name__ == "__main__":
    _cli_main()
