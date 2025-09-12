# scene_tags.py (fixed)
import json
from pathlib import Path
import torch
import clip
import os
from typing import Dict, Any, Tuple, List
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
from torchvision.transforms import Compose, Resize, CenterCrop, ToTensor, Normalize, ColorJitter
from PIL import Image
from rapidfuzz import process

MODEL_PATH = Path(os.getenv("MODEL_PATH", "data/models/wideresnet18_places365.pth.tar"))

# Load places365 categories
_cat_file = Path("/Applications/portfolio-projects/smart-photo-organizer/app/utils")
if not _cat_file.exists():
    raise FileNotFoundError(f"categories_places365.txt not found at {_cat_file}")
with open(_cat_file, "r", encoding="utf-8") as f:
    places365_classes = [line.strip().split(' ')[0][3:] for line in f.readlines()]

# Load Places365 model (CPU)
model_places = models.resnet18(num_classes=365)
checkpoint = torch.load(MODEL_PATH, map_location="cpu")
state_dict = {k.replace("module.", ""): v for k, v in checkpoint["state_dict"].items()}
model_places.load_state_dict(state_dict)
model_places.eval()

# Image transform pipeline for Places365
places365_transform = Compose([
    Resize(256),
    CenterCrop(224),
    ColorJitter(brightness=0.2, contrast=0.2),
    ToTensor(),
    Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

def transform_image_places365(image_path: str):
    image = Image.open(image_path).convert("RGB")
    return places365_transform(image).unsqueeze(0)  # CPU tensor

def predict_places365(image_tensor, topk=5):
    with torch.no_grad():
        output = model_places(image_tensor)
        probs = F.softmax(output, dim=1)
        top_probs, top_idxs = probs.topk(topk)
    return top_probs.squeeze(), top_idxs.squeeze()

# CLIP (on GPU if available)
device = "cuda" if torch.cuda.is_available() else "cpu"
model_clip, preprocess_clip = clip.load("ViT-B/32", device=device)

def transform_image_clip(image_path: str):
    image = Image.open(image_path).convert("RGB")
    return preprocess_clip(image).unsqueeze(0).to(device)

def weighted_scene_fusion(places_tags: List[str], clip_tags: List[str], weight_clip: float = 0.55) -> List[str]:
    score_map = {}
    # higher rank (lower index) should produce higher score
    for i, tag in enumerate(places_tags):
        score_map[tag] = score_map.get(tag, 0.0) + (1 - weight_clip) * (len(places_tags) - i)
    for i, tag in enumerate(clip_tags):
        score_map[tag] = score_map.get(tag, 0.0) + weight_clip * (len(clip_tags) - i)
    # sort keys by score desc
    return sorted(score_map.keys(), key=lambda k: score_map[k], reverse=True)

custom_categories = [
    "person", "people", "human", "portrait", "crowd",
    "selfie", "family", "friends", "birthday", "trip",
    "museum", "art-gallery", "vacation", "travel", "holiday",
    "hiking", "art museum", "gallery", "exhibition", "beach",
    "desert", "forest", "party", "wedding", "road trip",
    "camping", "mountains", "lake"
]
# lowercase helper set for matching
_custom_categories_lower = [c.lower() for c in custom_categories]

places_to_custom = {
    "art gallery": "museum",
    "museum/indoor": "museum",
    "exhibition room": "museum",
    "library/indoor": "museum",
    "beach": "beach trip",
    "mountain": "hiking",
    "valley": "hiking",
    "tent/outdoor": "camping",
    "forest/path": "hiking",
    "desert/road": "road trip"
}
# lower keys for robust mapping
_places_to_custom_lower = {k.lower(): v for k, v in places_to_custom.items()}

def map_to_custom_category(tags: List[str]) -> str:
    """
    Return best-match custom category string for the provided tags list.
    If none match confidently, return empty string.
    """
    if not tags:
        return ""
    best_score = 0
    best_match = ""
    for tag in tags:
        if not tag:
            continue
        # use rapidfuzz against our custom list
        match, score, _ = process.extractOne(tag, custom_categories)
        if score > best_score:
            best_score = score
            best_match = match
    return best_match if best_score >= 50 else ""  # threshold

def combined_scene_tagging(image_path: str, topk: int = 5) -> Tuple[List[str], str]:
    """
    Returns: (tags_list, final_label)
      - tags_list: ranked fused tags (topk)
      - final_label: mapped custom category (string)
    """
    try:
        # Places365 (CPU)
        img_tensor = transform_image_places365(image_path)
        _, idxs = predict_places365(img_tensor, topk=topk)
        # map indices -> scene names (safe indexing)
        if isinstance(idxs, torch.Tensor):
            idx_list = idxs.tolist()
        else:
            idx_list = list(idxs)
        predicted_scenes = []
        for idx in idx_list:
            try:
                predicted_scenes.append(places365_classes[int(idx)])
            except Exception:
                continue
        # normalize
        predicted_scenes = [s.lower() for s in predicted_scenes if isinstance(s, str)]

        # map to custom where we have straightforward mapping
        mapped_scenes = [_places_to_custom_lower.get(s, s) for s in predicted_scenes]

        # CLIP refinement
        all_tags = list(dict.fromkeys(mapped_scenes + _custom_categories_lower))  # preserve order, unique
        # prepare textual prompts
        text_prompts = [f"a photo of {t}" for t in all_tags]
        text_inputs = clip.tokenize(text_prompts).to(device)
        image_clip = transform_image_clip(image_path)

        with torch.no_grad():
            image_features = model_clip.encode_image(image_clip)
            text_features = model_clip.encode_text(text_inputs)

            image_features /= image_features.norm(dim=-1, keepdim=True)
            text_features /= text_features.norm(dim=-1, keepdim=True)

            similarity = (image_features @ text_features.T).squeeze(0)
            top_sim_values, top_sim_idxs = similarity.topk(min(topk, len(all_tags)))
            clip_tags = [all_tags[i] for i in top_sim_idxs.tolist()]

        # fuse
        fused = weighted_scene_fusion(mapped_scenes, clip_tags, weight_clip=0.55)
        fused_topk = fused[:topk]

        # final mapped label (map fused tags to our friendly custom category)
        final_label = map_to_custom_category(fused_topk)
        return fused_topk, final_label
    except Exception as e:
        print(f"⚠️ combined_scene_tagging failed for {image_path}: {e}")
        return [], ""

def add_scene_tags(metadata: Dict[str, Any], folder_path: str | Path) -> Dict[str, Any]:
    """
    Iterate over metadata keys and write:
      - metadata[image]['tags'] = [tag1, tag2, ...]
      - metadata[image]['label'] = final_label (single string)
    """
    folder_path = Path(folder_path)
    if not folder_path.exists():
        print(f"⚠️ add_scene_tags: folder_path does not exist: {folder_path}")
        # still ensure metadata has tags keys
        for k in metadata.keys():
            v = metadata.get(k)
            if not isinstance(v, dict):
                metadata[k] = {"datetime": v}
            metadata[k].setdefault("tags", [])
            metadata[k].setdefault("label", "")
        return metadata

    for image_name in list(metadata.keys()):
        image_path = folder_path / image_name
        # ensure dict
        if not isinstance(metadata.get(image_name), dict):
            metadata[image_name] = {"datetime": metadata.get(image_name)}
        try:
            if not image_path.exists():
                # no image file, set empty tags
                metadata[image_name]["tags"] = []
                metadata[image_name]["label"] = ""
                print(f"⚠️ image not found, skipping: {image_name}")
                continue

            tags, label = combined_scene_tagging(str(image_path), topk=5)
            metadata[image_name]["tags"] = tags
            metadata[image_name]["label"] = label
            print(f"✅ Tagged {image_name}: label={label}, tags={tags}")
        except Exception as e:
            print(f"⚠️ Failed to tag {image_name}: {e}")
            metadata[image_name].setdefault("tags", [])
            metadata[image_name].setdefault("label", "")

    return metadata

# CLI usage for standalone testing
if __name__ == "__main__":
    folder = Path(os.getenv("/Applications/portfolio-projects/smart-photo-organizer/organized_output/Sabzevar,_Iran_2023-07-26_1/IMG_6832"))
    # output_file = folder / "metadata_with_tags.json"
    tags_data = {}
    print(tags_data)

    # if not folder.exists():
    #     print(f"Folder not found: {folder}")
    # else:
    #     for image_path in folder.glob("*.[jp][pn]g"):
    #         try:
    #             t, lbl = combined_scene_tagging(str(image_path), topk=5)
    #             tags_data[image_path.name] = {"tags": t, "label": lbl}
    #         except Exception as e:
    #             print(f"Error {image_path}: {e}")
    #             tags_data[image_path.name] = {"tags": [], "label": ""}

    #     with open(output_file, "w", encoding="utf-8") as f:
    #         json.dump(tags_data, f, indent=4, ensure_ascii=False)
    #     print(f"Saved tags to {output_file}")
