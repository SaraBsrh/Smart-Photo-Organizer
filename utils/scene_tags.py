import json
from pathlib import Path
from datetime import datetime
import torch
import clip
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
from torchvision.models.resnet import ResNet, BasicBlock
from torchvision.transforms import Compose, Resize, CenterCrop, ToTensor, Normalize, ColorJitter
from PIL import Image
from rapidfuzz import process
from persiantools.jdatetime import JalaliDate

MODEL_PATH = "/Applications/portfolio-projects/smart-photo-organizer/data/models/wideresnet18_places365.pth.tar"

# Load places365 categories
with open('utils/categories_places365.txt') as f:
    places365_classes = [line.strip().split(' ')[0][3:] for line in f.readlines()]

# Load Places365 model
model_places = models.resnet18(num_classes=365)
checkpoint = torch.load(MODEL_PATH, map_location='cpu')
state_dict = {k.replace("module.", ""): v for k, v in checkpoint["state_dict"].items()}
model_places.load_state_dict(state_dict)
model_places.eval()

# Image transform pipeline for Places365
places365_transform = Compose([
    Resize(256),
    CenterCrop(224),  # focus on center
    ColorJitter(brightness=0.2, contrast=0.2),  # light enhancement
    ToTensor(),
    Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

def transform_image_places365(image_path):
    image = Image.open(image_path).convert('RGB')
    return places365_transform(image).unsqueeze(0)  # add batch dim

def predict_places365(image_tensor, topk=5):
    with torch.no_grad():
        output = model_places(image_tensor)
        probs = F.softmax(output, dim=1)
        top_probs, top_idxs = probs.topk(topk)
    return top_probs.squeeze(), top_idxs.squeeze()

device = "cuda" if torch.cuda.is_available() else "cpu"
model_clip, preprocess_clip = clip.load("ViT-B/32", device=device)

def transform_image_clip(image_path):
    image = Image.open(image_path).convert("RGB")
    return preprocess_clip(image).unsqueeze(0).to(device)

def weighted_scene_fusion(places_tags, clip_tags, weight_clip=0.55):
    score_map = {}
    for i, tag in enumerate(places_tags):
        score_map[tag] = score_map.get(tag, 0) + (1 - weight_clip) * (len(places_tags) - i)
    for i, tag in enumerate(clip_tags):
        score_map[tag] = score_map.get(tag, 0) + weight_clip * (len(clip_tags) - i)
    return sorted(score_map, key=score_map.get, reverse=True)

custom_categories = [
        "person", "people", "human", "portrait", "crowd", 
        "selfie", "family", "friends", "birthday", "trip", 
        "museum", "art-gallery", "trip", "vacation", "travel",
        "holiday", "hiking", "museum", "art museum", "gallery",
        "exhibition", "beach", "desert", "forest", "party", "birthday",
        "wedding", "road trip", "camping", "mountains", "lake"
]

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

def map_to_custom_category(tags):
    """Match each tag individually to custom categories, pick the best match."""
    best_score = 0
    best_match = tags[0]
    for tag in tags:
        match, score, _ = process.extractOne(tag, custom_categories)
        if score > best_score:
            best_score = score
            best_match = match
    return best_match

def combined_scene_tagging(image_path, topk=5):
    # Step 1: Get Places365 predictions
    image_tensor = transform_image_places365(image_path)
    probs, idxs = predict_places365(image_tensor, topk=topk)

    # Map indices to scene names
    predicted_scenes = [places365_classes[idx] for idx in idxs.tolist()]

    # Apply Places→Custom mapping where possible
    mapped_scenes = [places_to_custom.get(scene, scene) for scene in predicted_scenes]
    print(f"Places365 raw scenes: {predicted_scenes}")
    print(f"Mapped scenes: {mapped_scenes}")

    # Step 2: Refine tags with CLIP using context-rich prompts
    all_tags = list(set(mapped_scenes + custom_categories))
    text_inputs = clip.tokenize([f"a photo of {tag}" for tag in all_tags]).to(device)
    image_clip = transform_image_clip(image_path)

    with torch.no_grad():
        image_features = model_clip.encode_image(image_clip)
        text_features = model_clip.encode_text(text_inputs)

    image_features /= image_features.norm(dim=-1, keepdim=True)
    text_features /= text_features.norm(dim=-1, keepdim=True)

    similarity = (image_features @ text_features.T).squeeze(0)
    top_sim_values, top_sim_idxs = similarity.topk(topk)
    clip_tags = [all_tags[i] for i in top_sim_idxs.tolist()]

    print(f"CLIP top-{topk} tags: {clip_tags}")

    # Step 3: Weighted fusion of Places365 and CLIP tags
    fused_tags = weighted_scene_fusion(mapped_scenes, clip_tags, weight_clip=0.55)
    print(f"Fused tags: {fused_tags[:topk]}")

    # Step 4: Final best category
    final_category = map_to_custom_category(fused_tags[:topk])
    print(f"Final mapped category: {final_category}")

    return fused_tags[:topk]

if __name__ == "__main__":
    folder = Path("/Users/sara/Desktop/backed-data/sample-data")
    output_file = folder / "metadata_with_tags.json"

    tags_data = {}

    for image_path in folder.glob("*.[jp][pn]g"):  # jpg or png
        print(f"Processing {image_path.name}...")
        try:
            tags = combined_scene_tagging(str(image_path), topk=5)
            tags_data[image_path.name] = tags
        except Exception as e:
            print(f"Error processing {image_path.name}: {e}")

    # Save to JSON
    with open(output_file, "w") as f:
        json.dump(tags_data, f, indent=4)

    print(f"\n✅ Tags saved to {output_file}")