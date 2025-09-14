---

# 📸 Smart Photo Organizer

An AI-powered photo organizer that automatically analyzes your images, tags them with meaningful scene categories (e.g., *beach trip*, *birthday*, *museum*), and organizes them into structured folders. Built with **PyTorch**, **CLIP**, **Places365**, and a **Streamlit UI**.

---

## 🚀 Features

* **Scene Detection**: Uses Places365 (ResNet18 trained on 365 scene classes) to detect environments like *mountain*, *forest*, *library*.
* **CLIP Refinement**: Combines CLIP’s text-image similarity with Places365 predictions for richer tagging.
* **Custom Categories**: Maps raw predictions into user-friendly tags (e.g., `museum/indoor → museum`).
* **Weighted Fusion**: Blends Places365 and CLIP outputs with adjustable weighting for better accuracy.
* **Metadata Integration**: Adds scene tags into photo metadata JSON.
* **Automatic Organization**: Sorts photos into structured folders (e.g., `/organized_output/travel/beach_trip/`).
* **Streamlit UI**: Browse, search, and manage organized photo collections with an interactive interface.

---

## 🛠️ Tech Stack

* **Python 3.10+**
* **PyTorch** (deep learning)
* **CLIP** (OpenAI vision-language model)
* **Places365 ResNet18** (scene classification)
* **Torchvision** (image preprocessing)
* **RapidFuzz** (fuzzy string matching for tags)
* **Pillow** (image loading)
* **Streamlit** (web interface)

---

## 📂 Project Structure

```
smart-photo-organizer/
├── app/
│   ├── ui.py                   # Streamlit app
│   ├── main.py
│   └──  utils/
│       ├── categories_places365.txt
│       ├── geocode.py
│       ├── metadata.py
│       ├── scene_tags.py
│       └── zip_utils.py 
│
│   └──  expermental/
│       └── face-recognition.py
│
│   └──  detectors/
│       ├── face_detection.py
│       └── object_detection.py
│
├── models/
│   ├── yolov8n.pt
│   └── wideresnet18_places365.pth.tar   # Pretrained Places365 weights
│
├── uploaded_photos/            # Input folder (user photos)
├── organized_output/           # Output folder (tagged + organized photos)
├── pyproject.toml
├── uv.lock
├── requirements.txt
└── README.md
```
## ⚡ Installation

1. **Clone this repo**

```bash
git clone https://github.com/yourusername/smart-photo-organizer.git
cd smart-photo-organizer
```

2. **Create a virtual environment**

```bash
python -m venv .venv
source .venv/bin/activate  # macOS/Linux
.venv\Scripts\activate     # Windows
```

3. **Install dependencies**

```bash
pip install -r requirements.txt
```

4. **Download Places365 model**

* Already included (`models/wideresnet18_places365.pth.tar`).
* Or download from [MIT Places365](http://places2.csail.mit.edu/download.html).

---

## ▶️ Usage

### 1. Run Auto-Tagging Script

```bash
python app/scene_tags.py
```

This generates `metadata_with_tags.json` for all images in `uploaded_photos/`.

### 2. Organize Photos with Metadata

```bash
python app/main.py
```

Organized results will be stored in `organized_output/`.

### 3. Start Streamlit App

```bash
streamlit run app/ui.py
```

Open [http://localhost:8501](http://localhost:8501) in your browser to explore your organized collection.

---

## 🧩 Example Workflow

1. Drop your photos into `/uploaded_photos`.
2. Run `scene_tags.py` → tags are generated (`beach trip`, `birthday`, `wedding`, etc.).
3. Run `main.py` → photos organized into subfolders by tag/date.
4. Launch Streamlit (`ui.py`) → browse and search your collection interactively.

---

## 🔧 Configuration

* **Adjust Tagging Weighting**: In `scene_tags.py`, modify:

```python
fused_tags = weighted_scene_fusion(mapped_scenes, clip_tags, weight_clip=0.55)
```

* **Custom Categories**: Edit the `custom_categories` list or `places_to_custom` mapping in `scene_tags.py` to define your own photo groupings.

---

## 📌 Notes

* If you get `JSONDecodeError` when loading metadata, it’s likely due to control characters in EXIF data. Run the sanitizer in `metadata.py` to clean metadata before saving.
* GPU acceleration will be used automatically if available (`torch.cuda.is_available()`).

---

## 🤝 Contributing

Pull requests are welcome! For major changes, please open an issue first to discuss what you’d like to change.

---

## 📜 License

MIT License. Free to use and modify.

---
