# ui.py
import streamlit as st
from pathlib import Path
import tempfile
import os
import shutil
import json
from PIL import Image, ExifTags
import io
import time
import zipfile
from datetime import datetime

# Import your main pipeline (adjust import if main.py is not in PYTHONPATH)
from main import organize_photos

# ---------------- Config / Defaults ----------------
SAMPLE_PHOTO_DIR = Path("/Users/sara/Desktop/backed-data/sample-data")  # your existing dataset
DEFAULT_OUTPUT_DIR = Path("/Users/sara/Desktop/backed-data/organized")  # where organized results will go

# filenames that main.organize_photos expects (used for uploads fallback)
METADATA_FILES = ["metadata.json", "metadata_with_location.json", "metadata_with_tags.json"]

# ---------------- Helpers ----------------
def get_exif_datetime(path: Path):
    """Return EXIF DateTimeOriginal string (e.g. '2024:06:03 21:46:06') if present, else None."""
    try:
        img = Image.open(path)
        exif = img._getexif()
        if not exif:
            return None
        for tag, val in exif.items():
            name = ExifTags.TAGS.get(tag)
            if name in ("DateTimeOriginal", "DateTime"):
                return val
    except Exception:
        return None
    return None

def find_script_in_project(name: str):
    """Search project tree for a script filename and return its Path or None."""
    cwd = Path.cwd()
    for p in cwd.glob("**/" + name):
        # skip virtualenv and .git etc
        if "site-packages" in str(p) or ".venv" in str(p):
            continue
        return p
    return None

def copy_scripts_to_folder(target_dir: Path, script_names):
    """Try to copy scripts (object-detection.py, face-detection.py) into target_dir so main can use them."""
    info = {}
    for name in script_names:
        src = find_script_in_project(name)
        if src:
            dst = target_dir / name
            shutil.copy(src, dst)
            info[name] = {"found": True, "src": str(src), "dst": str(dst)}
        else:
            info[name] = {"found": False}
    return info

def render_image_grid(image_paths, cols=5, thumb_w=200):
    """Display images in a responsive grid in Streamlit."""
    if not image_paths:
        st.info("No images to display.")
        return
    columns = st.columns(cols)
    for i, img_path in enumerate(image_paths):
        col = columns[i % cols]
        try:
            with Image.open(img_path) as im:
                col.image(im, use_column_width=True)
        except Exception:
            col.write(f"Failed to open {img_path}")

def zip_dir_and_get_bytes(folder: Path):
    """Create a zip in memory and return bytes."""
    mem_zip = io.BytesIO()
    with zipfile.ZipFile(mem_zip, mode="w", compression=zipfile.ZIP_DEFLATED) as zf:
        for root, _, files in os.walk(folder):
            for file in files:
                file_path = Path(root) / file
                # maintain relative paths inside zip
                zf.write(file_path, arcname=str(file_path.relative_to(folder)))
    mem_zip.seek(0)
    return mem_zip.read()

# ---------------- Streamlit UI ----------------
st.set_page_config(page_title="Smart Photo Organizer", layout="wide")
st.title("📸 Smart Photo Organizer")

st.sidebar.header("Navigation")
page = st.sidebar.radio("Go to", ["Welcome", "Upload & Organize", "Search Organized"])

# --- Welcome ---
if page == "Welcome":
    st.header("What this app does")
    st.markdown(
        """
        **Smart Photo Organizer** groups your photos by **date**, **location**, and **situation** (tags).
        It can also detect **objects** and **faces** and save metadata for searching.

        Workflow:
        1. Choose to use the **sample dataset** (already on disk) or **upload photos**.
        2. The app will run object & face detection (if not present) and group photos into folders.
        3. Use the _Search Organized_ tab to find photos by object, tag, location or date.
        """
    )
    st.markdown("**Tips:** Upload a batch of photos (JPEG/PNG). If you upload photos without metadata files, the app will extract EXIF `DateTimeOriginal` where available and run detections for you.")

# --- Upload & Organize ---
elif page == "Upload & Organize":
    st.header("Upload photos or use existing dataset")

    mode = st.radio("Source", ["Use sample dataset (on disk)", "Upload photos (temporary)"])

    run_detection = st.checkbox("Run object & face detection (if needed)", value=True)
    force_detection = st.checkbox("Force re-run detection even if outputs exist", value=False)

    if mode == "Use sample dataset (on disk)":
        st.write(f"Using sample folder: `{SAMPLE_PHOTO_DIR}`")
        if not SAMPLE_PHOTO_DIR.exists():
            st.error(f"Sample folder not found: {SAMPLE_PHOTO_DIR}")
        else:
            if st.button("Organize sample dataset"):
                out_dir = DEFAULT_OUTPUT_DIR
                st.info("Running organization pipeline on sample dataset...")
                with st.spinner("Organizing photos..."):
                    organized = organize_photos(photo_dir=SAMPLE_PHOTO_DIR, output_dir=out_dir, run_detection=run_detection, force_detection=force_detection)
                st.success("Organization complete!")
                st.write(f"Output saved to: `{out_dir}`")
                st.experimental_set_query_params(last_output=str(out_dir))
                # show top-level folders
                st.subheader("Folders created")
                for folder in sorted(out_dir.iterdir()):
                    if folder.is_dir():
                        st.markdown(f"- **{folder.name}** ({len(list(folder.glob('*')))} files)")
                st.markdown("---")
                if (out_dir / "organized_metadata.json").exists():
                    st.download_button("Download organized_metadata.json", data=open(out_dir / "organized_metadata.json","rb"), file_name="organized_metadata.json")
                if st.button("Download output as ZIP"):
                    st.info("Creating ZIP - this may take a moment...")
                    zip_bytes = zip_dir_and_get_bytes(out_dir)
                    st.download_button("Download ZIP", data=zip_bytes, file_name=f"organized_output_{int(time.time())}.zip")

    else:  # Upload photos
        uploaded_files = st.file_uploader("Upload multiple photos", accept_multiple_files=True, type=["jpg","jpeg","png"])
        if uploaded_files:
            temp_dir = Path(tempfile.mkdtemp(prefix="spo_upload_"))
            st.write(f"Saved uploads to temporary folder: `{temp_dir}`")

            # save files
            file_paths = []
            for uf in uploaded_files:
                save_path = temp_dir / uf.name
                with open(save_path, "wb") as f:
                    f.write(uf.getbuffer())
                file_paths.append(save_path)

            # Build minimal metadata.json if not provided
            metadata_json_path = temp_dir / "metadata.json"
            if not any((temp_dir / m).exists() for m in METADATA_FILES):
                st.info("No metadata files found in upload; creating minimal metadata.json using EXIF where available.")
                meta = {}
                for p in file_paths:
                    dt = get_exif_datetime(p)
                    meta[p.name] = dt or ""
                with open(metadata_json_path, "w") as f:
                    json.dump(meta, f, indent=2)
            else:
                st.info("Found metadata files in upload; using them.")

            # Ensure object & face detection scripts exist in temp_dir so main.organize_photos can run them
            st.info("Ensuring detection scripts are available...")
            copy_info = copy_scripts_to_folder(temp_dir, ["object-detection.py", "face-detection.py"])
            for k, v in copy_info.items():
                if not v.get("found"):
                    st.warning(f"Script `{k}` not found in project; main() may skip related detection or expect scripts in the folder.")
                else:
                    st.success(f"Copied {k} from {v['src']} to upload folder.")

            # Run organize_photos on temp_dir
            if st.button("Organize uploaded photos"):
                out_dir = Path(tempfile.mkdtemp(prefix="spo_output_"))
                st.info("Running organization pipeline on uploaded photos...")
                with st.spinner("Organizing uploaded photos... this may take some time depending on number of images"):
                    organized = organize_photos(photo_dir=temp_dir, output_dir=out_dir, run_detection=run_detection, force_detection=force_detection)
                st.success("Organization complete!")
                st.write(f"Output saved to: `{out_dir}`")
                st.subheader("Folders created")
                for folder in sorted(out_dir.iterdir()):
                    if folder.is_dir():
                        st.markdown(f"- **{folder.name}** ({len(list(folder.glob('*')))} files)")
                if (out_dir / "organized_metadata.json").exists():
                    st.download_button("Download organized_metadata.json", data=open(out_dir / "organized_metadata.json","rb"), file_name="organized_metadata.json")
                if st.button("Download output as ZIP (uploaded)"):
                    st.info("Creating ZIP...")
                    zip_bytes = zip_dir_and_get_bytes(out_dir)
                    st.download_button("Download ZIP", data=zip_bytes, file_name=f"organized_uploaded_output_{int(time.time())}.zip")

# --- Search Organized ---
elif page == "Search Organized":
    st.header("Search photos by object / tag / location / date")

    # Let user choose organized folder (recent outputs)
    st.write("Select the organized folder (where `organized_metadata.json` is present).")
    candidate_dirs = []
    # search common output locations
    search_roots = [DEFAULT_OUTPUT_DIR, Path.cwd()]
    for root in search_roots:
        if root.exists():
            for p in root.glob("**/organized_metadata.json"):
                candidate_dirs.append(p.parent)
    # unique
    candidate_dirs = list({str(p): p for p in candidate_dirs}.values())

    if not candidate_dirs:
        st.warning("No organized outputs found on disk. Run 'Upload & Organize' first.")
    else:
        selected = st.selectbox("Choose organized output folder", options=[str(p) for p in candidate_dirs])
        sel_path = Path(selected)
        meta_file = sel_path / "organized_metadata.json"
        if not meta_file.exists():
            st.error("organized_metadata.json not found in the selected folder.")
        else:
            with open(meta_file, "r") as f:
                organized = json.load(f)

            # Collect filters
            # objects, tags, locations, dates
            all_objects = set()
            all_tags = set()
            all_locations = set()
            all_dates = set()

            # organized is folder_name -> list of photo entries (as created by main.organize_photos)
            photos_index = []  # flat list of entries
            for folder, items in organized.items():
                for entry in items:
                    photos_index.append(entry)
                    # objects may be list of dicts or list of strings depending on object script
                    objs = entry.get("objects", [])
                    if isinstance(objs, list):
                        for o in objs:
                            if isinstance(o, dict):
                                all_objects.add(o.get("label") or str(o))
                            else:
                                all_objects.add(str(o))
                    # tags
                    t = entry.get("tag", [])
                    if isinstance(t, list):
                        for x in t:
                            all_tags.add(x)
                    else:
                        if t:
                            all_tags.add(str(t))
                    # location & date
                    loc = entry.get("location")
                    if loc:
                        all_locations.add(loc)
                    dt = entry.get("date")
                    if dt:
                        all_dates.add(dt)

            # Filter widgets
            chosen_objects = st.multiselect("Objects", sorted([o for o in all_objects if o]))
            chosen_tags = st.multiselect("Tags / Situations", sorted([t for t in all_tags if t]))
            chosen_locations = st.multiselect("Locations", sorted([l for l in all_locations if l]))
            chosen_dates = st.multiselect("Dates", sorted([d for d in all_dates if d]))

            # Search
            def entry_matches(entry):
                if chosen_objects:
                    objs = entry.get("objects", [])
                    labels = []
                    for o in objs:
                        if isinstance(o, dict):
                            labels.append(o.get("label"))
                        else:
                            labels.append(str(o))
                    if not any(obj in labels for obj in chosen_objects):
                        return False
                if chosen_tags:
                    tagv = entry.get("tag", [])
                    tag_list = tagv if isinstance(tagv, list) else [tagv]
                    if not any(t in tag_list for t in chosen_tags):
                        return False
                if chosen_locations:
                    if entry.get("location") not in chosen_locations:
                        return False
                if chosen_dates:
                    if entry.get("date") not in chosen_dates:
                        return False
                return True

            matches = [e for e in photos_index if entry_matches(e)]
            st.markdown(f"### 🔎 Found {len(matches)} matching photos")
            # show thumbnails
            if matches:
                cols = st.columns(5)
                for i, entry in enumerate(matches):
                    try:
                        with Image.open(entry["copied_path"]) as im:
                            cols[i % 5].image(im, use_column_width=True, caption=Path(entry["copied_path"]).name)
                    except Exception:
                        cols[i % 5].write("Unable to open image.")

            # Offer CSV/JSON export of search results
            if matches:
                if st.button("Export search results (JSON)"):
                    js = json.dumps(matches, indent=2)
                    st.download_button("Download JSON", data=js, file_name="search_results.json", mime="application/json")
