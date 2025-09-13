import streamlit as st
from pathlib import Path
from PIL import Image
import json
import random
from main import organize_photos

st.set_page_config(page_title="Photo Organizer", layout="wide")

PAGES = ["Welcome", "Upload & Organize", "Search Organized"]

page = st.sidebar.radio("Navigate", PAGES)

# -------------------------
#  WELCOME PAGE
# -------------------------
if page == "Welcome":
    st.title("📸 Smart Photo Organizer")
    st.markdown(
        """
        Welcome to **Smart Photo Organizer**! 🎉  

        With this app, you can:  
        - Upload your photos  
        - Automatically organize them by **date, location, and scene tags**  
        - Search your collection by **objects, tags, locations, or dates**  

        👉 Use the sidebar to get started!
        """
    )

# -------------------------
#  UPLOAD & ORGANIZE PAGE
# -------------------------
elif page == "Upload & Organize":
    st.header("Upload and Organize Photos")

    uploaded_files = st.file_uploader(
        "Upload photos (jpg, png)", type=["jpg", "jpeg", "png"], accept_multiple_files=True
    )

    run_detection = st.checkbox("Run object/face detection", value=True)
    force_detection = st.checkbox("Force re-run detection", value=False)

    if uploaded_files:
        tmp_dir = Path("uploaded_photos")
        tmp_dir.mkdir(exist_ok=True)

        for f in uploaded_files:
            out_path = tmp_dir / f.name
            with open(out_path, "wb") as out_f:
                out_f.write(f.getbuffer())
        st.success(f"Uploaded {len(uploaded_files)} files to {tmp_dir}")

        if st.button("Organize now"):
            output_dir = Path("organized_output")
            output_dir.mkdir(exist_ok=True)

            organize_photos(
                str(tmp_dir), str(output_dir),
                run_detection=run_detection, force_detection=force_detection
            )

            # Save output dir in session state
            st.session_state["last_output_dir"] = str(output_dir)

            meta_file = output_dir / "organized_metadata.json"
            if meta_file.exists():
                st.success(f"✅ Photos organized into {output_dir}")
                st.info(f"Metadata saved to {meta_file}")

                # -------------------------
                # Show Folders + 5 random photos
                # -------------------------
                st.subheader("📂 Organized Folders Preview")
                for folder in sorted(output_dir.iterdir()):
                    if folder.is_dir():
                        st.markdown(f"### 📁 {folder.name}")

                        images = list(folder.glob("*.[jp][pn]g"))
                        if images:
                            sample_imgs = random.sample(images, min(5, len(images)))
                            cols = st.columns(len(sample_imgs))
                            for i, img_path in enumerate(sample_imgs):
                                try:
                                    with Image.open(img_path) as im:
                                        cols[i].image(im, caption=img_path.name, width=150)
                                except Exception:
                                    cols[i].write("Image not previewable")
                        else:
                            st.write("_No images in this folder_")
            else:
                st.error("❌ Organization finished but metadata file not found!")

# -------------------------
#  SEARCH ORGANIZED PAGE
# -------------------------
elif page == "Search Organized":
    st.header("Search photos by object / tag / location / date")

    candidate_dirs = []
    if "last_output_dir" in st.session_state:
        last_dir = Path(st.session_state["last_output_dir"])
        if (last_dir / "organized_metadata.json").exists():
            candidate_dirs.append(last_dir)

    for p in Path.cwd().rglob("organized_metadata.json"):
        candidate_dirs.append(p.parent)

    candidate_dirs = list({str(p): p for p in candidate_dirs}.values())

    if not candidate_dirs:
        st.warning("No organized outputs found. Run 'Upload & Organize' first.")
    else:
        selected = st.selectbox(
            "Choose organized output folder", options=[str(p) for p in candidate_dirs]
        )
        sel_path = Path(selected)
        meta_file = sel_path / "organized_metadata.json"

        if not meta_file.exists():
            st.error("organized_metadata.json not found in the selected folder.")
        else:
            with open(meta_file, "r") as f:
                organized = json.load(f)

            # Build flat index
            photos_index = []
            all_objects, all_tags, all_locations, all_dates = set(), set(), set(), set()

            for folder, items in organized.items():
                for entry in items:
                    photos_index.append(entry)

                    # Objects
                    objs = entry.get("objects", [])
                    if isinstance(objs, list):
                        for o in objs:
                            if isinstance(o, dict):
                                all_objects.add(o.get("label") or str(o))
                            else:
                                all_objects.add(str(o))

                    # Tags (combine group_tag + tags list)
                    t = entry.get("tags", [])
                    if isinstance(t, list):
                        all_tags.update([x for x in t if x])
                    if entry.get("group_tag"):
                        all_tags.add(entry["group_tag"])

                    # Location
                    loc = entry.get("location")
                    if loc:
                        all_locations.add(loc)

                    # Date
                    dt = entry.get("date")
                    if dt:
                        all_dates.add(dt)

            # Filter widgets
            chosen_objects = st.multiselect("Objects", sorted([o for o in all_objects if o]))
            chosen_tags = st.multiselect("Tags / Scene Labels", sorted([t for t in all_tags if t]))
            chosen_locations = st.multiselect("Locations", sorted([l for l in all_locations if l]))
            chosen_dates = st.multiselect("Dates", sorted([d for d in all_dates if d]))

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
                    tag_list = entry.get("tags", [])
                    if entry.get("group_tag"):
                        tag_list = tag_list + [entry["group_tag"]]
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

            if matches:
                cols = st.columns(5)
                for i, entry in enumerate(matches):
                    try:
                        img_path = entry.get("copied_path") or entry.get("original_filename")
                        if img_path and Path(img_path).exists():
                            with Image.open(img_path) as im:
                                cols[i % 5].image(
                                    im, width="stretch", caption=Path(img_path).name
                                )
                        else:
                            cols[i % 5].write("Image not found")
                    except Exception:
                        cols[i % 5].write("Unable to open image.")

            if matches:
                if st.button("Export search results (JSON)"):
                    js = json.dumps(matches, indent=2)
                    st.download_button(
                        "Download JSON", data=js,
                        file_name="search_results.json", mime="application/json"
                    )
