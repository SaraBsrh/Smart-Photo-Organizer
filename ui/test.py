import streamlit as st
from PIL import Image
import os
import tempfile

# --- Page setup ---
st.set_page_config(page_title="Smart Photo Organizer", layout="wide")

# --- Sidebar Navigation ---
st.sidebar.title("📂 Navigation")
page = st.sidebar.radio("Go to", ["Welcome", "Upload Photos"])

# --- Welcome Page ---
if page == "Welcome":
    st.title("📸 Smart Photo Organizer")
    st.write(
        """
        Welcome to **Smart Photo Organizer**!  
        This app helps you organize your photos by:
        - Grouping photos based on **location, date, and situation**  
        - Detecting **people in photos**  
        - Searching by **objects or events**  

        Go to **Upload Photos** to start organizing your photos.
        """
    )

# --- Upload Photos Page ---
elif page == "Upload Photos":
    st.title("📁 Upload Your Photos")
    st.write("Select multiple images to organize:")

    uploaded_files = st.file_uploader(
        "Choose images", type=["png", "jpg", "jpeg"], accept_multiple_files=True
    )

    if uploaded_files:
        st.success(f"{len(uploaded_files)} files uploaded successfully!")

        # Save uploaded files temporarily
        temp_dir = tempfile.mkdtemp()
        file_paths = []
        for uploaded_file in uploaded_files:
            file_path = os.path.join(temp_dir, uploaded_file.name)
            with open(file_path, "wb") as f:
                f.write(uploaded_file.getbuffer())
            file_paths.append(file_path)

        st.write("Processing photos...")

        # --- Call your photo organization function here ---
        # Example placeholder function (replace with your code)
        def organize_photos(file_paths):
            """
            Dummy function: group photos by color label (simulate grouping)
            Return: dict with group_name -> list of file paths
            """
            import random

            groups = ["Group A", "Group B", "Group C"]
            grouped = {g: [] for g in groups}
            for f in file_paths:
                group = random.choice(groups)
                grouped[group].append(f)
            return grouped

        grouped_photos = organize_photos(file_paths)

        # --- Display Results ---
        st.success("Photos organized successfully!")
        for group_name, images in grouped_photos.items():
            st.markdown(f"### 📁 {group_name} ({len(images)} photos)")
            cols = st.columns(5)
            for idx, img_path in enumerate(images):
                image = Image.open(img_path)
                cols[idx % 5].image(image, use_column_width=True)
