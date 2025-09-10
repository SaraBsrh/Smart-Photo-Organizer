# import os
# import shutil
# import numpy as np
# from pathlib import Path
# from sklearn.cluster import DBSCAN
# from insightface.app import FaceAnalysis
# import cv2

# # --- Paths ---
# faces_detected_dir = Path("/Users/sara/Desktop/backed-data/faces_detected")   # cropped faces (from RetinaFace)
# faces_clustered_dir = Path("/Users/sara/Desktop/backed-data/faces_clustered") # clustered output
# faces_clustered_dir.mkdir(exist_ok=True)

# # --- Load ArcFace (ONNX model) ---
# app = FaceAnalysis(name="buffalo_l")  # buffalo_l = strong ArcFace model
# app.prepare(ctx_id=0, det_size=(640, 640))  # ctx_id=0 -> GPU, -1 -> CPU

# # --- Step 1: Load faces ---
# face_images = []
# face_filenames = []

# for file in faces_detected_dir.glob("*.jpg"):
#     img = cv2.imread(str(file))
#     if img is None:
#         print(f"❌ Error loading {file}")
#         continue
#     face_images.append(img)
#     face_filenames.append(file.name)

# if not face_images:
#     raise ValueError("No valid face images found in faces-detected/")

# print(f"Loaded {len(face_images)} faces")

# # --- Step 2: Extract embeddings ---
# embeddings = []
# for img in face_images:
#     faces = app.get(img)   # get embeddings for detected faces
#     if len(faces) > 0:
#         emb = faces[0].embedding  # we assume only 1 face per cropped image
#         embeddings.append(emb)
#     else:
#         embeddings.append(np.zeros((512,)))  # fallback in case of failure

# embeddings = np.array(embeddings)
# print(f"Generated {embeddings.shape[0]} embeddings")

# # --- Step 3: Cluster with DBSCAN ---
# clustering = DBSCAN(eps=0.9, min_samples=4, metric="cosine").fit(embeddings)
# labels = clustering.labels_
# num_clusters = len(set(labels)) - (1 if -1 in labels else 0)
# print(f"Found {num_clusters} clusters")

# # --- Step 4: Save clustered faces ---
# for label, filename in zip(labels, face_filenames):
#     if label == -1:
#         folder = faces_clustered_dir / "unknown"
#     else:
#         folder = faces_clustered_dir / f"person_{label}"
#     folder.mkdir(exist_ok=True)
#     src = faces_detected_dir / filename
#     dst = folder / filename
#     shutil.copy(src, dst)

# print(f"✅ Faces grouped into {faces_clustered_dir}")
