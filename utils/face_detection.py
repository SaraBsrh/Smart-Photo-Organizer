from retinaface import RetinaFace
import cv2
from pathlib import Path

# Input and output folders
PHOTO_DIR = Path("/Users/sara/Desktop/backed-data/sample-data")
OUTPUT_DIR = Path("/Users/sara/Desktop/backed-data/faces_detected")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

def detect_faces(image_path):
    img = cv2.imread(str(image_path))
    detections = RetinaFace.detect_faces(str(image_path))

    if isinstance(detections, dict):  # If faces are detected
        for i, (face_id, face_data) in enumerate(detections.items()):
            facial_area = face_data["facial_area"]  # [x1, y1, x2, y2]
            x1, y1, x2, y2 = facial_area
            face = img[y1:y2, x1:x2]

            # Save cropped face
            out_path = OUTPUT_DIR / f"{image_path.stem}_face{i}.jpg"
            cv2.imwrite(str(out_path), face)
            print(f"✅ Saved: {out_path}")
    else:
        print(f"❌ No faces detected in {image_path.name}")

if __name__ == "__main__":
    for image_path in PHOTO_DIR.glob("*.jpg"):
        detect_faces(image_path)
