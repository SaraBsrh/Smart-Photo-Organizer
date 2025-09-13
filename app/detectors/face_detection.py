from retinaface import RetinaFace
import argparse
import cv2
import os
from pathlib import Path

try:
    from retinaface import RetinaFace
    HAVE_RETINA = True
except Exception:
    HAVE_RETINA = False

def detect_and_crop_faces(photo_dir: Path, out_dir: Path):
    out_dir.mkdir(parents=True, exist_ok=True)
    count = 0
    for img_path in sorted(photo_dir.glob("*.*")):
        if img_path.suffix.lower() not in (".jpg", ".jpeg", ".png"):
            continue
        img = cv2.imread(str(img_path))
        if img is None:
            continue
        if not HAVE_RETINA:
            # fallback: try OpenCV Haar cascade (less accurate) if retinaface unavailable
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
            faces = cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=4)
            if len(faces) == 0:
                continue
            for i, (x, y, w, h) in enumerate(faces):
                crop = img[y:y+h, x:x+w]
                out_path = out_dir / f"{img_path.stem}_face{i}.jpg"
                cv2.imwrite(str(out_path), crop)
                count += 1
            continue

        # Use RetinaFace
        try:
            dets = RetinaFace.detect_faces(str(img_path))
        except Exception:
            dets = None

        if isinstance(dets, dict):
            for i, (face_id, face_data) in enumerate(dets.items()):
                try:
                    x1, y1, x2, y2 = face_data["facial_area"]
                    crop = img[y1:y2, x1:x2]
                    out_path = out_dir / f"{img_path.stem}_face{i}.jpg"
                    cv2.imwrite(str(out_path), crop)
                    count += 1
                except Exception:
                    continue
    print(f"✅ Face crops written to {out_dir} (total crops: {count})")
    return count

def main():
    parser = argparse.ArgumentParser(description="Detect faces and save crops.")
    parser.add_argument("photo_dir", nargs="?", default=".", help="Folder with photos")
    parser.add_argument("out_dir", nargs="?", default="./faces_detected", help="Folder to save face crops")
    args = parser.parse_args()
    detect_and_crop_faces(Path(args.photo_dir), Path(args.out_dir))

if __name__ == "__main__":
    main()
