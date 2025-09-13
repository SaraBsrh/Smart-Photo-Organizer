import os
import json
from pathlib import Path
import argparse
from ultralytics import YOLO

try:
    from ultralytics import YOLO
except Exception as e:
    YOLO = None

def detect_objects_in_folder(input_dir: Path, output_json: Path, model_weights: str | Path = "yolov8n.pt"):
    results_map = {}
    if YOLO is None:
        print("ERROR: ultralytics not installed. Install `ultralytics` or skip object detection.")
        return results_map

    model = YOLO(str(model_weights))
    for image_path in sorted(input_dir.glob("*.*")):
        if image_path.suffix.lower() not in (".jpg", ".jpeg", ".png"):
            continue
        res = model(str(image_path))
        objects = []
        for r in res:
            for box in r.boxes:
                cls_index = int(box.cls)
                label = model.names.get(cls_index, str(cls_index))
                conf = float(box.conf) if hasattr(box, "conf") else float(box.confidence) if hasattr(box, "confidence") else 0.0
                bbox = box.xywh.tolist() if hasattr(box, "xywh") else []
                objects.append({"label": label, "confidence": conf, "bbox": bbox})
        results_map[image_path.name] = objects

    # Save output
    with open(output_json, "w") as f:
        json.dump(results_map, f, indent=2)
    print(f"✅ Object detection results saved to {output_json}")
    return results_map

def main():
    parser = argparse.ArgumentParser(description="Run object detection over a folder.")
    parser.add_argument("photo_dir", nargs="?", default=".", help="Folder with photos")
    parser.add_argument("output_json", nargs="?", default="./object_detection_results.json", help="Output JSON path")
    parser.add_argument("--weights", default=os.getenv("YOLO_WEIGHTS", "yolov8n.pt"), help="YOLO model weights")
    args = parser.parse_args()
    input_dir = Path(args.photo_dir)
    output_json = Path(args.output_json)
    input_dir = input_dir.resolve()
    output_json.parent.mkdir(parents=True, exist_ok=True)
    detect_objects_in_folder(input_dir, output_json, model_weights=args.weights)

if __name__ == "__main__":
    main()
