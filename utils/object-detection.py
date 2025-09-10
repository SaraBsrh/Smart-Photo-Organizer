import os
import json
from ultralytics import YOLO

# Initialize the model
model = YOLO('yolov8n.pt')  # Using the small, fast model

# Define paths
input_dir = '/Users/sara/Desktop/backed-data/sample-data'
output_json = '/Users/sara/Desktop/backed-data/sample-data/object_detection_results.json'

# Initialize an empty dictionary to store results
detection_results = {}

# Process each image in the directory
for image_name in os.listdir(input_dir):
    if image_name.endswith(('.jpg', '.jpeg', '.png')):  # Ensure it's an image file
        image_path = os.path.join(input_dir, image_name)
        results = model(image_path)

        # Extract relevant information
        objects = []
        for result in results:
            for obj in result.boxes:
                cls_index = int(obj.cls)          # Convert tensor to int
                label = model.names[cls_index]   # Get class name
                confidence = float(obj.conf)     # Convert tensor to float
                bbox = obj.xywh.tolist()         # Bounding box
                objects.append({"label": label, "confidence": confidence, "bbox": bbox})

        # Store the results in the dictionary
        detection_results[image_name] = objects

# Save the results to a JSON file
with open(output_json, 'w') as f:
    json.dump(detection_results, f, indent=4)

print(f"Detection results saved to {output_json}")
