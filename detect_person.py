from ultralytics import YOLO
from PIL import Image

# Load a pretrained YOLOv8 model (on COCO dataset)
model = YOLO("yolov8n.pt")  # use yolov8s.pt or yolov8m.pt for better accuracy

# Run detection
results = model("test.jpg")

# Check if any detection is a person
person_detected = any(
    result.names[int(cls)] == "person"
    for result in results
    for cls in result.boxes.cls
)

print("Person detected:" if person_detected else "No person detected")
