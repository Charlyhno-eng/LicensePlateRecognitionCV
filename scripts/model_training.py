from ultralytics import YOLO

model = YOLO("yolov8n.pt")

model.train(
    data="data.yaml",
    epochs=40,
    imgsz=416,
    batch=64,
    name="plate_detector",
)
