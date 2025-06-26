from ultralytics import YOLO

# Load the YOLO8 model
model = YOLO("yolov8mymodel.pt")

# Export the model to ONNX format
model.export(format='onnx')
