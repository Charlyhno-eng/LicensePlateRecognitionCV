from ultralytics import YOLO

model = YOLO("yolov8mymodel.pt")

model.export(format="onnx")
