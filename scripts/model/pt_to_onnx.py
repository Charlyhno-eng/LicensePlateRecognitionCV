from ultralytics import YOLO

model = YOLO("yolov8mymodel.pt")

model.export(format="onnx", opset=12, dynamic=False, imgsz=640)
