from ultralytics import YOLO
import cv2

image_path = "images_test/image9.jpg"

model = YOLO("yolov8mymodel.pt")

results = model.predict(
    source=image_path,
    conf=0.25,
    save=True,
    save_txt=False,
    project="images_test",
    name="plate_output",
    exist_ok=True
)

img = cv2.imread(image_path)

for result in results:
    for box in result.boxes:
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        conf = float(box.conf[0])
        print(f"Licence plate detected : ({x1}, {y1}), ({x2}, {y2}) with {conf:.2%} of confidence")
        cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
