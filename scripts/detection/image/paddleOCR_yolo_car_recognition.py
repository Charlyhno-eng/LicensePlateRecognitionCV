from ultralytics import YOLO
import cv2
from paddleocr import PaddleOCR
import time
import psutil
import os

start_time = time.time()
process = psutil.Process(os.getpid())

model = YOLO("yolov8mymodel.pt")
ocr = PaddleOCR(use_textline_orientation=True, lang='en')

PATH = "images_test/plate_input/image9.jpg"
img = cv2.imread(PATH)

# YOLO Prediction
results = model.predict(
    source=PATH,
    conf=0.25,
    save=False,
    save_txt=False
)

# Loop on every detection of YOLO
for result in results:
    for i, box in enumerate(result.boxes):
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        conf = float(box.conf[0])
        print(f"Plate detected : ({x1}, {y1}) - ({x2}, {y2}) with {conf:.2%} of confidence")

        # Reading the text on the plate
        cropped_resized = img[y1:y2, x1:x2]
        result = ocr.predict(cropped_resized)
        license_plate_text = ' '.join(result[0]['rec_texts'])
        print("Plate :", license_plate_text.strip())

end_time = time.time()
elapsed_time = end_time - start_time
memory_usage = process.memory_info().rss / (1024 ** 2)

print(f"Execution time : {elapsed_time:.2f} seconds")
print(f"RAM used : {memory_usage:.2f} Mo")
