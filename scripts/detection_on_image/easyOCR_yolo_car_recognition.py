from ultralytics import YOLO
import cv2
import easyocr
import time
import psutil
import os
import re

start_time = time.time()
process = psutil.Process(os.getpid())

model = YOLO("runs/detect/plate_detector8/weights/best.pt")
ocr = easyocr.Reader(['en'], gpu=False)

PATH = "images_test/image9.jpg"
img = cv2.imread(PATH)

# Run YOLO detection
results = model.predict(
    source=PATH,
    conf=0.25,
    save=False,
    save_txt=False
)

# Process each detected plate
for result in results:
    for i, box in enumerate(result.boxes):
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        conf = float(box.conf[0])
        print(f"Plate detected: ({x1}, {y1}) - ({x2}, {y2}) with {conf:.2%} confidence")

        cropped_resized = img[y1:y2, x1:x2]
        result = ocr.readtext(cropped_resized)

        # Normalize and print text
        license_plate_text = ''.join([text for (_, text, _) in result])
        normalized_text = re.sub(r'[^A-Z0-9\- ]', '', license_plate_text.upper())
        print(f"Detected text (plate {i+1}): {normalized_text}")

end_time = time.time()
elapsed_time = end_time - start_time
memory_usage = process.memory_info().rss / (1024 ** 2)

print(f"\nExecution time: {elapsed_time:.2f} seconds")
print(f"RAM used: {memory_usage:.2f} MB")
