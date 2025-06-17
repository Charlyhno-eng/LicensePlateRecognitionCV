import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)) + "/../..")

import cv2
import pytesseract
from ultralytics import YOLO
import re
import time
import psutil
from plate_format.plate_format_ro import is_valid_plate, normalize_plate_format

INPUT_IMAGE_PATH = "images_test/plate_input_ro/image2.jpg"
OUTPUT_IMAGE_PATH = "images_test/plate_output_ro/image2.jpg"

model = YOLO("yolov8mymodel.pt")

def preprocess_plate(plate_crop):
    """Preprocess plate image to improve OCR accuracy."""
    clahe = cv2.createCLAHE(clipLimit=1.8, tileGridSize=(8,8))
    gray = cv2.cvtColor(plate_crop, cv2.COLOR_BGR2GRAY)
    gray = clahe.apply(gray)
    gray = cv2.resize(gray, None, fx=1.4, fy=1.4, interpolation=cv2.INTER_LINEAR)
    blur = cv2.bilateralFilter(gray, 11, 16, 16)
    thresh = cv2.adaptiveThreshold(
        blur, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 13, 2
    )
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 1))
    morph = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)
    return morph

def extract_valid_plate(plate_crop):
    """Extract text with Tesseract and validate using plate format."""
    processed = preprocess_plate(plate_crop)

    config = '--psm 7 -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789'
    raw_text = pytesseract.image_to_string(processed, config=config)
    raw_text = raw_text.strip().replace("\n", "").replace("\f", "")
    raw_text = re.sub(r'[^A-Z0-9 ]', '', raw_text.upper())

    if is_valid_plate(raw_text):
        return normalize_plate_format(raw_text)
    return None

def main():
    start_time = time.time()
    process = psutil.Process(os.getpid())

    img = cv2.imread(INPUT_IMAGE_PATH)
    if img is None:
        print(f"Failed to load image: {INPUT_IMAGE_PATH}")
        return

    results = model.predict(source=img, conf=0.25, save=False, save_txt=False)

    for result in results:
        for i, box in enumerate(result.boxes):
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            conf = float(box.conf[0])
            print(f"Plate detected #{i+1}: ({x1}, {y1}) - ({x2}, {y2}) confidence={conf:.2%}")

            plate_crop = img[y1:y2, x1:x2]
            if plate_crop.size == 0:
                print("Warning: empty crop, skipping.")
                continue

            plate_text = extract_valid_plate(plate_crop)
            if plate_text:
                print(f"Recognized valid plate #{i+1}: {plate_text}")
                cv2.putText(
                    img,
                    plate_text,
                    (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.9,
                    (0, 255, 0),
                    2,
                    cv2.LINE_AA
                )
            else:
                print(f"No valid plate recognized for plate #{i+1}")
            cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)

    cv2.imwrite(OUTPUT_IMAGE_PATH, img)

    elapsed = time.time() - start_time
    mem_mb = process.memory_info().rss / (1024**2)
    print(f"\nExecution time: {elapsed:.2f} s")
    print(f"Memory usage: {mem_mb:.2f} MB")
    print(f"Annotated image saved to: {OUTPUT_IMAGE_PATH}")

if __name__ == "__main__":
    main()
