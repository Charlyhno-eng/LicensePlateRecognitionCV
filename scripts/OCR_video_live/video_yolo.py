import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)) + "/..")

import time
import re
from ultralytics import YOLO
import cv2
import easyocr
from utils.plate_format_ro import is_valid_ro_plate

model = YOLO("runs/detect/plate_detector8/weights/best.pt")
ocr = easyocr.Reader(['en'], gpu=False)

def clean_text(text):
    text = text.upper()
    text = re.sub(r'[^A-Z0-9]', '', text)
    return text

def normalize_plate(text):
    text = text.strip().upper().replace("-", "").replace(" ", "")
    if len(text) >= 6:
        if text[0] == 'B':
            return f"{text[:1]} {text[1:4]} {text[4:7]}"
        else:
            return f"{text[:2]} {text[2:5]} {text[5:7]}"
    return text

def display_camera_with_detection():
    cap = cv2.VideoCapture(0)
    last_detection_time = 0
    ocr_interval = 2
    recent_texts = set()

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        results = model.predict(source=frame, conf=0.25, verbose=False)

        for result in results:
            boxes = result.boxes
            for box in boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                width = x2 - x1
                height = y2 - y1

                if width < 60 or height < 20:
                    continue

                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

                current_time = time.time()
                if current_time - last_detection_time > ocr_interval:
                    plate_crop = frame[y1:y2, x1:x2]
                    if plate_crop.size == 0:
                        continue

                    ocr_result = ocr.readtext(plate_crop)

                    for res in ocr_result:
                        bbox, raw_text, conf = res
                        if conf < 0.5:
                            continue

                        cleaned = clean_text(raw_text)
                        normalized = normalize_plate(cleaned)
                        valid = is_valid_ro_plate(normalized)

                        if valid and normalized not in recent_texts:
                            print(normalized)
                            recent_texts.add(normalized)
                            last_detection_time = current_time

        cv2.imshow("Camera with Plate Detection", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

        if len(recent_texts) > 10:
            recent_texts.clear()

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    display_camera_with_detection()
