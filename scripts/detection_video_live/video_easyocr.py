import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)) + "/..")

import time
import cv2
import psutil
from ultralytics import YOLO
import easyocr
from plate_format.plate_format_ro import is_valid_plate, normalize_plate_format

model = YOLO("yolov8mymodel.pt")

ocr = easyocr.Reader(['en'], gpu=False)

# Plate cache to avoid reprocessing the same plate multiple times
last_detected_plates = {}
max_plate_age_seconds = 10

def get_memory_usage():
    """
    Returns used and total RAM in GB, including cache/buffers (like htop or Stacer).
    """
    mem = psutil.virtual_memory()
    used_gb = (mem.total - mem.available) / (1024 ** 3)
    total_gb = mem.total / (1024 ** 3)
    return used_gb, total_gb

def extract_valid_plate(plate_crop):
    """
    Runs OCR on the cropped plate image, cleans result, validates format.
    """
    ocr_result = ocr.readtext(plate_crop)
    for res in ocr_result:
        raw_text = res[1].strip()
        raw_text = ''.join(c for c in raw_text if c.isalnum() or c.isspace())

        if is_valid_plate(raw_text):
            return normalize_plate_format(raw_text)
    return None

def display_camera_with_detection():
    cap = cv2.VideoCapture(0)

    last_detection_time = 0
    ocr_interval_second = 3

    while True:
        ret, frame = cap.read()
        if not ret:
            continue

        current_time = time.time()

        # Resize to speed up processing
        frame = cv2.resize(frame, (640, 480))

        results = model.predict(source=frame, conf=0.25, imgsz=640, verbose=False)

        for result in results:
            for box in result.boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                width = x2 - x1
                height = y2 - y1

                if width < 60 or height < 20:
                    continue

                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

                key = (x1, y1, x2, y2)
                if key in last_detected_plates and current_time - last_detected_plates[key] < max_plate_age_seconds:
                    continue

                if current_time - last_detection_time > ocr_interval_second:
                    plate_crop = frame[y1:y2, x1:x2]
                    if plate_crop.size == 0:
                        continue

                    plate = extract_valid_plate(plate_crop)
                    if plate:
                        print("License Plate :", plate)
                        last_detected_plates[key] = current_time
                        last_detection_time = current_time

        used_gb, total_gb = get_memory_usage()
        cv2.putText(frame, f"RAM: {used_gb:.2f} / {total_gb:.2f} GB", (10, 25), cv2.FONT_HERSHEY_SIMPLEX,
                            0.6, (0, 0, 180), 2)

        cv2.imshow("Camera with Plate Detection", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    display_camera_with_detection()
