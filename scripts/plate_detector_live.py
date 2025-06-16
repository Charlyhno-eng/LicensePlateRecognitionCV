import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)) + "/..")

import time
import cv2
import pytesseract
from ultralytics import YOLO
from plate_format.plate_format_ro import is_valid_plate, normalize_plate_format

model = YOLO("yolov8mymodel.pt")

last_detected_plates = {}
max_plate_age_seconds = 10

clahe = cv2.createCLAHE(clipLimit=1.8, tileGridSize=(8, 8))

def preprocess_plate(plate_crop):
    """
    Applies a set of preprocessing steps to enhance plate image for OCR.
    Includes contrast enhancement, denoising, binarization, and deskewing.
    """
    gray = cv2.cvtColor(plate_crop, cv2.COLOR_BGR2GRAY)
    gray = clahe.apply(gray)
    blur = cv2.bilateralFilter(gray, 11, 16, 16)
    thresh = cv2.adaptiveThreshold(blur, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 13, 2)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 1))
    morph = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)
    return morph

def extract_valid_plate(plate_crop):
    """
    Runs OCR on the plate image and returns a valid Romanian plate string if found.
    Also displays the preprocessed plate image for debugging.
    """
    processed = preprocess_plate(plate_crop)
    config = '--psm 7 -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789'
    raw_text = pytesseract.image_to_string(processed, config=config)
    raw_text = raw_text.strip().replace("\n", " ").replace("\f", "")
    raw_text = ''.join(c for c in raw_text if c.isalnum() or c.isspace())

    if is_valid_plate(raw_text):
        return normalize_plate_format(raw_text)

    return None

def display_camera_with_detection():
    cap = cv2.VideoCapture(0)

    last_detection_time = 0
    ocr_interval_second = 3

    frame_count = 0
    detect_every_n_frames = 2

    while True:
        ret, frame = cap.read()
        if not ret:
            continue

        frame_count += 1
        if frame_count % detect_every_n_frames != 0:
            time.sleep(0.01)
            continue

        current_time = time.time()

        resize_width = 640
        height, width = frame.shape[:2]
        scale = resize_width / width
        frame_small = cv2.resize(frame, (resize_width, int(height * scale)))

        results = model.predict(source=frame_small, conf=0.25, imgsz=640, verbose=False)
        for result in results:
            for box in result.boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                width = x2 - x1
                height = y2 - y1

                if width < 60 or height < 20:
                    continue

                key = (x1, y1, x2, y2)
                if key in last_detected_plates and current_time - last_detected_plates[key] < max_plate_age_seconds:
                    continue

                if current_time - last_detection_time > ocr_interval_second:
                    plate_crop = frame[y1:y2, x1:x2]
                    if plate_crop.size == 0:
                        continue

                    plate = extract_valid_plate(plate_crop)
                    if plate:
                        print(f"[{time.strftime('%H:%M:%S')}] Plate detected : {plate}")
                        last_detected_plates[key] = current_time
                        last_detection_time = current_time

        time.sleep(0.02)

    cap.release()

if __name__ == "__main__":
    display_camera_with_detection()
