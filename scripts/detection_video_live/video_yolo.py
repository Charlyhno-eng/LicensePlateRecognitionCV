import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)) + "/..")

from ultralytics import YOLO
import cv2
import easyocr
import time
from plate_format.plate_format_ro import is_valid_plate, normalize_plate_format

model = YOLO("runs/detect/plate_detector8/weights/best.pt")
ocr = easyocr.Reader(['en'], gpu=True)

def display_camera_with_detection():
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

    last_detection_time = 0
    ocr_interval = 3

    while True:
        ret, frame = cap.read()

        # Resize to speed up processing
        frame = cv2.resize(frame, (640, 480))

        # Detection with reduced image size
        results = model.predict(source=frame, conf=0.25, imgsz=416, verbose=False)

        for result in results:
            for box in result.boxes:
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
                        raw_text = res[1].strip()

                        if is_valid_plate(raw_text):
                            print("License Plate :", normalize_plate_format(raw_text))


        cv2.imshow("Camera with Plate Detection", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    display_camera_with_detection()
