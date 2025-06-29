import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)) + "/../..")

import time
import cv2
import pytesseract
import onnx
import onnxruntime as onnxr
import numpy as np
from plate_format.plate_format_ro import is_valid_plate, normalize_plate_format

IMAGE_SIZE = 416
ONNX_PATH = "yolov8mymodel.onnx"

last_detected_plates = {}
max_plate_age_seconds = 10

url = "http://192.168.100.67:4747/video"
cap = cv2.VideoCapture(url)

# Create a session
onnx_model = onnx.load(ONNX_PATH)
onnx.checker.check_model(onnx_model)

session = onnxr.InferenceSession(ONNX_PATH)
input_name = session.get_inputs()[0].name
output_names = [out.name for out in session.get_outputs()]

def findIntersectionOverUnion(box1, box2):
    box1_w = box1[2]/2.0
    box1_h = box1[3]/2.0
    box2_w = box2[2]/2.0
    box2_h = box2[3]/2.0

    b1_1, b1_2 = box1[0] - box1_w, box1[1] - box1_h
    b1_3, b1_4 = box1[0] + box1_w, box1[1] + box1_h
    b2_1, b2_2 = box2[0] - box2_w, box2[1] - box2_h
    b2_3, b2_4 = box2[0] + box2_w, box2[1] + box2_h

    x1, y1 = max(b1_1, b2_1), max(b1_2, b2_2)
    x2, y2 = min(b1_3, b2_3), min(b1_4, b2_4)

    intersect = max(0, x2 - x1) * max(0, y2 - y1)
    area1 = (b1_3 - b1_1) * (b1_4 - b1_2)
    area2 = (b2_3 - b2_1) * (b2_4 - b2_2)
    union = area1 + area2 - intersect

    return intersect / union if union > 0 else 0

def preprocess_plate(plate_crop):
    """
    Applies a set of preprocessing steps to enhance plate image for OCR.
    Includes contrast enhancement, denoising, binarization, and deskewing.
    """
    clahe = cv2.createCLAHE(clipLimit=1.8, tileGridSize=(8,8))
    gray = cv2.cvtColor(plate_crop, cv2.COLOR_BGR2GRAY)
    gray = clahe.apply(gray)
    gray = cv2.resize(gray, None, fx=1.4, fy=1.4, interpolation=cv2.INTER_LINEAR)
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
    last_detection_time = 0
    ocr_interval_second = 3
    conf_thres = 0.25
    iou_thres = 0.7

    while True:
        ret, frame = cap.read()
        if not ret:
            continue

        current_time = time.time()

        # Prepare the image for ONNX
        resized = cv2.resize(frame, (IMAGE_SIZE, IMAGE_SIZE))
        input_tensor = resized.astype(np.float32) / 255.0
        input_tensor = np.transpose(input_tensor, (2, 0, 1))
        input_tensor = np.expand_dims(input_tensor, axis=0)

        # ONNX Inference
        outputs = session.run(output_names, {input_name: input_tensor})

        he, wi, _ = frame.shape
        x_scale = wi / IMAGE_SIZE
        y_scale = he / IMAGE_SIZE

        detections = []
        for detection in outputs[0]:
            boxes_ = detection[:4, :]
            prob_ = detection[4:7, :]

            class_id_ = np.argmax(prob_, axis=0)
            prob_score_ = prob_[class_id_, np.arange(prob_.shape[1])]
            mask_ = prob_score_ >= conf_thres
            indices_ = np.where(mask_)[0]

            flag = np.zeros(len(indices_))
            for flag_index, i in enumerate(indices_):
                if flag[flag_index]:
                    continue

                box = boxes_[:, i]
                prob_score = prob_score_[i]
                class_id = class_id_[i]

                for flag_index_2, i_2 in enumerate(indices_):
                    if i_2 < i:
                        continue
                    if class_id_[i_2] != class_id:
                        continue

                    box_2 = boxes_[:, i_2]
                    iou = findIntersectionOverUnion(box, box_2)
                    if iou >= iou_thres:
                        flag[flag_index_2] = True

                detections.append({"bbox": box, "confidence": prob_score, "class_id": class_id})
                flag[flag_index] = True

        # Draw results and process plates
        for detect in detections:
            x, y, w, h = detect["bbox"]

            left = (x - w / 2.0) * x_scale
            top = (y - h / 2.0) * y_scale
            right = (x + w / 2.0) * x_scale
            bottom = (y + h / 2.0) * y_scale

            x1, y1, x2, y2 = int(left), int(top), int(right), int(bottom)
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
                    print("License Plate:", plate)
                    last_detected_plates[key] = current_time
                    last_detection_time = current_time

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyWindow("Preprocessed Plate")
    cv2.destroyAllWindows()

if __name__ == "__main__":
    display_camera_with_detection()
