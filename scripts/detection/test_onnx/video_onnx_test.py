import cv2
import numpy as np
import onnxruntime as ort

session = ort.InferenceSession("yolov8mymodel.onnx", providers=['CPUExecutionProvider'])
input_name = session.get_inputs()[0].name
output_name = session.get_outputs()[0].name

img_size = 640
conf_threshold = 0.25

def preprocess(frame):
    img = cv2.resize(frame, (img_size, img_size))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = img.astype(np.float32) / 255.0
    img = np.transpose(img, (2, 0, 1))
    img = np.expand_dims(img, axis=0)
    return img

def postprocess(predictions, orig_shape):
    detections = []
    pred = predictions[0].squeeze().T  # [8400, 5]
    h, w = orig_shape[:2]
    scale_h, scale_w = h / img_size, w / img_size

    for det in pred:
        x, y, box_w, box_h, conf = det
        if conf < conf_threshold:
            continue
        x1 = int((x - box_w / 2) * scale_w)
        y1 = int((y - box_h / 2) * scale_h)
        x2 = int((x + box_w / 2) * scale_w)
        y2 = int((y + box_h / 2) * scale_h)
        width, height = x2 - x1, y2 - y1
        if width < 60 or height < 20:
            continue
        if x1 < 0 or y1 < 0 or x2 > w or y2 > h:
            continue
        detections.append((x1, y1, x2, y2, conf))
    return detections

def display_camera_with_detection():
    cap = cv2.VideoCapture(0)

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        input_tensor = preprocess(frame)
        outputs = session.run([output_name], {input_name: input_tensor})
        boxes = postprocess(outputs, frame.shape)

        for x1, y1, x2, y2, conf in boxes:
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

        cv2.imshow("Détection de plaques", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyWindow("Preprocessed Plate")
    cv2.destroyAllWindows()

if __name__ == "__main__":
    display_camera_with_detection()
