import onnx
import onnxruntime as onnxr
import cv2 as cv
import numpy as np

IMAGE_SIZE = 416
ONNX_PATH = "yolov8mymodel.onnx"
IMAGE_PATH = "images_test/plate_input/image3.jpg"

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

# Create a session
onnx_model = onnx.load(ONNX_PATH)
onnx.checker.check_model(onnx_model)

session = onnxr.InferenceSession(ONNX_PATH)
input_name = session.get_inputs()[0].name
output_names = [out.name for out in session.get_outputs()]

# Read the image
image = cv.imread(IMAGE_PATH)
if image is None:
    raise FileNotFoundError(f"Image non trouvée : {IMAGE_PATH}")

resized = cv.resize(image, (IMAGE_SIZE, IMAGE_SIZE))
input_tensor = resized.astype(np.float32) / 255.0
input_tensor = np.transpose(input_tensor, (2, 0, 1))
input_tensor = np.expand_dims(input_tensor, axis=0)

outputs = session.run(output_names, {input_name: input_tensor})

he, wi, _ = image.shape
x_scale = wi / IMAGE_SIZE
y_scale = he / IMAGE_SIZE

conf_thres = 0.2
iou_thres = 0.7
detections = []

for detection in outputs[0]:
    boxes_ = detection[:4, :]
    prob_ = detection[4:7, :]  # Adapter si plus/moins de classes

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

# Draw results
for detect in detections:
    x, y, w, h = detect["bbox"]
    found_class = detect["class_id"]
    found_score = detect["confidence"]

    left = (x - w / 2.0) * x_scale
    top = (y - h / 2.0) * y_scale
    right = (x + w / 2.0) * x_scale
    bottom = (y + h / 2.0) * y_scale

    cv.rectangle(image, (int(left), int(top)), (int(right), int(bottom)), (0, 255, 0), 3)

cv.imshow("Detection Result", image)
cv.waitKey(0)
cv.destroyAllWindows()
