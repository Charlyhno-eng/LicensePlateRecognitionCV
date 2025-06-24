import cv2
from ultralytics import YOLO

model = YOLO("yolov8mymodel.pt")

def display_camera_with_detection():
    cap = cv2.VideoCapture(0)

    while True:
        ret, frame = cap.read()
        if not ret:
            continue

        results = model.predict(source=frame, conf=0.25, imgsz=640, verbose=False)
        for result in results:
            for box in result.boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

        cv2.imshow("Camera with Plate Detection", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyWindow("Preprocessed Plate")
    cv2.destroyAllWindows()

if __name__ == "__main__":
    display_camera_with_detection()
