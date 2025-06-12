import cv2
import numpy as np
import imutils
from paddleocr import PaddleOCR

ocr = PaddleOCR(use_textline_orientation=True, lang='en')

def recognize_license_plate(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    bfilter = cv2.bilateralFilter(gray, 11, 17, 17)
    edged = cv2.Canny(bfilter, 30, 200)

    keypoints = cv2.findContours(edged.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    contours = imutils.grab_contours(keypoints)
    contours = sorted(contours, key=cv2.contourArea, reverse=True)[:10]

    location = None
    for contour in contours:
        approx = cv2.approxPolyDP(contour, 10, True)
        if len(approx) == 4:
            location = approx
            break

    if location is not None:
        mask = np.zeros(gray.shape, np.uint8)
        cv2.drawContours(mask, [location], 0, 255, -1)
        new_image = cv2.bitwise_and(img, img, mask=mask)

        (x, y) = np.where(mask == 255)
        (x1, y1), (x2, y2) = (np.min(x), np.min(y)), (np.max(x), np.max(y))
        cropped_image = img[x1:x2+1, y1:y2+1]

        cropped_resized = cv2.resize(cropped_image, None, fx=2, fy=2, interpolation=cv2.INTER_LINEAR)
        result = ocr.predict(cropped_resized)
        license_plate_text = ' '.join([res['rec_texts'][0] for res in result if 'rec_texts' in res and res['rec_texts']])

        return license_plate_text.strip()
    else:
        return None

def display_camera():
    cap = cv2.VideoCapture(0)
    frame_count = 0
    recognition_frequency = 5

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        if frame_count % recognition_frequency == 0:
            result = recognize_license_plate(frame)

            if result:
                print("License Plate:", result)

        cv2.imshow("Camera", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

        frame_count += 1

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    display_camera()
