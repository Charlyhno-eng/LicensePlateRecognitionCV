import cv2
from matplotlib import pyplot as plt
import numpy as np
import imutils
import easyocr
import time
import psutil
import os

start_time = time.time()
process = psutil.Process(os.getpid())

ocr = easyocr.Reader(['en'])

PATH = 'images_test/plate_input/image9.jpg'
img = cv2.imread(PATH)

# Image preprocessing
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
bfilter = cv2.bilateralFilter(gray, 11, 17, 17)
edged = cv2.Canny(bfilter, 30, 200)

# Edge detection
keypoints = cv2.findContours(edged.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
contours = imutils.grab_contours(keypoints)
contours = sorted(contours, key=cv2.contourArea, reverse=True)[:10]

# Search for the plate
location = None
for contour in contours:
    approx = cv2.approxPolyDP(contour, 10, True)
    if len(approx) == 4:
        location = approx
        break

# Masking to isolate the plate
mask = np.zeros(gray.shape, np.uint8)
new_image = cv2.drawContours(mask, [location], 0, 255, -1)
new_image = cv2.bitwise_and(img, img, mask=mask)

# Segmenting the plate
(x, y) = np.where(mask == 255)
(x1, y1), (x2, y2) = (np.min(x), np.min(y)), (np.max(x), np.max(y))
cropped_image = gray[x1:x2 + 1, y1:y2 + 1]

# Reading the text on the plate
result = ocr.readtext(cropped_image)
license_plate_text = ''.join([text for (bbox, text, prob) in result])
print("Plate :", license_plate_text)

plt.imshow(cv2.cvtColor(new_image, cv2.COLOR_BGR2RGB))
plt.show()

end_time = time.time()
elapsed_time = end_time - start_time
memory_usage = process.memory_info().rss / (1024 ** 2)

print(f"Execution time : {elapsed_time:.2f} seconds")
print(f"RAM used : {memory_usage:.2f} Mo")
