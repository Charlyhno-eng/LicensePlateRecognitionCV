# LicensePlateRecognitionCV

LicensePlateRecognitionCV is a computer vision project focused on detecting car license plates. This project tests various approaches to locate and analyze plates on images and videos.

## Installation

- sudo apt install tesseract-ocr (or equivalent for your system)
- python3 -m venv venv
- source venv/bin/activate
- pip install -r requirements.txt

## Usage

- python scripts/<file>.py

---

## OCR Methods Comparison

This project includes tests with three main OCR tools: **Tesseract**, **easyOCR**, and **PaddleOCR**. Here is a summary of the findings:

| OCR          | Overall Performance                         | Resource Usage                    | Advantages                               | Disadvantages                                  |
|--------------|---------------------------------------------|---------------------------------|-----------------------------------------|------------------------------------------------|
| **Tesseract**| Suitable for low-power systems like Raspberry Pi | Low memory and CPU usage          | Very lightweight, easy to deploy         | Requires significant image preprocessing for good results |
| **easyOCR**  | Good balance between performance and resources | Moderate to high, heavy for Raspberry Pi | Generally accurate and versatile          | Too heavy for real-time use on Raspberry Pi     |
| **PaddleOCR**| Best recognition results                    | Very resource-intensive          | Very accurate and robust                  | Too slow and heavy for small embedded systems   |

---

## General Remarks

- **Tesseract** is the most suitable for Raspberry Pi usage but demands strong image preprocessing (cleaning, contrast adjustment, etc.) for satisfactory results.
- **easyOCR** is a good compromise between accuracy and resource consumption but remains too heavy for embedded real-time applications.
- **PaddleOCR** delivers the best recognition accuracy but is too resource-demanding for limited hardware.

---

## Project Purpose

This project compiles the different tests carried out to compare these OCR solutions and improve license plate recognition. The final version optimized for Raspberry Pi is available here:

https://github.com/Charlyhno-eng/raspberry-plate-recognition

---

## Yolo Model Training

For plate detection, a Yolo model was trained using a public dataset:

https://www.kaggle.com/datasets/fareselmenshawii/license-plate-dataset

Training was done for 40 epochs with a batch size of 64 (see `script/model_training.py`). The trained model file is `yolov8mymodel.pt`.

You can adapt the system to other plate formats by adding a script defining your country’s plate format and calling it in `scripts/detection_video_live/video_yolo.py`.
