# LicensePlateReconitionCV

LicensePlateReconitionCV is a computer vision project. In this project, I'm testing different approaches to detecting car license plates. I analyze and search for them on images and videos.

## Installation
- sudo apt install tesseract-ocr # or equivalent for your system
- python3 -m venv venv
- source venv/bin/activate
- pip install -r requirements.txt

## Usage
python scripts/<file>.py

---

## Details

First, I tested image detection using only OCR. I was able to compare the differences between easyOCR and paddleOCR. The test was carried out on 9 images, some blurred, some tilted, and one of them contained two license plates. Here are the results :

| #V1            | **Plaque**                           | **easyOCR** | result     | **PaddleOCR** | result     |
| -------------- | ------------------------------------ | ----------- | ---------- | ------------- | ---------- |
| **image1**     | **_S443 JHP_**                       | 15443 HP    | ~          | S443JHP       | Ok         |
| **image2**     | **_WA03 BJF_**<br><br>**_SC04 VFS_** | cityBus     | bad result | CItyRus       | bad result |
| **image3**     | **_LHH-0887_**                       |             | -          |               | -          |
| **image4**     | **_LWJ 663_**                        |             | -          |               | -          |
| **image5**     | **_07-TH-FD_**                       | 07-TH-FD    | Ok         | 07-TH-FD      | Ok         |
| **image6**     | **_NA Y35396_**                      | (NA'Y35396  | ~          | NAY35396      | Ok         |
| **image7**     | **_1594 DRM_**                       |             | -          |               | -          |
| **image8**     | **_Q992 VHR_**                       |             | -          |               | -          |
| **image9**     | **_AKK-67K_**                        |             | -          |               | -          |
|                |                                      |             |            |               |            |
| **Note**       |                                      | ~2/9        |            | 3/9           |            |
| **Temps**      |                                      | [2s; 2.5s]  |            | [5s; 7s]      |            |
| **RAM**        |                                      | ~800Mo      |            | ~1100Mo       |            |
| **Conclusion** |                                      | + fast      |            | + accurate    |            |

Rather poor results. PaddleOCR is much slower but is much more accurate. More than half the time, the plates are not even identified. Extraneous characters are added (especially on the easyOCR side). No plate pattern in this dataset, so it confuses certain letters with numbers and vice versa.

### **Areas for improvement:**
- Improve license plate detection (V2 with Yolo)
- Remove stray characters (only for easyOCR)

| #V2            | **Plaque**                           | **easyOCR & Yolo**                                          | result         | **PaddleOCR  & Yolo**                           | result               |
| -------------- | ------------------------------------ | ----------------------------------------------------------- | -------------- | ----------------------------------------------- | -------------------- |
| **image1**     | **_S443 JHP_**                       | 5443 JHP                                                    | ~              | SREESSS                                         | bad result           |
| **image2**     | **_WA03 BJF_**<br><br>**_SC04 VFS_** | HAO3 BJF<br><br><br>SCO4 VES                                | ~<br><br><br>~ | SRRSOAE<br><br><br>SC04VFS                      | bad result<br><br>Ok |
| **image3**     | **_LHH-0887_**                       | LHOED                                                       | bad result     | LHH0887                                         | Ok                   |
| **image4**     | **_LWJ 663_**                        | L86637SEVILL                                                | bad result     | LWJ663 福 SEVILLA                                | Ok ~                 |
| **image5**     | **_07-TH-FD_**                       | 0Z-TH-FD                                                    | ~              | wwwNewMNlub 07-TH-FD                            | ~ Ok                 |
| **image6**     | **_NA Y35396_**                      | NAY35396                                                    | Ok             | NAY35396                                        | Ok                   |
| **image7**     | **_1594 DRM_**                       | 11594 DRH                                                   | ~              | 594DRK                                          | ~                    |
| **image8**     | **_Q992 VHR_**                       | 0992VHRI                                                    | ~              | Q992VHF                                         | ~                    |
| **image9**     | **_AKK-67K_**                        | AKK6ZK                                                      | ~              | HEAA                                            | bad result           |
|                |                                      |                                                             |                |                                                 |                      |
| **Note**       |                                      | 4.5/9                                                       |                | 5.5/9                                           |                      |
| **Temps**      |                                      | [2.5s; 4s]                                                  |                | [7s; 9.5s]                                      |                      |
| **RAM**        |                                      | ~800Mo                                                      |                | ~1650Mo                                         |                      |
| **Conclusion** |                                      | 2 to 3 times faster than paddleOCR but much less accurate |                | Better than the first version. +50% of RAM, +2s |                      |

After this second version, the results were significantly better. The biggest problem was recognizing the license plate patterns specific to each country. So I decided to start the video analysis part by implementing the patterns. The patterns are managed in the "plate_format" folder. So I tested with Romanian and French license plates. All while optimizing resources as much as possible because the ultimate goal is to run this program on Raspberry Pi. After several tests, Yolo8v + easyOCR is the best. PaddleOCR cost too much resources.

---

## Yolo Model Training

To train my license plate detection model, I retrieved a dataset from the internet (link below). For training, I did it on 40 epochs with a batch of 64 (see the file script/model_training.py). The trained model is the file yolov8mymodel.pt. Everything is available on the github. If you want to analyze license plates from countries other than Romania and France, just add a script with the format of your country's plate and call it in scripts/detection_video_live/video_yolo.py

### Dataset used
https://www.kaggle.com/datasets/fareselmenshawii/license-plate-dataset
