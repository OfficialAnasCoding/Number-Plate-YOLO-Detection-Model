# YOLO Training & Testing Project

This repository contains everything needed to test (and train) a YOLO number plate detection model.

## 📁 Project Contents

- **Training code** – Script used to train the YOLO model  
- **Testing code** – Script for running inference and evaluating the trained model  
- **Trained model** – A ready-to-use YOLO model for immediate testing  
- **Training dataset** – Included in case you want to retrain or fine-tune the model yourself  

## 🚀 Getting Started

### 1. Clone the repository
```bash
git clone https://github.com/OfficialAnasCoding/YOLOmodel.git
cd YOLOmodel
```
### 2. Install dependencies
```bash
pip install ultralytics opencv-python
```
### 3. Run program on test images
The program currently has some test car images for the model to predict on, but you are welcome to add/change the images in YOLO_Dataset/YOLOtesting
```bash
python3 YOLO_Testing.py
```
### 4. (Optional) train your own model
There is already a model trained on 80 epochs which you can use. However if you would like to try training your own you can. The models are trained on labelled images from the YOLO_Dataset folder
```bash
python3 YOLO_Train.py
```



