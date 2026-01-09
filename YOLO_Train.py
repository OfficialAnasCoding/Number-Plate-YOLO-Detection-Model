from ultralytics import YOLO

model = YOLO("yolov8n.pt")

model.train(data="/Users/anasibnsaifullah/Desktop/NumberPlateDetector/YOLO_model/YOLO_dataset/data.yaml", epochs=80, imgsz=512, device="mps", verbose=True)

