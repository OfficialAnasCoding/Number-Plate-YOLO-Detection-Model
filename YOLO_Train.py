from ultralytics import YOLO

# Loads a pre trained (knows general shapes of objects) model
model = YOLO("yolov8n.pt")

# Trains on training set, and shows it learning progress by comparing against the validation set
model.train(data="YOLO_Dataset/data.yaml", epochs=50, imgsz=512, device="cpu", verbose=True)

