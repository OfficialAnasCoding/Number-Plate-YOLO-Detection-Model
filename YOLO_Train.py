from ultralytics import YOLO
from pathlib import Path

# ---------------- PATH SETUP ---------------- #
ROOT = Path(__file__).resolve().parent

MODEL_PATH = ROOT / "yolov8n.pt"
DATA_YAML = ROOT / "YOLO_Dataset" / "data.yaml"

# ---------------- LOAD MODEL ---------------- #
model = YOLO(MODEL_PATH)

# ---------------- TRAIN ---------------- #
model.train(
    data=DATA_YAML,
    epochs=50,
    imgsz=512,
    device="cpu",
    verbose=True
)
