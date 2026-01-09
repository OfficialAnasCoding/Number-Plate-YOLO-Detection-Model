from ultralytics import YOLO
import cv2
from pathlib import Path

# ---------------- PATH SETUP ---------------- #
ROOT = Path(__file__).resolve().parent  # folder where this script lives

MODEL_PATH = ROOT / "runs" / "detect" / "train7" / "weights" / "best.pt"
TEST_FOLDER = ROOT / "YOLO_dataset" / "YOLOtesting"

# ---------------- CONFIG ---------------- #
model = YOLO(MODEL_PATH)
CONF_THRESHOLD = 0.25

# ---------------- RUN INFERENCE ---------------- #
for img_path in TEST_FOLDER.iterdir():
    if img_path.suffix.lower() not in [".jpg", ".jpeg", ".png"]:
        continue

    img = cv2.imread(str(img_path))
    if img is None:
        continue

    results = model(img, conf=CONF_THRESHOLD)
    boxes = results[0].boxes

    if boxes is None or len(boxes) == 0:
        print(f"No plate detected in {img_path.name}")
        continue

    # ---------------- SELECT HIGHEST CONFIDENCE BOX ---------------- #
    best_box = max(boxes, key=lambda b: float(b.conf[0]))

    x1, y1, x2, y2 = map(int, best_box.xyxy[0])
    conf = float(best_box.conf[0])

    # Safety clamp
    h, w = img.shape[:2]
    x1, y1 = max(0, x1), max(0, y1)
    x2, y2 = min(w, x2), min(h, y2)

    # Draw bounding box
    cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
    label = f"Plate {conf:.2f}"
    cv2.putText(img, label, (x1, y1 - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

    cv2.imshow("YOLO Plate Detection", img)
    if cv2.waitKey(0) == 27:  # ESC to quit
        break

cv2.destroyAllWindows()
print("Done displaying all plates.")
