from ultralytics import YOLO
import cv2
import os

# ---------------- CONFIG ---------------- #
# Load trained YOLO model
model = YOLO("/Users/anasibnsaifullah/Desktop/NumberPlateDetector/runs/detect/train6/weights/best.pt")

# Input folder (full car images)
test_folder = "YOLO_dataset/YOLOtesting"

# Confidence threshold
CONF_THRESHOLD = 0.25

# ---------------- RUN INFERENCE ---------------- #
for filename in os.listdir(test_folder):
    if not filename.lower().endswith((".jpg", ".jpeg", ".png")):
        continue

    img_path = os.path.join(test_folder, filename)
    img = cv2.imread(img_path)

    if img is None:
        continue

    # Run YOLO inference
    results = model(img, conf=CONF_THRESHOLD)

    boxes = results[0].boxes
    if boxes is None or len(boxes) == 0:
        print(f"No plate detected in {filename}")
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

    # Show the image
    cv2.imshow("YOLO Plate Detection", img)
    key = cv2.waitKey(0)  # Press any key to move to next image
    if key == 27:          # ESC to quit
        break

cv2.destroyAllWindows()
print("Done displaying all plates.")
