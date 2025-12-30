# Run this script after the blender script.
# Ensure correct folder locations.

import pandas as pd
import cv2
from ultralytics import YOLO
from pathlib import Path

# -------- CONFIG --------
CSV_PATH = "poses.csv"
IMAGE_DIR = Path("Images")
MODEL_PATH = "Coral-640-640-yolov11n.pt"
OUTPUT_CSV = "poses_detected.csv"

DEBUG_DIR = Path("Debug")
DET_DIR = DEBUG_DIR / "Detections"
NO_DET_DIR = DEBUG_DIR / "NoDetections"

CONF_THRESH = 0.01
# ------------------------

DET_DIR.mkdir(parents=True, exist_ok=True)
NO_DET_DIR.mkdir(parents=True, exist_ok=True)

model = YOLO(MODEL_PATH)
df = pd.read_csv(CSV_PATH)

df["cx"] = None
df["cy"] = None
df["w"] = None
df["h"] = None

for i, row in df.iterrows():
    img_path = f"{IMAGE_DIR}/{row['image']}"
    img = cv2.imread(str(img_path))

    if img is None:
        print(f"Missing image: {img_path}")
        continue

    h_img, w_img, _ = img.shape
    vis = img.copy()

    results = model(img, conf=CONF_THRESH, verbose=False)
    boxes = results[0].boxes

    if boxes is None or len(boxes) == 0:
        cv2.putText(
            vis,
            "NO DETECTION",
            (30, 50),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.2,
            (0, 0, 255),
            3
        )
        cv2.imwrite(str(NO_DET_DIR / row["image"]), vis)
        continue

    best = boxes[boxes.conf.argmax()]
    x1, y1, x2, y2 = best.xyxy[0].cpu().numpy()
    conf = float(best.conf[0])

    cv2.rectangle(
        vis,
        (int(x1), int(y1)),
        (int(x2), int(y2)),
        (0, 255, 0),
        2
    )
    cv2.putText(
        vis,
        f"{conf:.2f}",
        (int(x1), int(y1) - 10),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        (0, 255, 0),
        2
    )

    cv2.imwrite(str(DET_DIR / row["image"]), vis)

    # Normalized bbox
    cx = ((x1 + x2) / 2) / w_img
    cy = ((y1 + y2) / 2) / h_img
    w  = (x2 - x1) / w_img
    h  = (y2 - y1) / h_img

    df.at[i, "cx"] = cx
    df.at[i, "cy"] = cy
    df.at[i, "w"]  = w
    df.at[i, "h"]  = h

    if i % 500 == 0:
        print(f"Processed {i}/{len(df)}")

df = df.dropna()
df.to_csv(OUTPUT_CSV, index=False)

print("Done.")
