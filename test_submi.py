from ultralytics import YOLO
import os
import pandas as pd
import gc
import torch

MODEL_PATH = "/kaggle/working/runs/detect/train/weights/best.pt"  
TEST_IMG_DIR = "/kaggle/input/solidworks-ai-hackathon/test/test"

model = YOLO(MODEL_PATH)

CONF_TH = 0.2
IOU_TH = 0.55

CLASS_CONF = {
    0: 0.6,
    1: 0.6,
    2: 0.55,
    3: 0.65
}

results = []

image_names = sorted(os.listdir(TEST_IMG_DIR))
total = len(image_names)

for i, img_name in enumerate(image_names):
    img_path = os.path.join(TEST_IMG_DIR, img_name)

    pred = model(img_path, conf=CONF_TH, iou=IOU_TH, verbose=False)[0]

    counts = {0: 0, 1: 0, 2: 0, 3: 0}

    if pred.boxes is not None:
        for cls, conf in zip(pred.boxes.cls, pred.boxes.conf):
            cls = int(cls)
            if conf >= CLASS_CONF[cls]:
                counts[cls] += 1

    results.append([
        img_name,
        counts[0],
        counts[1],
        counts[2],
        counts[3]
    ])

    # free memory every iteration
    del pred
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    # progress print 
    if (i + 1) % 50 == 0:
        print(f"Processed {i+1}/{total}")

submission = pd.DataFrame(
    results,
    columns=["image_name", "bolt", "locatingpin", "nut", "washer"]
)

submission_path = "/kaggle/working/submission.csv"
submission.to_csv(submission_path, index=False)

print("submission.csv generated")
print(submission.head())
