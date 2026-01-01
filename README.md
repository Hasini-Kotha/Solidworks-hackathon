
# Overview

The task focuses on multi-class object detection and counting of mechanical components:
- Bolt
- Locating Pin
- Nut
- Washer

We employ a YOLOv8  architecture trained from scratch, followed by a carefully tuned inference and post-processing pipeline to generate the final submission.

---

# Key Highlights

- Training from scratch** (`yolov8n.yaml`) — no pretrained weights
- Exact hyperparameters preserved (training & inference)
- Memory-safe inference loop (GPU cleanup per image)
- Class-specific confidence thresholds
  


# The accuracy of the model is 0.9914285
