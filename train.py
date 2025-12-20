from ultralytics import YOLO

model = YOLO("yolov8n.yaml")  

model.train(
    data="data.yaml",

    epochs=25,                 
    imgsz=768,                
    batch=16,
    device="cuda",

    optimizer="AdamW",        
    lr0=0.001,                
    lrf=0.01,                
    cos_lr=True,              

    weight_decay=0.001,       
    momentum=0.9,

    patience=10,              

    augment=True,             
    mosaic=0.3,               
    mixup=0.05,               
    close_mosaic=10,         

    label_smoothing=0.0,

    save=True,
    verbose=True
)
