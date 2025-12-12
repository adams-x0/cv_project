from ultralytics import YOLO

# Load a pretrained YOLOv12 model (choose one: n, s, m, l, or x)
# model = YOLO("custom/yolov12x-seg.pt")  # 'n' is the smallest and fastest

model = YOLO("yolov12x-seg.pt")

# Run prediction on an image or video
results = model.predict(
    source="image2.jpg",
    save=True,
    save_crop=True,
    conf=0.5,
    # retina_masks=True,
)
