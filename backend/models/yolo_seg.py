from pathlib import Path
import cv2
import numpy as np
from ultralytics import YOLO  # YOLOv12 engine (installed via pip install -e .)

# ------------------------------------------------------------
# COCO CLASS NAMES (80 classes)
# ------------------------------------------------------------
COCO_CLASSES = [
    "person","bicycle","car","motorcycle","airplane","bus","train","truck","boat",
    "traffic light","fire hydrant","stop sign","parking meter","bench","bird","cat",
    "dog","horse","sheep","cow","elephant","bear","zebra","giraffe","backpack","umbrella",
    "handbag","tie","suitcase","frisbee","skis","snowboard","sports ball","kite",
    "baseball bat","baseball glove","skateboard","surfboard","tennis racket","bottle",
    "wine glass","cup","fork","knife","spoon","bowl","banana","apple","sandwich","orange",
    "broccoli","carrot","hot dog","pizza","donut","cake","chair","couch","potted plant",
    "bed","dining table","toilet","tv","laptop","mouse","remote","keyboard","cell phone",
    "microwave","oven","toaster","sink","refrigerator","book","clock","vase","scissors",
    "teddy bear","hair drier","toothbrush"
]

# ------------------------------------------------------------
# Generate UNIQUE COLOR for each class (BGR format)
# ------------------------------------------------------------
def generate_class_colors(num_classes):
    np.random.seed(42)  # consistent colors every time
    return (np.random.rand(num_classes, 3) * 255).astype(np.uint8)

CLASS_COLORS = generate_class_colors(len(COCO_CLASSES))


# ------------------------------------------------------------
# Load YOLOv12x-Seg model
# ------------------------------------------------------------
MODEL_PATH = "yolov12x-seg.pt"
model = YOLO(str(MODEL_PATH))


def run_yolo_seg(image_path: str):
    """
    YOLOv12x-seg pipeline:
      ✔ Unique per-class overlay color
      ✔ Unique per-class bounding box
      ✔ Class name + confidence
      ✔ Individual transparent stickers
    """
    img_bgr = cv2.imread(image_path)
    if img_bgr is None:
        return {"error": f"Could not read image at {image_path}"}

    h, w = img_bgr.shape[:2]

    results = model(image_path, verbose=False)
    if not results:
        return {"error": "No YOLO results returned"}

    r = results[0]

    if r.masks is None or r.masks.data is None:
        return {"error": "No segmentation masks detected"}

    masks = r.masks.data.cpu().numpy()
    boxes_xyxy = r.boxes.xyxy.cpu().numpy()
    confs = r.boxes.conf.cpu().numpy()
    class_ids = r.boxes.cls.cpu().numpy().astype(int)

    # overlay base
    overlay = img_bgr.copy()

    # BGRA image for stickers
    img_bgra = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2BGRA)

    # output paths
    outputs_dir = Path("outputs")
    outputs_dir.mkdir(exist_ok=True)
    stem = Path(image_path).stem
    sticker_paths = []

    # ------------------------------------------------------------
    # Loop over detections
    # ------------------------------------------------------------
    for i, mask in enumerate(masks):

        mask_bool = mask > 0.5
        if mask_bool.shape != (h, w):
            mask_bool = cv2.resize(
                mask_bool.astype(np.uint8), (w, h), interpolation=cv2.INTER_NEAREST
            ).astype(bool)

        cls = class_ids[i]
        class_name = COCO_CLASSES[cls] if 0 <= cls < len(COCO_CLASSES) else "object"
        conf = float(confs[i])

        # unique class color
        color_bgr = CLASS_COLORS[cls].tolist()  # e.g. [12, 200, 50]

        # highlight color (slightly brighter)
        highlight_color = np.clip(CLASS_COLORS[cls] * 1.3, 0, 255).astype(np.uint8)

        # blend overlay
        obj_region = overlay[mask_bool]
        blended = (0.6 * obj_region + 0.4 * highlight_color).astype(np.uint8)
        overlay[mask_bool] = blended

        # bounding box
        x1, y1, x2, y2 = boxes_xyxy[i].astype(int)
        cv2.rectangle(overlay, (x1, y1), (x2, y2), color_bgr, 2)

        # text label
        label_text = f"{class_name}  {int(conf * 100)}%"

        (tw, th), baseline = cv2.getTextSize(
            label_text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2
        )

        # text background
        cv2.rectangle(
            overlay,
            (x1, max(0, y1 - th - 10)),
            (x1 + tw + 10, y1),
            color_bgr,
            thickness=cv2.FILLED,
        )

        # text
        cv2.putText(
            overlay,
            label_text,
            (x1 + 5, y1 - 5),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 0, 0),
            2,
            cv2.LINE_AA,
        )

        # ------------------------------------------------------------
        # Create cropped sticker (transparent)
        # ------------------------------------------------------------
        crop = img_bgra[y1:y2, x1:x2].copy()
        mask_crop = mask_bool[y1:y2, x1:x2]
        crop[~mask_crop] = [0, 0, 0, 0]

        sticker_path = outputs_dir / f"{stem}_sticker_{i}.png"
        cv2.imwrite(str(sticker_path), crop)
        sticker_paths.append(f"/outputs/{sticker_path.name}")

    # save overlay
    overlay_path = outputs_dir / f"{stem}_overlay.png"
    cv2.imwrite(str(overlay_path), overlay)

    return {
        "message": "YOLOv12 segmentation successful",
        "overlay": f"/outputs/{overlay_path.name}",
        "stickers": sticker_paths,
    }
