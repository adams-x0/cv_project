from pathlib import Path
import cv2
import numpy as np

from ultralytics import RTDETR
from models.modules import load_sam2, refine_with_sam2, refine_mask_grabcut


# ------------------------------------------------------------
# COCO class names
# ------------------------------------------------------------
COCO_CLASSES = [
    "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck", "boat",
    "traffic light", "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat",
    "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella",
    "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard", "sports ball", "kite",
    "baseball bat", "baseball glove", "skateboard", "surfboard", "tennis racket", "bottle",
    "wine glass", "cup", "fork", "knife", "spoon", "bowl", "banana", "apple", "sandwich", "orange",
    "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair", "couch", "potted plant",
    "bed", "dining table", "toilet", "tv", "laptop", "mouse", "remote", "keyboard", "cell phone",
    "microwave", "oven", "toaster", "sink", "refrigerator", "book", "clock", "vase", "scissors",
    "teddy bear", "hair drier", "toothbrush"
]


# ------------------------------------------------------------
# Unique color per class
# ------------------------------------------------------------
def generate_class_colors(num):
    np.random.seed(42)
    return (np.random.rand(num, 3) * 255).astype(np.uint8)


CLASS_COLORS = generate_class_colors(len(COCO_CLASSES))


# ------------------------------------------------------------
# Load RT-DETR model
# ------------------------------------------------------------
RTDETR_MODEL_PATH = "rtdetr-x.pt"
detector = RTDETR(RTDETR_MODEL_PATH)


# ------------------------------------------------------------
# RT-DETR segmentation pipeline (with labels)
# ------------------------------------------------------------
def run_rtdetr_seg(image_path: str):

    load_sam2()  # Ensure SAM2 is loaded

    img_bgr = cv2.imread(image_path)
    if img_bgr is None:
        return {"error": f"Could not read image at {image_path}"}

    h, w = img_bgr.shape[:2]

    results = detector(image_path, verbose=False)
    r = results[0]

    boxes_xyxy = r.boxes.xyxy.cpu().numpy()
    confs = r.boxes.conf.cpu().numpy()
    class_ids = r.boxes.cls.cpu().numpy().astype(int)

    # print("DEBUG BOXES:", boxes_xyxy, "CLASSES:", class_ids)

    # No detections
    if len(boxes_xyxy) == 0:
        return {
            "message": "No objects detected",
            "overlay": None,
            "stickers": [],
            "labels": []
        }

    overlay = img_bgr.copy()
    img_bgra = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2BGRA)

    outputs_dir = Path("outputs")
    outputs_dir.mkdir(exist_ok=True)
    stem = Path(image_path).stem

    sticker_paths = []
    sticker_labels = []   # ⭐ NEW LIST FOR LABELS

    # ------------------------------------------------------------
    # Loop through detections (RT-DETR has no masks)
    # ------------------------------------------------------------
    for i in range(len(boxes_xyxy)):
        x1, y1, x2, y2 = boxes_xyxy[i].astype(int)
        cls = class_ids[i]
        conf = float(confs[i])
        class_name = COCO_CLASSES[cls] if 0 <= cls < len(
            COCO_CLASSES) else "object"

        # ------------------------------------------------------------
        # SAM2 segmentation from bounding box
        # ------------------------------------------------------------
        mask_bool = refine_with_sam2(img_bgr, (x1, y1, x2, y2))
        if mask_bool is None:
            continue

        # Colors
        color_bgr = CLASS_COLORS[cls].tolist()
        highlight_color = np.clip(
            CLASS_COLORS[cls] * 1.3, 0, 255).astype(np.uint8)

        # Overlay blend
        obj_region = overlay[mask_bool]
        overlay[mask_bool] = (0.6 * obj_region + 0.4 *
                              highlight_color).astype(np.uint8)

        # Bounding box
        cv2.rectangle(overlay, (x1, y1), (x2, y2), color_bgr, 2)

        label_text = f"{class_name}  {int(conf * 100)}%"
        (tw, th), _ = cv2.getTextSize(
            label_text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)

        cv2.rectangle(
            overlay,
            (x1, max(0, y1 - th - 10)),
            (x1 + tw + 10, y1),
            color_bgr,
            cv2.FILLED,
        )
        cv2.putText(
            overlay, label_text, (x1 + 5, y1 - 5),
            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2, cv2.LINE_AA
        )

        # ------------------------------------------------------------
        # Cropped transparent sticker
        # ------------------------------------------------------------
        crop = img_bgra[y1:y2, x1:x2].copy()
        mask_crop = mask_bool[y1:y2, x1:x2]
        crop[~mask_crop] = [0, 0, 0, 0]

        sticker_path = outputs_dir / f"{stem}_rtdetr_sticker_{i}.png"
        cv2.imwrite(str(sticker_path), crop)

        sticker_paths.append(f"/outputs/{sticker_path.name}")
        sticker_labels.append(class_name)   # ⭐ STORE CLASS NAME

    # Save overlay
    overlay_path = outputs_dir / f"{stem}_rtdetr_overlay.png"
    cv2.imwrite(str(overlay_path), overlay)

    return {
        "message": "RT-DETR + SAM2 segmentation successful",
        "overlay": f"/outputs/{overlay_path.name}",
        "stickers": sticker_paths,
        "labels": sticker_labels,     # ⭐ RETURN LABELS
    }
