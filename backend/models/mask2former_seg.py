from pathlib import Path
import cv2
import numpy as np
import torch
from PIL import Image

from transformers import AutoImageProcessor, Mask2FormerForUniversalSegmentation

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
MODEL_NAME = "facebook/mask2former-swin-base-coco-panoptic"

_processor = None
_model = None
_ID2LABEL = None
_CLASS_COLORS = None


def _load_mask2former():
    global _processor, _model, _ID2LABEL, _CLASS_COLORS

    if _processor is not None and _model is not None:
        return

    print(f"[Mask2Former] Loading {MODEL_NAME} on {DEVICE}...")
    _processor = AutoImageProcessor.from_pretrained(MODEL_NAME)
    _model = Mask2FormerForUniversalSegmentation.from_pretrained(
        MODEL_NAME).to(DEVICE)
    _model.eval()

    raw_id2label = _model.config.id2label
    # ensure integer keys
    if isinstance(list(raw_id2label.keys())[0], str):
        _ID2LABEL = {int(k): v for k, v in raw_id2label.items()}
    else:
        _ID2LABEL = raw_id2label

    num_classes = max(_ID2LABEL.keys()) + 1
    rng = np.random.default_rng(42)
    _CLASS_COLORS = (rng.random((num_classes, 3)) * 255).astype(np.uint8)

    print(f"[Mask2Former] Loaded. {len(_ID2LABEL)} classes.")


def run_mask2former_seg(image_path: str, score_thresh: float = 0.6):
    _load_mask2former()

    img_bgr = cv2.imread(image_path)
    if img_bgr is None:
        return {"error": f"Could not read image at {image_path}"}

    h, w = img_bgr.shape[:2]
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    pil_image = Image.fromarray(img_rgb)

    # Prepare inputs
    inputs = _processor(images=pil_image, return_tensors="pt")
    inputs = {k: v.to(DEVICE) for k, v in inputs.items()}

    with torch.no_grad():
        outputs = _model(**inputs)

    panoptic = _processor.post_process_panoptic_segmentation(
        outputs, target_sizes=[(h, w)]
    )[0]

    seg = panoptic["segmentation"].cpu().numpy().astype(np.int32)
    segments_info = panoptic["segments_info"]

    overlay = img_bgr.copy()
    img_bgra = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2BGRA)

    outputs_dir = Path("outputs")
    outputs_dir.mkdir(exist_ok=True)
    stem = Path(image_path).stem

    sticker_paths = []
    sticker_labels = []    # ⭐ NEW LABEL ARRAY

    for i, s in enumerate(segments_info):
        seg_id = s["id"]
        label_id = s["label_id"]
        score = float(s.get("score", 1.0))
        is_thing = bool(s.get("isthing", True))

        if score < score_thresh:
            continue
        if not is_thing:
            continue

        label = _ID2LABEL.get(label_id, f"class_{label_id}")

        mask_bool = (seg == seg_id)
        if mask_bool.sum() < 50:
            continue

        ys, xs = np.where(mask_bool)
        y1, y2 = int(ys.min()), int(ys.max())
        x1, x2 = int(xs.min()), int(xs.max())

        # reject tiny & huge bbox regions
        if (x2 - x1) < 5 or (y2 - y1) < 5:
            continue
        box_area = (x2 - x1) * (y2 - y1)
        if box_area > (h * w * 0.4):
            continue

        # overlay coloring
        color = _CLASS_COLORS[label_id % len(_CLASS_COLORS)].tolist()
        highlight = np.clip(np.array(color) * 1.3, 0, 255).astype(np.uint8)

        obj_region = overlay[mask_bool]
        overlay[mask_bool] = (0.6 * obj_region + 0.4 *
                              highlight).astype(np.uint8)

        # draw bbox + label
        cv2.rectangle(overlay, (x1, y1), (x2, y2), color, 2)

        label_text = f"{label} {int(score * 100)}%"
        (tw, th), _ = cv2.getTextSize(
            label_text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
        cv2.rectangle(
            overlay,
            (x1, max(0, y1 - th - 10)),
            (x1 + tw + 10, y1),
            color,
            cv2.FILLED,
        )
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

        # create sticker
        crop = img_bgra[y1:y2 + 1, x1:x2 + 1].copy()
        mask_crop = mask_bool[y1:y2 + 1, x1:x2 + 1]
        crop[~mask_crop] = [0, 0, 0, 0]

        sticker_path = outputs_dir / f"{stem}_m2f_sticker_{i}.png"
        cv2.imwrite(str(sticker_path), crop)

        sticker_paths.append(f"/outputs/{sticker_path.name}")
        sticker_labels.append(label)   # ⭐ ADD LABEL

    # no detections → behave like YOLO
    if len(sticker_paths) == 0:
        return {
            "message": "No objects detected",
            "overlay": None,
            "stickers": [],
            "labels": []
        }

    overlay_path = outputs_dir / f"{stem}_m2f_overlay.png"
    cv2.imwrite(str(overlay_path), overlay)

    return {
        "message": "Mask2Former panoptic segmentation successful",
        "overlay": f"/outputs/{overlay_path.name}",
        "stickers": sticker_paths,
        "labels": sticker_labels,   # ⭐ RETURN LABELS
    }
