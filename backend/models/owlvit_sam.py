import torch
from pathlib import Path
import cv2
import numpy as np
from transformers import OwlViTProcessor, OwlViTForObjectDetection
from segment_anything import sam_model_registry, SamPredictor
from PIL import Image

# Load OWL-ViT
processor = OwlViTProcessor.from_pretrained("google/owlvit-base-patch32")
owlvit_model = OwlViTForObjectDetection.from_pretrained("google/owlvit-base-patch32")

# Load SAM
sam = sam_model_registry["vit_b"](checkpoint="sam_vit_b_01ec64.pth")
sam.to("cpu")  # change to "cuda" if you have GPU
predictor = SamPredictor(sam)


def run_owlvit_sam(image_path: str, prompt: str, score_thresh: float = 0.2):
    # 1) Load image (PIL + OpenCV)
    pil_image = Image.open(image_path).convert("RGB")
    img_cv = cv2.imread(image_path)
    if img_cv is None:
        return {"error": f"Could not read image at {image_path}"}
    img_cv = cv2.cvtColor(img_cv, cv2.COLOR_BGR2RGB)
    h, w = img_cv.shape[:2]

    # 2) OWL-ViT detection
    inputs = processor(text=[prompt], images=pil_image, return_tensors="pt")
    with torch.no_grad():
        outputs = owlvit_model(**inputs)

    target_sizes = torch.tensor([pil_image.size[::-1]])  # (H, W)
    results = processor.post_process_object_detection(
        outputs=outputs, threshold=score_thresh, target_sizes=target_sizes
    )[0]

    boxes = results["boxes"]        # (N, 4)
    scores = results["scores"]      # (N,)

    if len(boxes) == 0:
        return {"error": "No object detected by OWL-ViT"}

    boxes_np = boxes.detach().numpy()
    scores_np = scores.detach().numpy()

    # 3) Prepare SAM
    predictor.set_image(img_cv)

    # overlay + BGRA version
    overlay = img_cv.copy()
    highlight_color = np.array([0, 255, 0], dtype=np.uint8)  # green tint
    img_bgra = cv2.cvtColor(cv2.cvtColor(img_cv, cv2.COLOR_RGB2BGR), cv2.COLOR_BGR2BGRA)

    outputs_dir = Path("outputs")
    outputs_dir.mkdir(exist_ok=True)
    stem = Path(image_path).stem

    stickers_relpaths = []

    for i, (box, score) in enumerate(zip(boxes_np, scores_np)):
        # SAM expects (x1, y1, x2, y2)
        box_xyxy = box.astype(np.float32)
        x1, y1, x2, y2 = box_xyxy.astype(int)

        # Clip to image bounds
        x1 = max(0, min(x1, w - 1))
        x2 = max(0, min(x2, w))
        y1 = max(0, min(y1, h - 1))
        y2 = max(0, min(y2, h))

        # 4) SAM segmentation for this box
        masks, _, _ = predictor.predict(
            point_coords=None,
            point_labels=None,
            box=box_xyxy[None, :],
            multimask_output=False,
        )
        mask = masks[0] > 0.5  # (H, W) boolean

        # Ensure mask size matches image size
        if mask.shape != (h, w):
            mask = cv2.resize(
                mask.astype(np.uint8), (w, h), interpolation=cv2.INTER_NEAREST
            ).astype(bool)

        # ---------------------------
        # 1) Semi-transparent overlay
        # ---------------------------
        obj_region = overlay[mask]
        blended = (0.6 * obj_region + 0.4 * highlight_color).astype(np.uint8)
        overlay[mask] = blended

        # draw bounding box
        cv2.rectangle(overlay, (x1, y1), (x2, y2), (0, 255, 0), 2)

        # confidence label (e.g., "92%")
        label = f"{int(float(score) * 100)}%"

        (tw, th), baseline = cv2.getTextSize(
            label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1
        )
        cv2.rectangle(
            overlay,
            (x1, max(y1 - th - 4, 0)),
            (x1 + tw + 4, y1),
            (0, 255, 0),
            thickness=cv2.FILLED,
        )
        cv2.putText(
            overlay,
            label,
            (x1 + 2, y1 - 2),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (0, 0, 0),
            thickness=1,
            lineType=cv2.LINE_AA,
        )

        # ---------------------------
        # 2) Cropped transparent sticker
        # ---------------------------
        crop = img_bgra[y1:y2, x1:x2].copy()
        mask_crop = mask[y1:y2, x1:x2]

        crop[~mask_crop] = [0, 0, 0, 0]
        sticker_path = outputs_dir / f"{stem}_owl_sticker_{i}.png"
        cv2.imwrite(str(sticker_path), crop)
        stickers_relpaths.append(f"/outputs/{sticker_path.name}")

    # save overlay (convert back to BGR for saving)
    overlay_bgr = cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR)
    overlay_path = outputs_dir / f"{stem}_owl_overlay.png"
    cv2.imwrite(str(overlay_path), overlay_bgr)

    return {
        "message": "OWL-ViT + SAM segmentation success",
        "overlay": f"/outputs/{overlay_path.name}",
        "stickers": stickers_relpaths,
    }
