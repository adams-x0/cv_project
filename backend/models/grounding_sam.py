from pathlib import Path
import cv2
import numpy as np
import torch
from PIL import Image, ImageOps

from groundingdino.util.inference import Model as GDINOModel
from sam2.build_sam import build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor


# ============================================================
# GLOBAL SETTINGS & FIXES
# ============================================================

torch.utils.checkpoint.use_reentrant = False   # remove checkpoint warning


def autocast():
    return torch.amp.autocast("cuda")


DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
BASE = Path(__file__).resolve().parent


# ============================================================
# CONFIG PATHS
# ============================================================

GDINO_CONFIG = BASE / \
    "Grounded-SAM-2/grounding_dino/groundingdino/config/GroundingDINO_SwinT_OGC.py"
GDINO_CKPT = BASE / "Grounded-SAM-2/gdino_checkpoints/groundingdino_swint_ogc.pth"

SAM2_CONFIG = BASE / "sam2/sam2/configs/sam2.1/sam2.1_hiera_b+.yaml"
SAM2_CKPT = BASE / "sam2/checkpoints/sam2.1_hiera_base_plus.pt"


# ============================================================
# LOAD MODELS
# ============================================================

print("[GroundingDINO] Loading...")
grounding_dino = GDINOModel(
    model_config_path=str(GDINO_CONFIG),
    model_checkpoint_path=str(GDINO_CKPT),
    device=DEVICE
)
print("[GroundingDINO] Loaded.")

print("[SAM2] Loading...")
sam2_model = build_sam2(
    config_file=str(SAM2_CONFIG),
    ckpt_path=str(SAM2_CKPT),
)
sam2_model.to(DEVICE)
sam2_predictor = SAM2ImagePredictor(sam2_model)
print("[SAM2] Ready.]")


# ============================================================
# MAIN PIPELINE — PORTRAIT FIXED
# ============================================================

def run_grounded_sam(image_path: str, text_prompt: str):
    if not text_prompt.strip():
        return {"error": "Text prompt required for GroundingDINO."}

    # -------------------------------------------
    # Load image (FIX PORTRAIT ROTATION)
    # -------------------------------------------
    # EXIF transpose makes portrait images correct BEFORE OpenCV sees them.
    image_pil = ImageOps.exif_transpose(Image.open(image_path)).convert("RGB")
    image_np = np.ascontiguousarray(np.array(image_pil))  # no rotation issues

    orig_h, orig_w = image_np.shape[:2]
    image_bgr = cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR)

    # -------------------------------------------
    # GroundingDINO: text -> bounding boxes
    # -------------------------------------------
    with torch.no_grad():
        boxes, phrases = grounding_dino.predict_with_caption(
            image_bgr,
            caption=text_prompt,
            box_threshold=0.45,
            text_threshold=0.45,
        )

    if boxes is None or len(boxes) == 0:
        return {"overlay": None, "stickers": [], "labels": [], "message": "No objects detected."}

    abs_boxes = []
    for b in boxes:
        xyxy = np.array(b[0]).astype(float)
        if xyxy.shape[0] == 4:
            abs_boxes.append(xyxy.astype(int).tolist())

    if not abs_boxes:
        return {"overlay": None, "stickers": [], "labels": [], "message": "No valid boxes extracted."}

    # -------------------------------------------
    # SAM2 segmentation
    # -------------------------------------------
    sam2_predictor.set_image(image_np)

    masks = []
    for box in abs_boxes:
        with torch.no_grad():
            with autocast():
                m, _, _ = sam2_predictor.predict(
                    box=np.array(box)[None, :],
                    multimask_output=False
                )

        mask = m[0].squeeze()

        # Ensure the mask matches correct shape
        if mask.shape != (orig_h, orig_w):
            mask = cv2.resize(mask.astype(np.float32), (orig_w, orig_h))

        masks.append(mask)

    # -------------------------------------------
    # Build overlay
    # -------------------------------------------
    overlay = image_np.copy()

    for (x1, y1, x2, y2), mask in zip(abs_boxes, masks):
        mask_bool = mask > 0.5
        cv2.rectangle(overlay, (x1, y1), (x2, y2), (0, 255, 0), 2)

        overlay[mask_bool] = (
            overlay[mask_bool] * 0.6 + np.array([0, 255, 0]) * 0.4
        ).astype(np.uint8)

    outputs = Path("outputs")
    outputs.mkdir(exist_ok=True)

    stem = Path(image_path).stem
    overlay_path = outputs / f"{stem}_grounded_overlay.png"
    cv2.imwrite(str(overlay_path), cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR))

    # -------------------------------------------
    # Stickers (transparent PNGs)
    # -------------------------------------------
    stickers = []
    img_bgra = cv2.cvtColor(image_np, cv2.COLOR_RGB2BGRA)

    for i, ((x1, y1, x2, y2), mask) in enumerate(zip(abs_boxes, masks)):
        crop = img_bgra[y1:y2, x1:x2].copy()
        mask_crop = mask[y1:y2, x1:x2] > 0.5
        crop[~mask_crop] = [0, 0, 0, 0]

        path = outputs / f"{stem}_grounded_sticker_{i}.png"
        cv2.imwrite(str(path), crop)
        stickers.append(f"/outputs/{path.name}")

    return {
        "message": "GroundingDINO + SAM2 segmentation successful.",
        "overlay": f"/outputs/{overlay_path.name}",
        "stickers": stickers,
        "labels": [text_prompt] * len(stickers),
    }
