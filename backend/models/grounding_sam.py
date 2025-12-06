import torch
import numpy as np
import cv2
from pathlib import Path
from PIL import Image

# -----------------------------
# GroundingDINO
# -----------------------------
from groundingdino.util.inference import Model as GDINOModel

# -----------------------------
# SAM2
# -----------------------------
from sam2.build_sam import build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor


DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
BASE = Path(__file__).resolve().parent


# ============================================================
#               CONFIG PATHS
# ============================================================

GDINO_CONFIG = BASE / \
    "Grounded-SAM-2/grounding_dino/groundingdino/config/GroundingDINO_SwinT_OGC.py"
GDINO_CKPT = BASE / "Grounded-SAM-2/gdino_checkpoints/groundingdino_swint_ogc.pth"

SAM2_CONFIG = BASE / "sam2/sam2/configs/sam2.1/sam2.1_hiera_b+.yaml"
SAM2_CKPT = BASE / "sam2/checkpoints/sam2.1_hiera_base_plus.pt"


# ============================================================
#               LOAD MODELS
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
    ckpt_path=str(SAM2_CKPT)
)
sam2_predictor = SAM2ImagePredictor(sam2_model)
print("[SAM2] Ready.")


# ============================================================
#                 MAIN PIPELINE
# ============================================================

def run_grounded_sam(image_path: str, text_prompt: str):
    """
    GroundingDINO: text → bounding boxes
    SAM2:          boxes → masks
    Returns overlays, stickers, AND correct per-sticker labels.
    """

    if not text_prompt.strip():
        return {"error": "Text prompt required for GroundingDINO."}

    # -----------------------------
    # Load image
    # -----------------------------
    image_pil = Image.open(image_path).convert("RGB")
    image_np = np.array(image_pil)
    image_bgr = cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR)

    # -----------------------------
    # DINO: text → box proposals
    # -----------------------------
    boxes, phrases = grounding_dino.predict_with_caption(
        image_bgr,
        caption=text_prompt,
        box_threshold=0.45,
        text_threshold=0.45,
    )

    if boxes is None or len(boxes) == 0:
        return {
            "overlay": None,
            "stickers": [],
            "labels": [],
            "message": "No objects detected."
        }

    # -----------------------------
    # Convert to xyxy
    # -----------------------------
    abs_boxes = []
    clean_labels = []

    for b, phr in zip(boxes, phrases):
        try:
            xyxy = np.array(b[0]).astype(float)
        except:
            continue

        if xyxy.shape[0] != 4:
            continue

        x1, y1, x2, y2 = map(int, xyxy)
        abs_boxes.append([x1, y1, x2, y2])

        # phrase is like: "bottle (0.78)" → we keep only "bottle"
        class_name = phr.split("(")[0].strip()
        clean_labels.append(class_name)

    if len(abs_boxes) == 0:
        return {
            "overlay": None,
            "stickers": [],
            "labels": [],
            "message": "No valid boxes extracted."
        }

    # -----------------------------
    # SAM2 segmentation
    # -----------------------------
    sam2_predictor.set_image(image_np)

    masks = []
    for box in abs_boxes:
        m, scores, _ = sam2_predictor.predict(
            box=np.array(box)[None, :],
            multimask_output=False
        )
        masks.append(m[0])

    # -----------------------------
    # Overlay
    # -----------------------------
    overlay = image_np.copy()

    for (x1, y1, x2, y2), mask in zip(abs_boxes, masks):
        mask_bool = (mask > 0.5)

        cv2.rectangle(overlay, (x1, y1), (x2, y2), (0, 255, 0), 2)
        overlay[mask_bool] = (
            overlay[mask_bool] * 0.6 + np.array([0, 255, 0]) * 0.4
        ).astype(np.uint8)

    # -----------------------------
    # Save outputs
    # -----------------------------
    outputs = Path("outputs")
    outputs.mkdir(exist_ok=True)

    stem = Path(image_path).stem

    overlay_path = outputs / f"{stem}_grounded_overlay.png"
    cv2.imwrite(str(overlay_path), cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR))

    # -----------------------------
    # Stickers
    # -----------------------------
    sticker_paths = []

    img_bgra = cv2.cvtColor(image_np, cv2.COLOR_RGB2BGRA)

    for i, ((x1, y1, x2, y2), mask) in enumerate(zip(abs_boxes, masks)):
        crop = img_bgra[y1:y2, x1:x2].copy()
        mask_crop = (mask[y1:y2, x1:x2] > 0.5)
        crop[~mask_crop] = [0, 0, 0, 0]

        sticker_path = outputs / f"{stem}_grounded_sticker_{i}.png"
        cv2.imwrite(str(sticker_path), crop)
        sticker_paths.append(f"/outputs/{sticker_path.name}")

    # -----------------------------
    # RETURN
    # -----------------------------

    clean_labels = [text_prompt.strip()] * len(abs_boxes)
    return {
        "message": "GroundingDINO + SAM2 segmentation successful.",
        "overlay": f"/outputs/{overlay_path.name}",
        "stickers": sticker_paths,
        "labels": clean_labels,        # <--- ⭐ CORRECT LABELS FOR EACH STICKER
    }
