import torch
import numpy as np
import cv2
from pathlib import Path
from PIL import Image
from transformers import OwlViTProcessor, OwlViTForObjectDetection

torch.utils.checkpoint.use_reentrant = False

# -------------------------------
# SAM2 imports
# -------------------------------
from sam2.build_sam import build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

BASE = Path(__file__).resolve().parent

# -------------------------------
# SAM2 MODEL + CONFIG PATHS
# -------------------------------
SAM2_CONFIG = BASE / "sam2/sam2/configs/sam2.1/sam2.1_hiera_b+.yaml"
SAM2_WEIGHTS = BASE / "sam2/checkpoints/sam2.1_hiera_base_plus.pt"

print("[SAM2] Loading...")

sam2_model = build_sam2(
    config_file=str(SAM2_CONFIG),
    ckpt_path=str(SAM2_WEIGHTS)
).to(DEVICE)

sam2_predictor = SAM2ImagePredictor(sam2_model)

print("[SAM2] Ready.")


# -------------------------------
# Load OWL-ViT
# -------------------------------
processor = OwlViTProcessor.from_pretrained("google/owlvit-base-patch32")
owlvit_model = OwlViTForObjectDetection.from_pretrained(
    "google/owlvit-base-patch32"
).to(DEVICE)


# -------------------------------------------------------------------
# Helper: run OWL-ViT → get best bounding box for the given prompt
# -------------------------------------------------------------------
def owlvit_detect(image_path: str, text_prompt: str):
    image = Image.open(image_path).convert("RGB")
    inputs = processor(text=[text_prompt], images=image,
                       return_tensors="pt").to(DEVICE)

    with torch.no_grad():
        outputs = owlvit_model(**inputs)

    target_sizes = torch.tensor([image.size[::-1]]).to(DEVICE)
    results = processor.post_process_object_detection(
        outputs, threshold=0.03, target_sizes=target_sizes
    )[0]

    if len(results["scores"]) == 0:
        return None

    idx = results["scores"].argmax().item()
    score = results["scores"][idx].item()
    box = results["boxes"][idx].cpu().numpy().astype(int)

    return box, score


# -------------------------------------------------------------------
# Main function: OWL-ViT + SAM2 segmentation
# -------------------------------------------------------------------
def run_owlvit_sam(image_path: str, text_prompt: str):
    result = owlvit_detect(image_path, text_prompt)

    if result is None:
        return {
            "overlay": None,
            "stickers": [],
            "labels": [text_prompt],
            "message": "Object not found."
        }

    box_xyxy, score = result

    img_bgr = cv2.imread(image_path)
    h, w = img_bgr.shape[:2]

    # ---------------------------
    # Use SAM2 for segmentation
    # ---------------------------
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    sam2_predictor.set_image(img_rgb)

    masks, scores, logits = sam2_predictor.predict(
        box=box_xyxy[None, :],
        multimask_output=False
    )

    if masks is None or len(masks) == 0:
        return {
            "overlay": None,
            "stickers": [],
            "labels": [text_prompt],
            "message": "SAM2 produced no masks."
        }

    # Normalize mask (support tensor or numpy)
    mask = masks[0]
    if isinstance(mask, torch.Tensor):
        mask = (mask > 0.5).cpu().numpy()
    else:
        mask = (mask > 0.5).astype(bool)

    # ---------------------------
    # Draw overlay with bounding box
    # ---------------------------
    overlay = img_bgr.copy()
    color = (0, 255, 0)

    x1, y1, x2, y2 = box_xyxy
    cv2.rectangle(overlay, (x1, y1), (x2, y2), color, 2)

    obj_region = overlay[mask]
    blended = (0.6 * obj_region + 0.4 * np.array(color)).astype(np.uint8)
    overlay[mask] = blended

    # ---------------------------
    # Crop sticker (transparent)
    # ---------------------------
    img_bgra = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2BGRA)
    crop = img_bgra[y1:y2, x1:x2].copy()

    mask_crop = mask[y1:y2, x1:x2]
    crop[~mask_crop] = [0, 0, 0, 0]

    # Save outputs
    outputs = Path("outputs")
    outputs.mkdir(exist_ok=True)

    stem = Path(image_path).stem

    overlay_path = outputs / f"{stem}_overlay.png"
    sticker_path = outputs / f"{stem}_sticker.png"

    cv2.imwrite(str(overlay_path), overlay)
    cv2.imwrite(str(sticker_path), crop)

    return {
        "overlay": f"/outputs/{overlay_path.name}",
        "stickers": [f"/outputs/{sticker_path.name}"],
        "labels": [text_prompt],
        "message": "OWL-ViT + SAM2 segmentation successful."
    }
