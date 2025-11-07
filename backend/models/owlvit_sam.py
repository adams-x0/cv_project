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
sam.to("cpu")  # change to "cuda" if using GPU
predictor = SamPredictor(sam)


def run_owlvit_sam(image_path, prompt):
    # 1️⃣ Load image
    image = Image.open(image_path).convert("RGB")
    inputs = processor(text=[prompt], images=image, return_tensors="pt")
    outputs = owlvit_model(**inputs)

    # 2️⃣ Detect objects
    target_sizes = torch.tensor([image.size[::-1]])
    results = processor.post_process_object_detection(
        outputs=outputs, threshold=0.2, target_sizes=target_sizes
    )[0]
    boxes = results["boxes"]

    if len(boxes) == 0:
        return {"error": "No object detected"}

    # 3️⃣ SAM segmentation
    img_cv = cv2.imread(image_path)
    img_cv = cv2.cvtColor(img_cv, cv2.COLOR_BGR2RGB)
    predictor.set_image(img_cv)

    mask_union = np.zeros(img_cv.shape[:2], dtype=bool)
    for box in boxes:
        box = box.detach().numpy()
        mask, _, _ = predictor.predict(
            point_coords=None, point_labels=None, box=box[None, :], multimask_output=False
        )
        mask_union = np.logical_or(mask_union, mask[0])

    # 4️⃣ Create green overlay (existing feature)
    overlay = img_cv.copy()
    overlay[mask_union] = [0, 255, 0]
    output_overlay = Path("outputs") / (Path(image_path).stem + "_overlay.png")
    cv2.imwrite(str(output_overlay), cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR))

    # ✅ 5️⃣ Create STICKER (cropped + transparent PNG)
    sticker = img_cv.copy()
    sticker = cv2.cvtColor(sticker, cv2.COLOR_RGB2RGBA)  # Add alpha channel
    sticker[~mask_union] = [0, 0, 0, 0]  # Make background transparent
    output_sticker = Path("outputs") / (Path(image_path).stem + "_sticker.png")
    cv2.imwrite(str(output_sticker), cv2.cvtColor(sticker, cv2.COLOR_RGBA2BGRA))

    return {
        "message": "Success",
        "overlay": f"/outputs/{output_overlay.name}",
        "sticker": f"/outputs/{output_sticker.name}"
    }
