import torch
import cv2
import numpy as np


def refine_mask_grabcut(crop_bgra: np.ndarray, mask_bool: np.ndarray, iterations: int = 3):
    """
    Refine a segmentation mask using GrabCut.
    
    Parameters
    ----------
    crop_bgra : np.ndarray
        Cropped RGBA region (H x W x 4).
    mask_bool : np.ndarray
        Boolean mask for the crop (H x W).
    iterations : int
        How many GrabCut refinement iterations to run.

    Returns
    -------
    refined_crop : np.ndarray
        New RGBA crop with a cleaner boundary.
    refined_mask : np.ndarray
        Boolean refined mask.
    """

    # Clone crop
    crop = crop_bgra.copy()

    # Create GrabCut mask
    # GC_PR_FGD = probably foreground
    # GC_PR_BGD = probably background
    gc_mask = np.where(mask_bool, cv2.GC_PR_FGD, cv2.GC_PR_BGD).astype("uint8")

    bgModel = np.zeros((1, 65), np.float64)
    fgModel = np.zeros((1, 65), np.float64)

    # Run GrabCut using mask
    cv2.grabCut(
        crop[:, :, :3],   # BGR
        gc_mask,
        None,             # No rectangle mode
        bgModel,
        fgModel,
        iterations,
        cv2.GC_INIT_WITH_MASK
    )

    # Final refined mask
    refined_mask = np.where(
        (gc_mask == cv2.GC_FGD) | (gc_mask == cv2.GC_PR_FGD),
        True,
        False
    )

    # Apply mask to get RGBA result
    refined_crop = crop.copy()
    refined_crop[~refined_mask] = [0, 0, 0, 0]

    return refined_crop, refined_mask


# Try importing SAM2
try:
    from sam2.build_sam import build_sam2
    from sam2.sam2_image_predictor import SAM2ImagePredictor
    SAM2_IMPORTED = True
except Exception as e:
    print("[SAM2] Import failed:", e)
    SAM2_IMPORTED = False

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


# ------------------------------------------------------------
# Initialize SAM2 (only once)
# ------------------------------------------------------------
sam2_predictor = None
SAM2_READY = False


def load_sam2():
    from pathlib import Path
    global sam2_predictor, SAM2_READY

    BASE = Path(__file__).resolve().parent  # backend/models

    config_path = str(
        (BASE / "sam2/sam2/configs/sam2.1/sam2.1_hiera_b+.yaml").resolve())
    checkpoint_path = str(
        (BASE / "sam2/checkpoints/sam2.1_hiera_base_plus.pt").resolve())

    try:
        sam_model = build_sam2(config_path, checkpoint_path, device=DEVICE)
        sam2_predictor_local = SAM2ImagePredictor(sam_model)
        sam2_predictor = sam2_predictor_local
        SAM2_READY = True
        print(
            f"[SAM2] Loaded with:\n  config={config_path}\n  weights={checkpoint_path}")
    except Exception as e:
        print("[SAM2] Failed to load SAM2:", e)
        SAM2_READY = False


# ------------------------------------------------------------
# SAM2 mask refinement function
# ------------------------------------------------------------
def refine_with_sam2(full_image_bgr, bbox_xyxy):
    """
    full_image_bgr: original image BGR (H,W,3)
    bbox_xyxy: (x1, y1, x2, y2)
    Returns boolean mask (H,W) or None.
    """

    if not SAM2_READY or sam2_predictor is None:
        return None

    try:
        # Convert full image to RGB and set for SAM2 only once
        img_rgb = cv2.cvtColor(full_image_bgr, cv2.COLOR_BGR2RGB)
        sam2_predictor.set_image(img_rgb)

        box = np.array([bbox_xyxy], dtype=np.float32)

        masks, scores, _ = sam2_predictor.predict(
            box=box,
            multimask_output=False
        )

        if masks is None or len(masks) == 0:
            return None

        return masks[0].astype(bool)

    except Exception as e:
        print("[SAM2] Error during mask refinement:", e)
        return None
