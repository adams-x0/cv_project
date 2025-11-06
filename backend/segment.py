import pathlib
import shutil
import numpy as np
from PIL import Image
from ultralytics import YOLO

def run_segmentation(image_path, model_path="yolov12x-seg.pt", save_stickers=True):
    """
    Runs instance segmentation on a single image using a YOLO segmentation model.
    Produces:
        - A full overlay image showing all detected masks + boxes
        - Optionally, separate PNG 'sticker' cutouts for each detected object with alpha masks

    Args:
        image_path (str): Path to input image file.
        model_path (str): Path to YOLO segmentation model weights (.pt).
        save_stickers (bool): Whether to output individual RGBA object stickers.

    Returns:
        List[str]: Paths to all generated output image files (overlay + stickers).
    """
    
    # Directory where output overlay + stickers will be stored
    OUTDIR = pathlib.Path("runs/segment/web_output")

    # If a previous output folder exists → remove it to keep output clean
    if OUTDIR.exists():
        shutil.rmtree(OUTDIR, ignore_errors=True)

    OUTDIR.mkdir(parents=True, exist_ok=True)

    # Load YOLO segmentation model
    model = YOLO(model_path)

    # Run inference — no saving inside YOLO; we handle saving manually
    results = model.predict(
        source=image_path,
        save=False,
        conf=0.5,               # minimum confidence threshold
        retina_masks=True,      # high quality mask mode
        overlap_mask=True,      # allow masks to overlap naturally
        verbose=False,
    )

    output_files = []       # stores paths we return

    # Loop over inference results (usually just 1 image)
    for res in results:
        
        # Extract simple name (e.g., "image2")
        base = pathlib.Path(res.path).stem

        # Obtain YOLO's rendered overlay (numpy array)
        overlay = res.plot()

        # Convert BGR → RGB if needed
        overlay_rgb = overlay[..., ::-
                           1] if overlay.shape[2] == 3 else overlay
        
        # Save annotated overlay image
        overlay_path = OUTDIR / f"{base}_overlay.png"
        Image.fromarray(overlay_rgb).save(overlay_path)
        output_files.append(str(overlay_path))

        # If no segmentation masks detected → continue to next result
        if res.masks is None or len(res.masks) == 0:
            continue

        # Extract mask, bounding boxes, confidence, classes
        masks = res.masks.data.cpu().numpy()
        boxes = res.boxes.xyxy.cpu().numpy()
        scores = res.boxes.conf.cpu().numpy()
        cls_ids = res.boxes.cls.cpu().numpy().astype(int)
        names = getattr(model.model, "names", {}) or {}

       # Load original image (again converting BGR → RGB)
        rgb = res.orig_img[..., ::-
                        1] if res.orig_img.shape[2] == 3 else res.orig_img
       
       # Generate stickers (cropped RGBA object cutouts)
        if save_stickers:
            H, W = masks.shape[1], masks.shape[2]
            for i, m in enumerate(masks):
                label = names.get(int(cls_ids[i]), str(int(cls_ids[i])))
                score = float(scores[i])
                x1, y1, x2, y2 = boxes[i]

                # Ensure we don't go outside image boundaries
                xi1, yi1 = max(0, int(x1)), max(0, int(y1))
                xi2, yi2 = min(W, int(x2)), min(H, int(y2))

                # Convert mask to 0–255 alpha channel
                alpha = (m > 0.5).astype(np.uint8) * 255

                # Stack RGB + alpha = RGBA object image
                rgba = np.dstack([rgb, alpha])  # HxWx4

                # Extract object region only
                crop = rgba[yi1:yi2, xi1:xi2, :]

                # Only save if not empty
                if crop.size > 0:
                    sticker = Image.fromarray(crop, mode="RGBA")
                    sticker_path = OUTDIR / \
                        f"{base}_{i:02d}_{label}_{score:.2f}_sticker.png"
                    sticker.save(sticker_path)
                    output_files.append(str(sticker_path))

    return output_files



# def main():
#     outputs = run_segmentation("image2.jpg")

#     if outputs:
#         print("Generated files:")
#         for f in outputs:
#             print("  ", f)
#     else:
#         print("No objects detected.")

# if __name__ == "__main__":
#     main()