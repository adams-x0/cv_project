# pred_masks.py
import os
import pathlib
import numpy as np
from PIL import Image
from ultralytics import YOLO

MODEL = r"yolov12x-seg.pt"            # or your custom path
SOURCE = r"image2.jpg"                  # image/video/path/glob
OUTDIR = pathlib.Path("runs/segment/predict_masks")  # custom output folder
SAVE_STICKERS = True                   # also save RGBA crops with alpha

OUTDIR.mkdir(parents=True, exist_ok=True)

model = YOLO(MODEL)

# Run segmentation (no save_masks available in this fork)
results = model.predict(
    source=SOURCE,
    save=True,                 # annotated overlay image(s)
    conf=0.5,
    retina_masks=True,         # get full-res masks (not box-cropped)
    overlap_mask=True,         # keep masks in image coords
    verbose=False,
)

for res in results:                               # handle batch or folder
    if res.masks is None or len(res.masks) == 0:
        continue

    # base name of the input image/frame
    base = pathlib.Path(res.path).stem

    # tensors -> numpy
    masks = res.masks.data.cpu().numpy()         # [N, H, W] (float 0..1)
    boxes = res.boxes.xyxy.cpu().numpy()         # [N, 4]
    scores = res.boxes.conf.cpu().numpy()        # [N]
    cls_ids = res.boxes.cls.cpu().numpy().astype(int)
    names = getattr(model.model, "names", {}) or {}

    # Load original RGB to make RGBA stickers
    # BGR->RGB if needed
    rgb = res.orig_img[..., ::-
                       1] if res.orig_img.shape[2] == 3 else res.orig_img

    for i, m in enumerate(masks):
        label = names.get(int(cls_ids[i]), str(int(cls_ids[i])))
        score = float(scores[i])
        x1, y1, x2, y2 = boxes[i]

        # --- 1) Save full-size binary mask (white=object) ---
        mask_u8 = (m > 0.5).astype(np.uint8) * 255
        mask_img = Image.fromarray(mask_u8, mode="L")
        mask_path = OUTDIR / f"{base}_{i:02d}_{label}_{score:.2f}.png"
        mask_img.save(mask_path)

        # --- 2) (Optional) Save RGBA sticker crop with transparency ---
        if SAVE_STICKERS:
            h, w = mask_u8.shape
            xi1, yi1 = max(0, int(x1)), max(0, int(y1))
            xi2, yi2 = min(w, int(x2)), min(h, int(y2))

            alpha = mask_u8
            if rgb.shape[:2] != alpha.shape:
                # fall back to full image size from res.orig_shape if needed
                pass

            rgba = np.dstack([rgb, alpha])  # HxWx4
            crop = rgba[yi1:yi2, xi1:xi2, :]
            if crop.size > 0:
                sticker = Image.fromarray(crop, mode="RGBA")
                sticker_path = OUTDIR / \
                    f"{base}_{i:02d}_{label}_{score:.2f}_sticker.png"
                sticker.save(sticker_path)

print(f"Saved masks (and stickers) to: {OUTDIR.resolve()}")
