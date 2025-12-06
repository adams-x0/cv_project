from fastapi import FastAPI, UploadFile, File, Form
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from pathlib import Path
from uuid import uuid4

from models.owlvit_sam import run_owlvit_sam
from models.yolo_seg import run_yolo_seg
from models.rtdetr_seg import run_rtdetr_seg
from models.grounding_sam import run_grounded_sam
from models.mask2former_seg import run_mask2former_seg



app = FastAPI()

app.mount("/outputs", StaticFiles(directory="outputs"), name="outputs")

# Allow frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Folders
UPLOAD_DIR = Path("uploads")
OUTPUT_DIR = Path("outputs")
UPLOAD_DIR.mkdir(exist_ok=True)
OUTPUT_DIR.mkdir(exist_ok=True)


@app.post("/segment/owlvit")
async def segment_with_owlvit(image: UploadFile = File(...), prompt: str = Form(...)):
    img_name = f"{uuid4().hex}_{image.filename}"
    img_path = UPLOAD_DIR / img_name

    with img_path.open("wb") as f:
        f.write(await image.read())

    result = run_owlvit_sam(str(img_path), prompt)
    return result


@app.post("/segment/grounded")
async def segment_with_grounded(image: UploadFile = File(...), prompt: str = Form(...)):
    img_name = f"{uuid4().hex}_{image.filename}"
    img_path = UPLOAD_DIR / img_name

    with img_path.open("wb") as f:
        f.write(await image.read())

    return run_grounded_sam(str(img_path), prompt)


@app.post("/segment/yolo")
async def segment_with_yolo(image: UploadFile = File(...)):
    img_name = f"{uuid4().hex}_{image.filename}"
    img_path = UPLOAD_DIR / img_name

    with img_path.open("wb") as f:
        f.write(await image.read())

    result = run_yolo_seg(str(img_path))
    return result                                   

# ----------------------------------------------------------
# NEW: RT-DETR + SAM2 endpoint
# ----------------------------------------------------------

@app.post("/segment/rtdetr")
async def segment_with_rtdetr(image: UploadFile = File(...)):
    img_name = f"{uuid4().hex}_{image.filename}"
    img_path = UPLOAD_DIR / img_name

    with img_path.open("wb") as f:
        f.write(await image.read())

    return run_rtdetr_seg(str(img_path))


@app.post("/segment/mask2former")
async def segment_with_mask2former(image: UploadFile = File(...)):
    img_name = f"{uuid4().hex}_{image.filename}"
    img_path = UPLOAD_DIR / img_name

    with img_path.open("wb") as f:
        f.write(await image.read())

    return run_mask2former_seg(str(img_path))


# Health Check
# ----------------------------------------------------------
@app.get("/")
async def root():
    return {"message": "Backend running with OWL-ViT + SAM + YOLOv8-Seg"}