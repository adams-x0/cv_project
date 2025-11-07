from fastapi import FastAPI, UploadFile, File, Form
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from pathlib import Path
from uuid import uuid4

from models.owlvit_sam import run_owlvit_sam

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
