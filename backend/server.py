import shutil
from pathlib import Path
from uuid import uuid4
import os

from fastapi import FastAPI, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles

from segment import run_segmentation

# ----------------------------------------------------
# Create FastAPI app instance
# ----------------------------------------------------
app = FastAPI(title="Segmentation API", version="1.0")

# ----------------------------------------------------
# CORS (Cross-Origin Resource Sharing)
# Allows the frontend (React at localhost:5174) to call this backend
# ----------------------------------------------------
ALLOW_ORIGINS = ["*"]       # For development: allow requests from anywhere

app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOW_ORIGINS,        # Which website domains can access this server

    allow_credentials=True,
    allow_methods=["*"],            # Allow all HTTP methods (GET, POST, etc.)
    allow_headers=["*"],            # Allow all request headers
)

# ----------------------------------------------------
# Static File Mount
# We expose the "runs" directory at the URL path "/static/"
#
# Example:
#   Actual file on disk:  runs/segment/web_output/image_overlay.png
#   Accessible URL:       http://localhost:8000/static/segment/web_output/image_overlay.png
# ----------------------------------------------------
app.mount("/static/", StaticFiles(directory="runs"), name="static")

# ----------------------------------------------------
# Folder to temporarily store uploaded images
# We recreate it fresh each time the server starts
# ----------------------------------------------------
UPLOADS = Path("uploads")
if UPLOADS.exists():
    shutil.rmtree(UPLOADS, ignore_errors=True)

UPLOADS.mkdir(exist_ok=True)


# ----------------------------------------------------
# Convert local filesystem paths into public URLs
#
# Since /static/ maps to /runs/, we must strip the leading "runs/"
# so the browser can load the file correctly.
# ----------------------------------------------------
def to_static_url(path_str: str | None) -> str | None:
    if not path_str:
        return None
    
    # Normalize Windows backslashes -> forward slashes
    norm = os.path.normpath(path_str).replace("\\", "/")

    # Remove optional "./" prefix
    if norm.startswith("./"):
        norm = norm[2:]

    # Remove "runs/" so URL matches /static/<rest_of_path>
    if norm.startswith("runs/"):
        norm = norm[len("runs/"):]  # 'segment/web_output/a.png'

    # Final public URL
    return "/static/" + norm.lstrip("/")  # guarantee single leading slash

# ----------------------------------------------------
# Main API Endpoint
# Receives uploaded image → runs YOLO segmentation → returns result URLs
# ----------------------------------------------------
@app.post("/api/segment")
async def segment(image: UploadFile = File(...)):
    # 1) Save uploaded image to disk with a unique name to avoid overwriting files
    saved = UPLOADS / f"{uuid4().hex}_{image.filename}"
    with saved.open("wb") as f:
        f.write(await image.read())

    # 2) Run segmentation pipeline (returns list of output file paths)
    outputs = run_segmentation(str(saved))
    print("run_segmentation outputs:", outputs)  # DEBUG LOG

    # 3) Separate overlay image and sticker images
    overlay_path = None
    sticker_paths = []


    for p in outputs:
        s = str(p).replace("\\", "/")       # normalize for checking suffix
        if s.endswith("_overlay.png"):
            overlay_path = str(p)
        elif s.endswith("_sticker.png"):
            sticker_paths.append(str(p))

    # If no overlay was generated, print a warning
    if overlay_path is None:
        print("WARN: No overlay found. Did the model produce outputs?")

    # 4) Convert local file paths → public URLs for frontend display
    overlay_url = to_static_url(overlay_path)
    sticker_urls = [to_static_url(p) for p in sticker_paths]

    print("Resolved URLs:", overlay_url, sticker_urls)  # DEBUG LOG

    # 5) Return JSON response to frontend
    return JSONResponse(
        {
            "overlay_url": overlay_url,         # full annotated overlay
            "sticker_urls": sticker_urls,       # individual cutout stickers
        }
    )
