# Instance Segmentation App

This project combines **OWL‑ViT**, **RT-DETR**, **SAM / SAM2**, **Mask2Former**, **YOLOv12**, and **Grounded‑DINO** for advanced instance segmentation. The backend is built with **FastAPI**, and the frontend uses **React + Vite**.

> **Note**: These instructions assume **Conda** is installed and available in your PATH. Conda is the only supported environment manager for this project.

---

## 1. Create & Activate Conda Environment

```bash
conda create -n segmentation_env python=3.10 -y
conda activate segmentation_env
```

---

## 2. Install Backend Dependencies

```bash
cd backend
pip install -r requirements.txt
```

### PyTorch (CPU fallback)
If PyTorch fails to install automatically, install the CPU-only version manually:

```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
```

---

## 3. Install YOLOv12 Engine (**Required**)

```bash
cd yolov12
pip install -r requirements.txt
pip install -e .
```

⚠️ **Important**
- **Do NOT** install `ultralytics` from pip.
- Installing `ultralytics` will break YOLOv12 compatibility.

---

## 4. Download Models & Weights (**Required**)

All model weights should be placed inside the **`backend/`** directory unless stated otherwise.

### Segment Anything (SAM)
Run inside the `backend` folder:

```bash
wget https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth
```

---

### YOLOv12 (Instance Segmentation)

- Repository:
  https://github.com/sunsmarterjie/yolov12

- Pretrained weights:
  https://drive.google.com/drive/folders/1EG65LoOyMW0_On00ATcZpS_H9AHs05B6

Download the trained YOLOv12 weights and place them in the appropriate backend directory.

---

### SAM2

```bash
cd models
git clone https://github.com/facebookresearch/sam2.git
cd sam2
pip install -e .
```

---

### Grounded‑SAM‑2 / Grounding‑DINO

```bash
cd models
git clone https://github.com/IDEA-Research/Grounded-SAM-2.git
pip install git+https://github.com/IDEA-Research/GroundingDINO.git
```

Download Grounding‑DINO checkpoints:

```bash
cd Grounded-SAM-2/checkpoints
dos2unix download_ckpts.sh
bash download_ckpts.sh
```

---

## 5. Run the FastAPI Backend

```bash
cd backend
uvicorn main:app --reload --port 8000
```

### API Documentation
Once running, interactive API docs are available at:

```
http://localhost:8000/docs
```

---

## 6. Set Up & Run Frontend (React + Vite)

```bash
cd ../frontend
npm install
```

### Optional Environment Variable
Create a `.env` file inside the `frontend/` directory:

```env
VITE_BACKEND_URL=http://localhost:8000
```

Start the frontend:

```bash
npm run dev
```

Frontend will be available at:

```
http://localhost:5173
```

---

## Troubleshooting

- Ensure all model weights are downloaded before starting the backend.
- If CUDA is required, install a compatible PyTorch version manually.
- If you encounter permission or script errors on Windows, ensure `dos2unix` is installed or run commands via WSL.

---

## License & Credits

This project builds on open‑source work from:
- Meta AI (SAM / SAM2)
- IDEA Research (Grounding‑DINO)
- YOLOv12 contributors

Refer to individual repositories for license details.

