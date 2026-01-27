# Setup Instructions (Using Conda Only) – OWL-ViT + SAM Segmentation App

------------------------------------------------------------
 1. CREATE & ACTIVATE CONDA ENVIRONMENT
------------------------------------------------------------
conda create -n segmentation_env python=3.10 -y
conda activate segmentation_env

------------------------------------------------------------
 2. INSTALL BACKEND DEPENDENCIES
------------------------------------------------------------
cd backend
pip install -r requirements.txt

# If torch fails to install, use:
# (CPU version)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
------------------------------------------------------------
 3. INSTALL YOLOv12 ENGINE (REQUIRED)
------------------------------------------------------------
cd yolov12
pip install -r requirements.txt
pip install -e .

# DO NOT install "ultralytics" from pip (it breaks YOLOv12)

------------------------------------------------------------
 4. DOWNLOAD WEIGHTS (REQUIRED)
------------------------------------------------------------
# Make sure the files are located in backend folder

# SAM
# Run this inside backend folder or click on link
wget https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth

# YOLO
# Download the model weights from here (Instance segmentation)
 https://github.com/sunsmarterjie/yolov12

# download trained yolo weight
https://drive.google.com/drive/folders/1EG65LoOyMW0_On00ATcZpS_H9AHs05B6?usp=drive_link

 # SAM2
cd models
git clone https://github.com/facebookresearch/sam2.git && cd sam2
pip install -e .

# Grounding-Dino
cd models
git clone https://github.com/IDEA-Research/Grounded-SAM-2.git
pip install git+https://github.com/IDEA-Research/GroundingDINO.git
cd Grounded-SAM-2/checkpoints
dos2unix download_ckpts.sh
bash download_ckpts.sh





------------------------------------------------------------
 5. RUN THE FASTAPI BACKEND
------------------------------------------------------------
cd backend
uvicorn main:app --reload --port 8000

# API docs available at:
# http://localhost:8000/docs

------------------------------------------------------------
 6. SET UP & RUN FRONTEND (React + Vite)
------------------------------------------------------------
cd ../frontend
npm install

# (Optional) Create a .env file in frontend folder:
# VITE_BACKEND_URL=http://localhost:8000

npm run dev

# Frontend will start at:
# http://localhost:5173

