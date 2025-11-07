# ✅ Setup Instructions (Using Conda Only) – OWL-ViT + SAM Segmentation App

------------------------------------------------------------
📌 1. CREATE & ACTIVATE CONDA ENVIRONMENT
------------------------------------------------------------
conda create -n segmentation_env python=3.10 -y
conda activate segmentation_env

------------------------------------------------------------
📌 2. INSTALL BACKEND DEPENDENCIES
------------------------------------------------------------
cd backend
pip install --upgrade pip
pip install -r requirements.txt

# If torch fails to install, use:
# (CPU version)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu

------------------------------------------------------------
📌 3. DOWNLOAD SAM MODEL FILE (REQUIRED)
------------------------------------------------------------
# Run this inside backend folder
wget https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth

# Make sure the file is located at:
# backend/sam_vit_b_01ec64.pth

------------------------------------------------------------
📌 4. RUN THE FASTAPI BACKEND
------------------------------------------------------------
cd backend
uvicorn main:app --reload --port 8000

# API docs available at:
# http://localhost:8000/docs

------------------------------------------------------------
📌 5. SET UP & RUN FRONTEND (React + Vite)
------------------------------------------------------------
cd ../frontend
npm install

# (Optional) Create a .env file in frontend folder:
# VITE_BACKEND_URL=http://localhost:8000

npm run dev

# Frontend will start at:
# http://localhost:5173
