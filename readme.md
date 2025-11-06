# Make sure you are in the backend folder
cd backend

# 1) Create conda environment
conda create -n yolov12 python=3.11 -y

# 2) Activate environment
conda activate yolov12

# 3) Install PyTorch (GPU version if supported)
conda install pytorch torchvision pytorch-cuda=12.1 -c pytorch -c nvidia  # GPU
# OR:
# conda install pytorch torchvision cpuonly -c pytorch                    # CPU-only

# 4) Install backend dependencies
pip install -r requirements.txt

# 5) Install YOLOv12 from source (IMPORTANT)
cd yolov12
pip install -r requirements.txt
pip install -e .
cd ..

# 6) Run FastAPI server
uvicorn server:app --reload --host 0.0.0.0 --port 8000

# 7) Download the model weights from here (Instance segmentation)
it should be in the backend directory
https://github.com/sunsmarterjie/yolov12

# 8) in frontend requirements_npm.txt
copy the line there (npm install react-router-dom) (for now) run it where you usually run npm run dev

