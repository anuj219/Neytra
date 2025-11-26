# api/server.py

from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from ai.pipeline import process_frame
from PIL import Image
import numpy as np
import cv2
import io
import os

app = FastAPI()

# Get the path to the mobile-client directory
# server.py is at: Face/backend/api/server.py
# mobile-client is at: Face/mobile-client
# So we need to go up 2 levels: api -> backend -> Face
BACKEND_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # Face/backend
FACE_DIR = os.path.dirname(BACKEND_DIR)  # Face
MOBILE_CLIENT_DIR = os.path.join(FACE_DIR, "mobile-client")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"]
)

@app.post("/frame")
async def receive_frame(file: UploadFile = File(...)):
    img_bytes = await file.read()
    img = Image.open(io.BytesIO(img_bytes)).convert("RGB")
    frame_rgb = np.array(img)
    # Convert RGB (PIL format) to BGR (OpenCV format)
    frame = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)

    results = process_frame(frame)

    return {"results": results}


@app.get("/")
def root():
    # Serve the mobile client HTML file
    index_path = os.path.join(MOBILE_CLIENT_DIR, "index.html")
    if os.path.exists(index_path):
        return FileResponse(index_path)
    return {"status": "running", "message": "Mobile client not found"}

@app.get("/index.html")
def serve_index():
    """Serve the mobile client index.html"""
    index_path = os.path.join(MOBILE_CLIENT_DIR, "index.html")
    if os.path.exists(index_path):
        return FileResponse(index_path)
    return {"error": "index.html not found"}

# Mount static files directory (for any additional assets)
if os.path.exists(MOBILE_CLIENT_DIR):
    app.mount("/static", StaticFiles(directory=MOBILE_CLIENT_DIR), name="static")
