# api/server.py

from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from ai.pipeline import process_frame
from PIL import Image
import numpy as np
import cv2
import io

app = FastAPI()

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
    return {"status": "running"}
