
# from fastapi import FastAPI, UploadFile, File
# from fastapi.middleware.cors import CORSMiddleware
# from fastapi.staticfiles import StaticFiles
# from fastapi.responses import FileResponse
# from ai.pipeline import process_frame
# from PIL import Image
# import numpy as np
# import cv2
# import io
# import os

# app = FastAPI()

# # Get the path to the mobile-client directory
# # server.py is at: Face/backend/api/server.py
# # mobile-client is at: Face/mobile-client
# # So we need to go up 2 levels: api -> backend -> Face
# BACKEND_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # Face/backend
# FACE_DIR = os.path.dirname(BACKEND_DIR)  # Face
# MOBILE_CLIENT_DIR = os.path.join(FACE_DIR, "mobile-client")

# app.add_middleware(
#     CORSMiddleware,
#     allow_origins=["*"],
#     allow_methods=["*"],
#     allow_headers=["*"]
# )

# @app.post("/frame")
# async def receive_frame(file: UploadFile = File(...)):
#     img_bytes = await file.read()
#     img = Image.open(io.BytesIO(img_bytes)).convert("RGB")
#     frame_rgb = np.array(img)
#     # Convert RGB (PIL format) to BGR (OpenCV format)
#     frame = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)

#     results = process_frame(frame)

#     return {"results": results}


# @app.get("/")
# def root():
#     # Serve the mobile client HTML file
#     index_path = os.path.join(MOBILE_CLIENT_DIR, "index.html")
#     if os.path.exists(index_path):
#         return FileResponse(index_path)
#     return {"status": "running", "message": "Mobile client not found"}

# @app.get("/index.html")
# def serve_index():
#     """Serve the mobile client index.html"""
#     index_path = os.path.join(MOBILE_CLIENT_DIR, "index.html")
#     if os.path.exists(index_path):
#         return FileResponse(index_path)
#     return {"error": "index.html not found"}

# # Mount static files directory (for any additional assets)
# if os.path.exists(MOBILE_CLIENT_DIR):
#     app.mount("/static", StaticFiles(directory=MOBILE_CLIENT_DIR), name="static")


# api/server.py

from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel
from typing import Any
from ai.pipeline import process_frame
from PIL import Image
import numpy as np
import cv2
import io
import os
import json
from groq import Groq
from dotenv import load_dotenv

app = FastAPI()

# ============ PATH SETUP ============
BACKEND_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FACE_DIR = os.path.dirname(BACKEND_DIR)
MOBILE_CLIENT_DIR = os.path.join(FACE_DIR, "mobile-client")
ENV_PATH = os.path.join(FACE_DIR, ".env")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"]
)

@app.post("/frame")
async def receive_frame(file: UploadFile = File(...)):
    """Process camera frame for object/face detection"""
    try:
        img_bytes = await file.read()
        img = Image.open(io.BytesIO(img_bytes)).convert("RGB")
        frame_rgb = np.array(img)
        frame = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)

        results = process_frame(frame)

        return {"results": results}
    except Exception as e:
        print(f"[FRAME ERROR] {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/voice-command", response_model=ModeResponse)
async def process_voice_command(voice_input: VoiceCommand):
    """
    Process voice command and determine intent using Groq LLM.

    Returns:
        - intent: What the user wants (e.g., "object_detection")
        - mode: Which system mode to activate
        - response: Natural language response to speak back
        - action: Specific action to trigger (optional)
    """
    command = voice_input.command

    print(f"\n[VOICE] Received: '{command}'")

    # Use Groq for intent detection
    mode_data = detect_mode_groq(command)

    print(f"[MODE] → {mode_data['mode']}")
    if "prompt" in mode_data:
        print(f"[PROMPT] → {mode_data['prompt']}")
    print(f"{'='*60}\n")
    return ModeResponse(**mode_data)

@app.post("/analyze")
async def analyze_scene(file: UploadFile = File(...)):
    """
    Endpoint for LLM-based scene analysis.
    Receives an image, sends it to Gemini, and returns the description.
    """
    print("[API] Received /analyze request")
    img_bytes = await file.read()
    
    # Simulate a prompt for now (as per requirements)
    # In the future, this could come from the client (audio transcription)
    prompt = "In one short sentence, describe what objects and environment are visible in this scene, focusing on what's useful for navigation and awareness."
    
    description = generate_scene_description(img_bytes, prompt)
    print(f"[API] Sending response: {description[:50]}...")
    
    return {"text": description}


@app.get("/")
def root():
    """Serve mobile client interface"""
    index_path = os.path.join(MOBILE_CLIENT_DIR, "index.html")
    if os.path.exists(index_path):
        return FileResponse(index_path)
    return {"status": "running", "message": "Neytra Server Running"}


@app.get("/index.html")
def serve_index():
    """Serve the mobile client index.html"""
    index_path = os.path.join(MOBILE_CLIENT_DIR, "index.html")
    if os.path.exists(index_path):
        return FileResponse(index_path)
    return {"error": "index.html not found"}


@app.get("/health")
def health_check():
    """Check if server is running"""
    return {
        "status": "healthy",
        "llm_provider": "Groq (Llama 3.3)",
        "mobile_client": os.path.exists(MOBILE_CLIENT_DIR),
        "groq_configured": bool(
            GROQ_API_KEY and GROQ_API_KEY != "your-groq-api-key-here"
        ),
    }


# Mount static files
if os.path.exists(MOBILE_CLIENT_DIR):
    app.mount("/static", StaticFiles(directory=MOBILE_CLIENT_DIR), name="static")


if __name__ == "__main__":
    import uvicorn

    print("\n🚀 Starting Neytra Server with Groq LLM...")
    print(f"📁 Mobile client path: {MOBILE_CLIENT_DIR}")
    print(
        f"🤖 Groq API configured: {bool(GROQ_API_KEY and GROQ_API_KEY != 'your-groq-api-key-here')}\n"
    )
    uvicorn.run(app, host="0.0.0.0", port=8000)
