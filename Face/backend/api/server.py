# # api/server.py

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
    CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"]
)

# ============ GROQ SETUP ============
load_dotenv(ENV_PATH)

print(f"[ENV] Loading from: {ENV_PATH}")
print(f"[ENV] File exists: {os.path.exists(ENV_PATH)}")

# ============ GROQ SETUP ============
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

if not GROQ_API_KEY or GROQ_API_KEY == "":
    print("⚠️  WARNING: GROQ_API_KEY not found!")
    print(f"   Checked: {ENV_PATH}")
    print("   Make sure your .env file contains: GROQ_API_KEY=your-key-here")
else:
    print(f"✓ GROQ_API_KEY loaded successfully (length: {len(GROQ_API_KEY)})")

groq_client = Groq(api_key=GROQ_API_KEY)

# ============ DATA MODELS ============


class VoiceCommand(BaseModel):
    command: str


class ModeResponse(BaseModel):
    mode: str = None
    prompt: str = None


# ============ GROQ INTENT DETECTION ============


def detect_mode_groq(command: str) -> dict:
    """
    Advanced intent detection using Groq's Llama 3.3.
    """
    try:
        prompt = f"""You are an AI assistant for Neytra, a smart assistive device for visually impaired users.
    Analyze this voice command and determine the user's intent.

    Command: "{command}"

    Available intents and modes:
    1. scan (mode: scan )  
    User wants to detect objects or understand surroundings in a normal, non-urgent context.  
    Actions: give the mode only.

    2. quickscan (mode: quickscan )  
    User wants fast, rapid object detection for safety or urgent scenarios such as crossing a road, avoiding obstacles, or detecting incoming objects.  
    Actions: give the mode only.

    3. face_recognition (mode: face)  
    User wants to identify who is in front of them from stored faces. Or wants to save new Faces in the stored faces database.  
    Actions: give the mode only.

    4. llm (mode: vision)  
    User wants to ask a question about a specific image using a vision LLM, requiring reasoning, explanation, or analysis.  
    Actions: give the mode as well as the prompt that should be given to the vision model.

    Respond ONLY with valid JSON in this exact format (no markdown, no extra text):
    {{
    "mode": "mode_name",
    "prompt":"prompt"(only for vision mode)
    }}"""

        # Call Groq API
        completion = groq_client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[
                {
                    "role": "system",
                    "content": "You are a helpful intent classifier. Always respond with valid JSON only, no markdown formatting.",
                },
                {"role": "user", "content": prompt},
            ],
            temperature=0.3,  # Lower temperature for more consistent output
            max_completion_tokens=256,
            top_p=0.9,
            stream=False,  # Disable streaming for easier parsing
            response_format={"type": "json_object"},  # Force JSON response
        )

        # Extract response
        response_text = completion.choices[0].message.content
        print(f"[GROQ] Raw response: {response_text}")

        # Parse JSON
        result = json.loads(response_text)

        # Validate required fields
        required_fields = ["mode"]
        if not all(field in result for field in required_fields):
            raise ValueError("Missing required fields in Groq response")

        print(f"[GROQ] Parsed mode: {result['mode']}")
        return result

    except json.JSONDecodeError as e:
        print(f"[GROQ ERROR] JSON parsing failed: {e}")
        print(f"[GROQ ERROR] Response was: {response_text}")
        return fallback_intent(command)

    except Exception as e:
        print(f"[GROQ ERROR] {e}")
        return fallback_intent(command)


def fallback_intent(command: str) -> dict:
    """
    Fallback rule-based intent detection if Groq fails.
    Matches the same modes used by detect_intent_groq.
    """
    command = command.lower().strip()

    # --- QuickScan: urgent detection ---
    if any(word in command for word in ["fast", "quick", "urgent", "hurry", "danger", "cross", "road"]):
        return {
            "mode": "quickscan",
            "prompt": "",
        }

    # --- Scan: normal object detection ---
    if any(word in command for word in ["what do you see", "describe", "what's in front", "objects", "around me"]):
        return {
            "mode": "scan",
            "prompt": "",
        }

    # --- Face mode (recognition or enrollment both use 'face') ---
    if any(word in command for word in ["who is", "identify", "recognize", "who am i", "face"]):
        return {
            "mode": "face",
            "prompt": "",
        }
    if any(word in command for word in ["add person", "enroll", "remember", "new person", "save face"]):
        return {
            "mode": "face",
            "prompt": "",
        }

    # --- Vision LLM mode ---
    if any(word in command for word in ["explain this", "analyze this", "what is happening", "question about image", "tell me about this"]):
        return {
            "mode": "vision",
            "prompt": command,  # user question becomes prompt
        }

    # --- Greeting ---
    if any(word in command for word in ["hello", "hi", "hey"]):
        return {
            "mode": "interaction",
            "prompt": "",
        }

    # --- Fallback unknown ---
    return {
        "mode": "interaction",
        "prompt": "",
    }


# ============ API ENDPOINTS ============


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
