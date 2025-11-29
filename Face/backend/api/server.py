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
from fastapi.responses import FileResponse, JSONResponse
from pydantic import BaseModel
from typing import Any, Optional
from ai.pipeline import process_frame, process_frame_scan, process_frame_quickscan, process_frame_face
from PIL import Image
import numpy as np
import cv2
import io
import os
import json
from groq import Groq
from dotenv import load_dotenv
from ai.llm import generate_scene_description
from ai.navigation import get_navigation_guidance
from ai.detector import yolo_model

app = FastAPI()
yolo_model()
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
    mode: str
    prompt: Optional[str] = None
    endpoint: Optional[str] = None

# ========================
# MODE → ENDPOINT MAPPING
# ========================

MODE_ENDPOINT_MAP = {
    "scan": "/api/scan",
    "quickscan": "/api/quickscan",
    "face": "/api/face",
    "vision": "/api/vision"
}

# ========================
# VOICE COMMAND → MODE DETECTION
# ========================

@app.post("/voice-command", response_model=ModeResponse)
async def process_voice_command(voice_input: VoiceCommand):
    """
    Process voice command and determine intent using Groq LLM.
    Returns the mode and which endpoint to call next.
    """
    command = voice_input.command
    print(f"\n[VOICE] Received: '{command}'")

    # Use Groq for intent detection
    mode_data = detect_mode_groq(command)

    # Add the endpoint to call
    mode = mode_data.get("mode")
    endpoint = MODE_ENDPOINT_MAP.get(mode, "/api/scan")
    mode_data["endpoint"] = endpoint

    print(f"[MODE] → {mode}")
    print(f"[ENDPOINT] → {endpoint}")
    if "prompt" in mode_data and mode_data["prompt"]:
        print(f"[PROMPT] → {mode_data['prompt']}")
    print(f"{'='*60}\n")
    
    return ModeResponse(**mode_data)



# ============ GROQ INTENT DETECTION ============


def detect_mode_groq(command: str) -> dict:
    """
    Advanced intent detection using Groq's Llama 3.3.
    """
    try:
        prompt = f"""You are Neytra’s intent-classification AI. Your job is to analyze a voice command and determine what the user actually wants.

Command: "{command}"

You must choose exactly ONE mode from the list below. Read each definition carefully. If ANY part of the command matches a higher-priority mode, choose that one even if other modes are also partially relevant.

---------------------------------------
INTENT DEFINITIONS (IN PRIORITY ORDER)
---------------------------------------

1. vision   (LLM reasoning / analysis)
User is asking a QUESTION, seeking an EXPLANATION, REASONING, DESCRIPTION, SUMMARY, or ANALYSIS about what the camera sees OR about an object they want identified in detail.  
Triggers include:
- “what is this…”
- “describe…”
- “explain…”
- “what does it look like…”
- “read this…”
- “analyze this…”
- “what is happening…”
- “tell me about this object”
If the user is asking ANY question or wants detailed information → mode = "vision".  
Also include a “prompt” field explaining exactly what to send to the vision model.

2. face     (face recognition / enrollment)
User wants to identify people OR save new faces.
Triggers:
- “who is this person”
- “identify the person”
- “recognize them”
- “save this face”
- “add this person”
If the context involves humans, identity, or memory → mode = "face".

3. quickscan   (rapid object detection / safety)
User expresses urgency, danger, or fast scanning.
Triggers:
- “quick scan”
- “what’s around me quickly”
- “is anything coming”
- “fast detection”
- “alert me”
If there is urgency, movement, safety → mode = "quickscan".

4. scan      (standard object/environment detection)
Default mode ONLY when:
- The user wants to detect surroundings or objects
- No reasoning question is asked
- No face-recognition intent
- No urgency

Examples:
- “scan the room”
- “what objects are near me”
- “detect surroundings”
If the command is general-purpose and NOT a question → mode = "scan".

---------------------------------------
IMPORTANT HARD RULES
---------------------------------------
- If the user ASKED A QUESTION → ALWAYS choose "vision".
- If the command contains words like “who” + “person” → choose "face".
- If the command includes “quick”, “fast”, “urgent”, “alert”, “coming”, “danger” → choose "quickscan".
- ONLY return "scan" if NONE of the above categories match.

---------------------------------------
OUTPUT FORMAT
---------------------------------------
Respond ONLY with this JSON format and nothing else:

{{
  "mode": "mode_name",
  "prompt": "prompt_for_vision_mode_or_empty"
}}
"""

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

@app.post("/api/scan")
async def scan_endpoint(file: UploadFile = File(...)):
    """
    SCAN MODE: Standard object detection
    Frontend sends image every 10 seconds
    Returns detected objects and their locations
    """
    try:
        print("\n[API] /api/scan - Processing frame")
        
        # Read and convert image
        img_bytes = await file.read()
        img = Image.open(io.BytesIO(img_bytes)).convert("RGB")
        frame_rgb = np.array(img)
        frame = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)

        # Process with scan mode
        results = process_frame_scan(frame)

        # Get navigation guidance
        guidance = get_navigation_guidance(results, frame_width=frame.shape[1])
        if guidance:
            print(f"[NAVIGATION] Guidance: {guidance}")
        else:
            print("[NAVIGATION] Path clear")

        return JSONResponse({
            "mode": "scan",
            "status": "success",
            "detections": results,
            "count": len(results),
            "navigation": guidance
        })
    
    except Exception as e:
        print(f"[SCAN ERROR] {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/quickscan")
async def quickscan_endpoint(file: UploadFile = File(...)):
    """
    QUICKSCAN MODE: Fast object detection for urgent scenarios
    Optimized for speed, prioritizes obstacles and people
    """
    try:
        print("\n[API] /api/quickscan - Fast processing")
        
        img_bytes = await file.read()
        img = Image.open(io.BytesIO(img_bytes)).convert("RGB")
        frame_rgb = np.array(img)
        frame = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)

        # Process with quickscan mode
        results = process_frame_quickscan(frame)

        # Get navigation guidance
        guidance = get_navigation_guidance(results, frame_width=frame.shape[1])

        return JSONResponse({
            "mode": "quickscan",
            "status": "success",
            "detections": results,
            "count": len(results),
            "priority": "high",
            "navigation": guidance
        })
    
    except Exception as e:
        print(f"[QUICKSCAN ERROR] {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/face")
async def face_recognition_endpoint(file: UploadFile = File(...)):
    """
    FACE RECOGNITION MODE: Identify known faces or enroll new ones
    Returns recognized faces from database
    """
    try:
        print("\n[API] /api/face - Face recognition")
        
        img_bytes = await file.read()
        img = Image.open(io.BytesIO(img_bytes)).convert("RGB")
        frame_rgb = np.array(img)
        frame = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)

        # Process with face recognition mode
        results = process_frame_face(frame)

        # Separate faces and objects
        faces = [r for r in results if r["type"] == "face"]
        objects = [r for r in results if r["type"] == "object"]

        return JSONResponse({
            "mode": "face",
            "status": "success",
            "faces": faces,
            "objects": objects,
            "total_detections": len(results)
        })
    
    except Exception as e:
        print(f"[FACE ERROR] {e}")
        raise HTTPException(status_code=500, detail=str(e))


# @app.post("/api/vision")
# async def vision_llm_endpoint(
#     file: UploadFile = File(...),
#     prompt: str = Form(None)
# ):
#     """
#     VISION LLM MODE: AI reasoning about the scene
#     Uses vision model to answer questions about the image
#     """
#     try:
#         print(f"\n[API] /api/vision - LLM analysis")
#         print(f"[PROMPT] {prompt}")
        
#         img_bytes = await file.read()
        
#         # Use custom prompt if provided, otherwise use default
#         if not prompt:
#             prompt = "Describe what you see in this image, focusing on objects and environment useful for navigation."
        
#         # Call vision LLM
#         description = generate_scene_description(img_bytes, prompt)
        
#         return JSONResponse({
#             "mode": "vision",
#             "status": "success",
#             "prompt": prompt,
#             "description": description
#         })
    
#     except Exception as e:
#         print(f"[VISION ERROR] {e}")
#         raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/vision")
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


# ========================
# HEALTH & STATIC FILES
# ========================

@app.get("/")
def root():
    """Serve mobile client interface"""
    index_path = os.path.join(MOBILE_CLIENT_DIR, "index.html")
    if os.path.exists(index_path):
        return FileResponse(index_path)
    return {"status": "running", "message": "Neytra Server Running"}

@app.get("/health")
def health_check():
    """Check if server is running"""
    from ai.detector import yolo_model
    
    return {
        "status": "healthy",
        "llm_provider": "Groq (Llama 3.3)",
        "yolo_model_loaded": yolo_model is not None,
        "mobile_client": os.path.exists(MOBILE_CLIENT_DIR),
        "groq_configured": bool(GROQ_API_KEY and GROQ_API_KEY != "your-groq-api-key-here"),
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
