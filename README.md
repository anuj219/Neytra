<!--
Complete detailed overview of system as on 21-04-2026
2 days after SummerHacks26 hackathon

now working on improving some features
 -->

Comprehensive Analysis: How the NeytraN Face Recognition System Works
Based on a thorough analysis of all relevant files in the codebase, here's a detailed technical breakdown of how the current system operates:

1. Overall System Architecture & Purpose
NeytraN is an intelligent visual assistance system designed for visually impaired individuals. It combines real-time face recognition, obstacle detection, scene understanding, and voice-based interaction to provide independence and situational awareness.

Architecture Flow:


📱 Mobile/Browser Client (index.html)    ↓ HTTP/WebSocket💻 FastAPI Backend Server (port 8000)    ├─ 🎯 YOLO Object Detection    ├─ 👤 Face Recognition Pipeline    ├─ 🧠 LLM Integration (Gemini + Groq)    └─ 📊 Enrollment & Database Management    ↓💾 Database (faces.pkl)
2. Key Components & Their Roles
A. Frontend - mobile-client/index.html
Responsibility: Real-time video capture, voice control, UI rendering
Technologies: WebRTC/Web Speech API, Canvas for video processing
Key Functions:
startCamera() - Initializes device camera
captureAndSendFrame() - Captures 640x480 frames, sends to backend
handleModeResponse() - Processes detection results
Mode-based frame intervals: SCAN (3s), QUICKSCAN (2s), FACE (3s)
Enrollment Flow: Unknown face → voice confirmation → name capture → save
Connection: Hardcoded backend URL (192.168.29.118:8000)
B. Backend API - backend/api/server.py
Core Endpoints:

Endpoint	Mode	Purpose
POST /api/scan	SCAN	Standard object detection + conditional face recognition
POST /api/quickscan	QUICKSCAN	Fast detection, prioritizes obstacles & people
POST /api/face	FACE	Full face recognition pipeline
POST /api/vision	VISION	LLM scene analysis with user questions
POST /voice-command	ALL	Intent detection via Groq Llama 3.3
POST /api/enroll/*	FACE	4-step enrollment workflow
Voice Command Processing:

Uses Groq API for intent classification (mode: vision/face/quickscan/scan)
Fallback rule-based classifier if Groq fails
Priority: vision > face > quickscan > scan
C. Detection Pipeline - backend/ai/detector.py
YOLO Detection:

Model: models/bestLatest.pt (custom-trained or YOLOv8)
Confidence threshold: 0.5
Resolution: 640px (standard), 320px (fast mode)
Fallback: Uses face_recognition.face_locations() when YOLO misses people

detect_yolo(frame, fast_mode=False)  # Returns [{"label", "bbox", "confidence"}]detect_faces_fallback(frame)         # Dlib-based face detection
D. Recognition Pipeline - backend/ai/recognizer.py
Face Encoding Comparison:


compare_face_to_db(face_encoding, db)# Returns: (best_name, best_distance)# Threshold: 0.48 (distances < 0.48 = match)
Database Format:

Modern: {"Alice": [enc1, enc2, ...], "Bob": [enc1, ...]}
Legacy Support: {"encodings": [...], "names": [...]}
Storage: faces.pkl (pickle binary)
E. Main Processing Pipeline - backend/ai/pipeline.py
Three Processing Modes:

process_frame_scan(frame)

YOLO object detection
Face fallback if no people detected
Returns objects with bboxes
process_frame_quickscan(frame)

Lower resolution (320px) for speed
Prioritizes obstacles & people
Useful for navigation
process_frame_face(frame) [Most Complex]

YOLO detection
Face extraction from person bboxes
Per-person caching (3s cooldown to avoid re-recognition)
Unknown tracker: Temporal tracking for unknown faces
Detection window: 15 seconds
Enrollment prompt: 3+ detections in window
Tracks via encoding distance (0.6 threshold) + bbox center distance (150px)
Announcement cooldown: 30 seconds per person
Returns: faces with names, distances, encodings (unknown only)
Per-Person Caching Logic:


recent_people = [{'box': bbox, 'name': name, 'time': timestamp}]# Avoids expensive face_recognition.face_encodings() calls# Reused if bbox within 80px in 3s window
Unknown Face Tracking:


unknown_face_trackers = {    "unknown_1": {        "timestamps": [t1, t2, t3],      # Within 15s window        "encodings": [enc1, enc2, ...],  # Up to 5 recent        "bbox": [x1, y1, x2, y2],        "last_seen": time,        "last_encoding": enc    }}# Enrollment triggered when detection_count >= 3
F. Enrollment System - backend/ai/enrollment.py
State Machine:


enrollment_state = {    "pending_encoding": None,    "pending_name": None,    "in_enrollment": False}
Key Functions:

initiate_enrollment(face_encoding) - Starts workflow
ask_for_name() - Requests name via voice
confirm_name(name) - Confirms captured name
complete_enrollment(confirmed) - Saves to database
save_unknown_face(encoding, name) - Direct save
encode_face_for_transmission(enc) - Converts numpy → base64 (HTTP transmission)
decode_face_from_transmission(b64) - Converts base64 → numpy
API Endpoints:

POST /api/enroll/initiate - Detects unknown face
POST /api/enroll/capture-face - Extracts encoding
POST /api/enroll/save - Saves with name
POST /api/enroll/cancel - Cancels workflow
G. Navigation Engine - backend/ai/navigation.py
Zone-Based Guidance:


┌─────────────────────────────────┐│  LEFT (0-33%)│CENTER│RIGHT(66%)││              │(33-66%)        │└─────────────────────────────────┘
Logic:

Path clear: No output
Center blocked, sides clear: "Move left/right"
Multiple directions blocked: "Path blocked"
10s cooldown between announcements
Special handling: Distinguishes people from obstacles
Output:


guidance = "Person ahead. Move left."  # or None
H. Encounter Tracking - backend/ai/encounter.py
Per-Person State:


encounter_state = {    "name": {        "last_seen": timestamp,        "last_detected": timestamp,        "last_announced": timestamp,        "in_frame": bool    }}
Rules:

First detection: Always announce
Subsequent detections: Only if 30+ seconds since last announcement
Prevents announcement spam
Functions:

update_presence(name) - Returns True if should announce
mark_absent(names_in_frame) - Tracks who left scene
I. LLM Integration - backend/ai/llm.py
Google Gemini 2.5 Flash:


generate_scene_description(image_bytes, prompt_text)# Returns natural language description
Usage: "What am I looking at?" queries, scene context for navigation

Groq Integration (Intent):

Model: Llama 3.3 70B
JSON-forced response format
Temperature: 0.3 (consistent)
Fallback: Rule-based classifier
3. Data Flow & Processing Pipeline
Typical Frame Processing (FACE Mode):

1. FRAME CAPTURE (Client)   ├─ Resolution: 640x480   └─ Interval: 3 seconds2. TRANSMISSION (Client → Server)   └─ Format: JPEG (70% quality) via FormData3. DETECTION (Server)   ├─ YOLO: Detects objects/persons (confidence > 0.5)   ├─ Fallback: dlib if no persons found   └─ Result: [{label, bbox, confidence}]4. FACE RECOGNITION (for person detections)   ├─ Crop person bbox   ├─ Find face in upper 50% (narrowed ROI)   ├─ Extract face encodings (128-D vector)   └─ Compare against database (distance < 0.48)5. DECISION MAKING   ├─ KNOWN: check encounter_state (30s cooldown)   ├─ UNKNOWN: track temporally + extract encoding   │   └─ If 3+ detections in 15s: trigger enrollment   └─ Cache result (3s reuse window)6. RESPONSE ASSEMBLY   ├─ Faces: [name, distance, bbox, announce, encoding]   ├─ Objects: [label, bbox]   ├─ Navigation: "Person ahead. Move right."   └─ Enrollment: "Unknown person. Add to memory?"7. CLIENT HANDLING   ├─ Display faces & objects   ├─ Voice: Announce names (with cooldown)   ├─ Voice: Navigation guidance   └─ UI: Enrollment dialog if triggered
Enrollment Flow (when unknown detected):

Frontend                          Backend─────────────────────────────────────────────Unknown detected│Speech: "Unknown person. Add?"User: "Yes"                                  ├─ Detect 3+ times in 15s                                  └─ Send prompt_enrollment=TrueUI shows enrollment modal│User speaks name│"Are you sure?"User: "Yes"     ↓ base64-encoded face_encoding/api/enroll/save {name, encoding}                                  ├─ Decode base64 → numpy                                  ├─ Load faces.pkl                                  ├─ Add to database                                  ├─ Save faces.pkl                                  └─ Reload in-memory DB     ← Success response│UI: "John added to memory"Immediately recognizes John in next frame
4. Face Detection, Recognition & Enrollment
Detection Process:
YOLO Detection → Bounding boxes for all objects
Person Filter → Check if label is "person", "face", "human"
Fallback Detection → If no persons: run face_recognition.face_locations()
ROI Extraction → Crop detected person region
Recognition Process:
Face Location → Find face landmarks in cropped region
Face Encoding → Convert face to 128-D embedding
Database Comparison → Calculate distances to all stored encodings
Threshold Check → If min distance < 0.48: MATCH
Announcement → Only if last announced > 30 seconds ago
Enrollment Process:

Unknown Face    ↓Track temporally (15s window)    ↓ (3+ detections)Prompt enrollment    ↓User confirms    ↓Extract face encoding    ↓Capture name (voice)    ↓Confirm name    ↓Save to faces.pkl    ↓Reload database in memory    ↓Recognized in next frame
Key Detail: Unknown trackers prevent repeated enrollment prompts for the same person by matching new detections to previous encodings within 0.6 distance threshold.

5. Backend API & Server Functionality
Server Architecture:

# Location: backend/start-server.py# Command: python -m uvicorn api.server:app --host 0.0.0.0 --port 8000# Accessible: http://192.168.x.x:8000 (same WiFi network)
CORS Enabled: All origins allowed (development)
Request/Response Pattern:
Request:


POST /api/faceContent-Type: multipart/form-datafile: [JPEG image bytes]
Response (Success):


{  "mode": "face",  "status": "success",  "faces": [    {      "type": "face",      "name": "Alice",      "distance": 0.38,      "bbox": [100, 50, 200, 200],      "position": "center",      "announce": true,      "face_encoding": null    },    {      "type": "face",      "name": "unknown",      "bbox": [300, 100, 380, 250],      "position": "right",      "announce": false,      "face_encoding": "base64_string...",      "prompt_enrollment": true,      "detection_count": 3,      "tracker_id": "unknown_1"    }  ],  "objects": [    {"type": "object", "label": "chair", "bbox": [50, 100, 150, 200]}  ],  "enrollment_prompt": "Unknown person. Add to memory?"}
Error Handling:
HTTP 500 on exception
Detailed error messages in logs
Graceful fallbacks (e.g., YOLO → face_recognition)
6. Mobile/Web Client Implementation
Frontend Architecture:
mobile-client/index.html is a single-page application with:

State Management:


let streaming = false;let currentMode = 'scan';let isProcessing = false;let enrollmentState = {  active: false,  pendingFaceEncoding: null,  pendingName: null,  wasStreaming: false};
Core Functions:

startCamera() - Requests WebRTC permissions, initializes video stream
switchMode(mode) - Changes detection mode, updates UI
startStreaming() - Begins periodic frame captures
captureAndSendFrame() - Draws video to canvas, sends to server
handleModeResponse(data) - Processes detections
announcement(text) - Text-to-speech via Web Speech API
Enrollment dialog handling
Voice Recognition:

Web Speech API for speech-to-text
Manual microphone button for enrollment names
Affirmative/negative phrase detection
UI Components:

Mode pills (SCAN, QUICKSCAN, FACE, VISION)
Video player with bracket overlays
Detection cards (faces, objects)
Enrollment alert with confirm/cancel
Voice command feedback
Performance:

640x480 frames, 70% JPEG quality
Skips if processing already in progress
Frame intervals: 2-3 seconds depending on mode
7. Database & Data Storage
Database Format:
faces.pkl location options:

Preferred: Neytra/Face/faces.pkl (shared)
Fallback: faces.pkl (local)
Modern Format:


{    "Alice": [numpy_array_128d, numpy_array_128d],    "Bob": [numpy_array_128d],    "Charlie": [numpy_array_128d, numpy_array_128d, numpy_array_128d]}
Legacy Format (Auto-converted):


{    "encodings": [enc1, enc2, enc3, ...],    "names": ["Alice", "Bob", "Alice", ...]}
Data Type: Each encoding is a numpy float32 array (128 dimensions)

Operations:

load_database() - Deserializes pickle, handles both formats
save_database(db) - Serializes dict to pickle
reload_database() - Called after enrollment to update in-memory copy
Durability: Synchronous file writes ensure data safety

8. AI/ML Models & Their Usage
YOLOv8 Object Detection:
Model File: models/bestLatest.pt (or bestv2.pt, bestv3.pt)

Custom-trained on people/objects for assistive tech
Input: 640x480 or 320x240 images
Output: Bounding boxes with class labels
Classes: person, chair, table, door, stairs, etc.
Inference Time: ~50-100ms per frame (depending on resolution)
Configuration:


yolo_model = YOLO(MODEL_PATH)results = yolo_model(frame, imgsz=640, verbose=False)# imgsz=320 for fast mode (quickscan)
face_recognition Library (dlib-based):
Face Detection:

CNN-based face detector (fast on GPU, accurate)
Returns: bounding boxes + landmarks
Used as fallback when YOLO misses people
Face Encoding:

Generates 128-D embedding from face
Unique per individual
Distance < 0.48 = same person
Distance Calculation:


distances = face_recognition.face_distance(encodings_list, new_encoding)# Returns euclidean distances
Google Gemini 2.5 Flash (Vision LLM):
Capability: Analyze images, answer questions, describe scenes
API Key: GOOGLE_API_KEY (from .env)
Usage: Scene description for "What am I looking at?" queries

Groq Llama 3.3 70B (Intent Classification):
Capability: NLP intent detection from voice commands
API Key: GROQ_API_KEY (from .env)
Temperature: 0.3 (deterministic)
Response Format: JSON (forced)
Fallback: Rule-based classifier if Groq unavailable

9. Configuration & Resources
Environment Variables (.env):

GOOGLE_API_KEY=your_gemini_api_keyGROQ_API_KEY=your_groq_api_key
Dependencies (requirements.txt):

fastapi                  # Web frameworkuvicorn                  # ASGI serveropencv-python           # Image processingnumpy                    # Numerical computingpillow                   # Image loadingface_recognition        # Face encoding/detectionpyttsx3                 # Text-to-speechpython-multipart        # Form data handlingultralytics             # YOLO v8speechrecognition       # Microphone inputkeyboard                # Keyboard eventsgoogle-generativeai     # Gemini APIpython-dotenv           # .env loadinggroq                    # Groq API
Models Directory:

models/├─ bestLatest.pt    (Primary YOLO model)├─ bestv2.pt        (Backup model)└─ bestv3.pt        (Backup model)
Resource Constraints: