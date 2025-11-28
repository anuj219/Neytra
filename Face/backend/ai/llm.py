# ai/llm.py

import os
import google.generativeai as genai
from PIL import Image
import io

# Configure the API key
# Ideally, this should be loaded from environment variables for security
api_key = os.environ.get("GOOGLE_API_KEY")
if not api_key:
    print("Warning: GOOGLE_API_KEY environment variable not set. LLM features will fail.")
else:
    genai.configure(api_key=api_key)

def generate_scene_description(image_bytes, prompt_text="Describe this scene for a visually impaired person in short."):
    """
    Sends the image and prompt to Google Gemini 1.5 Flash and returns the text response.
    """
    if not api_key:
        return "Error: Server API key not configured."

    try:
        print(f"[LLM] Generating description for image ({len(image_bytes)} bytes)...")
        # Load image from bytes
        image = Image.open(io.BytesIO(image_bytes))

        # Initialize the model
        model = genai.GenerativeModel('gemini-2.5-flash')

        # Generate content
        print(f"[LLM] Sending request to Gemini with prompt: '{prompt_text}'")
        response = model.generate_content([prompt_text, image])
        
        print(f"[LLM] Response received: {response.text[:100]}...") # Log first 100 chars
        # Return the text
        return response.text
    except Exception as e:
        print(f"[LLM] Gemini Error: {e}")
        return f"Error analyzing image: {str(e)}"
