#!/usr/bin/env python3
"""
Start the FastAPI server accessible from your local network.
This allows your phone to connect to the backend API.
"""

import uvicorn
import sys
import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Add the backend directory to the path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

if __name__ == "__main__":
    print("🚀 Starting FastAPI server...")
    print("📱 Server will be accessible at: http://YOUR_COMPUTER_IP:8000")
    print("💻 Or locally at: http://localhost:8000")
    print("\n⚠️  Make sure:")
    print("   1. Your computer and phone are on the same network")
    print("   2. Windows Firewall allows connections on port 8000")
    print("\nPress Ctrl+C to stop the server\n")
    
    # Run on 0.0.0.0 to accept connections from any network interface
    # This allows your phone to connect via your computer's local IP
    uvicorn.run(
        "api.server:app",
        host="0.0.0.0",  # Listen on all network interfaces
        port=8000,
        reload=False
    )


