# Mobile Client Setup Guide

## Quick Start

### Step 1: Find Your Computer's IP Address

**On Windows (PowerShell):**
```powershell
ipconfig | findstr IPv4
```

Look for the IP address of your active network adapter (usually starts with `192.168.x.x` or `10.x.x.x`).

### Step 2: Update the Backend URL in index.html

Edit `index.html` and change line 59:
```javascript
const backendURL = "http://YOUR_COMPUTER_IP:8000";
```

Replace `YOUR_COMPUTER_IP` with the IP address from Step 1 (e.g., `192.168.137.1`).

### Step 3: Start the FastAPI Backend Server

Open a terminal in `Face/backend/` and run:
```bash
python start-server.py
```

Or manually:
```bash
cd Face/backend
uvicorn api.server:app --host 0.0.0.0 --port 8000
```

### Step 4: Start the Web Server (for serving HTML)

Open another terminal in `Face/mobile-client/` and run:
```bash
python server.py
```

This will start a web server on port 8080.

### Step 5: Access from Your Phone

1. Make sure your phone is connected to the **same Wi-Fi network** as your computer
2. Open your phone's browser
3. Go to: `http://YOUR_COMPUTER_IP:8080/index.html`

Replace `YOUR_COMPUTER_IP` with your computer's IP address from Step 1.

### Troubleshooting

**Phone can't connect:**
- ✅ Check that both devices are on the same Wi-Fi network
- ✅ Verify Windows Firewall allows connections on ports 8000 and 8080
- ✅ Try accessing `http://YOUR_COMPUTER_IP:8000` from your phone's browser - you should see `{"status":"running"}`
- ✅ Make sure both servers are running (check terminal windows)

**Connection Error in browser:**
- ✅ Verify the `backendURL` in `index.html` matches your computer's IP
- ✅ Check that the FastAPI server is running and accessible
- ✅ Try accessing the backend directly: `http://YOUR_COMPUTER_IP:8000/` from your phone

**Camera not working:**
- ✅ Make sure you grant camera permissions in your phone's browser
- ✅ Use HTTPS or localhost (some browsers require HTTPS for camera access)
- ✅ Try a different browser (Chrome usually works best)




