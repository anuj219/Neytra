# Mobile Client Setup Guide

## ⚡ Quick Start (NEW - Easier Method)

### Step 1: Find Your Computer's IP Address

**On Windows (PowerShell):**
```powershell
ipconfig | findstr IPv4
```

Look for the IP address of your active network adapter (usually starts with `192.168.x.x` or `10.x.x.x`).

### Step 2: Start the FastAPI Backend Server

Open a terminal in `Face/backend/` and run:
```bash
python start-server.py
```

Or manually:
```bash
cd Face/backend
uvicorn api.server:app --host 0.0.0.0 --port 8000
```

### Step 3: Start the Web Server

Open another terminal in `Face/mobile-client/` and run:
```bash
python server.py
```

This will start a web server on port 8080.

### Step 4: Access from Your Phone

1. Make sure your phone is connected to the **same Wi-Fi network** as your computer
2. Open your phone's browser
3. Go to: `http://YOUR_COMPUTER_IP:8080/index.html`
   - Replace `YOUR_COMPUTER_IP` with your computer's IP from Step 1
   - Example: `http://192.168.1.100:8080/index.html`

### Step 5: Configure Backend (if needed)

If the app shows "Disconnected":
- **Click the ⚙️ settings icon** in the top-right corner
- Enter your computer's IP and port: `YOUR_COMPUTER_IP:8000`
- Click Save
- The app will remember this for future sessions

---

## 🔗 Alternative: Use Query Parameter

You can also pass the backend URL directly in the browser URL:

```
http://YOUR_COMPUTER_IP:8080/index.html?backend=YOUR_COMPUTER_IP:8000
```

Example:
```
http://192.168.1.100:8080/index.html?backend=192.168.1.100:8000
```

---

## ✅ Troubleshooting

### Phone can't connect:
- ✅ Check that both devices are on the **same Wi-Fi network**
- ✅ Verify Windows Firewall allows connections on ports **8000** and **8080**
- ✅ Try accessing backend directly from phone: `http://YOUR_COMPUTER_IP:8000` 
  - You should see `{"status":"running"}`
- ✅ Make sure both servers are running (check terminal windows)

### Connection Error in browser:
- ✅ Use the **⚙️ settings icon** to configure the backend URL
- ✅ Verify the backend server is running (`python start-server.py`)
- ✅ Try accessing the backend directly from your phone first

### Camera not working:
- ✅ Make sure you **grant camera permissions** in your phone's browser
- ✅ Try a different browser (Chrome/Firefox usually work best)
- ✅ Check that the app says "Ready" in the video status

### Different resolutions on mobile:
- ✅ The app now **automatically adapts** to your phone's camera resolution
- ✅ Works with landscape and portrait orientations
- ✅ Clear your browser cache if issues persist (Ctrl+Shift+Delete)

---

## 📱 What's Different on Mobile?

✨ **New Features in Latest Version:**
- ⚙️ Settings modal for easy backend URL configuration
- 📷 Auto-detecting camera resolution for optimal performance
- 🔄 Responsive canvas sizing (no more hardcoded resolutions)
- 💾 Remembers your backend URL in local storage

---

## 🚀 Pro Tips

**For Desktop/Laptop Access:**
- Use `http://localhost:8080` if running browser on same computer as the servers

**For Faster Configuration:**
- Use the query parameter method: `?backend=192.168.1.100:8000`

**For Testing:**
- Open browser's **Developer Tools** (F12) to see detailed logs in the Console





