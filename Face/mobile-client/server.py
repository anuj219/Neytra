#!/usr/bin/env python3
"""
Simple HTTP server to serve the mobile client HTML file.
Run this script and access from your phone at http://YOUR_IP:8080
"""

import http.server
import socketserver
import os
import sys

PORT = 8080

class MyHTTPRequestHandler(http.server.SimpleHTTPRequestHandler):
    def do_GET(self):
        # Redirect root and common paths to index.html
        if self.path == '/' or self.path == '/index.html' or self.path == '/Images/index.html':
            self.path = '/index.html'
        return super().do_GET()
    
    def end_headers(self):
        # Add CORS headers to allow cross-origin requests
        self.send_header('Access-Control-Allow-Origin', '*')
        self.send_header('Access-Control-Allow-Methods', 'GET, POST, OPTIONS')
        self.send_header('Access-Control-Allow-Headers', '*')
        super().end_headers()

    def log_message(self, format, *args):
        # Custom log format
        sys.stderr.write("%s - - [%s] %s\n" %
                        (self.address_string(),
                         self.log_date_time_string(),
                         format%args))

if __name__ == "__main__":
    # Change to the directory containing this script
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    
    with socketserver.TCPServer(("", PORT), MyHTTPRequestHandler) as httpd:
        print(f"🚀 Mobile client server running on port {PORT}")
        print(f"📁 Serving from: {os.getcwd()}")
        print(f"\n📱 Access from your phone at:")
        print(f"   http://YOUR_COMPUTER_IP:{PORT}/index.html")
        print(f"   http://YOUR_COMPUTER_IP:{PORT}/")
        print(f"   http://YOUR_COMPUTER_IP:{PORT}/Images/index.html (also works)")
        print(f"\n💻 Or locally at: http://localhost:{PORT}/index.html")
        print("\n⚠️  Make sure:")
        print("   1. Your FastAPI backend is running on port 8000")
        print("   2. Your computer and phone are on the same network")
        print("   3. Windows Firewall allows connections on port 8080")
        print("\nPress Ctrl+C to stop the server\n")
        
        try:
            httpd.serve_forever()
        except KeyboardInterrupt:
            print("\n\nServer stopped.")

