"""
Simple Local HTTP Server for Task 6 Web App.
Run with: python app/server.py
Access at: http://localhost:8000
"""
import http.server
import socketserver
import webbrowser
from pathlib import Path

PORT = 8000
APP_DIR = Path(__file__).resolve().parent

class CustomHandler(http.server.SimpleHTTPRequestHandler):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, directory=str(APP_DIR), **kwargs)

def main():
    with socketserver.TCPServer(("", PORT), CustomHandler) as httpd:
        print(f"Server started at http://localhost:{PORT}")
        print("Press Ctrl+C to stop.")
        try:
            httpd.serve_forever()
        except KeyboardInterrupt:
            print("\nServer stopped.")

if __name__ == '__main__':
    main()
