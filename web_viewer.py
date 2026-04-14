"""
🌐 Web Viewer — browse captures, detections, and logs from your browser.

Usage:
    python web_viewer.py            # start on port 8080
    python web_viewer.py --port 9000

Then open http://squirrel-defense.local:8080 in your browser.
"""

import argparse
import os
from http.server import HTTPServer, SimpleHTTPRequestHandler
from socketserver import ThreadingMixIn

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

FOLDERS = {
    "captures": os.path.join(BASE_DIR, "captures"),
    "detections": os.path.join(BASE_DIR, "detections"),
    "logs": os.path.join(BASE_DIR, "logs"),
    "output": os.path.join(BASE_DIR, "output"),
}


def list_files(folder_path):
    """List files in a folder, newest first."""
    if not os.path.isdir(folder_path):
        return []
    files = []
    for f in os.listdir(folder_path):
        full = os.path.join(folder_path, f)
        if os.path.isfile(full):
            files.append((f, os.path.getmtime(full), os.path.getsize(full)))
    files.sort(key=lambda x: x[1], reverse=True)
    return files


def format_size(size):
    for unit in ("B", "KB", "MB"):
        if size < 1024:
            return f"{size:.0f} {unit}"
        size /= 1024
    return f"{size:.1f} GB"


def render_index():
    html = """<!DOCTYPE html>
<html><head><title>Squirrel Defense</title>
<meta name="viewport" content="width=device-width, initial-scale=1">
<style>
body { font-family: system-ui, sans-serif; max-width: 900px; margin: 0 auto; padding: 20px; background: #1a1a2e; color: #eee; }
h1 { text-align: center; }
a { color: #64b5f6; }
.folder { background: #16213e; border-radius: 8px; padding: 16px; margin: 16px 0; }
.folder h2 { margin-top: 0; }
.file-list { list-style: none; padding: 0; }
.file-list li { padding: 4px 0; }
.count { color: #888; font-size: 0.9em; }
</style></head><body>
<h1>Squirrel Defense</h1>
"""
    for name, path in FOLDERS.items():
        files = list_files(path)
        html += f'<div class="folder"><h2>{name}/ <span class="count">({len(files)} files)</span></h2>'
        if files:
            html += '<ul class="file-list">'
            for fname, mtime, size in files[:50]:
                html += f'<li><a href="/files/{name}/{fname}">{fname}</a> — {format_size(size)}</li>'
            if len(files) > 50:
                html += f"<li>... and {len(files) - 50} more</li>"
            html += "</ul>"
        else:
            html += "<p>Empty</p>"
        html += "</div>"

    html += "</body></html>"
    return html


class ThreadedHTTPServer(ThreadingMixIn, HTTPServer):
    daemon_threads = True


class Handler(SimpleHTTPRequestHandler):
    def do_GET(self):
        try:
            self._handle_request()
        except (BrokenPipeError, ConnectionResetError):
            pass
        except Exception as e:
            try:
                self.send_error(500, str(e))
            except Exception:
                pass

    def _handle_request(self):
        if self.path == "/" or self.path == "":
            content = render_index().encode()
            self.send_response(200)
            self.send_header("Content-Type", "text/html")
            self.send_header("Content-Length", len(content))
            self.end_headers()
            self.wfile.write(content)
        elif self.path.startswith("/files/"):
            parts = self.path[7:].split("/", 1)
            if len(parts) == 2 and parts[0] in FOLDERS:
                folder_path = FOLDERS[parts[0]]
                file_path = os.path.join(folder_path, parts[1])
                # Prevent path traversal
                if os.path.realpath(file_path).startswith(os.path.realpath(folder_path)):
                    if os.path.isfile(file_path):
                        self.send_response(200)
                        if file_path.endswith(".jpg") or file_path.endswith(".png"):
                            self.send_header("Content-Type", "image/jpeg" if file_path.endswith(".jpg") else "image/png")
                        elif file_path.endswith(".log"):
                            self.send_header("Content-Type", "text/plain")
                        else:
                            self.send_header("Content-Type", "application/octet-stream")
                        with open(file_path, "rb") as f:
                            data = f.read()
                        self.send_header("Content-Length", len(data))
                        self.end_headers()
                        self.wfile.write(data)
                        return
            self.send_error(404)
        else:
            self.send_error(404)

    def log_message(self, format, *args):
        pass  # suppress request logging


def main():
    parser = argparse.ArgumentParser(description="Web Viewer")
    parser.add_argument("--port", type=int, default=8080)
    args = parser.parse_args()

    server = ThreadedHTTPServer(("0.0.0.0", args.port), Handler)
    print(f"🌐 Web viewer running at http://squirrel-defense.local:{args.port}")
    print(f"   Press Ctrl+C to stop.\n")
    server.serve_forever()


if __name__ == "__main__":
    main()
