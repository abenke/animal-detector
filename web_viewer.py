"""
🌐 Web Viewer — browse captures, detections, and logs from your browser.
Also provides a Controls page for running capture/calibrate/service commands.

Usage:
    python web_viewer.py            # start on port 8080
    python web_viewer.py --port 9000

Then open http://squirrel-defense.local:8080 in your browser.
"""

import argparse
import json
import os
import subprocess
import sys
from http.server import HTTPServer, SimpleHTTPRequestHandler
from socketserver import ThreadingMixIn
from urllib.parse import parse_qs

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PYTHON = sys.executable  # same interpreter that started this script (picks up venv)

FOLDERS = {
    "captures": os.path.join(BASE_DIR, "captures"),
    "detections": os.path.join(BASE_DIR, "detections"),
    "logs": os.path.join(BASE_DIR, "logs"),
    "output": os.path.join(BASE_DIR, "output"),
}

SERVICE = "squirrel-defense"


def list_files(folder_path):
    """List files in a folder (recursive), newest first.

    Returns tuples of (relative_path, mtime, size).
    """
    if not os.path.isdir(folder_path):
        return []
    files = []
    for root, _, filenames in os.walk(folder_path):
        for f in filenames:
            full = os.path.join(root, f)
            rel = os.path.relpath(full, folder_path)
            files.append((rel, os.path.getmtime(full), os.path.getsize(full)))
    files.sort(key=lambda x: x[1], reverse=True)
    return files


def format_size(size):
    for unit in ("B", "KB", "MB"):
        if size < 1024:
            return f"{size:.0f} {unit}"
        size /= 1024
    return f"{size:.1f} GB"


def run_command(cmd, timeout=60):
    """Run a command and return (exit_code, stdout, stderr)."""
    try:
        result = subprocess.run(
            cmd, cwd=BASE_DIR, capture_output=True, text=True, timeout=timeout
        )
        return result.returncode, result.stdout, result.stderr
    except subprocess.TimeoutExpired:
        return -1, "", f"Command timed out after {timeout}s"
    except Exception as e:
        return -1, "", str(e)


def service_status():
    """Return 'active', 'inactive', or 'unknown'."""
    code, out, _ = run_command(["systemctl", "is-active", SERVICE], timeout=5)
    return out.strip() or "unknown"


def get_current_focus():
    """Read lens_position from camera_config.json."""
    path = os.path.join(BASE_DIR, "camera_config.json")
    if os.path.exists(path):
        with open(path) as f:
            cfg = json.load(f)
        return cfg.get("lens_position")
    return None


def get_current_crop():
    """Read crop region from camera_config.json."""
    path = os.path.join(BASE_DIR, "camera_config.json")
    if os.path.exists(path):
        with open(path) as f:
            cfg = json.load(f)
        return cfg.get("crop")
    return None


# ============================================================
#  HTML rendering
# ============================================================

STYLES = """
body { font-family: system-ui, sans-serif; max-width: 900px; margin: 0 auto; padding: 20px; background: #1a1a2e; color: #eee; }
h1 { text-align: center; }
h2 { margin-top: 0; }
a { color: #64b5f6; }
nav { text-align: center; margin-bottom: 20px; }
nav a { display: inline-block; padding: 8px 16px; background: #16213e; border-radius: 6px; margin: 0 4px; text-decoration: none; }
nav a.active { background: #0f3460; font-weight: bold; }
.card { background: #16213e; border-radius: 8px; padding: 16px; margin: 16px 0; }
.file-list { list-style: none; padding: 0; }
.file-list li { padding: 4px 0; }
.count { color: #888; font-size: 0.9em; }
button { background: #0f3460; color: #eee; border: none; padding: 10px 18px; border-radius: 6px; cursor: pointer; font-size: 14px; margin: 4px 2px; }
button:hover { background: #1a4a80; }
button.danger { background: #a03030; }
button.danger:hover { background: #c04040; }
button.success { background: #2e7d32; }
button.success:hover { background: #388e3c; }
input[type=number], input[type=text] { background: #0a1020; color: #eee; border: 1px solid #444; padding: 6px 10px; border-radius: 4px; font-size: 14px; width: 100px; }
.status { padding: 8px 12px; border-radius: 4px; display: inline-block; font-weight: bold; }
.status.active { background: #2e7d32; }
.status.inactive { background: #555; }
pre { background: #0a1020; padding: 12px; border-radius: 6px; overflow-x: auto; white-space: pre-wrap; font-size: 13px; max-height: 300px; overflow-y: auto; }
.row { display: flex; gap: 8px; align-items: center; margin: 8px 0; flex-wrap: wrap; }
label { color: #aaa; }
"""

NAV = """
<nav>
  <a href="/" {home_active}>Files</a>
  <a href="/controls" {controls_active}>Controls</a>
</nav>
"""


def render_nav(page):
    return NAV.format(
        home_active='class="active"' if page == "home" else "",
        controls_active='class="active"' if page == "controls" else "",
    )


def render_index():
    html = f"""<!DOCTYPE html>
<html><head><title>Squirrel Defense</title>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<style>{STYLES}</style></head><body>
<h1>Squirrel Defense</h1>
{render_nav("home")}
"""
    for name, path in FOLDERS.items():
        files = list_files(path)
        html += f'<div class="card"><h2>{name}/ <span class="count">({len(files)} files)</span></h2>'
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


def render_controls(message=None, output=None):
    status = service_status()
    focus = get_current_focus()
    focus_display = (
        f"autofocus" if focus is not None and focus < 0
        else f"{focus:.2f} (~{1/focus:.1f}m)" if focus and focus > 0
        else "infinity" if focus == 0
        else "not set"
    )

    crop = get_current_crop()
    if crop:
        crop_display = f"left={crop['left']}, top={crop['top']}, right={crop['right']}, bottom={crop['bottom']} ({crop['right']-crop['left']}x{crop['bottom']-crop['top']})"
        crop_left, crop_top, crop_right, crop_bottom = crop["left"], crop["top"], crop["right"], crop["bottom"]
    else:
        crop_display = "not set"
        crop_left = crop_top = crop_right = crop_bottom = ""

    msg_html = f'<div class="card"><strong>{message}</strong></div>' if message else ""
    out_html = f'<div class="card"><h2>Output</h2><pre>{output}</pre></div>' if output else ""

    html = f"""<!DOCTYPE html>
<html><head><title>Squirrel Defense — Controls</title>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<style>{STYLES}</style></head><body>
<h1>Squirrel Defense</h1>
{render_nav("controls")}

{msg_html}

<div class="card">
  <h2>Service</h2>
  <div class="row">
    Status: <span class="status {status}">{status.upper()}</span>
  </div>
  <form method="post" action="/action" style="display:inline">
    <input type="hidden" name="action" value="service_start">
    <button class="success" type="submit">Start</button>
  </form>
  <form method="post" action="/action" style="display:inline">
    <input type="hidden" name="action" value="service_stop">
    <button class="danger" type="submit">Stop</button>
  </form>
  <form method="post" action="/action" style="display:inline">
    <input type="hidden" name="action" value="service_restart">
    <button type="submit">Restart</button>
  </form>
  <p class="count">Stop the service before running camera commands below.</p>
</div>

<div class="card">
  <h2>Camera</h2>
  <form method="post" action="/action" style="display:inline">
    <input type="hidden" name="action" value="snapshot">
    <button type="submit">📷 Take snapshot</button>
  </form>
  <form method="post" action="/action" style="display:inline">
    <input type="hidden" name="action" value="snapshot_nocrop">
    <button type="submit">📷 Snapshot (no crop)</button>
  </form>
  <form method="post" action="/action" style="display:inline">
    <input type="hidden" name="action" value="calibrate_crop">
    <button type="submit">✂️ Calibrate crop (grid)</button>
  </form>
</div>

<div class="card">
  <h2>Crop Region</h2>
  <p>Current: <strong>{crop_display}</strong></p>
  <form method="post" action="/action" class="row">
    <input type="hidden" name="action" value="set_crop">
    <label>Left:</label> <input type="number" name="left" value="{crop_left}" required>
    <label>Top:</label> <input type="number" name="top" value="{crop_top}" required>
    <label>Right:</label> <input type="number" name="right" value="{crop_right}" required>
    <label>Bottom:</label> <input type="number" name="bottom" value="{crop_bottom}" required>
    <button type="submit">Save</button>
  </form>
  <form method="post" action="/action" style="display:inline">
    <input type="hidden" name="action" value="clear_crop">
    <button class="danger" type="submit">Clear crop</button>
  </form>
  <p class="count">Tip: Run "Calibrate crop (grid)" first, open captures/calibration.jpg, then fill in pixel coordinates.</p>
</div>

<div class="card">
  <h2>Focus</h2>
  <p>Current focus: <strong>{focus_display}</strong></p>
  <form method="post" action="/action" style="display:inline">
    <input type="hidden" name="action" value="focus_sweep">
    <button type="submit">🔍 Run focus sweep</button>
  </form>
  <form method="post" action="/action" class="row">
    <input type="hidden" name="action" value="set_focus">
    <label>Set lens position:</label>
    <input type="number" step="0.01" name="value" placeholder="0.33" required>
    <button type="submit">Save</button>
  </form>
  <p class="count">Lens position ≈ 1/distance_in_meters. 0 = infinity, -1 = autofocus.</p>
</div>

{out_html}
</body></html>
"""
    return html


# ============================================================
#  Action handlers
# ============================================================

def handle_action(form):
    """Run an action and return (message, output)."""
    action = form.get("action", [""])[0]

    if action == "snapshot":
        code, out, err = run_command([PYTHON, "capture.py"])
        return f"Snapshot {'taken' if code == 0 else 'failed'}", (out + err).strip()

    if action == "snapshot_nocrop":
        code, out, err = run_command([PYTHON, "capture.py", "--no-crop"])
        return f"Snapshot {'taken' if code == 0 else 'failed'}", (out + err).strip()

    if action == "calibrate_crop":
        code, out, err = run_command([PYTHON, "capture.py", "--calibrate"])
        return f"Crop calibration {'done' if code == 0 else 'failed'}", (out + err).strip()

    if action == "focus_sweep":
        code, out, err = run_command([PYTHON, "calibrate_focus.py"], timeout=120)
        return f"Focus sweep {'done' if code == 0 else 'failed'}", (out + err).strip()

    if action == "set_focus":
        value = form.get("value", [""])[0]
        if not value:
            return "Missing value", ""
        code, out, err = run_command([PYTHON, "calibrate_focus.py", "--set-focus", value])
        return f"Focus {'set' if code == 0 else 'failed'}", (out + err).strip()

    if action == "set_crop":
        try:
            left = form.get("left", [""])[0]
            top = form.get("top", [""])[0]
            right = form.get("right", [""])[0]
            bottom = form.get("bottom", [""])[0]
            code, out, err = run_command(
                [PYTHON, "capture.py", "--set-crop", left, top, right, bottom]
            )
            return f"Crop {'set' if code == 0 else 'failed'}", (out + err).strip()
        except Exception as e:
            return "Error setting crop", str(e)

    if action == "clear_crop":
        code, out, err = run_command([PYTHON, "capture.py", "--clear-crop"])
        return f"Crop {'cleared' if code == 0 else 'failed'}", (out + err).strip()

    if action in ("service_start", "service_stop", "service_restart"):
        cmd = action.replace("service_", "")
        code, out, err = run_command(["sudo", "-n", "systemctl", cmd, SERVICE], timeout=15)
        msg = f"Service {cmd} {'succeeded' if code == 0 else 'failed'}"
        output = (out + err).strip()
        if "a password is required" in output.lower() or "sudo:" in output.lower():
            output += (
                "\n\nTo enable service control from the web UI, set up passwordless sudo:\n"
                "  sudo visudo -f /etc/sudoers.d/squirrel-defense\n"
                "Add this line:\n"
                f"  ttboss ALL=(ALL) NOPASSWD: /bin/systemctl start {SERVICE}, "
                f"/bin/systemctl stop {SERVICE}, /bin/systemctl restart {SERVICE}"
            )
        return msg, output

    return "Unknown action", ""


# ============================================================
#  HTTP server
# ============================================================

class ThreadedHTTPServer(ThreadingMixIn, HTTPServer):
    daemon_threads = True


class Handler(SimpleHTTPRequestHandler):
    def do_GET(self):
        try:
            self._handle_get()
        except (BrokenPipeError, ConnectionResetError):
            pass
        except Exception as e:
            try:
                self.send_error(500, str(e))
            except Exception:
                pass

    def do_POST(self):
        try:
            self._handle_post()
        except (BrokenPipeError, ConnectionResetError):
            pass
        except Exception as e:
            try:
                self.send_error(500, str(e))
            except Exception:
                pass

    def _send_html(self, body, code=200):
        content = body.encode("utf-8")
        self.send_response(code)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.send_header("Content-Length", len(content))
        self.end_headers()
        self.wfile.write(content)

    def _handle_get(self):
        if self.path in ("/", ""):
            self._send_html(render_index())
        elif self.path == "/controls":
            self._send_html(render_controls())
        elif self.path.startswith("/files/"):
            self._serve_file()
        else:
            self.send_error(404)

    def _handle_post(self):
        if self.path == "/action":
            length = int(self.headers.get("Content-Length", 0))
            body = self.rfile.read(length).decode("utf-8")
            form = parse_qs(body)
            message, output = handle_action(form)
            self._send_html(render_controls(message=message, output=output))
        else:
            self.send_error(404)

    def _serve_file(self):
        parts = self.path[7:].split("/", 1)
        if len(parts) == 2 and parts[0] in FOLDERS:
            folder_path = FOLDERS[parts[0]]
            file_path = os.path.join(folder_path, parts[1])
            if os.path.realpath(file_path).startswith(os.path.realpath(folder_path)):
                if os.path.isfile(file_path):
                    self.send_response(200)
                    if file_path.endswith(".jpg"):
                        self.send_header("Content-Type", "image/jpeg")
                    elif file_path.endswith(".png"):
                        self.send_header("Content-Type", "image/png")
                    elif file_path.endswith(".log"):
                        self.send_header("Content-Type", "text/plain; charset=utf-8")
                    else:
                        self.send_header("Content-Type", "application/octet-stream")
                    with open(file_path, "rb") as f:
                        data = f.read()
                    self.send_header("Content-Length", len(data))
                    self.end_headers()
                    self.wfile.write(data)
                    return
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
