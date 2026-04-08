"""
MMM Web Server — Browser-based audio sanitization via drag-and-drop.

Launch with: mmm server [--host 127.0.0.1] [--port 8778] [--max-size 500]
"""

import atexit
import os
import shutil
import tempfile
import threading
import time
import uuid
from pathlib import Path
from typing import Dict, Any

from flask import Flask, request, jsonify, send_file, Response
from werkzeug.utils import secure_filename

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

ALLOWED_EXTENSIONS: frozenset[str] = frozenset({"mp3", "wav", "flac"})
DEFAULT_MAX_FILE_SIZE: int = 500 * 1024 * 1024  # 500 MB
DEFAULT_STALE_AGE: int = 3600  # 1 hour

# ---------------------------------------------------------------------------
# Embedded frontend — CSS
# ---------------------------------------------------------------------------

CSS_STYLES = """\
*,*::before,*::after{box-sizing:border-box;margin:0;padding:0}
body{
  font-family:-apple-system,BlinkMacSystemFont,'Segoe UI',Roboto,monospace;
  background:#0d1117;color:#c9d1d9;min-height:100vh;
  display:flex;flex-direction:column;align-items:center;
}
header{text-align:center;padding:2rem 1rem 0.5rem}
header h1{font-size:1.6rem;color:#58a6ff}
header .subtitle{color:#8b949e;font-size:0.9rem;margin-top:0.3rem}
main{width:100%;max-width:560px;padding:1rem}
.warning-banner{
  background:#1c1208;border:1px solid #d29922;border-radius:6px;
  padding:0.7rem 1rem;font-size:0.75rem;color:#d29922;
  margin-bottom:1.2rem;line-height:1.4;
}
.dropzone{
  border:2px dashed #30363d;border-radius:10px;padding:3rem 1rem;
  text-align:center;cursor:pointer;transition:border-color .2s,background .2s;
}
.dropzone:hover,.dropzone.dragover{border-color:#58a6ff;background:#161b22}
.dropzone-icon{font-size:2.5rem;margin-bottom:0.5rem}
.dropzone p{color:#8b949e;font-size:0.9rem}
.dropzone .hint{font-size:0.75rem;margin-top:0.3rem}
.dropzone .filename{color:#58a6ff;font-weight:600;margin-top:0.6rem;word-break:break-all}
.options-panel{
  background:#161b22;border:1px solid #30363d;border-radius:8px;
  padding:1rem;margin-top:1rem;
}
.option-group{display:flex;align-items:center;gap:0.6rem;margin-bottom:0.8rem}
.option-group label{color:#8b949e;font-size:0.85rem;min-width:110px}
.option-group select{
  background:#0d1117;color:#c9d1d9;border:1px solid #30363d;
  border-radius:4px;padding:0.3rem 0.5rem;font-size:0.85rem;
}
.option-group input[type=checkbox]{accent-color:#58a6ff}
.toggle-label{color:#8b949e;font-size:0.8rem}
.btn-primary{
  display:block;width:100%;padding:0.7rem;margin-top:0.5rem;
  background:#238636;color:#fff;border:none;border-radius:6px;
  font-size:0.95rem;font-weight:600;cursor:pointer;transition:background .2s;
}
.btn-primary:hover{background:#2ea043}
.btn-primary:disabled{background:#21262d;color:#484f58;cursor:not-allowed}
.status-area{text-align:center;margin-top:1.5rem}
.spinner{
  width:36px;height:36px;border:3px solid #30363d;border-top-color:#58a6ff;
  border-radius:50%;margin:0 auto 0.8rem;animation:spin .8s linear infinite;
}
@keyframes spin{to{transform:rotate(360deg)}}
#statusText{color:#8b949e;font-size:0.9rem}
.progress-bar{
  width:100%;height:6px;background:#21262d;border-radius:3px;
  margin-top:0.8rem;overflow:hidden;
}
.progress-fill{
  height:100%;background:#58a6ff;border-radius:3px;
  transition:width .3s;width:0%;
}
.result-area{text-align:center;margin-top:1.5rem}
.success-icon{font-size:2rem;margin-bottom:0.5rem}
#resultText{color:#3fb950;font-size:0.95rem;margin-bottom:0.8rem}
.stats-panel{
  background:#161b22;border:1px solid #30363d;border-radius:6px;
  padding:0.8rem;margin-bottom:1rem;text-align:left;font-size:0.8rem;
}
.stats-panel .stat-row{display:flex;justify-content:space-between;padding:0.2rem 0}
.stats-panel .stat-label{color:#8b949e}
.stats-panel .stat-value{color:#c9d1d9;font-weight:500}
.btn-download{
  display:inline-block;padding:0.6rem 1.5rem;background:#1f6feb;color:#fff;
  border-radius:6px;text-decoration:none;font-weight:600;font-size:0.9rem;
  transition:background .2s;
}
.btn-download:hover{background:#388bfd}
.btn-secondary{
  display:inline-block;padding:0.5rem 1.2rem;margin-top:0.8rem;
  background:transparent;color:#8b949e;border:1px solid #30363d;
  border-radius:6px;font-size:0.85rem;cursor:pointer;transition:border-color .2s;
}
.btn-secondary:hover{border-color:#58a6ff;color:#58a6ff}
.error-area{text-align:center;margin-top:1.5rem}
.error-icon{font-size:2rem;margin-bottom:0.5rem}
#errorText{color:#f85149;font-size:0.9rem;margin-bottom:0.8rem}
footer{margin-top:auto;padding:1.5rem;text-align:center;color:#484f58;font-size:0.7rem}
"""

# ---------------------------------------------------------------------------
# Embedded frontend — JavaScript
# ---------------------------------------------------------------------------

JS_APP = """\
(function(){
  const $ = id => document.getElementById(id);
  const dropzone = $('dropzone');
  const fileInput = $('fileInput');
  const options = $('options');
  const status = $('status');
  const result = $('result');
  const error = $('error');
  let selectedFile = null;

  // --- Drop zone ---
  dropzone.addEventListener('click', () => fileInput.click());
  dropzone.addEventListener('dragover', e => { e.preventDefault(); dropzone.classList.add('dragover'); });
  dropzone.addEventListener('dragleave', () => dropzone.classList.remove('dragover'));
  dropzone.addEventListener('drop', e => {
    e.preventDefault();
    dropzone.classList.remove('dragover');
    if (e.dataTransfer.files.length) handleFile(e.dataTransfer.files[0]);
  });
  fileInput.addEventListener('change', () => { if (fileInput.files.length) handleFile(fileInput.files[0]); });

  function handleFile(file) {
    const ext = file.name.split('.').pop().toLowerCase();
    if (!['mp3','wav','flac'].includes(ext)) {
      showError('Unsupported file type. Please use MP3, WAV, or FLAC.');
      return;
    }
    const maxBytes = parseInt(document.body.dataset.maxSize || '524288000', 10);
    if (file.size > maxBytes) {
      showError('File too large. Maximum size: ' + formatBytes(maxBytes));
      return;
    }
    selectedFile = file;
    // Show filename in dropzone
    let fn = dropzone.querySelector('.filename');
    if (!fn) { fn = document.createElement('p'); fn.className='filename'; dropzone.querySelector('.dropzone-content').appendChild(fn); }
    fn.textContent = file.name + ' (' + formatBytes(file.size) + ')';
    options.hidden = false;
    result.hidden = true;
    error.hidden = true;
  }

  // --- Process ---
  $('processBtn').addEventListener('click', processFile);

  function processFile() {
    if (!selectedFile) return;
    const fmt = $('formatSelect').value;
    const paranoid = $('paranoidToggle').checked;

    // Show status
    options.hidden = true;
    status.hidden = false;
    result.hidden = true;
    error.hidden = true;
    $('statusText').textContent = 'Uploading...';
    $('progressFill').style.width = '0%';
    $('processBtn').disabled = true;

    const formData = new FormData();
    formData.append('file', selectedFile);
    formData.append('format', fmt);
    formData.append('paranoid', paranoid ? 'true' : 'false');

    const xhr = new XMLHttpRequest();
    xhr.upload.addEventListener('progress', e => {
      if (e.lengthComputable) {
        const pct = Math.round((e.loaded / e.total) * 50); // Upload = first 50%
        $('progressFill').style.width = pct + '%';
        $('statusText').textContent = 'Uploading... ' + pct*2 + '%';
      }
    });
    xhr.addEventListener('load', () => {
      if (xhr.status === 200) {
        const data = JSON.parse(xhr.responseText);
        if (data.success) {
          showResult(data);
        } else {
          showError(data.error || 'Sanitization failed.');
        }
      } else if (xhr.status === 429) {
        showError('Server is busy processing another file. Please wait and try again.');
      } else {
        let msg = 'Server error.';
        try { msg = JSON.parse(xhr.responseText).error || msg; } catch(e) {}
        showError(msg);
      }
      $('processBtn').disabled = false;
    });
    xhr.addEventListener('error', () => {
      showError('Network error. Is the server still running?');
      $('processBtn').disabled = false;
    });

    // Start upload
    xhr.open('POST', '/api/upload');
    xhr.send(formData);

    // Simulate processing progress after upload completes
    xhr.upload.addEventListener('loadend', () => {
      $('statusText').textContent = 'Processing audio...';
      $('progressFill').style.width = '50%';
      let p = 50;
      const iv = setInterval(() => {
        if (p < 95) { p += Math.random() * 3; $('progressFill').style.width = Math.min(p, 95) + '%'; }
      }, 500);
      const origLoad = xhr.onload;
      xhr.addEventListener('loadend', () => { clearInterval(iv); $('progressFill').style.width = '100%'; });
    });
  }

  function showResult(data) {
    status.hidden = true;
    result.hidden = false;
    $('resultText').textContent = 'File sanitized successfully!';
    const stats = data.stats || {};
    $('statsPanel').innerHTML =
      '<div class="stat-row"><span class="stat-label">Metadata removed</span><span class="stat-value">' + (stats.metadata_removed || 0) + '</span></div>' +
      '<div class="stat-row"><span class="stat-label">Processing time</span><span class="stat-value">' + formatTime(stats.processing_time) + '</span></div>' +
      '<div class="stat-row"><span class="stat-label">Speed</span><span class="stat-value">' + (stats.processing_speed || 'N/A') + '</span></div>';
    $('downloadLink').href = '/api/download/' + data.download_token;
    $('downloadLink').download = data.filename || 'cleaned_audio';
  }

  function showError(msg) {
    status.hidden = true;
    options.hidden = true;
    error.hidden = false;
    $('errorText').textContent = msg;
    $('processBtn').disabled = false;
  }

  $('resetBtn').addEventListener('click', resetUI);
  $('retryBtn').addEventListener('click', resetUI);

  function resetUI() {
    selectedFile = null;
    fileInput.value = '';
    const fn = dropzone.querySelector('.filename');
    if (fn) fn.remove();
    options.hidden = true;
    status.hidden = true;
    result.hidden = true;
    error.hidden = true;
    $('processBtn').disabled = false;
  }

  function formatBytes(b) {
    if (b < 1024) return b + ' B';
    if (b < 1048576) return (b/1024).toFixed(1) + ' KB';
    return (b/1048576).toFixed(1) + ' MB';
  }
  function formatTime(s) {
    if (s == null) return 'N/A';
    return parseFloat(s).toFixed(1) + 's';
  }
})();
"""

# ---------------------------------------------------------------------------
# Embedded frontend — HTML shell
# ---------------------------------------------------------------------------

HTML_TEMPLATE = """\
<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>MMM - Audio Sanitizer</title>
<style>{css}</style>
</head>
<body data-max-size="{max_size}">
<header>
  <h1>Melodic Metadata Massacrer</h1>
  <p class="subtitle">Browser-based audio sanitizer</p>
</header>
<main>
  <div class="warning-banner">
    LEGAL DISCLAIMER: This tool is for AUTHORIZED SECURITY RESEARCH ONLY.
    Use only on files you own or have explicit permission to modify.
    You are responsible for compliance with applicable laws.
  </div>

  <div id="dropzone" class="dropzone">
    <div class="dropzone-content">
      <div class="dropzone-icon">&#127925;</div>
      <p>Drag &amp; drop audio file here</p>
      <p class="hint">or click to select &middot; MP3, WAV, FLAC &middot; max {max_size_mb} MB</p>
      <input type="file" id="fileInput" accept=".mp3,.wav,.flac" hidden>
    </div>
  </div>

  <div id="options" class="options-panel" hidden>
    <div class="option-group">
      <label for="formatSelect">Output format:</label>
      <select id="formatSelect">
        <option value="preserve" selected>Preserve original</option>
        <option value="mp3">MP3</option>
        <option value="wav">WAV</option>
      </select>
    </div>
    <div class="option-group">
      <label for="paranoidToggle">Paranoid mode:</label>
      <input type="checkbox" id="paranoidToggle">
      <span class="toggle-label">Maximum destruction</span>
    </div>
    <button id="processBtn" class="btn-primary">Sanitize</button>
  </div>

  <div id="status" class="status-area" hidden>
    <div class="spinner" id="spinner"></div>
    <p id="statusText">Processing...</p>
    <div class="progress-bar"><div class="progress-fill" id="progressFill"></div></div>
  </div>

  <div id="result" class="result-area" hidden>
    <div class="success-icon">&#9989;</div>
    <p id="resultText"></p>
    <div id="statsPanel" class="stats-panel"></div>
    <a id="downloadLink" class="btn-download" href="#">Download cleaned file</a><br>
    <button id="resetBtn" class="btn-secondary">Process another file</button>
  </div>

  <div id="error" class="error-area" hidden>
    <div class="error-icon">&#10060;</div>
    <p id="errorText"></p>
    <button id="retryBtn" class="btn-secondary">Try again</button>
  </div>
</main>
<footer>MMM v2.0.0 &mdash; Authorized Security Research Tool</footer>
<script>{js}</script>
</body>
</html>
"""


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _validate_extension(filename: str) -> bool:
    """Check file extension against the allowlist."""
    if not filename or "." not in filename:
        return False
    ext = filename.rsplit(".", 1)[-1].lower()
    return ext in ALLOWED_EXTENSIONS


def _cleanup_old_files(directory: Path, max_age: int = DEFAULT_STALE_AGE) -> None:
    """Remove files older than *max_age* seconds from *directory*."""
    now = time.time()
    try:
        for entry in directory.iterdir():
            if entry.is_file():
                age = now - entry.stat().st_mtime
                if age > max_age:
                    entry.unlink(missing_ok=True)
    except OSError:
        pass


# ---------------------------------------------------------------------------
# Flask application factory
# ---------------------------------------------------------------------------

def create_app(
    max_file_size: int = DEFAULT_MAX_FILE_SIZE,
) -> Flask:
    """Create and configure the Flask application."""
    app = Flask(__name__)
    app.config["MAX_CONTENT_LENGTH"] = max_file_size
    app.config["SECRET_KEY"] = os.urandom(32)

    # Secure temp directory — cleaned up on exit
    temp_dir = Path(tempfile.mkdtemp(prefix="mmm_server_"))
    app.config["UPLOAD_FOLDER"] = temp_dir
    atexit.register(shutil.rmtree, str(temp_dir), True)

    # One-at-a-time processing lock
    app.config["PROCESSING_LOCK"] = threading.Lock()

    # Download registry: token -> {"path": Path, "filename": str, "created": float}
    app.config["DOWNLOAD_REGISTRY"]: Dict[str, Dict[str, Any]] = {}

    max_size_mb = max_file_size // (1024 * 1024)

    # --- Routes ---

    @app.route("/")
    def index() -> Response:
        html = HTML_TEMPLATE.format(
            css=CSS_STYLES,
            js=JS_APP,
            max_size=max_file_size,
            max_size_mb=max_size_mb,
        )
        return Response(html, mimetype="text/html")

    @app.route("/api/status")
    def api_status() -> tuple:
        lock: threading.Lock = app.config["PROCESSING_LOCK"]
        busy = not lock.acquire(blocking=False)
        if not busy:
            lock.release()
        return jsonify({
            "busy": busy,
            "version": "2.0.0",
            "max_file_size_mb": max_size_mb,
        }), 200

    @app.route("/api/upload", methods=["POST"])
    def api_upload() -> tuple:
        lock: threading.Lock = app.config["PROCESSING_LOCK"]

        if not lock.acquire(blocking=False):
            return jsonify({"error": "Server is busy processing another file. Please wait."}), 429

        try:
            return _handle_upload(app)
        except Exception:
            app.logger.exception("Upload handler failed")
            return jsonify({"error": "Sanitization failed. Please try a different file."}), 500
        finally:
            lock.release()

    @app.route("/api/download/<token>")
    def api_download(token: str) -> tuple:
        registry: dict = app.config["DOWNLOAD_REGISTRY"]
        entry = registry.pop(token, None)  # Atomic remove from registry

        if entry is None:
            return jsonify({"error": "File not found or expired."}), 404

        file_path = Path(entry["path"])
        if not file_path.exists():
            return jsonify({"error": "File not found or expired."}), 404

        filename = entry["filename"]

        response = send_file(
            str(file_path),
            as_attachment=True,
            download_name=filename,
        )
        # Clean up file after response is sent to client
        response.call_on_close(lambda: file_path.unlink(missing_ok=True))
        return response

    @app.errorhandler(413)
    def _too_large(e):
        return jsonify({"error": f"File too large. Maximum size: {max_size_mb} MB."}), 413

    return app


def _handle_upload(app: Flask) -> tuple:
    """Process an uploaded file. Called inside the processing lock."""
    temp_dir: Path = app.config["UPLOAD_FOLDER"]
    registry: dict = app.config["DOWNLOAD_REGISTRY"]

    # Reap stale files before processing
    _cleanup_old_files(temp_dir)

    # Validate upload
    if "file" not in request.files:
        return jsonify({"error": "No file provided."}), 400

    f = request.files["file"]
    if not f.filename:
        return jsonify({"error": "No file selected."}), 400

    safe_name = secure_filename(f.filename)
    if not safe_name or not _validate_extension(safe_name):
        return jsonify({"error": "Unsupported file type. Use MP3, WAV, or FLAC."}), 400

    # Parse options
    output_format = request.form.get("format", "preserve")
    if output_format not in ("preserve", "mp3", "wav"):
        output_format = "preserve"

    paranoid = request.form.get("paranoid", "false").lower() == "true"

    # Save to temp directory with UUID prefix
    unique_name = f"{uuid.uuid4().hex}_{safe_name}"
    input_path = temp_dir / unique_name
    f.save(str(input_path))

    try:
        # Import preserving sanitizer (deferred to avoid import overhead)
        from .preserving_sanitizer import preserving_sanitize

        fmt = None if output_format == "preserve" else output_format
        result = preserving_sanitize(
            input_file=input_path,
            output_file=None,  # auto-generates .clean.<ext>
            paranoid_mode=paranoid,
            threat_count=0,
            output_format=fmt,
        )

        if not result.get("success"):
            return jsonify({"error": "Sanitization failed."}), 500

        # Register output for download
        output_path = Path(result["output_file"])
        token = uuid.uuid4().hex

        # Build a clean download filename
        stem = Path(safe_name).stem
        ext = output_path.suffix
        download_name = f"{stem}_clean{ext}"

        registry[token] = {
            "path": str(output_path),
            "filename": download_name,
            "created": time.time(),
        }

        return jsonify({
            "success": True,
            "download_token": token,
            "filename": download_name,
            "stats": result.get("stats", {}),
        }), 200

    finally:
        # Always clean up the input file
        input_path.unlink(missing_ok=True)


# ---------------------------------------------------------------------------
# Entry point (called from CLI)
# ---------------------------------------------------------------------------

def run_server(
    host: str = "127.0.0.1",
    port: int = 8778,
    max_file_size: int = DEFAULT_MAX_FILE_SIZE,
) -> None:
    """Create and run the MMM web server."""
    from .ui.console import ConsoleManager
    from .ui.banners import BannerManager

    console = ConsoleManager()
    banner = BannerManager()

    banner.show_main_banner()

    console.warning(
        "LEGAL DISCLAIMER: This tool is for AUTHORIZED SECURITY RESEARCH ONLY"
    )
    console.info("   Use only on files you own or have explicit permission to modify\n")

    if host != "127.0.0.1":
        console.warning(
            f"Binding to {host} — the server will be accessible from the network!"
        )

    max_mb = max_file_size // (1024 * 1024)
    console.success(f"Starting MMM web server at http://{host}:{port}")
    console.info(f"   Max upload size: {max_mb} MB")
    console.info("   Press Ctrl+C to stop\n")

    app = create_app(max_file_size=max_file_size)

    # debug=False is mandatory — Werkzeug debugger enables remote code execution
    app.run(host=host, port=port, debug=False, threaded=True)
