#!/usr/bin/env -S uv run
# /// script
# requires-python = ">=3.11"
# dependencies = ["huggingface-hub", "requests", "Pillow"]
# ///

"""Capture screen and describe it using Qwen3.5-0.8B via llama-server."""

import base64
import shutil
import subprocess
import sys
import time
from pathlib import Path

MODEL_REPO = "unsloth/Qwen3.5-0.8B-GGUF"
MODEL_FILE = "Qwen3.5-0.8B-Q4_K_M.gguf"
MMPROJ_FILE = "mmproj-F16.gguf"
MODEL_DIR = Path(__file__).resolve().parent / "Qwen3.5-0.8B-GGUF"
PORT = 8001


def check_tools():
    for tool in ("llama-server",):
        if not shutil.which(tool):
            sys.exit(f"[ERROR] '{tool}' not found. Install it first.")
    print("[OK] All tools available")


def download_model():
    from huggingface_hub import hf_hub_download

    model_dir = MODEL_DIR
    model_path = model_dir / MODEL_FILE
    mmproj_path = model_dir / MMPROJ_FILE

    if model_path.exists() and mmproj_path.exists():
        print(f"[OK] Model already downloaded at {model_dir}")
        return model_path, mmproj_path

    print(f"[DL] Downloading {MODEL_REPO} (Q4_K_M + mmproj)...")
    model_dir.mkdir(parents=True, exist_ok=True)

    for filename in (MODEL_FILE, MMPROJ_FILE):
        hf_hub_download(
            repo_id=MODEL_REPO,
            filename=filename,
            local_dir=str(model_dir),
        )

    print(f"[OK] Model downloaded to {model_dir}")
    return model_path, mmproj_path


def start_server(model_path, mmproj_path):
    import requests

    # Test if server is already running
    try:
        r = requests.get(f"http://localhost:{PORT}/health", timeout=1)
        if r.status_code == 200:
            print("[OK] llama-server is already running")
            return None
    except requests.ConnectionError:
        pass

    print(f"[RUN] Starting llama-server on port {PORT}...")
    log_file = open("/tmp/llama_server.log", "w")
    proc = subprocess.Popen(
        [
            "llama-server",
            "--model",
            str(model_path),
            "--mmproj",
            str(mmproj_path),
            "--ctx-size",
            "8192",
            "--port",
            str(PORT),
            "--jinja",
        ],
        stdout=log_file,
        stderr=log_file,
    )

    for i in range(30):
        try:
            r = requests.get(f"http://localhost:{PORT}/health", timeout=2)
            if r.status_code == 200:
                print("[OK] llama-server is ready")
                return proc
        except requests.ConnectionError:
            pass
        time.sleep(2)

    proc.kill()
    sys.exit("[ERROR] llama-server failed to start within 60s")


def capture_screen():
    desktop = Path.home() / "Desktop"
    print(f"[SCAN] Scanning {desktop} for screenshot files...")

    try:
        images = list(desktop.glob("*.png")) + list(desktop.glob("*.jpg"))
    except PermissionError:
        sys.exit(
            "[ERROR] Terminal lacks permission to read your Desktop. Please grant 'Files and Folders' or 'Full Disk Access' to Terminal in macOS System Settings > Privacy & Security."
        )

    if not images:
        sys.exit(
            f"[ERROR] No .png or .jpg image files found on your Desktop ({desktop})."
        )

    # Pick the most recently modified image
    latest_img = max(images, key=lambda p: p.stat().st_mtime)
    print(f"[OK] Selected latest image: {latest_img.name}")
    return latest_img


def describe_screen(img_path):
    import requests
    from PIL import Image
    import io

    # Resize large screenshots (Retina displays can be 5000+ px)
    # The 0.8B model works best with smaller images
    img = Image.open(img_path)
    max_dim = 1024
    if max(img.size) > max_dim:
        ratio = max_dim / max(img.size)
        new_size = (int(img.width * ratio), int(img.height * ratio))
        print(f"[RESIZE] Resizing {img.size} → {new_size} for model compatibility")
        img = img.resize(new_size, Image.LANCZOS)

    # Convert to JPEG bytes for smaller payload and guaranteed compatibility
    buf = io.BytesIO()
    img.convert("RGB").save(buf, format="JPEG", quality=85)
    b64 = base64.b64encode(buf.getvalue()).decode()
    print(f"[SIZE] Image payload size: {len(b64) // 1024} KB (base64)")

    content = [
        {
            "type": "image_url",
            "image_url": {"url": f"data:image/jpeg;base64,{b64}"},
        },
        {
            "type": "text",
            "text": "Describe this screenshot in detail. What is taking place? What applications are open?",
        },
    ]

    print("[AI] Analyzing screenshot with Qwen3.5-0.8B...")
    resp = requests.post(
        f"http://localhost:{PORT}/v1/chat/completions",
        json={
            "model": "qwen3.5-0.8b",
            "messages": [{"role": "user", "content": content}],
            "max_tokens": 4096,
            "temperature": 0.7,
            "top_p": 0.8,
        },
        timeout=180,
    )
    resp.raise_for_status()

    data = resp.json()
    print(f"\n[SCAN] Raw API response keys: {list(data.keys())}")
    if data.get("choices"):
        choice = data["choices"][0]
        print(f"[SCAN] Finish reason: {choice.get('finish_reason')}")
        print(f"[SCAN] Message role: {choice.get('message', {}).get('role')}")
        print(
            f"[SCAN] Content length: {len(choice.get('message', {}).get('content', ''))}"
        )
    else:
        print(f"[SCAN] Full response: {data}")

    answer = data["choices"][0]["message"]["content"]

    print("\n" + "=" * 60)
    print("[DESC] SCREENSHOT DESCRIPTION:")
    print("=" * 60)
    print(answer if answer else "(empty response from model)")
    print("=" * 60 + "\n")


def main():
    check_tools()
    model_path, mmproj_path = download_model()
    server = start_server(model_path, mmproj_path)

    try:
        img_path = capture_screen()
        describe_screen(img_path)
    except Exception as e:
        print(f"[ERROR] Error occurred: {e}")
    finally:
        if server:
            print("[STOP] Stopping llama-server...")
            server.terminate()
            server.wait(timeout=5)
        print("[DONE]")


if __name__ == "__main__":
    main()
