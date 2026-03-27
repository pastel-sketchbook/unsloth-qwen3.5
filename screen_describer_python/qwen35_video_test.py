#!/usr/bin/env -S uv run
"""Test Qwen3.5-0.8B video-to-text via llama-server using a YouTube video from the codebase."""
# /// script
# requires-python = ">=3.11"
# dependencies = ["huggingface-hub", "requests", "Pillow"]
# ///

import base64
import shutil
import subprocess
import sys
import time
from pathlib import Path

MODEL_REPO = "unsloth/Qwen3.5-0.8B-GGUF"
MODEL_FILE = "Qwen3.5-0.8B-Q4_K_M.gguf"
MMPROJ_FILE = "mmproj-F16.gguf"
MODEL_DIR = Path(__file__).resolve().parent.parent / "Qwen3.5-0.8B-GGUF"
YT_VIDEO_ID = "V2cZl5s4EKU"
PORT = 8001
FRAMES_DIR = Path("/tmp/qwen35_frames")
VIDEO_PATH = Path("/tmp/qwen35_yt_clip.mkv")
OUTPUT_MD = Path("video_text_output.md")


def check_tools():
    for tool in ("llama-server", "yt-dlp", "ffmpeg"):
        if not shutil.which(tool):
            sys.exit(f"❌ {tool} not found. Install it first.")
    print("✅ All tools available")


def download_model():
    from huggingface_hub import hf_hub_download

    model_dir = MODEL_DIR
    model_path = model_dir / MODEL_FILE
    mmproj_path = model_dir / MMPROJ_FILE

    if model_path.exists() and mmproj_path.exists():
        print(f"✅ Model already downloaded at {model_dir}")
        return model_path, mmproj_path

    print(f"⬇️  Downloading {MODEL_REPO} (Q4_K_M + mmproj)...")
    model_dir.mkdir(parents=True, exist_ok=True)

    for filename in (MODEL_FILE, MMPROJ_FILE):
        hf_hub_download(
            repo_id=MODEL_REPO,
            filename=filename,
            local_dir=str(model_dir),
        )

    print(f"✅ Model downloaded to {model_dir}")
    return model_path, mmproj_path


def download_yt_clip():
    if VIDEO_PATH.exists():
        print(f"✅ Video clip already at {VIDEO_PATH}")
        return

    print(f"⬇️  Downloading full video of https://youtube.com/watch?v={YT_VIDEO_ID}")
    subprocess.run(
        [
            "yt-dlp",
            "-f",
            "bestvideo[height<=480]+bestaudio/best[height<=480]",
            "--force-keyframes-at-cuts",
            "--merge-output-format",
            "mkv",
            "-o",
            str(VIDEO_PATH),
            "--no-warnings",
            f"https://www.youtube.com/watch?v={YT_VIDEO_ID}",
        ],
        check=True,
    )
    print("✅ Video clip downloaded")


def extract_frames():
    FRAMES_DIR.mkdir(parents=True, exist_ok=True)
    existing = list(FRAMES_DIR.glob("frame_*.jpg"))
    if existing:
        print(f"✅ {len(existing)} frames already extracted")
        return

    print("🎞️  Extracting frames at 1fps from clip...")
    subprocess.run(
        [
            "ffmpeg",
            "-y",
            "-i",
            str(VIDEO_PATH),
            "-vf",
            "fps=1",
            f"{FRAMES_DIR}/frame_%05d.jpg",
        ],
        check=True,
        capture_output=True,
    )
    count = len(list(FRAMES_DIR.glob("frame_*.jpg")))
    print(f"✅ Extracted {count} frames")


def deduplicate_frames():
    """Compare frames and return only unique ones (by perceptual hash)."""
    from PIL import Image

    frames = sorted(FRAMES_DIR.glob("frame_*.jpg"))
    if not frames:
        return []

    def frame_hash(path):
        img = Image.open(path).convert("L").resize((16, 16))
        pixels = list(img.getdata())
        avg = sum(pixels) / len(pixels)
        return "".join("1" if p > avg else "0" for p in pixels)

    unique = [frames[0]]
    prev_hash = frame_hash(frames[0])

    for f in frames[1:]:
        h = frame_hash(f)
        diff = sum(a != b for a, b in zip(prev_hash, h))
        print(f"  {f.name}: diff={diff}/256 {'✅ unique' if diff > 5 else '⏭️  skip'}")
        if diff > 5:
            unique.append(f)
            prev_hash = h

    print(
        f"🔍 {len(frames)} frames → {len(unique)} unique (skipped {len(frames) - len(unique)} duplicates)"
    )
    return unique


def start_server(model_path, mmproj_path):
    print(f"🚀 Starting llama-server on port {PORT}...")
    proc = subprocess.Popen(
        [
            "llama-server",
            "--model",
            str(model_path),
            "--mmproj",
            str(mmproj_path),
            "--ctx-size",
            "4096",
            "--port",
            str(PORT),
            "--jinja",
        ],
        stdout=sys.stderr,
        stderr=sys.stderr,
    )

    import requests

    for i in range(30):
        try:
            r = requests.get(f"http://localhost:{PORT}/health", timeout=2)
            if r.status_code == 200:
                print("✅ llama-server is ready")
                return proc
        except requests.ConnectionError:
            pass
        time.sleep(2)

    proc.kill()
    sys.exit("❌ llama-server failed to start within 60s")


def query_frame(frame_path):
    import requests

    b64 = base64.b64encode(frame_path.read_bytes()).decode()
    content = [
        {
            "type": "image_url",
            "image_url": {"url": f"data:image/jpeg;base64,{b64}"},
        },
        {
            "type": "text",
            "text": "Read and extract all visible text from this image. If there is no text, describe what is shown.",
        },
    ]

    resp = requests.post(
        f"http://localhost:{PORT}/v1/chat/completions",
        json={
            "model": "qwen3.5-0.8b",
            "messages": [{"role": "user", "content": content}],
            "max_tokens": 2048,
            "temperature": 0.7,
            "top_p": 0.8,
        },
        timeout=120,
    )
    resp.raise_for_status()
    return resp.json()["choices"][0]["message"]["content"]


def query_model():
    unique_frames = deduplicate_frames()
    if not unique_frames:
        sys.exit("❌ No frames found")

    all_frames = sorted(FRAMES_DIR.glob("frame_*.jpg"))
    results = []

    print(f"\n🎬 Processing {len(unique_frames)} unique frames (1 at a time)")
    print("=" * 60)

    for i, frame in enumerate(unique_frames, 1):
        sec = all_frames.index(frame) + 1
        print(f"\n📨 Frame {i}/{len(unique_frames)} (second {sec})...")
        try:
            answer = query_frame(frame)
            results.append((sec, answer))
            print(f"--- Second {sec} ---")
            print(answer)
        except Exception as e:
            print(f"⚠️  Frame at second {sec} failed: {e}")

    # Write to markdown
    with open(OUTPUT_MD, "w") as f:
        f.write("# Video Text Extraction\n\n")
        f.write(f"**Source:** https://youtube.com/watch?v={YT_VIDEO_ID}\n\n")
        f.write(
            f"**Frames processed:** {len(results)} unique out of {len(all_frames)} total\n\n"
        )
        f.write("---\n\n")
        for sec, text in results:
            f.write(f"## Second {sec}\n\n{text}\n\n---\n\n")

    print(f"\n📝 Output written to {OUTPUT_MD}")
    print("=" * 60)
    print("✅ All frames processed")


def main():
    check_tools()
    model_path, mmproj_path = download_model()
    download_yt_clip()
    extract_frames()

    server = start_server(model_path, mmproj_path)
    try:
        query_model()
    finally:
        print("\n🛑 Stopping llama-server...")
        server.terminate()
        server.wait(timeout=5)

    # Clean up caches
    print("🧹 Cleaning up caches...")
    shutil.rmtree(FRAMES_DIR, ignore_errors=True)
    VIDEO_PATH.unlink(missing_ok=True)
    print("✅ Done")


if __name__ == "__main__":
    main()
