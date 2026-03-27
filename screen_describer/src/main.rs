#!/usr/bin/env -S cargo +stable run
/*
 * Screen Describer Utility in Rust
 * Port of qwen35_screen_capture.py to Rust
 */

use anyhow::{Context, Result};
use std::fs;
use std::io::{Cursor, Read, Write};
use std::path::{Path, PathBuf};
use std::process::{Command, Stdio};
use std::thread;
use std::time::Duration;

use base64::{Engine as _, engine::general_purpose};
use image::DynamicImage;
use image::codecs::jpeg::JpegEncoder;
use image::imageops::FilterType;
use reqwest::blocking::Client;
use reqwest::header::{CONTENT_TYPE, HeaderMap, HeaderValue};
use serde_json::json;
use tracing::info;

const MODEL_REPO: &str = "unsloth/Qwen3.5-0.8B-GGUF";
const MODEL_FILE: &str = "Qwen3.5-0.8B-Q4_K_M.gguf";
const MMPROJ_FILE: &str = "mmproj-F16.gguf";
const PORT: u16 = 8001;

fn main() -> Result<()> {
    tracing_subscriber::fmt::init();

    check_tools()?;
    let (model_path, mmproj_path) = download_model()?;
    let _server = start_server(&model_path, &mmproj_path)?;

    let img_path = capture_screen()?;
    describe_screen(&img_path)?;

    Ok(())
}

fn check_tools() -> Result<()> {
    if Command::new("llama-server")
        .arg("--version")
        .output()
        .is_err()
    {
        anyhow::bail!("[ERROR] 'llama-server' not found. Install it first.");
    }
    info!("[OK] All tools available");
    Ok(())
}

fn model_dir() -> PathBuf {
    let manifest = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    manifest.parent().unwrap().join("Qwen3.5-0.8B-GGUF")
}

fn download_model() -> Result<(PathBuf, PathBuf)> {
    let model_dir = model_dir();
    let model_path = model_dir.join(MODEL_FILE);
    let mmproj_path = model_dir.join(MMPROJ_FILE);

    if model_path.exists() && mmproj_path.exists() {
        info!("[OK] Model already downloaded at {}", model_dir.display());
        return Ok((model_path, mmproj_path));
    }

    info!("[DL] Downloading {} (Q4_K_M + mmproj)...", MODEL_REPO);
    fs::create_dir_all(&model_dir)?;

    for filename in [MODEL_FILE, MMPROJ_FILE].iter() {
        let url = format!(
            "https://huggingface.co/{}/resolve/main/{}",
            MODEL_REPO, filename
        );
        let dest = model_dir.join(filename);
        download_file(&url, &dest).with_context(|| format!("Failed to download {}", filename))?;
    }

    info!("[OK] Model downloaded to {}", model_dir.display());
    Ok((model_path, mmproj_path))
}

fn download_file(url: &str, dest: &Path) -> Result<()> {
    let mut response = Client::builder()
        .timeout(Duration::from_secs(30))
        .build()?
        .get(url)
        .send()?;

    if !response.status().is_success() {
        anyhow::bail!(
            "[ERROR] Failed to download from {}: {}",
            url,
            response.status()
        );
    }

    let mut file = fs::File::create(dest)?;
    let mut content = Vec::new();
    response.read_to_end(&mut content)?;
    file.write_all(&content)?;
    Ok(())
}

fn start_server(model_path: &Path, mmproj_path: &Path) -> Result<Option<std::process::Child>> {
    if is_server_ready()? {
        info!("[OK] llama-server is already running");
        return Ok(None);
    }

    info!("[RUN] Starting llama-server on port {}...", PORT);
    let log_file = fs::OpenOptions::new()
        .create(true)
        .append(true)
        .open("/tmp/llama_server.log")?;

    let mut child = Command::new("llama-server")
        .arg("--model")
        .arg(model_path)
        .arg("--mmproj")
        .arg(mmproj_path)
        .arg("--ctx-size")
        .arg("8192")
        .arg("--port")
        .arg(PORT.to_string())
        .arg("--jinja")
        .stdout(Stdio::from(log_file.try_clone()?))
        .stderr(Stdio::from(log_file))
        .spawn()?;

    for _ in 0..30 {
        thread::sleep(Duration::from_secs(2));
        if is_server_ready()? {
            info!("[OK] llama-server is ready");
            return Ok(Some(child));
        }
    }

    child.kill()?;
    anyhow::bail!("[ERROR] llama-server failed to start within 60s");
}

fn is_server_ready() -> Result<bool> {
    match Client::builder()
        .timeout(Duration::from_secs(2))
        .build()?
        .get(format!("http://localhost:{}/health", PORT))
        .send()
    {
        Ok(response) => Ok(response.status().is_success()),
        Err(_) => Ok(false),
    }
}

fn capture_screen() -> Result<PathBuf> {
    let desktop = dirs::desktop_dir()
        .ok_or_else(|| anyhow::anyhow!("[ERROR] Could not locate desktop directory"))?;
    info!(
        "[SCAN] Scanning {} for screenshot files...",
        desktop.display()
    );

    let mut images = Vec::new();
    for entry in fs::read_dir(&desktop)? {
        let entry = entry?;
        let path = entry.path();
        if path.is_file() {
            let ext = path
                .extension()
                .and_then(|s| s.to_str())
                .unwrap_or("")
                .to_lowercase();
            if ext == "png" || ext == "jpg" || ext == "jpeg" {
                images.push(path);
            }
        }
    }

    if images.is_empty() {
        anyhow::bail!(
            "[ERROR] No .png or .jpg image files found on your Desktop ({}).",
            desktop.display()
        );
    }

    let latest_img = images
        .into_iter()
        .max_by_key(|p| {
            p.metadata()
                .ok()
                .and_then(|m| m.modified().ok())
                .unwrap_or(std::time::SystemTime::UNIX_EPOCH)
        })
        .ok_or_else(|| anyhow::anyhow!("[ERROR] Failed to determine latest image"))?;

    info!("[OK] Selected latest image: {}", latest_img.display());
    Ok(latest_img)
}

fn describe_screen(img_path: &Path) -> Result<()> {
    let img = image::open(img_path)?;
    let max_dim = 1024;
    let img = if img.width() > max_dim || img.height() > max_dim {
        let ratio = max_dim as f32 / img.width().max(img.height()) as f32;
        let new_width = (img.width() as f32 * ratio) as u32;
        let new_height = (img.height() as f32 * ratio) as u32;
        info!(
            "[RESIZE] Resizing {}x{} → {}x{} for model compatibility",
            img.width(),
            img.height(),
            new_width,
            new_height
        );
        DynamicImage::ImageRgb8(
            img.resize_exact(new_width, new_height, FilterType::Lanczos3)
                .into(),
        )
    } else {
        DynamicImage::ImageRgb8(img.to_rgb8())
    };

    let mut buf = Vec::new();
    let mut cursor = Cursor::new(&mut buf);
    let encoder = JpegEncoder::new_with_quality(&mut cursor, 85);
    img.write_with_encoder(encoder)?;
    let b64 = general_purpose::STANDARD.encode(&buf);
    info!(
        "[SIZE] Image payload size: {} KB (base64)",
        b64.len() / 1024
    );

    let content = json!([
        {
            "type": "image_url",
            "image_url": {"url": format!("data:image/jpeg;base64,{}", b64)}
        },
        {
            "type": "text",
            "text": "Describe this screenshot in detail. What is taking place? What applications are open?"
        }
    ]);

    info!("[AI] Analyzing screenshot with Qwen3.5-0.8B...");
    let client = Client::builder()
        .timeout(Duration::from_secs(180))
        .build()?;

    let mut headers = HeaderMap::new();
    headers.insert(CONTENT_TYPE, HeaderValue::from_static("application/json"));

    let resp = client
        .post(format!("http://localhost:{}/v1/chat/completions", PORT))
        .headers(headers)
        .body(
            json!({
                "model": "qwen3.5-0.8b",
                "messages": [{"role": "user", "content": content}],
                "max_tokens": 4096,
                "temperature": 0.7,
                "top_p": 0.8
            })
            .to_string(),
        )
        .send()?;

    if !resp.status().is_success() {
        anyhow::bail!("[ERROR] API request failed with status: {}", resp.status());
    }

    let data: serde_json::Value = resp.json()?;
    info!(
        "[SCAN] Raw API response keys: {:?}",
        data.as_object().map(|o| o.keys().collect::<Vec<_>>())
    );

    if let Some(choices) = data
        .get("choices")
        .and_then(|c| c.as_array())
        .and_then(|arr| arr.first())
    {
        if let Some(finish_reason) = choices.get("finish_reason") {
            info!("[SCAN] Finish reason: {}", finish_reason);
        }
        if let Some(message) = choices.get("message").and_then(|m| m.as_object()) {
            if let Some(role) = message.get("role") {
                info!("[SCAN] Message role: {}", role);
            }
            if let Some(content) = message.get("content").and_then(|c| c.as_str()) {
                info!("[SCAN] Content length: {}", content.len());
                print_description(content);
            }
        }
    } else {
        info!("[SCAN] Full response: {}", data);
    }

    Ok(())
}

fn print_description(content: &str) {
    println!("\n{}", "=".repeat(60));
    println!("[DESC] SCREENSHOT DESCRIPTION:");
    println!("{}", "=".repeat(60));
    termimad::print_text(content);
    println!("{}\n", "=".repeat(60));
}
