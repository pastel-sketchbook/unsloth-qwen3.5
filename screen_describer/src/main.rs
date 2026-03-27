/*
 * Screen Describer — direct GGUF inference via llama-cpp-2 (no llama-server)
 *
 * Uses the `mtmd` feature for multimodal (vision) support.
 */

use anyhow::{Context, Result};
use image::imageops::FilterType;
use llama_cpp_2::LogOptions;
use llama_cpp_2::context::params::LlamaContextParams;
use llama_cpp_2::llama_backend::LlamaBackend;
use llama_cpp_2::model::LlamaModel;
use llama_cpp_2::model::params::LlamaModelParams;
use llama_cpp_2::mtmd::{
    MtmdBitmap, MtmdContext, MtmdContextParams, MtmdInputText, mtmd_default_marker,
};
use llama_cpp_2::sampling::LlamaSampler;
use llama_cpp_2::send_logs_to_tracing;
use std::fs;
use std::num::NonZeroU32;
use std::path::{Path, PathBuf};
use std::pin::pin;
use tracing::info;

const MODEL_REPO: &str = "unsloth/Qwen3.5-0.8B-GGUF";
const MODEL_FILE: &str = "Qwen3.5-0.8B-Q4_K_M.gguf";
const MMPROJ_FILE: &str = "mmproj-F16.gguf";

fn main() -> Result<()> {
    tracing_subscriber::fmt::init();
    send_logs_to_tracing(LogOptions::default());

    let (model_path, mmproj_path) = ensure_model()?;
    let img_path = find_latest_screenshot()?;
    describe_screen(&model_path, &mmproj_path, &img_path)?;

    Ok(())
}

fn model_dir() -> PathBuf {
    let manifest = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    // Safety: CARGO_MANIFEST_DIR always points inside a workspace, so parent() is guaranteed.
    manifest.parent().unwrap().join("Qwen3.5-0.8B-GGUF")
}

fn ensure_model() -> Result<(PathBuf, PathBuf)> {
    let dir = model_dir();
    let model_path = dir.join(MODEL_FILE);
    let mmproj_path = dir.join(MMPROJ_FILE);

    if model_path.exists() && mmproj_path.exists() {
        info!("[OK] Model already present at {}", dir.display());
        return Ok((model_path, mmproj_path));
    }

    info!("[DL] Downloading {} (Q4_K_M + mmproj)...", MODEL_REPO);
    fs::create_dir_all(&dir)?;

    for filename in [MODEL_FILE, MMPROJ_FILE] {
        let url = format!("https://huggingface.co/{MODEL_REPO}/resolve/main/{filename}",);
        let dest = dir.join(filename);
        download_file(&url, &dest).with_context(|| format!("Failed to download {filename}"))?;
    }

    info!("[OK] Model downloaded to {}", dir.display());
    Ok((model_path, mmproj_path))
}

fn download_file(url: &str, dest: &Path) -> Result<()> {
    let mut resp = reqwest::blocking::get(url).with_context(|| format!("requesting {url}"))?;
    if !resp.status().is_success() {
        anyhow::bail!("Download failed: {}", resp.status());
    }
    let mut file =
        fs::File::create(dest).with_context(|| format!("creating {}", dest.display()))?;
    resp.copy_to(&mut file)
        .with_context(|| format!("writing {}", dest.display()))?;
    Ok(())
}

fn find_latest_screenshot() -> Result<PathBuf> {
    let desktop =
        dirs::desktop_dir().ok_or_else(|| anyhow::anyhow!("Could not locate desktop dir"))?;
    info!("[SCAN] Scanning {} for screenshots...", desktop.display());

    let mut images: Vec<PathBuf> = fs::read_dir(&desktop)?
        .filter_map(Result::ok)
        .map(|e| e.path())
        .filter(|p| {
            p.is_file()
                && matches!(
                    p.extension().and_then(|s| s.to_str()).unwrap_or(""),
                    "png" | "jpg" | "jpeg"
                )
        })
        .collect();

    if images.is_empty() {
        anyhow::bail!("No image files found on Desktop ({})", desktop.display());
    }

    images.sort_by_key(|p| {
        p.metadata()
            .ok()
            .and_then(|m| m.modified().ok())
            .unwrap_or(std::time::SystemTime::UNIX_EPOCH)
    });
    // Safety: `images` is non-empty — checked and bailed above.
    let latest = images.pop().unwrap();
    info!("[OK] Selected: {}", latest.display());
    Ok(latest)
}

fn resize_for_model(img: image::DynamicImage, max_dim: u32) -> image::DynamicImage {
    if img.width() <= max_dim && img.height() <= max_dim {
        return img;
    }
    // Image dimensions are bounded by max_dim (1024), so f32 precision loss
    // and truncation are harmless for resize calculations.
    #[allow(
        clippy::cast_precision_loss,
        clippy::cast_possible_truncation,
        clippy::cast_sign_loss
    )]
    let (nw, nh) = {
        let ratio = max_dim as f32 / img.width().max(img.height()) as f32;
        (
            (img.width() as f32 * ratio) as u32,
            (img.height() as f32 * ratio) as u32,
        )
    };
    info!("[RESIZE] {}x{} → {}x{}", img.width(), img.height(), nw, nh);
    img.resize_exact(nw, nh, FilterType::Lanczos3)
}

fn describe_screen(model_path: &Path, mmproj_path: &Path, img_path: &Path) -> Result<()> {
    // Resize image for model compatibility
    let img = image::open(img_path)?;
    let img = resize_for_model(img, 1024);

    // Convert to RGB bytes for MtmdBitmap
    let rgb = img.to_rgb8();
    let (w, h) = (rgb.width(), rgb.height());
    let raw_rgb = rgb.into_raw();

    // Init backend & load model
    let backend = LlamaBackend::init()?;
    let model_params = pin!(LlamaModelParams::default());

    info!("[LOAD] Loading model...");
    let model = LlamaModel::load_from_file(&backend, model_path, &model_params)
        .with_context(|| "Failed to load GGUF model")?;

    // Create llama context
    let ctx_params = LlamaContextParams::default().with_n_ctx(NonZeroU32::new(8192));
    let mut ctx = model
        .new_context(&backend, ctx_params)
        .with_context(|| "Failed to create llama context")?;

    // Init multimodal context
    let mmproj_str = mmproj_path
        .to_str()
        .context("mmproj path is not valid UTF-8")?;
    let mtmd_params = MtmdContextParams::default();
    let mtmd_ctx = MtmdContext::init_from_file(mmproj_str, &model, &mtmd_params)
        .with_context(|| "Failed to init multimodal context")?;

    // Create bitmap from image data
    let bitmap =
        MtmdBitmap::from_image_data(w, h, &raw_rgb).with_context(|| "Failed to create bitmap")?;

    // Build prompt with media marker
    let marker = mtmd_default_marker();
    let prompt = format!(
        "<|im_start|>user\n{marker}\nDescribe what you see on this screen. What is shown, what is happening, and what elements are visible?<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n"
    );

    let text_input = MtmdInputText {
        text: prompt,
        add_special: true,
        parse_special: true,
    };

    // Tokenize text + image into chunks
    let chunks = mtmd_ctx
        .tokenize(text_input, &[&bitmap])
        .with_context(|| "Failed to tokenize multimodal input")?;

    info!(
        "[AI] Processing {} chunks ({} tokens)...",
        chunks.len(),
        chunks.total_tokens()
    );

    // Evaluate all chunks (text + image embeddings)
    let n_batch = 2048;
    let n_past = chunks
        .eval_chunks(&mtmd_ctx, &ctx, 0, 0, n_batch, true)
        .with_context(|| "Failed to evaluate multimodal chunks")?;

    // Generate tokens
    let max_tokens = 1024_i32;
    let mut sampler = LlamaSampler::chain_simple([
        LlamaSampler::penalties(512, 1.5, 0.0, 0.0),
        LlamaSampler::temp(0.7),
        LlamaSampler::top_p(0.8, 1),
        LlamaSampler::dist(1234),
    ]);

    let mut decoder = encoding_rs::UTF_8.new_decoder();
    let mut output = String::new();
    let mut batch = llama_cpp_2::llama_batch::LlamaBatch::new(512, 1);
    let mut n_cur = n_past;
    let mut tok_count = 0u32;

    info!("[AI] Generating description...");

    // First sample: logits are from eval_chunks' last internal decode, use -1 for last token
    let mut logits_idx: i32 = -1;

    for _ in 0..max_tokens {
        let token = sampler.sample(&ctx, logits_idx);
        sampler.accept(token);

        if model.is_eog_token(token) {
            break;
        }

        let piece = model.token_to_piece(token, &mut decoder, true, None)?;
        output.push_str(&piece);

        tok_count += 1;
        if tok_count.is_multiple_of(32) {
            eprint!(".");
        }

        batch.clear();
        batch.add(token, n_cur, &[0], true)?;
        n_cur += 1;

        ctx.decode(&mut batch)
            .with_context(|| "decode failed during generation")?;

        // Subsequent samples use index 0 in our single-token batch
        logits_idx = 0;
    }

    eprintln!();
    info!("[OK] Generated {} tokens", tok_count);

    // Strip <think>...</think> blocks (Qwen3.5 thinking tokens)
    let display_output = strip_think_blocks(&output);

    println!("\n{}", "=".repeat(60));
    println!("[DESC] SCREENSHOT DESCRIPTION:");
    println!("{}", "=".repeat(60));
    termimad::print_text(&display_output);
    println!("{}\n", "=".repeat(60));

    Ok(())
}

fn strip_think_blocks(text: &str) -> String {
    let mut result = String::new();
    let mut rest = text;
    while let Some(start) = rest.find("<think>") {
        result.push_str(&rest[..start]);
        match rest[start..].find("</think>") {
            Some(end) => rest = &rest[start + end + "</think>".len()..],
            None => return result.trim().to_string(),
        }
    }
    result.push_str(rest);
    result.trim().to_string()
}
