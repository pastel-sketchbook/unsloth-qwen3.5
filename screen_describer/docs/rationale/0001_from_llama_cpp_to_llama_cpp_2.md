# 0001 — From llama.cpp (server) to llama-cpp-2 (in-process FFI)

**Date:** 2026-03-27  
**Status:** Accepted  
**Scope:** screen_describer inference backend

## Context

The original `screen_describer` ran inference by:

1. Checking that the `llama-server` binary was installed on the system.
2. Spawning it as a child process (`Command::new("llama-server")`) with model/mmproj args.
3. Polling `http://localhost:8001/health` in a retry loop (up to 60 s) until the server was ready.
4. Encoding the screenshot as base64 JPEG and sending it via HTTP POST to the OpenAI-compatible `/v1/chat/completions` endpoint.
5. Parsing the JSON response to extract the generated description.

This worked but had significant drawbacks.

## Problems

| # | Problem | Impact |
|---|---------|--------|
| 1 | **External binary dependency** | Users must install `llama-server` separately (brew, build from source, etc.). Breaks on version mismatches. |
| 2 | **60-second cold start** | Server startup + model loading blocks the main thread in a polling loop before any inference can begin. |
| 3 | **Unnecessary HTTP/JSON overhead** | Base64-encoding the image inflates payload ~33%. Serialising/deserialising JSON adds latency and complexity. |
| 4 | **Port conflicts** | Hard-coded port 8001 can collide with other services. |
| 5 | **Heavy dependencies** | `reqwest`, `serde`, `serde_json`, `base64` were all required just to talk to a localhost HTTP server. |
| 6 | **Process lifecycle management** | Had to handle server health checks, process cleanup on exit, and log file management. |

## Decision

Replace the HTTP-server architecture with **`llama-cpp-2`** (crate version 0.1.x), which statically links llama.cpp via `llama-cpp-sys-2` and exposes safe Rust wrappers.

Enable the **`mtmd`** feature flag to access the multimodal (vision) pipeline — the same `libmtmd` that powers `llama-server`'s vision support, but called directly as a library.

## How It Works Now

```
image file
  │
  ▼
image crate (resize + RGB extraction)
  │
  ▼
MtmdBitmap::from_image_data(w, h, &rgb_bytes)
  │
  ▼
MtmdContext::init_from_file(mmproj.gguf, &model, &params)
  │
  ▼
mtmd_ctx.tokenize(text_with_marker, &[&bitmap])  →  MtmdInputChunks
  │
  ▼
chunks.eval_chunks(&mtmd_ctx, &llama_ctx, ...)   →  n_past
  │
  ▼
LlamaSampler  →  token-by-token generation loop
```

Key changes:

- **Model + mmproj load directly** into the process via `LlamaModel::load_from_file` and `MtmdContext::init_from_file`.
- **Image pixels passed as raw RGB** — no base64, no JPEG re-encoding for the model.
- **Multimodal tokenization** via `mtmd_ctx.tokenize()` replaces the JSON content array.
- **`eval_chunks`** handles both text token decoding and image embedding encoding in one call.
- **Token generation loop** uses `LlamaSampler` with repetition penalty, temp, and top-p — all in-process.
- **Chat template wrapping** — the prompt is wrapped in `<|im_start|>user`/`<|im_end|>` markers so the model generates a proper assistant response.
- **Think-block stripping** — Qwen3.5 emits `<think>…</think>` reasoning tokens; these are stripped before display.

## Dependencies Changed

| Removed | Added |
|---------|-------|
| `base64` | `llama-cpp-2` (with `mtmd` feature) |
| `serde` / `serde_json` | `encoding_rs` |
| `reqwest` (for API calls) | — |

> `reqwest` is retained solely for model downloading, not inference.

## Consequences

**Positive:**
- Single binary — no external `llama-server` install required.
- No cold-start delay from server boot; model loads once, inference begins immediately.
- ~33% smaller image payload (raw RGB vs base64 JPEG).
- Metal acceleration on macOS is automatic (llama-cpp-sys-2 enables it on Apple Silicon).
- Simpler error handling — no HTTP status codes, no JSON parsing, no port management.

**Negative:**
- Compile time increases (~80 s) because llama.cpp is built from source via `build.rs`.
- Binary size is larger since llama.cpp + ggml are statically linked.
- The `mtmd` API is marked experimental in llama-cpp-2 and may see breaking changes.

## Lessons Learned

- After `eval_chunks(logits_last: true)`, the first sample must use logits index `-1` (last token from the internal batch), not `0`.
- A repetition penalty sampler (`LlamaSampler::penalties`) is essential for small models like Qwen3.5-0.8B to prevent degenerate looping. A window of 512 tokens with a penalty of 1.5 was needed; the initial 64/1.3 was too weak.
- Raw prompts without chat-template markers (`<|im_start|>`/`<|im_end|>`) cause the model to hallucinate and repeat. Always wrap in the expected chat format.
- Qwen3.5 is a "thinking" model that emits `<think>…</think>` blocks before the actual answer. The `/no_think` prompt suffix does not work reliably with this model. The effective approach is to pre-fill the assistant turn with an empty think block (`<think>\n\n</think>\n\n`) so the model skips reasoning entirely. A fallback `strip_think_blocks` post-processor handles any residual think content in both the live stream and final display.
