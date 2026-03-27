# 0001 — Zig port: direct C API via @cImport

**Date:** 2026-03-27  
**Status:** Accepted  
**Scope:** screen_describer_zig — alternative implementation of the screen describer

## Context

The primary `screen_describer` is written in Rust and uses the `llama-cpp-2` crate, which provides safe Rust FFI wrappers around llama.cpp and statically links the entire C++ library at build time. It works well but has trade-offs: ~80 s compile times (llama.cpp is rebuilt from source via `build.rs`), a large static binary, and a dependency chain (`llama-cpp-2` → `llama-cpp-sys-2` → vendored C++ source) that lags behind upstream llama.cpp releases.

The question: can the same tool be built against the system-installed llama.cpp directly, with zero vendored C++ and sub-second compile times?

## Decision

Create an alternative implementation in Zig that calls the llama.cpp C API directly via `@cImport`, dynamically linking against the system-installed libraries (`brew install llama.cpp`).

## Key Architectural Differences from Rust Version

| Aspect | Rust (`screen_describer`) | Zig (`screen_describer_zig`) |
|--------|---------------------------|------------------------------|
| **FFI approach** | `llama-cpp-2` crate (safe wrappers) | `@cImport` of `llama.h`, `mtmd.h`, `mtmd-helper.h` |
| **Linking** | Static — llama.cpp compiled from source via `build.rs` | Dynamic — links against system `libllama`, `libmtmd` (ggml pulled in transitively) |
| **Image loading** | Manual: `image` crate resizes to 1024px, extracts RGB bytes, passes to `MtmdBitmap::from_image_data` | Delegated: `mtmd_helper_bitmap_init_from_file` lets the library handle decoding/prep |
| **Memory management** | Rust ownership + RAII | Arena allocator for the full program lifetime; C resources freed via `defer` |
| **Model download** | `reqwest::blocking::get` (HTTP client compiled in) | `curl -fSL` child process (no HTTP dependency) |
| **Build time** | ~80 s (compiles llama.cpp from source) | ~1 s (links against pre-built system libs) |
| **Binary size** | ~40 MB (static llama.cpp + ggml + Metal) | ~200 KB (just the Zig code; libraries loaded at runtime) |

## How It Works

```
image file on disk
  │
  ▼
mtmd_helper_bitmap_init_from_file(mtmd_ctx, path)   ← library handles decode
  │
  ▼
mtmd_tokenize(mtmd_ctx, chunks, &text_input, &bitmaps, 1)
  │
  ▼
mtmd_helper_eval_chunks(mtmd_ctx, ctx, chunks, ...)  → n_past
  │
  ▼
llama_sampler_sample / llama_decode  →  token-by-token generation loop
```

The pipeline is the same as Rust — load model, load image, tokenize multimodal input, evaluate chunks, sample tokens — but every call goes through the raw C function signatures exposed by `@cImport` rather than through Rust wrapper types.

## Design Choices Worth Noting

### Arena allocator over individual allocations

The program has a single entry point, does one job, and exits. An arena over `page_allocator` means every Zig-side allocation is freed in one shot at process exit. This eliminates the need for per-allocation tracking and simplifies error paths — there is no risk of leaking Zig memory on early return.

C-side resources (model, context, sampler, batch, bitmap, chunks) are still freed individually via `defer` calls to the corresponding `*_free` functions, since those manage their own memory.

### mtmd_helper_bitmap_init_from_file instead of manual resize

The Rust version uses the `image` crate to resize screenshots to 1024px max before passing raw RGB bytes. The Zig version calls `mtmd_helper_bitmap_init_from_file`, which is a convenience function in llama.cpp's mtmd-helper API that handles image loading internally. This avoids pulling in an image processing library and keeps the dependency footprint at zero beyond llama.cpp itself.

### curl child process for downloads

Rather than compiling in an HTTP client library, the Zig version shells out to `curl` for the one-time model download. On macOS (the target platform), `curl` is always present. This keeps compile time fast and binary size minimal. The download is a one-time operation — once the model files exist on disk, `curl` is never invoked.

## Consequences

**Positive:**
- Sub-second compile times (~1 s vs ~80 s). Iteration speed during development is dramatically better.
- Tiny binary (~200 KB). No vendored C++ code.
- Always uses the same llama.cpp version as the system. `brew upgrade llama.cpp` upgrades the inference backend without recompilation.
- `@cImport` provides direct access to the full C API surface with no wrapper lag — new llama.cpp features are available immediately.
- Zero Zig dependencies beyond the standard library.

**Negative:**
- Requires `brew install llama.cpp` as a prerequisite. Not a self-contained binary.
- No compile-time safety net from Rust-style wrappers — C pointer misuse is possible (e.g., the `batch.seq_id[0].?[0]` vs `batch.seq_id[0][0]` issue caught during development, where Zig's type system flagged a `[*c]i32` being incorrectly treated as an optional).
- Dynamic linking means the binary is tied to the system's ABI. Major llama.cpp version bumps could break things silently at runtime rather than at compile time.
- `curl` dependency for downloads is implicit — it would fail on a system without `curl` (not a concern on macOS, but limits portability).

## Lessons Learned

- Zig's `@cImport` translates C pointer types faithfully. A `llama_seq_id **` becomes `[*c][*c]i32`, not an optional. Attempting to use `.?` on a non-optional C pointer is a compile error, which is correct — but the type names in error messages can be opaque until you internalize the `[*c]` convention.
- `std.ArrayList(u8)` in Zig 0.15 uses the unmanaged API pattern: `initCapacity(allocator, n)`, `appendSlice(allocator, items)`, `deinit(allocator)`. There is no `.init(allocator)` — use `initCapacity` with capacity 0 if you don't want to pre-allocate.
- The `mtmd_helper_eval_chunks` function returns `n_past` via an out-pointer. After calling it with `logits_last: true`, the first `llama_sampler_sample` call must use logits index `-1` (last token from the internal batch), matching the Rust version's behavior. Subsequent samples after single-token `llama_decode` use index `0`.
- Zig's `std.fs.File.stdout().writer(&buf)` returns a buffered writer via the `interface` field. All writes go through `stdout_w.interface.writeAll` / `.flush()`, not through the writer directly. This is different from Rust's `std::io::stdout().flush()` pattern and easy to get wrong on first encounter.
- When dynamically linking against Homebrew-installed llama.cpp, only link `libllama` and `libmtmd` directly. `libggml` and `libggml-base` are transitive dependencies — linking them explicitly causes a dyld "duplicate linked dylib" error at runtime on macOS. The symptom is an immediate crash before `main()` even runs, with no useful backtrace. The fix is to let the dynamic linker resolve transitive deps automatically.
- Newer versions of llama.cpp (post-b8500) require calling `ggml_backend_load_all()` before `llama_backend_init()` and model loading. Without this, `llama_model_load_from_file` fails with "no backends are loaded". This loads the compute backends (Metal, CPU, BLAS) from the ggml plugin directory. The Rust `llama-cpp-2` crate handles this internally; when using the C API directly you must call it yourself. Requires `@cInclude("ggml-backend.h")`.
