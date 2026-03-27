# Screen Describer

Takes a screenshot from `~/Desktop`, feeds it to the
[Qwen3.5-0.8B](https://huggingface.co/unsloth/Qwen3.5-0.8B-GGUF) vision
model via llama.cpp, and prints a styled description to the terminal.

The same tool is implemented three times — in Python, Rust, and Zig — as a
deliberate learning progression.

## Why Python → Rust → Zig

Each language sits at a different abstraction level, and implementing in this
order makes each step build naturally on the last:

1. **Python** — the sketch. No compilation, no memory management, minimal
   boilerplate. The goal is to validate the idea as fast as possible: can this
   model describe a screenshot, and is the output useful? Python's `llama-server`
   HTTP interface means zero FFI concerns. If the concept doesn't work, you've
   lost an afternoon, not a week.

2. **Rust** — the production version. Once the concept is proven, Rust
   reimplements it with in-process inference via `llama-cpp-2` FFI bindings
   (no HTTP server). The type system forces you to handle every error path, the
   borrow checker catches lifetime mistakes at compile time, and `cargo` pulls in
   image processing and markdown rendering as crate dependencies. You trade
   Python's iteration speed for guarantees: if it compiles, the memory and
   concurrency model is sound. The cost is an ~80 s build (llama.cpp is compiled
   from source by `build.rs`) and a ~40 MB binary.

3. **Zig** — the direct port. With the algorithm already proven in Rust, Zig
   strips away every abstraction layer. `@cImport` calls the llama.cpp C API
   directly — no wrapper crate, no vendored source, no build system complexity.
   Dynamic linking against the system `libllama` (via Homebrew) drops compile
   time to ~1 s and binary size to ~200 KB. You learn exactly what the C API
   expects because there is nothing between your code and it. The trade-off is
   that you give up Rust's safety net: pointer misuse is a runtime crash, not a
   compile error.

The progression follows a pattern: **validate → harden → minimize**. Each
rewrite removes a layer of indirection and forces you to understand what the
previous layer was doing for you.

## Project Layout

```
screen_describer_python/     Python — llama-server HTTP, uv script
screen_describer/            Rust   — llama-cpp-2 FFI, static linking
screen_describer_zig/        Zig    — @cImport C API, dynamic linking
Qwen3.5-0.8B-GGUF/          Shared model cache (auto-downloaded)
Taskfile.yaml                Task runner for all three languages
```

## Prerequisites

- macOS (all three target macOS; Metal GPU acceleration)
- [Homebrew](https://brew.sh)
- [Task](https://taskfile.dev) — `brew install go-task`

| Language | Additional requirements |
|----------|------------------------|
| Python   | [uv](https://docs.astral.sh/uv/) (`brew install uv`), `llama-server` (`brew install llama.cpp`) |
| Rust     | [Rust toolchain](https://rustup.rs) (`brew install rustup && rustup-init`), cmake |
| Zig      | [Zig 0.15+](https://ziglang.org) (`brew install zig`), `brew install llama.cpp` |

## Usage

Place a screenshot (`.png` / `.jpg`) on your Desktop, then:

```sh
task run:python    # Python version (starts llama-server, HTTP API)
task run:rust      # Rust version (in-process inference, static linked)
task run:zig       # Zig version (in-process inference, dynamic linked)
```

Model weights are downloaded automatically on first run to `Qwen3.5-0.8B-GGUF/`.

## Development

```sh
task fmt            # Format all (Rust + Python + Zig)
task lint           # Lint all (clippy + ruff)
task check          # fmt + lint + loc count
task build:zig      # Build Zig binary without running
```

## Architecture Decisions

Each implementation documents its design rationale:

- [`screen_describer/docs/rationale/`](screen_describer/docs/rationale/) — Rust: why llama-cpp-2 over llama-server
- [`screen_describer_zig/docs/rationale/`](screen_describer_zig/docs/rationale/) — Zig: direct C API, dynamic linking trade-offs
