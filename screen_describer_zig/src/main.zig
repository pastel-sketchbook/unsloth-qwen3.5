// Screen Describer — direct GGUF inference via llama.cpp C API
//
// Uses libmtmd for multimodal (vision) support.
// Requires: brew install llama.cpp

const std = @import("std");
const c = @cImport({
    @cInclude("llama.h");
    @cInclude("ggml-backend.h");
    @cInclude("mtmd.h");
    @cInclude("mtmd-helper.h");
});

const model_repo = "unsloth/Qwen3.5-0.8B-GGUF";
const model_file = "Qwen3.5-0.8B-Q4_K_M.gguf";
const mmproj_file = "mmproj-F16.gguf";

pub fn main() !void {
    var arena = std.heap.ArenaAllocator.init(std.heap.page_allocator);
    defer arena.deinit();
    const allocator = arena.allocator();

    const model_path, const mmproj_path = try ensureModel(allocator);
    defer allocator.free(model_path);
    defer allocator.free(mmproj_path);

    const img_path = try findLatestScreenshot(allocator);
    defer allocator.free(img_path);

    try describeScreen(allocator, model_path, mmproj_path, img_path);
}

fn modelDir(allocator: std.mem.Allocator) ![:0]const u8 {
    const exe_path = try std.fs.selfExeDirPathAlloc(allocator);
    defer allocator.free(exe_path);

    // Navigate from zig-out/bin -> zig-out -> project -> workspace root
    var dir = try std.fs.openDirAbsolute(exe_path, .{});
    defer dir.close();
    var up1 = try dir.openDir("..", .{});
    defer up1.close();
    var up2 = try up1.openDir("..", .{});
    defer up2.close();
    var up3 = try up2.openDir("..", .{});
    defer up3.close();

    const real = try up3.realpathAlloc(allocator, "Qwen3.5-0.8B-GGUF");
    return try allocator.dupeZ(u8, real);
}

fn ensureModel(allocator: std.mem.Allocator) !struct { [:0]const u8, [:0]const u8 } {
    const dir = try modelDir(allocator);
    defer allocator.free(dir);

    const m_path = try std.fs.path.joinZ(allocator, &.{ dir, model_file });
    const mm_path = try std.fs.path.joinZ(allocator, &.{ dir, mmproj_file });

    const m_exists = blk: {
        std.fs.accessAbsolute(m_path, .{}) catch break :blk false;
        break :blk true;
    };
    const mm_exists = blk: {
        std.fs.accessAbsolute(mm_path, .{}) catch break :blk false;
        break :blk true;
    };

    if (m_exists and mm_exists) {
        log("[OK] Model already present at {s}", .{dir});
        return .{ m_path, mm_path };
    }

    log("[DL] Downloading {s} (Q4_K_M + mmproj)...", .{model_repo});
    std.fs.makeDirAbsolute(dir) catch |e| switch (e) {
        error.PathAlreadyExists => {},
        else => return e,
    };

    for ([_][]const u8{ model_file, mmproj_file }) |filename| {
        const url = try std.fmt.allocPrint(allocator, "https://huggingface.co/{s}/resolve/main/{s}", .{ model_repo, filename });
        defer allocator.free(url);

        const dest = try std.fs.path.join(allocator, &.{ dir, filename });
        defer allocator.free(dest);

        try downloadFile(allocator, url, dest);
    }

    log("[OK] Model downloaded to {s}", .{dir});
    return .{ m_path, mm_path };
}

fn downloadFile(allocator: std.mem.Allocator, url: []const u8, dest: []const u8) !void {
    log("[DL]   {s}", .{url});
    var child = std.process.Child.init(
        &.{ "curl", "-fSL", "-o", dest, url },
        allocator,
    );
    const term = try child.spawnAndWait();
    switch (term) {
        .Exited => |code| if (code != 0) return error.DownloadFailed,
        else => return error.DownloadFailed,
    }
}

fn findLatestScreenshot(allocator: std.mem.Allocator) ![:0]const u8 {
    const home = std.posix.getenv("HOME") orelse return error.NoHome;
    const desktop = try std.fs.path.join(allocator, &.{ home, "Desktop" });
    defer allocator.free(desktop);

    log("[SCAN] Scanning {s} for screenshots...", .{desktop});

    var dir = try std.fs.openDirAbsolute(desktop, .{ .iterate = true });
    defer dir.close();

    var best_path: ?[]const u8 = null;
    var best_mtime: i128 = std.math.minInt(i128);

    var iter = dir.iterate();
    while (try iter.next()) |entry| {
        if (entry.kind != .file) continue;

        const ext = std.fs.path.extension(entry.name);
        const is_image = std.mem.eql(u8, ext, ".png") or
            std.mem.eql(u8, ext, ".jpg") or
            std.mem.eql(u8, ext, ".jpeg");
        if (!is_image) continue;

        const stat = dir.statFile(entry.name) catch continue;
        const mtime = stat.mtime;

        if (mtime > best_mtime) {
            if (best_path) |old| allocator.free(old);
            best_path = try allocator.dupe(u8, entry.name);
            best_mtime = mtime;
        }
    }

    const name = best_path orelse {
        log("[ERR] No image files found on Desktop ({s})", .{desktop});
        return error.NoScreenshots;
    };
    defer allocator.free(name);

    const full = try std.fs.path.join(allocator, &.{ desktop, name });
    const result = try allocator.dupeZ(u8, full);
    allocator.free(full);

    log("[OK] Selected: {s}", .{result});
    return result;
}

fn describeScreen(allocator: std.mem.Allocator, model_path: [:0]const u8, mmproj_path: [:0]const u8, img_path: [:0]const u8) !void {
    c.ggml_backend_load_all();
    c.llama_backend_init();
    defer c.llama_backend_free();

    const model_params = c.llama_model_default_params();
    log("[LOAD] Loading model...", .{});
    const model = c.llama_model_load_from_file(model_path.ptr, model_params) orelse
        return error.ModelLoadFailed;
    defer c.llama_model_free(model);

    var ctx_params = c.llama_context_default_params();
    ctx_params.n_ctx = 8192;
    const ctx = c.llama_init_from_model(model, ctx_params) orelse
        return error.ContextCreateFailed;
    defer c.llama_free(ctx);

    var mtmd_params = c.mtmd_context_params_default();
    _ = &mtmd_params;
    const mtmd_ctx = c.mtmd_init_from_file(mmproj_path.ptr, model, mtmd_params) orelse
        return error.MtmdInitFailed;
    defer c.mtmd_free(mtmd_ctx);

    const bitmap = c.mtmd_helper_bitmap_init_from_file(mtmd_ctx, img_path.ptr) orelse
        return error.BitmapLoadFailed;
    defer c.mtmd_bitmap_free(bitmap);

    log("[OK] Image loaded: {s}", .{img_path});

    // Build prompt with media marker
    const marker: [*:0]const u8 = c.mtmd_default_marker();
    const marker_slice = std.mem.span(marker);
    const prompt_raw = try std.fmt.allocPrint(
        allocator,
        "<|im_start|>user\n{s}\nDescribe what you see on this screen. What is shown, what is happening, and what elements are visible?<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n",
        .{marker_slice},
    );
    defer allocator.free(prompt_raw);
    const prompt = try allocator.dupeZ(u8, prompt_raw);
    defer allocator.free(prompt);

    const text_input = c.mtmd_input_text{
        .text = prompt.ptr,
        .add_special = true,
        .parse_special = true,
    };

    const chunks = c.mtmd_input_chunks_init() orelse return error.ChunksInitFailed;
    defer c.mtmd_input_chunks_free(chunks);

    var bitmaps_arr = [_]?*const c.mtmd_bitmap{bitmap};
    const tok_result = c.mtmd_tokenize(mtmd_ctx, chunks, &text_input, &bitmaps_arr, 1);
    if (tok_result != 0) {
        log("[ERR] Tokenization failed: {d}", .{tok_result});
        return error.TokenizeFailed;
    }

    const n_chunks = c.mtmd_input_chunks_size(chunks);
    const total_tokens = c.mtmd_helper_get_n_tokens(chunks);
    log("[AI] Processing {d} chunks ({d} tokens)...", .{ n_chunks, total_tokens });

    // Evaluate all chunks
    var n_past: c.llama_pos = 0;
    const eval_result = c.mtmd_helper_eval_chunks(mtmd_ctx, ctx, chunks, 0, 0, 2048, true, &n_past);
    if (eval_result != 0) {
        log("[ERR] Chunk evaluation failed: {d}", .{eval_result});
        return error.EvalFailed;
    }

    // Set up sampler chain
    const sparams = c.llama_sampler_chain_default_params();
    const sampler = c.llama_sampler_chain_init(sparams) orelse return error.SamplerInitFailed;
    defer c.llama_sampler_free(sampler);

    c.llama_sampler_chain_add(sampler, c.llama_sampler_init_penalties(512, 1.5, 0.0, 0.0));
    c.llama_sampler_chain_add(sampler, c.llama_sampler_init_temp(0.7));
    c.llama_sampler_chain_add(sampler, c.llama_sampler_init_top_p(0.8, 1));
    c.llama_sampler_chain_add(sampler, c.llama_sampler_init_dist(1234));

    // Generate tokens
    const max_tokens: usize = 1024;
    var output = try std.ArrayList(u8).initCapacity(allocator, 1024);
    defer output.deinit(allocator);

    const vocab = c.llama_model_get_vocab(model);

    log("[AI] Generating description...", .{});

    var batch = c.llama_batch_init(512, 0, 1);
    defer c.llama_batch_free(batch);
    var n_cur = n_past;
    var logits_idx: i32 = -1;
    var tok_count: usize = 0;

    for (0..max_tokens) |_| {
        const token = c.llama_sampler_sample(sampler, ctx, logits_idx);
        c.llama_sampler_accept(sampler, token);

        if (c.llama_vocab_is_eog(vocab, token)) break;

        var piece_buf: [256]u8 = undefined;
        const piece_len = c.llama_token_to_piece(vocab, token, &piece_buf, 256, 0, true);
        if (piece_len > 0) {
            const piece = piece_buf[0..@intCast(piece_len)];
            try output.appendSlice(allocator, piece);
        }

        tok_count += 1;
        if (tok_count % 32 == 0) logProgress(".");

        batch.n_tokens = 0;
        batch.token[0] = token;
        batch.pos[0] = n_cur;
        batch.n_seq_id[0] = 1;
        batch.seq_id[0][0] = 0;
        batch.logits[0] = 1;
        batch.n_tokens = 1;
        n_cur += 1;

        const dec_result = c.llama_decode(ctx, batch);
        if (dec_result != 0) {
            log("[ERR] Decode failed: {d}", .{dec_result});
            return error.DecodeFailed;
        }

        logits_idx = 0;
    }

    log("", .{}); // newline after progress dots
    log("[OK] Generated {d} tokens", .{tok_count});

    const display = try stripThinkBlocks(allocator, output.items);

    var stdout_buf: [4096]u8 = undefined;
    var stdout_w = std.fs.File.stdout().writer(&stdout_buf);

    const sep = "=" ** 60;
    try stdout_w.interface.print("\n{s}\n[DESC] SCREENSHOT DESCRIPTION:\n{s}\n", .{ sep, sep });
    try renderMarkdown(&stdout_w.interface, display);
    try stdout_w.interface.print("{s}\n", .{sep});
    try stdout_w.interface.flush();
}

fn stripThinkBlocks(allocator: std.mem.Allocator, text: []const u8) ![]const u8 {
    var result = try std.ArrayList(u8).initCapacity(allocator, text.len);
    var rest = text;

    while (std.mem.indexOf(u8, rest, "<think>")) |start| {
        try result.appendSlice(allocator, rest[0..start]);
        const after_tag = rest[start..];
        if (std.mem.indexOf(u8, after_tag, "</think>")) |end| {
            rest = after_tag[end + "</think>".len ..];
        } else {
            // Unclosed <think> — drop the remainder
            return std.mem.trim(u8, result.items, " \t\n\r");
        }
    }

    try result.appendSlice(allocator, rest);
    return std.mem.trim(u8, result.items, " \t\n\r");
}

// ---------------------------------------------------------------------------
// Minimal terminal markdown renderer (ANSI escape codes)
//
// Handles: # headings, **bold**, *italic*, `inline code`, ```code blocks```,
//          --- horizontal rules, - list items
// ---------------------------------------------------------------------------

const ansi = struct {
    const reset = "\x1b[0m";
    const bold = "\x1b[1m";
    const italic = "\x1b[3m";
    const dim = "\x1b[2m";
    const cyan = "\x1b[36m";
    const yellow = "\x1b[33m";
    const green = "\x1b[32m";
    const bold_yellow = "\x1b[1;33m";
    const bold_cyan = "\x1b[1;36m";
};

fn renderMarkdown(writer: anytype, text: []const u8) !void {
    var in_code_block = false;
    var lines = std.mem.splitScalar(u8, text, '\n');

    while (lines.next()) |line| {
        // Fenced code blocks
        if (std.mem.startsWith(u8, std.mem.trimLeft(u8, line, " "), "```")) {
            if (in_code_block) {
                try writer.writeAll(ansi.reset);
                in_code_block = false;
            } else {
                try writer.writeAll(ansi.dim ++ ansi.cyan);
                in_code_block = true;
            }
            try writer.writeByte('\n');
            continue;
        }

        if (in_code_block) {
            try writer.writeAll("  ");
            try writer.writeAll(line);
            try writer.writeByte('\n');
            continue;
        }

        const trimmed = std.mem.trimLeft(u8, line, " ");

        // Horizontal rules
        if (isHorizontalRule(trimmed)) {
            try writer.writeAll(ansi.dim);
            try writer.writeAll("\xe2\x94\x80" ** 40); // ─ repeated
            try writer.writeAll(ansi.reset ++ "\n");
            continue;
        }

        // Headings
        if (trimmed.len > 0 and trimmed[0] == '#') {
            const level = std.mem.indexOfNone(u8, trimmed, "#") orelse trimmed.len;
            if (level <= 6 and level < trimmed.len and trimmed[level] == ' ') {
                const heading = trimmed[level + 1 ..];
                try writer.writeAll(ansi.bold_yellow);
                try writer.writeAll(heading);
                try writer.writeAll(ansi.reset ++ "\n");
                continue;
            }
        }

        // List items (- or *)
        if (trimmed.len > 1 and (trimmed[0] == '-' or trimmed[0] == '*') and trimmed[1] == ' ') {
            try writer.writeAll(ansi.green ++ "  \xe2\x80\xa2 " ++ ansi.reset); // • bullet
            try renderInline(writer, trimmed[2..]);
            try writer.writeByte('\n');
            continue;
        }

        // Regular paragraph line
        try renderInline(writer, line);
        try writer.writeByte('\n');
    }

    if (in_code_block) {
        try writer.writeAll(ansi.reset);
    }
}

fn renderInline(writer: anytype, text: []const u8) !void {
    var i: usize = 0;
    while (i < text.len) {
        // **bold**
        if (i + 1 < text.len and text[i] == '*' and text[i + 1] == '*') {
            if (std.mem.indexOf(u8, text[i + 2 ..], "**")) |end| {
                try writer.writeAll(ansi.bold);
                try writer.writeAll(text[i + 2 .. i + 2 + end]);
                try writer.writeAll(ansi.reset);
                i += end + 4;
                continue;
            }
        }

        // *italic*  (but not **)
        if (text[i] == '*' and (i + 1 >= text.len or text[i + 1] != '*')) {
            if (std.mem.indexOfScalar(u8, text[i + 1 ..], '*')) |end| {
                try writer.writeAll(ansi.italic);
                try writer.writeAll(text[i + 1 .. i + 1 + end]);
                try writer.writeAll(ansi.reset);
                i += end + 2;
                continue;
            }
        }

        // `inline code`
        if (text[i] == '`') {
            if (std.mem.indexOfScalar(u8, text[i + 1 ..], '`')) |end| {
                try writer.writeAll(ansi.cyan);
                try writer.writeAll(text[i + 1 .. i + 1 + end]);
                try writer.writeAll(ansi.reset);
                i += end + 2;
                continue;
            }
        }

        // Literal character
        try writer.writeByte(text[i]);
        i += 1;
    }
}

fn isHorizontalRule(line: []const u8) bool {
    if (line.len < 3) return false;
    const ch = line[0];
    if (ch != '-' and ch != '*' and ch != '_') return false;
    for (line) |b| {
        if (b != ch and b != ' ') return false;
    }
    return true;
}

fn logProgress(msg: []const u8) void {
    var buf: [256]u8 = undefined;
    var w = std.fs.File.stderr().writer(&buf);
    w.interface.writeAll(msg) catch {};
    w.interface.flush() catch {};
}

fn log(comptime fmt: []const u8, args: anytype) void {
    var buf: [4096]u8 = undefined;
    var w = std.fs.File.stderr().writer(&buf);
    w.interface.print(fmt ++ "\n", args) catch {};
    w.interface.flush() catch {};
}
