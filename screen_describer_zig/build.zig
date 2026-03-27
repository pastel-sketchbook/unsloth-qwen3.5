const std = @import("std");

pub fn build(b: *std.Build) void {
    const target = b.standardTargetOptions(.{});
    const optimize = b.standardOptimizeOption(.{});

    const mod = b.createModule(.{
        .root_source_file = b.path("src/main.zig"),
        .target = target,
        .optimize = optimize,
    });

    // Link against system-installed llama.cpp (brew install llama.cpp)
    // Note: ggml/ggml-base are transitive deps of libllama/libmtmd — linking
    // them explicitly causes a dyld "duplicate linked dylib" error at runtime.
    mod.linkSystemLibrary("llama", .{});
    mod.linkSystemLibrary("mtmd", .{});
    mod.linkSystemLibrary("c", .{});

    const exe = b.addExecutable(.{
        .name = "screen_describer",
        .root_module = mod,
    });

    b.installArtifact(exe);

    const run_cmd = b.addRunArtifact(exe);
    run_cmd.step.dependOn(b.getInstallStep());
    if (b.args) |args| {
        run_cmd.addArgs(args);
    }

    const run_step = b.step("run", "Run screen_describer");
    run_step.dependOn(&run_cmd.step);
}
