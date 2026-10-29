# Building cwTools

Overview and downloads: [README](../README.md). Most people want a release package, not a source build.

## Layout

| Path | What |
|---|---|
| `crates/cwtools-core` | shared OFX plumbing, colour (ACES 2.0 view LUT), ONNX Runtime loading and model cache, panel layout |
| `crates/cwsegment` | cwSegment: SAM3 segmentation (text / point / box prompts, multi-object, VLM Auto) |
| `crates/cwmatte` | cwMatte: BiRefNet automatic matte |
| `crates/cwrefine` | cwRefine: ViTMatte / MEMatte refinement of an input mask |
| `crates/cwtools-remote`, `server/` | Remote mode client and the inference server |
| `crates/cwfill` | cwFill: foreground colour solve and despill (own Cargo workspace and version) |
| `flame_inference/matchbox/cwExtract` | Matchbox that pulls single objects out of cwSegment's Object IDs output |
| `install/` | installer and package builder |

## Requirements

- Rust 1.85 or newer (the locked dependencies use edition 2024).
- Linux x86-64 or macOS. Build natively on the platform you ship for.
- At run time the plugins load the ONNX Runtime 1.22 that ships with Flame 2027.2;
  nothing from ONNX Runtime is linked at build time.

## Build

```sh
scripts/build.sh               # cwSegment, cwMatte, cwRefine -> dist/<Name>.ofx.bundle
scripts/build.sh cwRefine      # one plugin
cargo test --workspace         # tests (no GPU or models needed)
(cd crates/cwfill && scripts/build.sh && cargo test)
```

Packages for a release: `install/make_package.sh --help`.

## Versions

cwSegment, cwMatte and cwRefine share one version (`Cargo.toml`
`[workspace.package]`, tags `vX.Y.Z`, see `CHANGELOG.md`); Flame shows it as each
node's read-only **Version** parameter. cwFill has its own version and `cwfill-vX.Y.Z`
tags. While the version is 0.x only the minor number moves, because Flame matches
saved setups on the major version.

## Rules for contributors

- Every OFX entry point runs inside `catch_unwind`: a bug must never take Flame down.
- Only the `Output` clip is ever written (`crates/cwtools-core/tests/clip_write_guard.rs`).
- Panel layout is checked by `cwtools_core::params::grid` in each plugin's tests.
