//! Convention guard: a plugin writes ONLY into the Output clip.
//!
//! In OFX every clip except "Output" is an INPUT. cwRefine's "OutMatte" clip
//! and cwSegment's "second output" were shown by Flame as (mandatory) input
//! sockets, and cwProbe's "2Out" test had in fact written into an input
//! buffer. Flame splits the RGBA Output into Result RGB + the OutMatte alpha
//! socket: 4 independent channels per node is the maximum.
//!
//! Scans every shipped plugin crate (cwprobe is a dev-only diagnostic) for:
//! 1. clip definitions: only Source, Output and known input clips;
//! 2. `fetch_image(...)` bound to `out` must be the Output clip, and the
//!    Output clip may only be bound to `out`;
//! 3. write pointers (`let op* = unsafe { X.px(..) }`) only from `out`.

use std::path::{Path, PathBuf};

const PLUGIN_CRATES: &[&str] = &["cwsegment", "cwrefine", "cwmatte"];
/// Clip-name expressions a plugin may define / fetch (inputs + Output).
const ALLOWED_CLIPS: &[&str] = &["kOfxImageEffectSimpleSourceClipName", "kOfxImageEffectOutputClipName", "TRIMAP_CLIP_NAME"];

fn sources() -> Vec<(PathBuf, String)> {
    let crates = Path::new(env!("CARGO_MANIFEST_DIR")).parent().unwrap().to_path_buf();
    let mut out = Vec::new();
    for c in PLUGIN_CRATES {
        let dir = crates.join(c).join("src");
        for e in std::fs::read_dir(&dir).unwrap_or_else(|e| panic!("{}: {e}", dir.display())) {
            let p = e.unwrap().path();
            if p.extension().map(|x| x == "rs").unwrap_or(false) {
                let s = std::fs::read_to_string(&p).unwrap();
                out.push((p, s));
            }
        }
    }
    assert!(out.len() >= 10, "found only {} plugin sources", out.len());
    out
}

/// The first argument after `eh,` / `effect_handle,` of each call to `f`.
fn clip_args<'a>(s: &'a str, f: &str) -> Vec<(usize, &'a str, &'a str)> {
    let mut v = Vec::new();
    let mut from = 0;
    while let Some(i) = s[from..].find(f) {
        let at = from + i;
        from = at + f.len();
        if s[..at].ends_with("fn ") {
            continue;
        }
        let rest = &s[from..];
        let Some(comma) = rest.find(',') else { continue };
        let arg = rest[comma + 1..].trim_start();
        let end = arg.find(|c: char| c == ',' || c == ')').unwrap_or(arg.len());
        let line_start = s[..at].rfind('\n').map(|x| x + 1).unwrap_or(0);
        v.push((s[..at].matches('\n').count() + 1, arg[..end].trim(), &s[line_start..at]));
    }
    v
}

#[test]
fn only_output_is_written() {
    let mut bad = Vec::new();
    for (p, s) in sources() {
        let f = p.display();
        // 1. clip definitions
        for call in ["define_clip(", "(effect().clipDefine)("] {
            for (line, arg, _) in clip_args(&s, call) {
                let arg = arg.split(".as_ptr(").next().unwrap_or(arg);
                if arg == "name" {
                    continue; // the define_clip helper itself
                }
                if !ALLOWED_CLIPS.contains(&arg) {
                    bad.push(format!("{f}:{line}: clip '{arg}' defined - only Source / Output / known inputs"));
                }
            }
        }
        // 2. fetches
        for (line, arg, prefix) in clip_args(&s, "fetch_image(") {
            if !ALLOWED_CLIPS.contains(&arg) {
                bad.push(format!("{f}:{line}: fetch_image of '{arg}'"));
            }
            let binds_out = prefix.contains("Some(out)") || prefix.contains("let out ");
            let is_output = arg == "kOfxImageEffectOutputClipName";
            if binds_out != is_output {
                bad.push(format!("{f}:{line}: the Output clip must be bound to `out` (and only it)"));
            }
        }
        // 3. write pointers
        for (n, l) in s.lines().enumerate() {
            let t = l.trim_start();
            if t.starts_with("let op") && t.contains(".px(") && !t.contains("unsafe { out.px(") {
                bad.push(format!("{f}:{}: write pointer not from `out`: {t}", n + 1));
            }
            if t.contains("OutMatte\\0") || t.contains("SECOND_OUTPUT") {
                bad.push(format!("{f}:{}: extra output clip: {t}", n + 1));
            }
        }
    }
    assert!(bad.is_empty(), "clip write guard:\n{}", bad.join("\n"));
}
