use std::path::Path;
use std::process::Command;

fn main() {
    println!("cargo:rerun-if-changed=c/variadic_helpers.c");
    cc::Build::new().file("c/variadic_helpers.c").warnings(true).compile("ofx_variadic_helpers");

    // Build id shown in Flame (Version param, OFX version label, plugin log):
    // `git describe` in a checkout; in a build copy without .git (e.g. an rsync'd
    // tree) the BUILD_ID file that scripts/stamp_build_id.sh writes before the sync.
    let ws = Path::new(env!("CARGO_MANIFEST_DIR")).join("../..");
    for p in [".git/HEAD", ".git/index", ".git/refs/tags", "BUILD_ID"] {
        println!("cargo:rerun-if-changed={}", ws.join(p).display());
    }
    let git = Command::new("git")
        .args(["describe", "--tags", "--match", "v[0-9]*", "--dirty", "--always"])
        .current_dir(&ws)
        .output()
        .ok()
        .filter(|o| o.status.success())
        .map(|o| String::from_utf8_lossy(&o.stdout).trim().to_string())
        .filter(|s| !s.is_empty());
    let id = git
        .or_else(|| std::fs::read_to_string(ws.join("BUILD_ID")).ok().map(|s| s.trim().to_string()).filter(|s| !s.is_empty()))
        .unwrap_or_else(|| "unknown".to_string());
    println!("cargo:rustc-env=CWTOOLS_BUILD_ID={id}");
}
