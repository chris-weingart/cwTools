//! Finding onnxruntime 1.22 for [`crate::ort_rt`] (no ort dependency: the
//! pure parts take paths, so tests use temp trees).
//!
//! [`select_ort`] order: (1) a 1.22 library already loaded in the process
//! (share Flame's copy), (2) the running Flame version's folder under
//! `/opt/Autodesk/lib64` (macOS: also inside the running .app), (3) the newest
//! `/opt/Autodesk/lib64/<version>/` holding a 1.22. [`resolve_ort_lib`] puts
//! the node param and cwtools.conf `ort_lib` in front of that.

use std::path::{Path, PathBuf};

/// Where Flame keeps its shared libraries, one folder per version.
pub const AUTODESK_LIB64: &str = "/opt/Autodesk/lib64";

/// Is `name` (a file name) the main onnxruntime library, version 1.22.x?
/// Linux `libonnxruntime.so.1.22[.N]`, macOS `libonnxruntime.1.22[.N].dylib`.
/// Only 1.22 is accepted: the ort crate (rc.10) speaks the 1.22 C API.
pub fn is_ort_122(name: &str) -> bool {
    let ver = name
        .strip_prefix("libonnxruntime.so.")
        .or_else(|| name.strip_prefix("libonnxruntime.").and_then(|r| r.strip_suffix(".dylib")));
    match ver {
        Some("1.22") => true,
        Some(v) => v.strip_prefix("1.22.").map(|p| !p.is_empty() && p.bytes().all(|b| b.is_ascii_digit())).unwrap_or(false),
        None => false,
    }
}

/// Is `name` some main onnxruntime library (any version)? (for log lines)
fn is_any_ort(name: &str) -> bool {
    name.starts_with("libonnxruntime.") && (name.contains(".so") || name.ends_with(".dylib"))
}

fn file_name(p: &Path) -> String {
    p.file_name().map(|f| f.to_string_lossy().into_owned()).unwrap_or_default()
}

/// The acceptable onnxruntime in `dir` (most specific name, e.g. `.so.1.22.0`
/// over the `.so.1.22` link).
pub fn find_ort_in_dir(dir: &Path) -> Option<PathBuf> {
    let mut found: Vec<PathBuf> = std::fs::read_dir(dir).ok()?.flatten().map(|e| e.path()).filter(|p| is_ort_122(&file_name(p)) && p.is_file()).collect();
    found.sort_by_key(|p| (file_name(p).len(), file_name(p)));
    found.pop()
}

/// Paths of the shared objects in a `/proc/self/maps` text.
pub fn maps_paths(maps: &str) -> Vec<String> {
    let mut out: Vec<String> = Vec::new();
    for line in maps.lines() {
        // address perms offset dev inode [path]; the path may contain spaces
        let mut it = line.splitn(6, char::is_whitespace);
        let path = it.nth(5).map(str::trim).unwrap_or("");
        if path.starts_with('/') && !path.ends_with("(deleted)") && !out.iter().any(|p| p == path) {
            out.push(path.to_string());
        }
    }
    out
}

/// The acceptable onnxruntime among the libraries loaded in the process.
pub fn loaded_ort(loaded: &[String]) -> Option<PathBuf> {
    loaded.iter().map(PathBuf::from).find(|p| is_ort_122(&file_name(p)))
}

/// Flame version folder name from the running executable's path, e.g.
/// `/opt/Autodesk/flame_2027.2.pr251/bin/flame` -> `2027.2.pr251`,
/// `/Applications/Autodesk/Flame 2027.2 Update PR251.app/...` -> `2027.2.pr251`.
pub fn flame_version(exe: &Path) -> Option<String> {
    for comp in exe.components() {
        let s = comp.as_os_str().to_string_lossy().to_ascii_lowercase();
        let s = s.strip_suffix(".app").unwrap_or(&s);
        if !(s.contains("flame") || s.contains("flare")) {
            continue;
        }
        let toks: Vec<&str> = s.split(|c: char| c == '_' || c == ' ' || c == '-').filter(|t| !t.is_empty()).collect();
        let is_ver = |t: &str| t.len() >= 4 && t.as_bytes()[..4].iter().all(|b| b.is_ascii_digit()) && t.bytes().all(|b| b.is_ascii_alphanumeric() || b == b'.');
        let Some(i) = toks.iter().position(|t| is_ver(t)) else { continue };
        let mut v = toks[i].trim_end_matches('.').to_string();
        if !v.contains("pr") {
            if let Some(pr) = toks[i + 1..].iter().find(|t| t.len() > 2 && t.starts_with("pr") && t[2..].bytes().all(|b| b.is_ascii_digit())) {
                v = format!("{v}.{pr}");
            }
        }
        return Some(v);
    }
    None
}

/// Sort key for a lib64 version folder: the numeric release parts, then
/// pre-release (`prNNN`) before release. `2027.pr240` < `2027.0.1` <
/// `2027.2.pr250` < `2027.2.pr251` < `2027.2` (the PR betas precede their
/// release; real folder names on Flame workstations).
pub fn version_key(name: &str) -> (Vec<u64>, u8, u64) {
    let mut nums = Vec::new();
    let mut pr = None;
    for t in name.split(|c: char| c == '.' || c == '_' || c == '-' || c == ' ') {
        let tl = t.to_ascii_lowercase();
        if let Some(n) = tl.strip_prefix("pr").and_then(|n| n.parse::<u64>().ok()) {
            pr = Some(n);
        } else if let Ok(n) = tl.parse::<u64>() {
            nums.push(n);
        }
    }
    match pr {
        Some(n) => (nums, 0, n),
        None => (nums, 1, 0),
    }
}

/// The newest `<root>/<version>/` that holds an acceptable onnxruntime.
pub fn newest_in_lib64(root: &Path) -> Option<PathBuf> {
    let mut dirs: Vec<((Vec<u64>, u8, u64), String, PathBuf)> = std::fs::read_dir(root)
        .ok()?
        .flatten()
        .filter(|e| e.path().is_dir())
        .filter_map(|e| {
            let n = e.file_name().to_string_lossy().into_owned();
            find_ort_in_dir(&e.path()).map(|lib| (version_key(&n), n, lib))
        })
        .collect();
    dirs.sort();
    dirs.pop().map(|(_, _, lib)| lib)
}

/// Pick onnxruntime 1.22 (pure: inputs are the loaded libraries, the
/// executable path and the lib64 root). Order: (1) already loaded in the
/// process (share Flame's copy), (2) the running Flame version's folder
/// (macOS: also the app's Contents/Frameworks), (3) the newest version
/// folder. Returns (library, why).
pub fn select_ort(loaded: &[String], exe: Option<&Path>, lib64: &Path) -> Result<(PathBuf, String), String> {
    if let Some(p) = loaded_ort(loaded) {
        return Ok((p, "already loaded in this process".to_string()));
    }
    let wrong: Vec<&String> = loaded.iter().filter(|p| is_any_ort(&file_name(Path::new(p)))).collect();
    if let Some(exe) = exe {
        if let Some(v) = flame_version(exe) {
            if let Some(p) = find_ort_in_dir(&lib64.join(&v)) {
                return Ok((p, format!("matches the running Flame {v} ({})", exe.display())));
            }
        }
        // macOS app bundle: <X.app>/Contents/Frameworks
        if let Some(app) = exe.ancestors().find(|a| a.extension().map(|e| e == "app").unwrap_or(false)) {
            for sub in ["Contents/Frameworks", "Contents/lib", "Contents/MacOS"] {
                if let Some(p) = find_ort_in_dir(&app.join(sub)) {
                    return Ok((p, format!("inside the running application {}", app.display())));
                }
            }
        }
    }
    if let Some(p) = newest_in_lib64(lib64) {
        return Ok((p, format!("newest 1.22 under {}", lib64.display())));
    }
    let mut e = format!("no onnxruntime 1.22 found (not loaded in Flame, none under {}); set ort_lib in cwtools.conf or the ONNX Runtime field", lib64.display());
    if !wrong.is_empty() {
        e.push_str(&format!(" (loaded but not 1.22: {wrong:?})"));
    }
    Err(e)
}

/// Libraries loaded in this process (Linux /proc/self/maps, macOS dyld).
pub fn loaded_libraries() -> Vec<String> {
    #[cfg(target_os = "macos")]
    {
        extern "C" {
            fn _dyld_image_count() -> u32;
            fn _dyld_get_image_name(image_index: u32) -> *const std::os::raw::c_char;
        }
        let mut out = Vec::new();
        let n = unsafe { _dyld_image_count() };
        for i in 0..n {
            let p = unsafe { _dyld_get_image_name(i) };
            if !p.is_null() {
                out.push(unsafe { std::ffi::CStr::from_ptr(p) }.to_string_lossy().into_owned());
            }
        }
        out
    }
    #[cfg(not(target_os = "macos"))]
    {
        std::fs::read_to_string("/proc/self/maps").map(|m| maps_paths(&m)).unwrap_or_default()
    }
}

/// The library to load for an "ONNX Runtime" param value: the param if set
/// (and present), else `ort_lib` from cwtools.conf if it is a path, else
/// auto-detect. Returns (library, why).
pub fn resolve_ort_lib(param: &str) -> Result<(String, String), String> {
    let auto = |s: &str| s.trim().is_empty() || s.trim().eq_ignore_ascii_case("auto");
    let mut note = String::new();
    if !auto(param) {
        let p = param.trim();
        if Path::new(p).is_file() {
            return Ok((p.to_string(), "ONNX Runtime field".to_string()));
        }
        // A setup saved on another workstation: find this one's instead.
        note = format!(" (ONNX Runtime field {p} does not exist here)");
    }
    let cfg = crate::config::get();
    if !auto(&cfg.ort_lib) {
        let src = cfg.source.as_ref().map(|s| s.display().to_string()).unwrap_or_default();
        return Ok((cfg.ort_lib.trim().to_string(), format!("ort_lib in {src}{note}")));
    }
    let exe = std::env::current_exe().ok();
    select_ort(&loaded_libraries(), exe.as_deref(), Path::new(AUTODESK_LIB64))
        .map(|(p, why)| (p.to_string_lossy().into_owned(), format!("auto: {why}{note}")))
        .map_err(|e| format!("{e}{note}"))
}


#[cfg(test)]
mod tests {
    use super::*;

    fn tree(name: &str, files: &[&str]) -> PathBuf {
        let d = std::env::temp_dir().join(format!("cwtools_ort_{}_{name}", std::process::id()));
        let _ = std::fs::remove_dir_all(&d);
        for f in files {
            let p = d.join(f);
            std::fs::create_dir_all(p.parent().unwrap()).unwrap();
            std::fs::write(p, b"x").unwrap();
        }
        std::fs::create_dir_all(&d).unwrap();
        d
    }

    #[test]
    fn only_1_22_is_accepted() {
        for ok in ["libonnxruntime.so.1.22.0", "libonnxruntime.so.1.22", "libonnxruntime.so.1.22.12", "libonnxruntime.1.22.0.dylib", "libonnxruntime.1.22.dylib"] {
            assert!(is_ort_122(ok), "{ok}");
        }
        for bad in [
            "libonnxruntime.so.1.20.1", "libonnxruntime.so.1.23.0", "libonnxruntime.so", "libonnxruntime.so.1.220", "libonnxruntime.so.1.22.x",
            "libonnxruntime_providers_cuda.so", "libonnxruntime.1.23.0.dylib", "libonnxruntime.dylib", "libonnxruntime.so.1.22.0.bak",
        ] {
            assert!(!is_ort_122(bad), "{bad}");
        }
    }

    #[test]
    fn maps_parsing_finds_loaded_ort() {
        let maps = "7f00-7f01 r--p 00000000 08:02 1 /opt/Autodesk/lib64/2027.2.pr251/libonnxruntime.so.1.22.0\n\
                    7f01-7f02 r-xp 00001000 08:02 1 /opt/Autodesk/lib64/2027.2.pr251/libonnxruntime.so.1.22.0\n\
                    7f02-7f03 rw-p 00000000 00:00 0 \n\
                    7f03-7f04 r--p 00000000 00:00 0 [heap]\n\
                    7f04-7f05 r--p 00000000 08:02 2 /opt/my libs/libfoo.so (deleted)\n\
                    7f05-7f06 r--p 00000000 08:02 3 /opt/my libs/libbar.so\n";
        let p = maps_paths(maps);
        assert_eq!(p, vec!["/opt/Autodesk/lib64/2027.2.pr251/libonnxruntime.so.1.22.0".to_string(), "/opt/my libs/libbar.so".to_string()]);
        assert_eq!(loaded_ort(&p), Some(PathBuf::from("/opt/Autodesk/lib64/2027.2.pr251/libonnxruntime.so.1.22.0")));
        assert_eq!(loaded_ort(&["/x/libonnxruntime.so.1.20.0".to_string()]), None);
    }

    #[test]
    fn flame_version_from_exe() {
        let v = |s: &str| flame_version(Path::new(s));
        assert_eq!(v("/opt/Autodesk/flame_2027.2.pr251/bin/flame").as_deref(), Some("2027.2.pr251"));
        assert_eq!(v("/opt/Autodesk/flame_2026.2.2/bin/flame").as_deref(), Some("2026.2.2"));
        assert_eq!(v("/opt/Autodesk/flare_2027.0.1/bin/flare").as_deref(), Some("2027.0.1"));
        assert_eq!(v("/opt/Autodesk/flameassist_2027/bin/flameassist").as_deref(), Some("2027"));
        assert_eq!(v("/Applications/Autodesk/Flame 2027.2 Update PR251.app/Contents/MacOS/Flame").as_deref(), Some("2027.2.pr251"));
        assert_eq!(v("/Applications/Autodesk/Flame 2027.1.app/Contents/MacOS/Flame").as_deref(), Some("2027.1"));
        // real macOS app names <-> lib64 folders 2027.pr240 / 2026.2.2 / 2027.0.1
        assert_eq!(v("/Applications/Autodesk/Flame 2027 PR240.app/Contents/MacOS/flame").as_deref(), Some("2027.pr240"));
        assert_eq!(v("/Applications/Autodesk/Flare 2026.2.2 Update.app/Contents/MacOS/flare").as_deref(), Some("2026.2.2"));
        assert_eq!(v("/opt/Autodesk/flame_2027.pr240/bin/flame").as_deref(), Some("2027.pr240"));
        assert_eq!(v("/usr/bin/python3"), None);
        assert_eq!(v("/home/me/target/debug/deps/cwtools_core-1234"), None);
    }

    #[test]
    fn version_folders_sort_numerically() {
        assert!(version_key("2027.2.pr251") > version_key("2027.0.1"));
        let order = ["2026.2.2", "2027.pr235", "2027.pr240", "2027.0.1", "2027.2.pr250", "2027.2.pr251", "2027.2", "2027.2.1"];
        for w in order.windows(2) {
            assert!(version_key(w[0]) < version_key(w[1]), "{} < {}", w[0], w[1]);
        }
        assert!(version_key("2027.0.10") > version_key("2027.0.9"));
        assert!(version_key("2026.2.2") < version_key("2027"));
    }

    #[test]
    fn selection_order_on_fake_trees() {
        let root = tree(
            "sel",
            &[
                "lib64/2026.2.2/libonnxruntime.so.1.22.0",
                "lib64/2027.0.1/libonnxruntime.so.1.22.0",
                "lib64/2027.0.1/libonnxruntime.so.1.22",
                "lib64/2027.2.pr251/libonnxruntime.so.1.23.0", // wrong version: never picked
                "lib64/2028.0/README",                          // no ORT at all
                "lib64/2027.pr240/libonnxruntime.1.22.0.dylib", // a 2027 beta: older than 2027.0.1
                "Flame 2027.2 Update PR251.app/Contents/Frameworks/libonnxruntime.1.22.0.dylib",
            ],
        );
        let lib64 = root.join("lib64");
        assert_eq!(find_ort_in_dir(&lib64.join("2027.0.1")), Some(lib64.join("2027.0.1/libonnxruntime.so.1.22.0")));
        assert_eq!(find_ort_in_dir(&lib64.join("2027.2.pr251")), None);
        // (3) newest folder with a 1.22
        assert_eq!(newest_in_lib64(&lib64), Some(lib64.join("2027.0.1/libonnxruntime.so.1.22.0")));
        let (p, why) = select_ort(&[], Some(Path::new("/usr/bin/python3")), &lib64).unwrap();
        assert_eq!((p, why.contains("newest")), (lib64.join("2027.0.1/libonnxruntime.so.1.22.0"), true));
        // (2) the running Flame's version beats newest
        let (p, why) = select_ort(&[], Some(Path::new("/opt/Autodesk/flame_2026.2.2/bin/flame")), &lib64).unwrap();
        assert_eq!(p, lib64.join("2026.2.2/libonnxruntime.so.1.22.0"));
        assert!(why.contains("2026.2.2"), "{why}");
        // running Flame's folder has no 1.22 -> app bundle (macOS) -> newest
        let app_exe = root.join("Flame 2027.2 Update PR251.app/Contents/MacOS/Flame");
        let (p, why) = select_ort(&[], Some(&app_exe), &lib64).unwrap();
        assert_eq!(p, root.join("Flame 2027.2 Update PR251.app/Contents/Frameworks/libonnxruntime.1.22.0.dylib"));
        assert!(why.contains("application"), "{why}");
        let (p, _) = select_ort(&[], Some(Path::new("/opt/Autodesk/flame_2027.2.pr251/bin/flame")), &lib64).unwrap();
        assert_eq!(p, lib64.join("2027.0.1/libonnxruntime.so.1.22.0"));
        // (1) already loaded wins over everything; a loaded 1.20 does not
        let loaded = vec!["/somewhere/libonnxruntime.so.1.22.0".to_string()];
        let (p, why) = select_ort(&loaded, Some(Path::new("/opt/Autodesk/flame_2026.2.2/bin/flame")), &lib64).unwrap();
        assert_eq!((p, why.as_str()), (PathBuf::from("/somewhere/libonnxruntime.so.1.22.0"), "already loaded in this process"));
        let loaded = vec!["/somewhere/libonnxruntime.so.1.20.0".to_string()];
        assert_eq!(select_ort(&loaded, None, &lib64).unwrap().0, lib64.join("2027.0.1/libonnxruntime.so.1.22.0"));
        // nothing anywhere: a clear error that names the wrong loaded version
        let e = select_ort(&loaded, None, &root.join("nope")).unwrap_err();
        assert!(e.contains("ort_lib") && e.contains("1.20"), "{e}");
        let _ = std::fs::remove_dir_all(&root);
    }
}
