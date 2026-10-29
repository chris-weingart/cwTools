//! Install-wide settings: `cwtools.conf`, read once per process.
//!
//! Format: `key = value` lines, `#` comments, blank lines. Unknown keys are
//! ignored (logged). The first EXISTING file wins:
//!
//! 1. `$CWTOOLS_CONFIG`
//! 2. `cwtools.conf` in the directory that contains this plugin's
//!    `<Name>.ofx.bundle` (the loaded `.ofx` is found with `dladdr`)
//! 3. the system file: Linux `/opt/cwTools/cwtools.conf`, macOS
//!    `/Library/Application Support/cwTools/cwtools.conf`
//!
//! Keys (all optional): `models_dir`, `refine_models`, `matte_models`,
//! `segment_models`, `remote_server` (cwRefine / cwMatte Device = Remote),
//! `segment_server` (cwSegment Remote), `ort_lib` (`auto` or a path).
//!
//! A node param that is set explicitly always wins over the config; an empty
//! param means "from cwtools.conf".

use std::path::{Path, PathBuf};
use std::sync::OnceLock;

pub const ENV_VAR: &str = "CWTOOLS_CONFIG";
pub const FILE_NAME: &str = "cwtools.conf";

#[cfg(target_os = "macos")]
pub const SYSTEM_DIR: &str = "/Library/Application Support/cwTools";
#[cfg(not(target_os = "macos"))]
pub const SYSTEM_DIR: &str = "/opt/cwTools";

pub const KEYS: [&str; 7] = ["models_dir", "refine_models", "matte_models", "segment_models", "remote_server", "segment_server", "ort_lib"];

/// The resolved settings (defaults filled in).
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Config {
    /// The file the values came from (None = no file found, all defaults).
    pub source: Option<PathBuf>,
    pub models_dir: String,
    pub refine_models: String,
    pub matte_models: String,
    pub segment_models: String,
    /// cwtools-infer server for cwRefine / cwMatte Remote ("" = none configured).
    pub remote_server: String,
    /// SAM3 server for cwSegment Remote ("" = none configured).
    pub segment_server: String,
    /// "auto" or a library path.
    pub ort_lib: String,
}

/// `<SYSTEM_DIR>/models`.
pub fn default_models_dir() -> String {
    Path::new(SYSTEM_DIR).join("models").to_string_lossy().into_owned()
}

/// `<SYSTEM_DIR>/cwtools.conf`.
pub fn system_file() -> PathBuf {
    Path::new(SYSTEM_DIR).join(FILE_NAME)
}

/// Parse `key = value` text. Returns the known keys (last one wins) and the
/// notes worth logging (unknown keys, malformed lines). Keys are
/// case-insensitive; values are trimmed, and may be quoted ("..." / '...').
pub fn parse(text: &str) -> (Vec<(String, String)>, Vec<String>) {
    let mut kv: Vec<(String, String)> = Vec::new();
    let mut notes = Vec::new();
    for (no, raw) in text.lines().enumerate() {
        let line = raw.trim();
        if line.is_empty() || line.starts_with('#') {
            continue;
        }
        let Some((k, v)) = line.split_once('=') else {
            notes.push(format!("line {}: not 'key = value', ignored: {line}", no + 1));
            continue;
        };
        let k = k.trim().to_ascii_lowercase();
        let mut v = v.trim();
        if v.len() >= 2 && ((v.starts_with('"') && v.ends_with('"')) || (v.starts_with('\'') && v.ends_with('\''))) {
            v = &v[1..v.len() - 1];
        }
        if !KEYS.contains(&k.as_str()) {
            notes.push(format!("line {}: unknown key '{k}' ignored", no + 1));
            continue;
        }
        kv.retain(|(x, _)| *x != k);
        kv.push((k, v.to_string()));
    }
    (kv, notes)
}

impl Config {
    /// Settings from parsed pairs, defaults for whatever is missing or empty.
    pub fn from_pairs(source: Option<PathBuf>, kv: &[(String, String)]) -> Config {
        let get = |k: &str| kv.iter().find(|(x, _)| x == k).map(|(_, v)| v.trim().to_string()).filter(|v| !v.is_empty());
        let models_dir = get("models_dir").unwrap_or_else(default_models_dir);
        let sub = |k: &str, d: &str| get(k).unwrap_or_else(|| Path::new(&models_dir).join(d).to_string_lossy().into_owned());
        Config {
            refine_models: sub("refine_models", "cwRefine"),
            matte_models: sub("matte_models", "cwMatte"),
            segment_models: sub("segment_models", "cwSegment"),
            remote_server: get("remote_server").unwrap_or_default(),
            segment_server: get("segment_server").unwrap_or_default(),
            ort_lib: get("ort_lib").unwrap_or_else(|| "auto".to_string()),
            models_dir,
            source,
        }
    }

    /// Built-in defaults (no file).
    pub fn defaults() -> Config {
        Config::from_pairs(None, &[])
    }

    /// One line for the log.
    pub fn describe(&self) -> String {
        let q = |s: &str| if s.is_empty() { "(none)".to_string() } else { s.to_string() };
        format!(
            "refine_models={} matte_models={} segment_models={} remote_server={} segment_server={} ort_lib={}",
            self.refine_models,
            self.matte_models,
            self.segment_models,
            q(&self.remote_server),
            q(&self.segment_server),
            self.ort_lib
        )
    }
}

/// The directory that contains `<Name>.ofx.bundle`, given the path of the
/// loaded `.ofx` (`.../<Name>.ofx.bundle/Contents/<arch>/<Name>.ofx`). None
/// when the library is not inside a bundle (tests, examples).
pub fn bundle_parent(lib: &Path) -> Option<PathBuf> {
    let bundle = lib.parent()?.parent()?.parent()?;
    let name = bundle.file_name()?.to_string_lossy();
    if !name.ends_with(".ofx.bundle") {
        return None;
    }
    bundle.parent().map(Path::to_path_buf)
}

/// The search order (pure): candidate files with where each came from.
pub fn candidates(env: Option<&str>, own_lib: Option<&Path>, system: &Path) -> Vec<(PathBuf, &'static str)> {
    let mut c = Vec::new();
    if let Some(e) = env.map(str::trim).filter(|e| !e.is_empty()) {
        c.push((PathBuf::from(e), "$CWTOOLS_CONFIG"));
    }
    if let Some(dir) = own_lib.and_then(bundle_parent) {
        c.push((dir.join(FILE_NAME), "next to the plugin bundle"));
    }
    c.push((system.to_path_buf(), "system default"));
    c
}

/// First existing candidate (pure apart from the existence checks).
pub fn find(env: Option<&str>, own_lib: Option<&Path>, system: &Path) -> Option<(PathBuf, &'static str)> {
    candidates(env, own_lib, system).into_iter().find(|(p, _)| p.is_file())
}

/// Read + parse one file into a Config (notes = what to log).
pub fn load_file(path: &Path) -> Result<(Config, Vec<String>), String> {
    let text = std::fs::read_to_string(path).map_err(|e| format!("{}: {e}", path.display()))?;
    let (kv, notes) = parse(&text);
    Ok((Config::from_pairs(Some(path.to_path_buf()), &kv), notes))
}

/// Search and load (pure apart from the file system); `log` gets every line.
pub fn resolve(env: Option<&str>, own_lib: Option<&Path>, system: &Path, log: &mut dyn FnMut(String)) -> Config {
    if let Some(e) = env.map(str::trim).filter(|e| !e.is_empty()) {
        if !Path::new(e).is_file() {
            log(format!("config: ${ENV_VAR}={e} does not exist, ignored"));
        }
    }
    match find(env, own_lib, system) {
        Some((p, why)) => match load_file(&p) {
            Ok((c, notes)) => {
                for n in notes {
                    log(format!("config: {}: {n}", p.display()));
                }
                log(format!("config: using {} ({why}): {}", p.display(), c.describe()));
                c
            }
            Err(e) => {
                log(format!("config: cannot read {e}; using built-in defaults"));
                Config::defaults()
            }
        },
        None => {
            let looked: Vec<String> = candidates(env, own_lib, system).iter().map(|(p, _)| p.display().to_string()).collect();
            let c = Config::defaults();
            log(format!("config: no {FILE_NAME} found (looked in {}); built-in defaults: {}", looked.join(", "), c.describe()));
            c
        }
    }
}

// ---------------------------------------------------------------------------
// The loaded plugin library (dladdr)
// ---------------------------------------------------------------------------

#[repr(C)]
struct DlInfo {
    dli_fname: *const std::os::raw::c_char,
    dli_fbase: *mut std::os::raw::c_void,
    dli_sname: *const std::os::raw::c_char,
    dli_saddr: *mut std::os::raw::c_void,
}

extern "C" {
    fn dladdr(addr: *const std::os::raw::c_void, info: *mut DlInfo) -> std::os::raw::c_int;
}

/// Path of the shared object (the plugin's `.ofx`) this code is linked into.
pub fn own_library() -> Option<PathBuf> {
    let mut info = DlInfo { dli_fname: std::ptr::null(), dli_fbase: std::ptr::null_mut(), dli_sname: std::ptr::null(), dli_saddr: std::ptr::null_mut() };
    let addr = own_library as fn() -> Option<PathBuf> as *const std::os::raw::c_void;
    let ok = unsafe { dladdr(addr, &mut info) };
    if ok == 0 || info.dli_fname.is_null() {
        return None;
    }
    let s = unsafe { std::ffi::CStr::from_ptr(info.dli_fname) }.to_string_lossy().into_owned();
    if s.is_empty() {
        None
    } else {
        Some(PathBuf::from(s))
    }
}

static CONFIG: OnceLock<Config> = OnceLock::new();

/// The process-wide settings (searched, read and logged on first use).
pub fn get() -> &'static Config {
    CONFIG.get_or_init(|| {
        let env = std::env::var(ENV_VAR).ok();
        let lib = own_library();
        resolve(env.as_deref(), lib.as_deref(), &system_file(), &mut |l| crate::log::log(&l))
    })
}

/// The ONNX Runtime library this process would load with an empty field
/// (cwtools.conf ort_lib, else auto-detect), shown as the field's default;
/// "" when none is found (the field then stays empty = auto at render).
pub fn shown_ort_lib() -> String {
    crate::ort_find::resolve_ort_lib("").map(|(p, _)| p).unwrap_or_default()
}

/// `param` if set, else `fallback` (the config value).
pub fn or_config(param: &str, fallback: &str) -> String {
    let p = param.trim();
    if p.is_empty() {
        fallback.to_string()
    } else {
        p.to_string()
    }
}

/// A models folder: `param` if set, else `installed` (the config value). A
/// folder set in the node that does not exist on this machine (a setup saved
/// on another workstation) falls back to `installed` when that exists.
pub fn folder_or_installed(param: &str, installed: &str) -> String {
    let p = or_config(param, installed);
    if p != installed && !Path::new(&p).is_dir() && Path::new(installed).is_dir() {
        // read_values runs per render: log each fallback once per process
        static SEEN: std::sync::Mutex<Vec<String>> = std::sync::Mutex::new(Vec::new());
        if let Ok(mut g) = SEEN.lock() {
            if !g.contains(&p) {
                crate::log::log(&format!("models folder {p} not found here, using the installed {installed} (cwtools.conf)"));
                g.push(p.clone());
            }
        }
        return installed.to_string();
    }
    p
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn missing_node_folder_falls_back_to_installed() {
        let inst = std::env::temp_dir().join(format!("cwtools_inst_{}", std::process::id()));
        std::fs::create_dir_all(&inst).unwrap();
        let i = inst.to_string_lossy().into_owned();
        assert_eq!(folder_or_installed("", &i), i);
        assert_eq!(folder_or_installed("/no/such/folder", &i), i);
        let here = std::env::temp_dir().to_string_lossy().into_owned();
        assert_eq!(folder_or_installed(&here, &i), here.trim().to_string()); // existing explicit folder wins
        assert_eq!(folder_or_installed("/no/such/folder", "/no/installed"), "/no/such/folder"); // nothing better: keep it
        std::fs::remove_dir_all(&inst).ok();
    }

    pub(crate) fn tmp(name: &str) -> PathBuf {
        let d = std::env::temp_dir().join(format!("cwtools_cfg_{}_{name}", std::process::id()));
        let _ = std::fs::remove_dir_all(&d);
        std::fs::create_dir_all(&d).unwrap();
        d
    }

    #[test]
    fn parses_keys_comments_and_unknowns() {
        let (kv, notes) = parse(
            "# comment\n\n  models_dir = /m \nREMOTE_SERVER=http://h:8201\nsegment_server = \"http://s:8199\"\nbogus = 1\nnot a pair\nmodels_dir = /m2\n",
        );
        assert_eq!(kv, vec![
            ("remote_server".to_string(), "http://h:8201".to_string()),
            ("segment_server".to_string(), "http://s:8199".to_string()),
            ("models_dir".to_string(), "/m2".to_string()),
        ]);
        assert_eq!(notes.len(), 2);
        assert!(notes[0].contains("bogus") && notes[1].contains("not a pair"));
    }

    #[test]
    fn defaults_derive_from_models_dir() {
        let c = Config::defaults();
        assert_eq!(c.models_dir, default_models_dir());
        assert!(c.refine_models.ends_with("/models/cwRefine"));
        assert!(c.matte_models.ends_with("/models/cwMatte"));
        assert!(c.segment_models.ends_with("/models/cwSegment"));
        assert_eq!((c.remote_server.as_str(), c.segment_server.as_str(), c.ort_lib.as_str()), ("", "", "auto"));
        let (kv, _) = parse("models_dir = /x\nmatte_models = /b\nort_lib =\n");
        let c = Config::from_pairs(None, &kv);
        assert_eq!((c.refine_models.as_str(), c.matte_models.as_str(), c.segment_models.as_str()), ("/x/cwRefine", "/b", "/x/cwSegment"));
        assert_eq!(c.ort_lib, "auto"); // empty = default
    }

    #[test]
    fn bundle_parent_of_the_ofx() {
        assert_eq!(bundle_parent(Path::new("/p/ofx/cwRefine.ofx.bundle/Contents/Linux-x86-64/cwRefine.ofx")), Some(PathBuf::from("/p/ofx")));
        assert_eq!(bundle_parent(Path::new("/L/OFX/Plugins/cwMatte.ofx.bundle/Contents/MacOS/cwMatte.ofx")), Some(PathBuf::from("/L/OFX/Plugins")));
        assert_eq!(bundle_parent(Path::new("/tmp/target/debug/deps/cwtools_core-123")), None);
        assert_eq!(bundle_parent(Path::new("x.ofx")), None);
    }

    #[test]
    fn search_order_env_bundle_system() {
        let d = tmp("search");
        let env = d.join("env.conf");
        let plug = d.join("plugins");
        let lib = plug.join("cwRefine.ofx.bundle/Contents/Linux-x86-64/cwRefine.ofx");
        let sys = d.join("sys.conf");
        std::fs::create_dir_all(lib.parent().unwrap()).unwrap();
        let envs = env.to_string_lossy().into_owned();
        // nothing exists
        assert_eq!(find(Some(&envs), Some(&lib), &sys), None);
        let mut log = Vec::new();
        let c = resolve(Some(&envs), Some(&lib), &sys, &mut |l| log.push(l));
        assert_eq!(c, Config::defaults());
        assert!(log.iter().any(|l| l.contains("does not exist")) && log.iter().any(|l| l.contains("no cwtools.conf")));
        // system only
        std::fs::write(&sys, "remote_server = http://sys:1\n").unwrap();
        assert_eq!(find(Some(&envs), Some(&lib), &sys).unwrap().0, sys);
        // next to the bundle beats system
        std::fs::write(plug.join(FILE_NAME), "remote_server = http://bundle:1\n").unwrap();
        assert_eq!(find(Some(&envs), Some(&lib), &sys).unwrap().0, plug.join(FILE_NAME));
        // env beats both
        std::fs::write(&env, "remote_server = http://env:1\nfoo = bar\n").unwrap();
        let mut log = Vec::new();
        let c = resolve(Some(&envs), Some(&lib), &sys, &mut |l| log.push(l));
        assert_eq!(c.remote_server, "http://env:1");
        assert_eq!(c.source.as_deref(), Some(env.as_path()));
        assert!(log.iter().any(|l| l.contains("unknown key 'foo'")));
        assert!(log.iter().any(|l| l.contains("using") && l.contains("$CWTOOLS_CONFIG")));
        // empty env var = not set
        assert_eq!(resolve(Some(" "), Some(&lib), &sys, &mut |_| {}).remote_server, "http://bundle:1");
        // not in a bundle: straight to system
        assert_eq!(resolve(None, None, &sys, &mut |_| {}).remote_server, "http://sys:1");
        let _ = std::fs::remove_dir_all(&d);
    }

    #[test]
    fn explicit_param_wins() {
        assert_eq!(or_config("", "/cfg"), "/cfg");
        assert_eq!(or_config("  ", "/cfg"), "/cfg");
        assert_eq!(or_config(" /mine ", "/cfg"), "/mine");
    }

    #[test]
    fn own_library_resolves() {
        // In a test binary dladdr finds the executable itself (not a bundle).
        let p = own_library().expect("dladdr");
        assert!(p.exists(), "{}", p.display());
        assert_eq!(bundle_parent(&p), None);
    }
}
