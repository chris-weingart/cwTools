//! Tiny file logger. Default `/tmp/<Plugin>.log` (override: `<PLUGIN>_LOG`),
//! verbose switch `<PLUGIN>_VERBOSE`. Also mirrored to stderr (visible in the
//! shell that launched Flame). The plugin name is set once with [`set_name`]
//! (first thing in `OfxGetPlugin`); it defaults to "cwTools".
use std::io::Write;
use std::sync::{Mutex, OnceLock};

static LOCK: Mutex<()> = Mutex::new(());
static NAME: OnceLock<&'static str> = OnceLock::new();

/// Set the plugin name used for the log file, line prefix, env vars and
/// node-message id. First call wins.
pub fn set_name(name: &'static str) {
    let _ = NAME.set(name);
}

pub fn name() -> &'static str {
    NAME.get().copied().unwrap_or("cwTools")
}

fn env_var(suffix: &str) -> Option<String> {
    std::env::var(format!("{}_{suffix}", name().to_uppercase())).ok()
}

pub fn log_path() -> String {
    env_var("LOG").unwrap_or_else(|| format!("/tmp/{}.log", name()))
}

pub fn log(msg: &str) {
    let _g = LOCK.lock().unwrap_or_else(|e| e.into_inner());
    let ts = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_secs_f64())
        .unwrap_or(0.0);
    let line = format!("[{} {ts:.3}] {msg}\n", name());
    eprint!("{line}");
    if let Ok(mut f) = std::fs::OpenOptions::new().create(true).append(true).open(log_path()) {
        // Best-effort: a failed log write must never affect rendering.
        let _ = f.write_all(line.as_bytes());
    }
}

pub fn verbose() -> bool {
    env_var("VERBOSE").map(|v| v != "0").unwrap_or(false)
}

/// True when env var `<PLUGIN>_<suffix>` is set to something other than "" / "0".
pub fn env_flag(suffix: &str) -> bool {
    env_var(suffix).map(|v| v != "0" && !v.is_empty()).unwrap_or(false)
}
