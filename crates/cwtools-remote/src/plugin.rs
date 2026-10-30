//! Plugin-side helpers shared by cwRefine and cwMatte Remote mode: server
//! address from the params (cwSegment pattern), artist-facing error
//! messages, the Test Server report, a per-process cache of the server's
//! model list, and a quick reachability probe (for the Local OOM hint).

use crate::{get_json, RemoteError};
use serde_json::Value;
use std::sync::Mutex;
use std::time::{Duration, Instant};

/// No built-in server: Server URL empty = `remote_server` from cwtools.conf
/// (cwtools_core::config).
pub const DEFAULT_SERVER: &str = "";
/// Node message when Remote has no server (Remote fails with it, nothing is sent).
pub const NO_SERVER_MSG: &str = "No Remote server configured: set remote_server in cwtools.conf or the Server URL field";
/// Per-frame request timeout (s): HR BiRefNet can take ~4-9 s per frame over a
/// slow (~0.75 MB/s) upload, so the 15 s cap of the core HTTP
/// helper is too tight.
pub const DEFAULT_TIMEOUT_S: f64 = 30.0;
pub const MIN_TIMEOUT_S: f64 = 2.0;
pub const MAX_TIMEOUT_S: f64 = 120.0;

/// Base URL: the Server URL string, or `configured` when it is empty
/// (cwtools.conf `remote_server`; "" = none, which Remote reports as
/// [`NO_SERVER_MSG`]). `http://` is added when the scheme is missing.
pub fn server_url(url: &str, configured: &str) -> String {
    let u = url.trim();
    let u = if u.is_empty() { configured.trim() } else { u };
    if u.is_empty() {
        String::new()
    } else if u.contains("://") {
        u.trim_end_matches('/').to_string()
    } else {
        format!("http://{}", u.trim_end_matches('/'))
    }
}

pub fn clamp_timeout(t: f64) -> f64 {
    if t.is_finite() {
        t.clamp(MIN_TIMEOUT_S, MAX_TIMEOUT_S)
    } else {
        DEFAULT_TIMEOUT_S
    }
}

/// One line for the node message (the caller appends what passes through).
pub fn artist_message(e: &RemoteError, server: &str, model: &str, timeout_s: f64) -> String {
    match (e.status, e.code.as_str()) {
        (0, "no_server") => NO_SERVER_MSG.to_string(),
        (0, "unreachable") => format!("Remote: server {server} unreachable ({}) - check the network / the server, press Test Server", short(&e.msg)),
        (0, "timeout") => format!("Remote: no answer from {server} within {timeout_s:.0} s - raise Timeout (s) or use a smaller model / crop"),
        (409, _) => format!(
            "Remote: the server's {model} is a different file than this node's (sha256 mismatch) - install the same model on both sides ({})",
            short(&e.msg)
        ),
        (404, _) => format!("Remote: server has no model {model} - copy it to the server's model folder ({})", short(&e.msg)),
        (503, "busy") => "Remote: server busy (queue full) - re-render in a moment".to_string(),
        (503, "gpu_oom") => format!("Remote: server GPU out of memory for {model} - re-render later or use a smaller model / crop"),
        (0, "sha_mismatch") => "Remote: response is for a different model file - re-render".to_string(),
        (0, "nonfinite_output") => "Remote: server returned NaN/Inf (GPU fault?) - re-render".to_string(),
        (0, _) => format!("Remote: {} error talking to {server}: {}", e.code, short(&e.msg)),
        (s, c) => format!("Remote: server error {s} {c}: {}", short(&e.msg)),
    }
}

fn short(s: &str) -> String {
    let s = s.replace('\n', " ");
    if s.chars().count() > 120 {
        format!("{}...", s.chars().take(120).collect::<String>())
    } else {
        s
    }
}

/// What the server knows about one model (from GET /models).
#[derive(Clone, Debug)]
pub struct ModelInfo {
    pub name: String,
    pub sha256: String,
    pub state: String,
    pub servable: bool,
    pub input: String,
    pub output: String,
    /// Input shape; dynamic dims are -1.
    pub in_shape: Vec<i64>,
    pub in_dtype: String,
    /// Sidecar JSON as served (Value::Null if none).
    pub sidecar: Value,
}

fn parse_models(v: &Value) -> Vec<ModelInfo> {
    let mut out = Vec::new();
    for m in v["models"].as_array().into_iter().flatten() {
        let dim = |d: &Value| d.as_i64().unwrap_or(-1);
        out.push(ModelInfo {
            name: m["name"].as_str().unwrap_or("").to_string(),
            sha256: m["sha256"].as_str().unwrap_or("").to_lowercase(),
            state: m["state"].as_str().unwrap_or("").to_string(),
            servable: m["servable"].as_bool().unwrap_or(false),
            input: m["inputs"][0]["name"].as_str().unwrap_or("").to_string(),
            output: m["outputs"][0]["name"].as_str().unwrap_or("").to_string(),
            in_shape: m["inputs"][0]["shape"].as_array().map(|a| a.iter().map(dim).collect()).unwrap_or_default(),
            in_dtype: m["inputs"][0]["dtype"].as_str().unwrap_or("").to_string(),
            sidecar: m["sidecar"].clone(),
        });
    }
    out
}

static MODELS: Mutex<Vec<(String, Instant, Vec<ModelInfo>)>> = Mutex::new(Vec::new());
const MODELS_TTL: Duration = Duration::from_secs(300);

/// The server's entry for `name` (GET /models, cached per server for 5 min;
/// refetched once when the name or sha is not in the cached list).
pub fn model_info(server: &str, name: &str, sha: &str, timeout_s: f64) -> Result<ModelInfo, RemoteError> {
    let find = |list: &[ModelInfo]| list.iter().find(|m| m.name == name).cloned();
    if let Ok(g) = MODELS.lock() {
        if let Some((_, t, list)) = g.iter().find(|(s, _, _)| s == server) {
            if t.elapsed() < MODELS_TTL {
                if let Some(m) = find(list) {
                    if m.sha256 == sha {
                        return Ok(m);
                    }
                }
            }
        }
    }
    let v = get_json(server, "/models", timeout_s)?;
    let list = parse_models(&v);
    let found = find(&list);
    if let Ok(mut g) = MODELS.lock() {
        g.retain(|(s, _, _)| s != server);
        g.push((server.to_string(), Instant::now(), list));
    }
    found.ok_or_else(|| RemoteError { status: 404, code: "unknown_model".into(), msg: format!("{name} is not in the server's /models list"), missing: Vec::new() })
}

/// Forget the cached model lists (Test Server, after a 409).
pub fn forget_models() {
    if let Ok(mut g) = MODELS.lock() {
        g.clear();
    }
}

/// Test Server: /health summary + whether `model` with `sha` is servable.
/// Ok = everything fine for this node; Err = what is wrong (still informative).
pub fn test_server(server: &str, model: &str, sha: Result<String, String>) -> Result<String, String> {
    if server.trim().is_empty() {
        return Err(NO_SERVER_MSG.to_string());
    }
    let t0 = Instant::now();
    let h = get_json(server, "/health", 5.0).map_err(|e| format!("{server}: {}", artist_message(&e, server, model, 5.0)))?;
    let rtt = t0.elapsed().as_secs_f64() * 1e3;
    let vram = &h["vram"];
    let gb = |k: &str| vram[k].as_f64().unwrap_or(f64::NAN) / 1024.0;
    let loaded: Vec<String> = h["sessions"]
        .as_array()
        .into_iter()
        .flatten()
        .map(|s| format!("{} ({})", s["model"].as_str().unwrap_or("?"), s["ep"].as_str().unwrap_or("?")))
        .collect();
    let q = &h["queue"];
    let head = format!(
        "server OK {:.0} ms, ORT {}, GPU {:.1}/{:.1} GB used, server {:.1} GB, loaded: {}, queue {}",
        rtt,
        h["onnxruntime"].as_str().unwrap_or("?"),
        gb("gpu_used_mb"),
        gb("gpu_total_mb"),
        gb("process_mb"),
        if loaded.is_empty() { "none".to_string() } else { loaded.join(", ") },
        q["waiting"].as_i64().unwrap_or(0),
    );
    let sha = match sha {
        Ok(s) => s,
        Err(e) => return Err(format!("{head}; this node's model {model}: cannot hash ({e})")),
    };
    forget_models();
    match model_info(server, model, &sha, 15.0) {
        Ok(m) if m.sha256 != sha => Err(format!(
            "{head}; model {model}: DIFFERENT FILE on the server (sha {} vs this node's {}) - renders will fail",
            &m.sha256[..m.sha256.len().min(12)],
            &sha[..sha.len().min(12)]
        )),
        Ok(m) if !m.servable => Err(format!("{head}; model {model}: on the server but not servable (state {})", m.state)),
        Ok(m) => Ok(format!("{head}; model {model} sha {} available ({})", &sha[..sha.len().min(12)], m.state)),
        Err(e) if e.status == 404 => Err(format!("{head}; model {model}: NOT on the server")),
        Err(e) => Err(format!("{head}; /models failed: {e}")),
    }
}

/// Quick probe (GET /health) for hints, e.g. "Device = Remote" after a Local OOM.
pub fn reachable(server: &str, timeout_s: f64) -> bool {
    get_json(server, "/health", timeout_s).map(|v| v["ok"].as_bool().unwrap_or(false)).unwrap_or(false)
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn urls() {
        // nothing set anywhere = no server
        assert_eq!(server_url("", ""), "");
        // empty URL = the configured server
        assert_eq!(server_url(" ", "http://cfg:8201/"), "http://cfg:8201");
        assert_eq!(server_url("", "cfg:8201"), "http://cfg:8201");
        // an explicit value wins over the config
        assert_eq!(server_url(" http://h:1/ ", "http://cfg:1"), "http://h:1");
        assert_eq!(server_url("10.0.0.2:8201", ""), "http://10.0.0.2:8201");
        assert_eq!(test_server("", "m", Ok("s".into())).unwrap_err(), NO_SERVER_MSG);
        assert_eq!(clamp_timeout(500.0), MAX_TIMEOUT_S);
        assert_eq!(clamp_timeout(f64::NAN), DEFAULT_TIMEOUT_S);
    }
    #[test]
    fn messages() {
        let e = |s: u16, c: &str| RemoteError { status: s, code: c.into(), msg: "m".into(), missing: Vec::new() };
        assert!(artist_message(&e(409, "sha_mismatch"), "S", "M", 30.0).contains("sha256 mismatch"));
        assert!(artist_message(&e(0, "timeout"), "S", "M", 30.0).contains("within 30 s"));
        assert!(artist_message(&e(0, "unreachable"), "S", "M", 30.0).contains("unreachable"));
        assert!(artist_message(&e(503, "busy"), "S", "M", 30.0).contains("busy"));
        assert!(artist_message(&e(503, "gpu_oom"), "S", "M", 30.0).contains("out of memory"));
        assert_eq!(artist_message(&e(0, "no_server"), "", "M", 30.0), NO_SERVER_MSG);
    }
    #[test]
    fn models_json() {
        let v: Value = serde_json::from_str(
            r#"{"models":[{"name":"a","sha256":"AB","state":"ok","servable":true,
            "inputs":[{"name":"x","dtype":"float32","shape":[1,4,"h","w"]}],"outputs":[{"name":"y"}],"sidecar":null}]}"#,
        )
        .unwrap();
        let m = &parse_models(&v)[0];
        assert_eq!((m.name.as_str(), m.sha256.as_str(), m.in_shape.clone()), ("a", "ab", vec![1, 4, -1, -1]));
    }
}
