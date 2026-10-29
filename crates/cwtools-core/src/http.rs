//! Shared HTTP plumbing for Remote modes: one keep-alive agent per plugin
//! process (reuses the connection across renders), bounded
//! timeouts, readable error strings. Callers pass the plate through on any
//! error (never hang, never crash).

use serde_json::Value;
use std::time::Duration;

/// Hard upper bound for any request (seconds). A render thread waits at most this long.
pub const MAX_TIMEOUT_S: f64 = 15.0;

/// One shared agent so HTTP keep-alive reuses the connection across renders;
/// the per-request timeout is set on each request.
pub fn shared_agent() -> &'static ureq::Agent {
    static A: std::sync::OnceLock<ureq::Agent> = std::sync::OnceLock::new();
    A.get_or_init(|| {
        ureq::AgentBuilder::new()
            .timeout_connect(Duration::from_secs(4))
            .timeout(Duration::from_secs(15))
            .max_idle_connections_per_host(4)
            .build()
    })
}

/// Per-request timeout, clamped to [1, MAX_TIMEOUT_S] seconds.
pub fn tmo(timeout_s: f64) -> Duration {
    Duration::from_secs_f64(timeout_s.clamp(1.0, MAX_TIMEOUT_S))
}

/// Server base URL without trailing slash / whitespace.
pub fn base(url: &str) -> String {
    url.trim().trim_end_matches('/').to_string()
}

pub trait JsonBody {
    fn into_json_value(self) -> Result<Value, String>;
}
impl JsonBody for ureq::Response {
    fn into_json_value(self) -> Result<Value, String> {
        let s = self.into_string().map_err(|e| e.to_string())?;
        serde_json::from_str(&s).map_err(|e| format!("bad JSON: {e}"))
    }
}

pub fn err_str(e: ureq::Error) -> String {
    match e {
        ureq::Error::Status(code, r) => {
            let body = r.into_string().unwrap_or_default();
            format!("HTTP {code}: {}", body.chars().take(200).collect::<String>())
        }
        ureq::Error::Transport(t) => format!("transport: {t}"),
    }
}

/// GET `<server>/health` and return the parsed JSON.
pub fn get_json(server: &str, path: &str, timeout_s: f64) -> Result<Value, String> {
    let r = shared_agent()
        .get(&format!("{}{path}", base(server)))
        .timeout(tmo(timeout_s))
        .call()
        .map_err(err_str)?;
    r.into_json_value()
}
