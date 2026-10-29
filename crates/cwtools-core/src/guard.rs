//! Panic guard: every OFX entry point (main entry, interacts) runs through
//! [`guarded`] so a Rust panic becomes `kOfxStatFailed` + a log line instead of
//! unwinding into Flame. Requires `panic = "unwind"` (workspace profile).

use crate::ofx_api::{kOfxStatFailed, OfxStatus};

/// Run `f`, converting a panic into kOfxStatFailed (and a log line).
pub fn guarded<F: FnOnce() -> OfxStatus>(what: &str, f: F) -> OfxStatus {
    match std::panic::catch_unwind(std::panic::AssertUnwindSafe(f)) {
        Ok(s) => s,
        Err(e) => {
            let msg = e
                .downcast_ref::<&str>()
                .map(|s| s.to_string())
                .or_else(|| e.downcast_ref::<String>().cloned())
                .unwrap_or_else(|| "unknown panic".into());
            crate::log::log(&format!("PANIC in {what}: {msg}"));
            kOfxStatFailed
        }
    }
}

#[cfg(test)]
mod tests {
    #[test]
    fn panic_becomes_failed() {
        let prev = std::panic::take_hook();
        std::panic::set_hook(Box::new(|_| {}));
        let s = super::guarded("test", || panic!("boom"));
        std::panic::set_hook(prev);
        assert_eq!(s, crate::ofx_api::kOfxStatFailed);
        assert_eq!(super::guarded("ok", || 0), 0);
    }
}
