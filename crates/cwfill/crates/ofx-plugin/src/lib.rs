//! cwFill — an OpenFX plugin that estimates clean foreground
//! colour (unpremult / despill extension) from a plate + matte, by porting
//! pymatting's multilevel foreground solver. Targets Autodesk Flame (Batch/BFX,
//! Filter context). See README for the design.

#![allow(non_snake_case)]

pub mod ofx_api;
pub mod params;
pub mod render; // pub (with params/ofx_api) so examples/parity.rs drives the real render core.
pub mod solver; // pub so `cargo test` and external benches can reach the math.
pub mod par;
mod cache;

mod actions;

use ofx_api::*;
use std::ffi::CStr;
use std::os::raw::{c_char, c_int, c_void};

// ---------------------------------------------------------------------------
// Global host + suite pointers. Set once during OfxActionLoad, read-only after.
// Single-threaded plugin init; access is wrapped in unsafe accessors.
// ---------------------------------------------------------------------------
pub(crate) struct Suites {
    pub host: *mut OfxHost,
    pub prop: *const OfxPropertySuiteV1,
    pub effect: *const OfxImageEffectSuiteV1,
    pub param: *const OfxParameterSuiteV1,
}

static mut SUITES: Suites = Suites {
    host: std::ptr::null_mut(),
    prop: std::ptr::null(),
    effect: std::ptr::null(),
    param: std::ptr::null(),
};

#[inline]
pub(crate) fn suites() -> &'static Suites {
    // Safe: written once in action_load before any other action runs.
    unsafe { &*std::ptr::addr_of!(SUITES) }
}

/// Set the suite pointers. Called once from action_load on the load action,
/// before any render/describe runs.
pub(crate) fn set_suites(
    prop: *const OfxPropertySuiteV1,
    effect: *const OfxImageEffectSuiteV1,
    param: *const OfxParameterSuiteV1,
) {
    unsafe {
        let s = &mut *std::ptr::addr_of_mut!(SUITES);
        s.prop = prop;
        s.effect = effect;
        s.param = param;
    }
}
#[inline]
pub(crate) fn prop() -> &'static OfxPropertySuiteV1 {
    unsafe { &*suites().prop }
}
#[inline]
pub(crate) fn effect() -> &'static OfxImageEffectSuiteV1 {
    unsafe { &*suites().effect }
}
#[inline]
pub(crate) fn param() -> &'static OfxParameterSuiteV1 {
    unsafe { &*suites().param }
}

// ---------------------------------------------------------------------------
// Plugin entry points (the three symbols a host looks for in an .ofx bundle).
// ---------------------------------------------------------------------------
static PLUGIN_IDENTIFIER: &[u8] = b"dev.weingart.cwFill\0";

static mut PLUGIN: OfxPlugin = OfxPlugin {
    pluginApi: std::ptr::null(),
    apiVersion: kOfxImageEffectPluginApiVersion,
    pluginIdentifier: std::ptr::null(),
    pluginVersionMajor: 0,
    pluginVersionMinor: 1,
    setHost: Some(set_host),
    mainEntry: Some(main_entry),
};

unsafe extern "C" fn set_host(host: *mut OfxHost) {
    SUITES.host = host;
}

#[no_mangle]
pub extern "C" fn OfxGetNumberOfPlugins() -> c_int {
    1
}

#[no_mangle]
pub extern "C" fn OfxGetPlugin(nth: c_int) -> *const OfxPlugin {
    if nth != 0 {
        return std::ptr::null();
    }
    unsafe {
        PLUGIN.pluginApi = kOfxImageEffectPluginApi.as_ptr() as *const c_char;
        PLUGIN.pluginIdentifier = PLUGIN_IDENTIFIER.as_ptr() as *const c_char;
        std::ptr::addr_of!(PLUGIN)
    }
}

// ---------------------------------------------------------------------------
// Action dispatch
// ---------------------------------------------------------------------------
unsafe extern "C" fn main_entry(
    action: *const c_char,
    handle: *const c_void,
    in_args: OfxPropertySetHandle,
    out_args: OfxPropertySetHandle,
) -> OfxStatus {
    if action.is_null() {
        return kOfxStatErrBadHandle;
    }
    let act = CStr::from_ptr(action);
    // Every action runs under catch_unwind (release profile panic = "unwind"):
    // a bug fails this action (kOfxStatFailed + a stderr line) instead of
    // taking the host down. Panics in the solver's scoped worker threads
    // re-raise at the scope join and land here too.
    guarded(&act.to_string_lossy(), || dispatch(act, handle, in_args, out_args))
}

/// Run `f`, converting a panic into kOfxStatFailed (and a stderr line).
pub(crate) fn guarded<F: FnOnce() -> OfxStatus>(what: &str, f: F) -> OfxStatus {
    match std::panic::catch_unwind(std::panic::AssertUnwindSafe(f)) {
        Ok(s) => s,
        Err(e) => {
            let msg = e
                .downcast_ref::<&str>()
                .map(|s| s.to_string())
                .or_else(|| e.downcast_ref::<String>().cloned())
                .unwrap_or_else(|| "unknown panic".into());
            eprintln!("cwFill: PANIC in {what}: {msg} (action failed, host kept running)");
            kOfxStatFailed
        }
    }
}

unsafe fn dispatch(
    act: &CStr,
    handle: *const c_void,
    in_args: OfxPropertySetHandle,
    out_args: OfxPropertySetHandle,
) -> OfxStatus {
    let cmp = |k: &[u8]| act.to_bytes_with_nul() == k;

    if cmp(kOfxActionLoad) {
        actions::action_load()
    } else if cmp(kOfxActionUnload) {
        cache::clear();
        kOfxStatOK
    } else if cmp(kOfxActionDescribe) {
        actions::action_describe(handle as OfxImageEffectHandle)
    } else if cmp(kOfxImageEffectActionDescribeInContext) {
        actions::action_describe_in_context(handle as OfxImageEffectHandle, in_args)
    } else if cmp(kOfxActionCreateInstance) {
        kOfxStatOK
    } else if cmp(kOfxActionDestroyInstance) {
        // Drop this instance's cached F/B solves (cache.rs).
        cache::purge_instance(handle as usize);
        kOfxStatOK
    } else if cmp(kOfxActionInstanceChanged) {
        actions::action_instance_changed(handle as OfxImageEffectHandle, in_args)
    } else if cmp(kOfxImageEffectActionGetClipPreferences) {
        actions::action_get_clip_preferences(handle as OfxImageEffectHandle, out_args)
    } else if cmp(kOfxImageEffectActionRender) {
        actions::action_render(handle as OfxImageEffectHandle, in_args)
    } else {
        // Unhandled actions return "default" so the host applies its own.
        kOfxStatReplyDefault
    }
}

// ---------------------------------------------------------------------------
// Small property-suite helpers (fixed-signature, safe to call directly).
// ---------------------------------------------------------------------------
pub(crate) fn prop_set_string(h: OfxPropertySetHandle, key: &[u8], index: c_int, val: &[u8]) {
    unsafe {
        (prop().propSetString)(h, key.as_ptr() as *const c_char, index, val.as_ptr() as *const c_char);
    }
}
pub(crate) fn prop_set_int(h: OfxPropertySetHandle, key: &[u8], index: c_int, val: c_int) {
    unsafe {
        (prop().propSetInt)(h, key.as_ptr() as *const c_char, index, val);
    }
}
pub(crate) fn prop_set_double(h: OfxPropertySetHandle, key: &[u8], index: c_int, val: f64) {
    unsafe {
        (prop().propSetDouble)(h, key.as_ptr() as *const c_char, index, val);
    }
}
pub(crate) fn prop_get_int(h: OfxPropertySetHandle, key: &[u8], index: c_int) -> c_int {
    let mut out: c_int = 0;
    unsafe {
        (prop().propGetInt)(h, key.as_ptr() as *const c_char, index, &mut out);
    }
    out
}
pub(crate) fn prop_get_pointer(h: OfxPropertySetHandle, key: &[u8], index: c_int) -> *mut c_void {
    let mut out: *mut c_void = std::ptr::null_mut();
    unsafe {
        (prop().propGetPointer)(h, key.as_ptr() as *const c_char, index, &mut out);
    }
    out
}
pub(crate) fn prop_get_rect_i(h: OfxPropertySetHandle, key: &[u8]) -> OfxRectI {
    OfxRectI {
        x1: prop_get_int(h, key, 0),
        y1: prop_get_int(h, key, 1),
        x2: prop_get_int(h, key, 2),
        y2: prop_get_int(h, key, 3),
    }
}
pub(crate) fn prop_get_double(h: OfxPropertySetHandle, key: &[u8], index: c_int) -> f64 {
    let mut out: f64 = 0.0;
    unsafe {
        (prop().propGetDouble)(h, key.as_ptr() as *const c_char, index, &mut out);
    }
    out
}

#[cfg(test)]
mod guard_tests {
    #[test]
    fn panic_becomes_failed() {
        let prev = std::panic::take_hook();
        std::panic::set_hook(Box::new(|_| {}));
        let s = super::guarded("test", || panic!("boom"));
        // a panic inside a scoped worker thread re-raises at the join
        let t = super::guarded("scoped", || {
            std::thread::scope(|sc| {
                sc.spawn(|| panic!("worker"));
            });
            0
        });
        std::panic::set_hook(prev);
        assert_eq!(s, super::kOfxStatFailed);
        assert_eq!(t, super::kOfxStatFailed);
        assert_eq!(super::guarded("ok", || 0), 0);
    }
}
