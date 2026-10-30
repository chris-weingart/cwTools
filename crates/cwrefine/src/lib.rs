//! cwRefine — OpenFX plugin that refines a seed matte with ViTMatte, running
//! the production cropped recipe (the reference ViTMatte server,
//! `_vitmatte_one_region`, hair recipe) with every knob exposed as a param.
//!
//! Pre/post processing (bbox, crop, PIL-exact Lanczos, /32 pad, OpenCV-exact
//! ellipse trimap, HF normalisation, paste) runs on the CPU in Rust
//! ([`recipe`], [`engine`]); only the ViTMatte core runs in onnxruntime
//! (Flame's bundled 1.22, CUDA EP), through a process-wide session shared by
//! all cwRefine nodes ([`local`]).
//!
//! Clips: Source (RGBA → Flame shows RGB + Alpha sockets; Alpha = the input
//! mask), optional Trimap (Alpha / grey), Output (Flame: Result RGB + OutMatte
//! alpha socket).
//! Every entry point is wrapped in catch_unwind: a panic must never reach Flame.

#![allow(non_snake_case)]

mod actions;
mod interact;
pub mod spline;
pub mod engine;
pub mod local;
pub mod params;
pub mod recipe;
pub mod remote;
mod render;

use cwtools_core::guarded;
use cwtools_core::ofx_api::*;
use cwtools_core::suites::set_host;
use std::ffi::CStr;
use std::os::raw::{c_char, c_int, c_void};

/// Spline trimap UI switch (cargo feature `spline`, off by default). Off: the
/// spline params and the overlay interact are not described / registered, the
/// render reads no spline param and uses the identity spline (production trimap).
/// The spline code stays compiled and tested.
pub const SPLINE_UI: bool = cfg!(feature = "spline");

pub const TRIMAP_CLIP_NAME: &[u8] = b"Trimap\0";

static PLUGIN_IDENTIFIER: &[u8] = b"dev.weingart.cw3Refine\0";

static mut PLUGIN: OfxPlugin = OfxPlugin {
    pluginApi: std::ptr::null(),
    apiVersion: kOfxImageEffectPluginApiVersion,
    pluginIdentifier: std::ptr::null(),
    pluginVersionMajor: cwtools_core::version::MAJOR,
    pluginVersionMinor: cwtools_core::version::MINOR,
    setHost: Some(set_host),
    mainEntry: Some(main_entry),
};

#[no_mangle]
pub extern "C" fn OfxGetNumberOfPlugins() -> c_int {
    1
}

#[no_mangle]
pub extern "C" fn OfxGetPlugin(nth: c_int) -> *const OfxPlugin {
    cwtools_core::log::set_name("cwRefine");
    if nth != 0 {
        return std::ptr::null();
    }
    unsafe {
        PLUGIN.pluginApi = kOfxImageEffectPluginApi.as_ptr() as *const c_char;
        PLUGIN.pluginIdentifier = PLUGIN_IDENTIFIER.as_ptr() as *const c_char;
        std::ptr::addr_of!(PLUGIN)
    }
}

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
    let name = act.to_string_lossy().into_owned();
    guarded(&name, || {
        let cmp = |k: &[u8]| act.to_bytes_with_nul() == k;
        let h = handle as OfxImageEffectHandle;
        if cmp(kOfxActionLoad) {
            cwtools_core::suites::load_suites()
        } else if cmp(kOfxActionUnload) {
            kOfxStatOK
        } else if cmp(kOfxActionDescribe) {
            actions::action_describe(h)
        } else if cmp(kOfxImageEffectActionDescribeInContext) {
            actions::action_describe_in_context(h)
        } else if cmp(kOfxActionCreateInstance) {
            params::fill_setup_fields(cwtools_core::suites::effect_param_set(h));
            kOfxStatOK
        } else if cmp(kOfxActionDestroyInstance) {
            spline::clear_geometry(h as usize);
            kOfxStatOK
        } else if cmp(kOfxImageEffectActionGetClipPreferences) {
            actions::action_get_clip_preferences(h, out_args)
        } else if cmp(kOfxActionInstanceChanged) {
            actions::action_instance_changed(h, in_args)
        } else if cmp(kOfxImageEffectActionRender) {
            render::render(h, in_args)
        } else {
            kOfxStatReplyDefault
        }
    })
}
