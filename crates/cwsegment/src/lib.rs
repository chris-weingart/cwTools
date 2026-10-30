//! cwSegment — OpenFX plugin that segments objects with SAM.
//!
//! Two backends:
//!  - Local: runs the SAM3 ONNX graphs through onnxruntime (Flame's bundled
//!    libonnxruntime 1.22, CUDA EP when available).
//!  - Remote: the same graphs on a cwtools-infer server (crate::sam_remote);
//!    everything around them is shared with Local.
//!
//! Built on cwtools-core (hand-written OFX FFI, C shim for variadics, suites,
//! params, images, ORT, HTTP, logging, panic guards).
//! Every entry point is wrapped in catch_unwind: a panic must never reach Flame.

#![allow(non_snake_case)]

mod actions;
pub mod clip_tok;
pub mod imgutil;
mod interact;
pub mod local;
pub mod objects;
mod ofx_api;
mod params;
pub mod prompt;
pub(crate) mod sam_remote;
pub mod track;
mod multi;
mod render;
pub mod vlm;

use ofx_api::*;
use std::ffi::CStr;
use std::os::raw::{c_char, c_int, c_void};

// Shared OFX plumbing lives in cwtools-core; re-exported so the module paths
// used throughout this crate (crate::prop(), crate::log, ...) stay put.
pub use cwtools_core::log;
pub(crate) use cwtools_core::suites::set_host;
pub(crate) use cwtools_core::{effect, guarded, param, set_node_message, suites};
pub(crate) use cwtools_core::props::*;

static PLUGIN_IDENTIFIER: &[u8] = b"dev.weingart.cw1Segment\0";

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
    log::set_name("cwSegment");
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
            actions::action_load()
        } else if cmp(kOfxActionUnload) {
            track::reset_all(); // background tracking stops before the plugin goes away
            kOfxStatOK
        } else if cmp(kOfxActionDescribe) {
            actions::action_describe(h)
        } else if cmp(kOfxImageEffectActionDescribeInContext) {
            actions::action_describe_in_context(h, in_args)
        } else if cmp(kOfxActionCreateInstance) {
            params::fill_setup_fields(actions::effect_param_set(h));
            kOfxStatOK
        } else if cmp(kOfxActionDestroyInstance) {
            track::reset(h as usize);
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
