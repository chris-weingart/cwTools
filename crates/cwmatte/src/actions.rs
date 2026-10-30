//! OFX actions: describe, describeInContext (clips + params), clip
//! preferences, instance changed (Release GPU Memory).

use crate::params;
use cwtools_core::image::{clip_pref_components_key, clip_pref_depth_key};
use cwtools_core::ofx_api::*;
use cwtools_core::props::{prop_get_string, prop_set_int, prop_set_string};
use cwtools_core::{effect, set_node_message};
use std::os::raw::c_char;

pub fn action_describe(effect_handle: OfxImageEffectHandle) -> OfxStatus {
    let mut props: OfxPropertySetHandle = std::ptr::null_mut();
    unsafe {
        if (effect().getPropertySet)(effect_handle, &mut props) != kOfxStatOK {
            return kOfxStatErrBadHandle;
        }
    }
    prop_set_string(props, kOfxPropLabel, 0, b"cwMatte\0");
    prop_set_string(props, kOfxPropShortLabel, 0, b"cwMatte\0");
    prop_set_string(props, kOfxPropLongLabel, 0, b"cwMatte\0");
    prop_set_string(props, kOfxImageEffectPluginPropGrouping, 0, b"cwTools\0");
    prop_set_string(props, kOfxPropIcon, 0, b"cwMatte.png\0");
    prop_set_string(props, kOfxPropIcon, 1, b"cwMatte.png\0");
    cwtools_core::version::describe_props(
        props,
        "Automatic matte with BiRefNet (no prompts): plate -> matte in alpha. Optional ROI pre-crop and garbage mask. Local ONNX via Flame's onnxruntime.",
    );
    prop_set_string(props, kOfxImageEffectPropSupportedContexts, 0, kOfxImageEffectContextFilter);
    for (i, d) in cwtools_core::image::SUPPORTED_DEPTHS.iter().enumerate() {
        prop_set_string(props, kOfxImageEffectPropSupportedPixelDepths, i as i32, d);
    }
    prop_set_int(props, kOfxImageEffectPropSupportsTiles, 0, 0);
    prop_set_int(props, kOfxImageEffectPropSupportsMultipleClipDepths, 0, 0);
    prop_set_int(props, kOfxImageEffectPropTemporalClipAccess, 0, 0);
    // Instance-safe: one render at a time per node; nodes may render in
    // parallel (the shared session + caches are Mutex-guarded).
    prop_set_string(props, kOfxImageEffectPluginRenderThreadSafety, 0, kOfxImageEffectRenderInstanceSafe);
    prop_set_int(props, kOfxImageEffectPluginPropHostFrameThreading, 0, 0);
    kOfxStatOK
}

fn define_clip(effect_handle: OfxImageEffectHandle, name: &[u8], comps: &[u8]) {
    let mut p: OfxPropertySetHandle = std::ptr::null_mut();
    unsafe {
        (effect().clipDefine)(effect_handle, name.as_ptr() as *const c_char, &mut p);
    }
    // SupportedComponents (the accepted set) — NOT PropComponents, or Flame
    // shows no socket (flame_ofx_host_facts).
    prop_set_string(p, kOfxImageEffectPropSupportedComponents, 0, comps);
}

pub fn action_describe_in_context(effect_handle: OfxImageEffectHandle) -> OfxStatus {
    // Source RGBA: Flame splits it into RGB (plate) + Alpha (optional
    // garbage mask / ROI) sockets.
    define_clip(effect_handle, kOfxImageEffectSimpleSourceClipName, kOfxImageComponentRGBA);
    define_clip(effect_handle, kOfxImageEffectOutputClipName, kOfxImageComponentRGBA);
    let mut param_set: OfxParamSetHandle = std::ptr::null_mut();
    unsafe {
        if (effect().getParamSet)(effect_handle, &mut param_set) != kOfxStatOK {
            return kOfxStatErrBadHandle;
        }
    }
    cwtools_core::params::define_all_with(param_set, params::PARAMS, &params::shown_default);
    kOfxStatOK
}

pub fn action_get_clip_preferences(effect_handle: OfxImageEffectHandle, out_args: OfxPropertySetHandle) -> OfxStatus {
    // one depth for all clips: the Source's (16-bit in = 16-bit out)
    let depth = cwtools_core::image::pref_depth(effect_handle);
    for name in ["Source", "Output"] {
        prop_set_string(out_args, &clip_pref_components_key(name), 0, kOfxImageComponentRGBA);
        prop_set_string(out_args, &clip_pref_depth_key(name), 0, depth);
    }
    kOfxStatOK
}

pub fn action_instance_changed(effect_handle: OfxImageEffectHandle, in_args: OfxPropertySetHandle) -> OfxStatus {
    if prop_get_string(in_args, kOfxPropType, 0) != "OfxTypeParameter" {
        return kOfxStatReplyDefault;
    }
    let name = prop_get_string(in_args, kOfxPropName, 0);
    let time = cwtools_core::props::prop_try_double(in_args, kOfxPropTime, 0).unwrap_or(0.0);
    if name == params::P_RELEASE {
        let v = params::read_values(cwtools_core::suites::effect_param_set(effect_handle), time);
        let n = cwtools_core::ort_rt::release_all();
        crate::local::clear_results();
        // Remote: POST /release is deliberately NOT called - it would unload the
        // models for every artist using the server, which already shrinks its
        // arena after each frame and unloads idle models after 10 minutes.
        let msg = if v.remote {
            format!(
                "cwMatte: Device = Remote - nothing local to release ({n} local session(s) from earlier Local renders freed). \
                 The server frees its own VRAM (after every frame; models unload after 10 min idle)."
            )
        } else {
            format!("cwMatte: released {n} model session(s) - VRAM freed, next render reloads")
        };
        set_node_message(effect_handle, Some(&msg));
        return kOfxStatOK;
    }
    if name == params::P_TEST_SERVER {
        let v = params::read_values(cwtools_core::suites::effect_param_set(effect_handle), time);
        let (_, model, sha) = crate::remote::identity(&v);
        let msg = match cwtools_remote::plugin::test_server(&v.server, &model, sha) {
            Ok(m) => format!("cwMatte Test Server {}: {m}", v.server),
            Err(m) => format!("cwMatte Test Server {} FAILED: {m}", v.server),
        };
        set_node_message(effect_handle, Some(&msg));
        return kOfxStatOK;
    }
    kOfxStatReplyDefault
}
