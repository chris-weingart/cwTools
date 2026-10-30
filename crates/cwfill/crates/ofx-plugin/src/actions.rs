//! OFX action handlers: load, describe, describeInContext, clip preferences,
//! render.

use crate::ofx_api::*;
use crate::params;
use crate::render;
use crate::{effect, prop_set_int, prop_set_string, suites};
use std::os::raw::c_char;

pub fn action_load() -> OfxStatus {
    let host = suites().host;
    if host.is_null() {
        return kOfxStatErrBadHandle;
    }
    let host_ref = unsafe { &*host };
    let fetch = match host_ref.fetchSuite {
        Some(f) => f,
        None => return kOfxStatErrMissingHostFeature,
    };
    unsafe {
        let prop = fetch(host_ref.host, kOfxPropertySuite.as_ptr() as *const c_char, 1);
        let eff = fetch(host_ref.host, kOfxImageEffectSuite.as_ptr() as *const c_char, 1);
        let par = fetch(host_ref.host, kOfxParameterSuite.as_ptr() as *const c_char, 1);
        if prop.is_null() || eff.is_null() || par.is_null() {
            return kOfxStatErrMissingHostFeature;
        }
        crate::set_suites(
            prop as *const OfxPropertySuiteV1,
            eff as *const OfxImageEffectSuiteV1,
            par as *const OfxParameterSuiteV1,
        );
    }
    kOfxStatOK
}

pub fn action_describe(effect_handle: OfxImageEffectHandle) -> OfxStatus {
    let mut props: OfxPropertySetHandle = std::ptr::null_mut();
    unsafe {
        if (effect().getPropertySet)(effect_handle, &mut props) != kOfxStatOK {
            return kOfxStatErrBadHandle;
        }
    }
    // Set all label variants so the host (Flame) shows the full name on the node
    // tile regardless of which it picks; grouping is the menu folder.
    prop_set_string(props, kOfxPropLabel, 0, b"cwFill\0");
    prop_set_string(props, kOfxPropShortLabel, 0, b"cwFill\0");
    prop_set_string(props, kOfxPropLongLabel, 0, b"cwFill\0");
    prop_set_string(props, kOfxImageEffectPluginPropGrouping, 0, b"cwTools\0");
    // Node icon (Flame's only honored OFX resource = a PNG in Contents/Resources).
    // Dim 0 = normal, dim 1 = pressed; build.sh copies resources/cwFill.png.
    prop_set_string(props, kOfxPropIcon, 0, b"cwFill.png\0");
    prop_set_string(props, kOfxPropIcon, 1, b"cwFill.png\0");
    prop_set_string(
        props,
        kOfxPropPluginDescription,
        0,
        b"Estimate the true foreground colour for matte edges (pixel fill) from a plate + matte. Multilevel solver, faithful port of pymatting.\0",
    );

    // Filter context ONLY. Flame's confirmed-working context (General is
    // undocumented and was NOT wiring input sockets — node loaded with params
    // but no inputs). Filter auto-provides the Source input, and because Flame
    // splits an RGBA clip into RGB + Alpha sockets, one Source clip already
    // delivers plate + matte. (Optional garbage/holdout clip via General is a
    // future revisit once General-context inputs are proven in Flame.)
    prop_set_string(props, kOfxImageEffectPropSupportedContexts, 0, kOfxImageEffectContextFilter);

    // Float, half, 16-bit and 8-bit. Non-float buffers are converted to f32
    // for the solve and back on output (render.rs); never read as f32 in place
    // (the defocus lesson). Flame gives all clips one depth, and
    // getClipPreferences asks for the Source's, so the result has the plate's
    // bit depth (only declaring half was not enough: Flame then chose float).
    prop_set_string(props, kOfxImageEffectPropSupportedPixelDepths, 0, kOfxBitDepthFloat);
    prop_set_string(props, kOfxImageEffectPropSupportedPixelDepths, 1, kOfxBitDepthHalf);
    prop_set_string(props, kOfxImageEffectPropSupportedPixelDepths, 2, b"OfxBitDepthShort\0");
    prop_set_string(props, kOfxImageEffectPropSupportedPixelDepths, 3, b"OfxBitDepthByte\0");

    // Global multilevel solve — no tiling.
    prop_set_int(props, kOfxImageEffectPropSupportsTiles, 0, 0);

    kOfxStatOK
}

pub fn action_describe_in_context(
    effect_handle: OfxImageEffectHandle,
    _in_args: OfxPropertySetHandle,
) -> OfxStatus {
    // Source clip: RGBA so Flame splits it into RGB + Alpha sockets in Batch.
    let mut src_props: OfxPropertySetHandle = std::ptr::null_mut();
    let mut out_props: OfxPropertySetHandle = std::ptr::null_mut();
    unsafe {
        (effect().clipDefine)(effect_handle, kOfxImageEffectSimpleSourceClipName.as_ptr() as *const c_char, &mut src_props);
        (effect().clipDefine)(effect_handle, kOfxImageEffectOutputClipName.as_ptr() as *const c_char, &mut out_props);
    }
    // RGBA ONLY. The matte rides in Source alpha — if a host is allowed to
    // negotiate RGB it will pass no matte and the plugin degrades to a
    // passthrough (Resolve picked RGB when both were advertised; Flame
    // negotiates RGBA either way). Render keeps an RGB copy-through as a
    // safety net for non-conforming hosts.
    prop_set_string(src_props, kOfxImageEffectPropSupportedComponents, 0, kOfxImageComponentRGBA);
    prop_set_string(out_props, kOfxImageEffectPropSupportedComponents, 0, kOfxImageComponentRGBA);

    // Parameters
    let mut param_set: OfxParamSetHandle = std::ptr::null_mut();
    unsafe {
        if (effect().getParamSet)(effect_handle, &mut param_set) != kOfxStatOK {
            return kOfxStatErrBadHandle;
        }
    }
    params::define_params(param_set);
    kOfxStatOK
}

pub fn action_get_clip_preferences(
    effect_handle: OfxImageEffectHandle,
    out_args: OfxPropertySetHandle,
) -> OfxStatus {
    // Explicitly demand RGBA on both clips (spec: out-args keyed per clip
    // name), at the Source's depth (16-bit in = 16-bit out, 8-bit = 8-bit).
    prop_set_string(out_args, b"OfxImageClipPropComponents_Source\0", 0, kOfxImageComponentRGBA);
    prop_set_string(out_args, b"OfxImageClipPropComponents_Output\0", 0, kOfxImageComponentRGBA);
    let depth = crate::render::Depth::of_source(effect_handle).ofx();
    prop_set_string(out_args, b"OfxImageClipPropDepth_Source\0", 0, depth);
    prop_set_string(out_args, b"OfxImageClipPropDepth_Output\0", 0, depth);
    kOfxStatOK
}

pub fn action_render(effect_handle: OfxImageEffectHandle, in_args: OfxPropertySetHandle) -> OfxStatus {
    render::render(effect_handle, in_args)
}

fn prop_string(h: OfxPropertySetHandle, key: &[u8]) -> String {
    let mut out: *mut c_char = std::ptr::null_mut();
    let st = unsafe { (crate::prop().propGetString)(h, key.as_ptr() as *const c_char, 0, &mut out) };
    if st != kOfxStatOK || out.is_null() {
        return String::new();
    }
    unsafe { std::ffi::CStr::from_ptr(out) }.to_string_lossy().into_owned()
}

/// Picking a Screen Colour switches Despill Screen to Custom Colour (the presets
/// ignore the pick, so a pick alone looked like it did nothing).
pub fn action_instance_changed(effect_handle: OfxImageEffectHandle, in_args: OfxPropertySetHandle) -> OfxStatus {
    if prop_string(in_args, kOfxPropType) != "OfxTypeParameter"
        || prop_string(in_args, kOfxPropName).as_bytes() != &params::P_SCREEN_COLOUR[..params::P_SCREEN_COLOUR.len() - 1]
        || prop_string(in_args, kOfxPropChangeReason) != "OfxChangeUserEdited"
    {
        return kOfxStatReplyDefault;
    }
    let mut ps: OfxParamSetHandle = std::ptr::null_mut();
    if unsafe { (effect().getParamSet)(effect_handle, &mut ps) } != kOfxStatOK || ps.is_null() {
        return kOfxStatReplyDefault;
    }
    if params::get_int(ps, params::P_DESPILL_SCREEN) != params::DESPILL_CUSTOM {
        params::set_int(ps, params::P_DESPILL_SCREEN, params::DESPILL_CUSTOM);
    }
    kOfxStatOK
}
