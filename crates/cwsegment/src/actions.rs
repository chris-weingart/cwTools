//! OFX action handlers: load, describe, describeInContext, clip preferences,
//! instance changed.

use crate::ofx_api::*;
use crate::params;
use crate::{effect, prop_get_string, prop_set_int, prop_set_pointer, prop_set_string};
use std::os::raw::{c_char, c_void};

pub fn action_load() -> OfxStatus {
    cwtools_core::suites::load_suites()
}

fn env_off(name: &str) -> bool {
    std::env::var(name).map(|v| v != "0" && !v.is_empty()).unwrap_or(false)
}

pub fn action_describe(effect_handle: OfxImageEffectHandle) -> OfxStatus {
    let mut props: OfxPropertySetHandle = std::ptr::null_mut();
    unsafe {
        if (effect().getPropertySet)(effect_handle, &mut props) != kOfxStatOK {
            return kOfxStatErrBadHandle;
        }
    }
    prop_set_string(props, kOfxPropLabel, 0, b"cwSegment\0");
    prop_set_string(props, kOfxPropShortLabel, 0, b"cwSegment\0");
    prop_set_string(props, kOfxPropLongLabel, 0, b"cwSegment\0");
    prop_set_string(props, kOfxImageEffectPluginPropGrouping, 0, b"cwTools\0");
    prop_set_string(props, kOfxPropIcon, 0, b"cwSegment.png\0");
    prop_set_string(props, kOfxPropIcon, 1, b"cwSegment.png\0");
    cwtools_core::version::describe_props(
        props,
        "SAM object segmentation: text / point / box / mask prompts -> mask. Local (ONNX) or Remote (cwtools-infer server).",
    );
    prop_set_string(props, kOfxImageEffectPropSupportedContexts, 0, kOfxImageEffectContextFilter);
    for (i, d) in cwtools_core::image::SUPPORTED_DEPTHS.iter().enumerate() {
        prop_set_string(props, kOfxImageEffectPropSupportedPixelDepths, i as i32, d);
    }
    prop_set_int(props, kOfxImageEffectPropSupportsTiles, 0, 0);
    prop_set_int(props, kOfxImageEffectPropSupportsMultipleClipDepths, 0, 0);
    // Track Range fetches other frames of the Source during one render.
    prop_set_int(props, kOfxImageEffectPropTemporalClipAccess, 0, 1);
    // Instance-safe: one render at a time per node; several nodes may render
    // in parallel (shared caches are Mutex-guarded).
    prop_set_string(props, kOfxImageEffectPluginRenderThreadSafety, 0, kOfxImageEffectRenderInstanceSafe);
    prop_set_int(props, kOfxImageEffectPluginPropHostFrameThreading, 0, 0);

    // Overlay interact (prompt display + viewer clicks). Unknown whether Flame
    // calls it at all; harmless if not. CWSEGMENT_NO_OVERLAY=1 disables.
    if !env_off("CWSEGMENT_NO_OVERLAY") {
        prop_set_pointer(
            props,
            kOfxImageEffectPluginPropOverlayInteractV1,
            0,
            crate::interact::overlay_main as *mut c_void,
        );
    }
    kOfxStatOK
}

pub fn action_describe_in_context(effect_handle: OfxImageEffectHandle, _in_args: OfxPropertySetHandle) -> OfxStatus {
    let mut src_props: OfxPropertySetHandle = std::ptr::null_mut();
    let mut out_props: OfxPropertySetHandle = std::ptr::null_mut();
    unsafe {
        (effect().clipDefine)(effect_handle, kOfxImageEffectSimpleSourceClipName.as_ptr() as *const c_char, &mut src_props);
    }
    // Source RGBA: Flame splits it into RGB (plate) + Alpha sockets. The Alpha
    // socket is the prompt mask (Use Prompt Mask).
    prop_set_string(src_props, kOfxImageEffectPropSupportedComponents, 0, kOfxImageComponentRGBA);

    // Only Source + Output: in OFX every clip except "Output" is an
    // INPUT. The optional Prompt clip (Flame labelled it "Back") and the
    // "second output" clip (Flame showed it as a mandatory input) are gone.

    unsafe {
        (effect().clipDefine)(effect_handle, kOfxImageEffectOutputClipName.as_ptr() as *const c_char, &mut out_props);
    }
    prop_set_string(out_props, kOfxImageEffectPropSupportedComponents, 0, kOfxImageComponentRGBA);

    let mut param_set: OfxParamSetHandle = std::ptr::null_mut();
    unsafe {
        if (effect().getParamSet)(effect_handle, &mut param_set) != kOfxStatOK {
            return kOfxStatErrBadHandle;
        }
    }
    params::define_params(param_set);
    kOfxStatOK
}

pub fn action_get_clip_preferences(effect_handle: OfxImageEffectHandle, out_args: OfxPropertySetHandle) -> OfxStatus {
    prop_set_string(out_args, b"OfxImageClipPropComponents_Source\0", 0, kOfxImageComponentRGBA);
    prop_set_string(out_args, b"OfxImageClipPropComponents_Output\0", 0, kOfxImageComponentRGBA);
    // one depth for all clips: the Source's (16-bit in = 16-bit out)
    let depth = cwtools_core::image::pref_depth(effect_handle);
    prop_set_string(out_args, b"OfxImageClipPropDepth_Source\0", 0, depth);
    prop_set_string(out_args, b"OfxImageClipPropDepth_Output\0", 0, depth);
    kOfxStatOK
}

pub use cwtools_core::suites::effect_param_set;

fn prop_get_double_or(h: OfxPropertySetHandle, k: &[u8], d: f64) -> f64 {
    crate::prop_try_double(h, k, 0).unwrap_or(d)
}

pub fn clear_points(ps: OfxParamSetHandle) {
    unsafe {
        (crate::param().paramEditBegin)(ps, b"cwSegment clear points\0".as_ptr() as *const c_char);
    }
    for i in 0..params::MAX_POINTS {
        params::set_bool(ps, &params::point_on_name(i), false);
    }
    params::set_bool(ps, params::P_USE_BOX, false);
    unsafe {
        (crate::param().paramEditEnd)(ps);
    }
}

/// Flame's number of clip time `t` (the plugin sees clip times from the clip start).
fn flame_frame(effect_handle: OfxImageEffectHandle, ps: OfxParamSetHandle, t: f64) -> i64 {
    let c0 = cwtools_core::image::clip_frame_range(effect_handle, kOfxImageEffectSimpleSourceClipName).map_or(0.0, |r| r.0);
    (t - c0).round() as i64 + params::get_int(ps, params::P_TRACK_BASE, t) as i64
}

/// Fill Track From / To with the whole clip in Batch numbers (when unset, or `force`).
/// Call inside a param edit.
fn fill_range(effect_handle: OfxImageEffectHandle, ps: OfxParamSetHandle, t: f64, force: bool) {
    let set = params::get_int(ps, params::P_TRACK_FROM, t) != 0 || params::get_int(ps, params::P_TRACK_TO, t) != 0;
    if set && !force {
        return;
    }
    let Some((c0, c1)) = cwtools_core::image::clip_frame_range(effect_handle, kOfxImageEffectSimpleSourceClipName) else { return };
    params::set_int(ps, params::P_TRACK_FROM, flame_frame(effect_handle, ps, c0) as i32);
    params::set_int(ps, params::P_TRACK_TO, flame_frame(effect_handle, ps, c1) as i32);
}

/// How many frames the next Track Range reads (as multi::start_batch picks them).
fn frames_to_fetch(effect_handle: OfxImageEffectHandle, ps: OfxParamSetHandle, t: f64, node: usize) -> usize {
    use crate::track::{DIR_BACK, DIR_BOTH, DIR_FWD};
    let Some((c0, c1)) = cwtools_core::image::clip_frame_range(effect_handle, kOfxImageEffectSimpleSourceClipName) else { return 0 };
    let (c0, c1) = (c0.round() as i64, c1.round() as i64);
    let (from, to) = (params::get_int(ps, params::P_TRACK_FROM, t) as i64, params::get_int(ps, params::P_TRACK_TO, t) as i64);
    let base = params::get_int(ps, params::P_TRACK_BASE, t) as i64;
    let clip = |n: i64| (n - base + c0).clamp(c0, c1);
    let (from, to) = if from == 0 && to == 0 { (c0, c1) } else { (clip(from).min(clip(to)), clip(to)) };
    let anchor = params::get_int(ps, params::P_TRACK_ANCHOR, t) as i64;
    let dir = params::get_int(ps, params::P_TRACK, t);
    let dir = if dir == crate::track::DIR_OFF { DIR_BOTH } else { dir };
    let done = crate::track::status(node).map(|(_, a, b, _)| (a, b));
    let mut n = usize::from(done.is_none());
    if dir == DIR_FWD || dir == DIR_BOTH {
        n += (to - done.map_or(anchor, |d| d.1)).max(0) as usize;
    }
    if dir == DIR_BACK || dir == DIR_BOTH {
        n += (done.map_or(anchor, |d| d.0) - from).max(0) as usize;
    }
    n.min(crate::track::MAX_BATCH)
}

/// Write the Track status line (a visible change: Flame renders after it).
fn track_info(ps: OfxParamSetHandle, what: &str) {
    let now = std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).map(|d| d.as_secs()).unwrap_or(0);
    let (h, m, sec) = ((now / 3600) % 24, (now / 60) % 60, now % 60);
    params::set_string(ps, params::P_TRACK_INFO, &format!("{what} ({h:02}:{m:02}:{sec:02} UTC)"));
}

pub fn action_instance_changed(effect_handle: OfxImageEffectHandle, in_args: OfxPropertySetHandle) -> OfxStatus {
    let ty = prop_get_string(in_args, kOfxPropType, 0);
    if ty != "OfxTypeParameter" {
        return kOfxStatReplyDefault;
    }
    let name = prop_get_string(in_args, kOfxPropName, 0);
    let ps = effect_param_set(effect_handle);
    if let Some(ch) = name.strip_prefix("pickBtn") {
        // Arm a viewer pick for this slot; the next click in the viewer sets it.
        if let Some(k) = params::SLOTS.iter().position(|c| *c == ch) {
            let nm = if ch == "A" { "Matte" } else { ch };
            unsafe { (crate::param().paramEditBegin)(ps, b"cwSegment arm pick\0".as_ptr() as *const c_char) };
            params::set_int(ps, params::P_CLICK_MODE, params::CLICK_PICK0 + k as i32);
            unsafe { (crate::param().paramEditEnd)(ps) };
            // no popup: the viewer shows every object (ID Map) framed in the slot colour until the click
            crate::log::log(&format!("pick armed for {nm}"));
            return kOfxStatOK;
        }
    }
    let time = prop_get_double_or(in_args, kOfxPropTime, 0.0);
    let node = effect_handle as usize;
    // Picking ends by itself when the user changes anything else (it forces the ID Map,
    // so a forgotten second Pick Objects press left every output on the ID Map).
    let user = prop_get_string(in_args, kOfxPropChangeReason, 0) == "OfxChangeUserEdited";
    if user && params::armed_track(ps, time) && !matches!(name.as_str(), "trackPick" | "trackList" | "trackInfo" | "trackAnchor" | "clickMode") {
        let list = params::get_string(ps, params::P_TRACK_LIST);
        unsafe { (crate::param().paramEditBegin)(ps, b"cwSegment picking done\0".as_ptr() as *const c_char) };
        params::set_int(ps, params::P_CLICK_MODE, 0);
        let anchor = params::get_int(ps, params::P_TRACK_ANCHOR, time) as f64;
        fill_range(effect_handle, ps, time, false);
        track_info(ps, &format!("picked {} on frame {}", list.trim(), flame_frame(effect_handle, ps, anchor)));
        unsafe { (crate::param().paramEditEnd)(ps) };
        crate::log::log(&format!("track: picking done ({name} changed)"));
    }
    match name.as_str() {
        "trackPick" => {
            // Toggle: armed = every viewer click adds an object (the viewer shows the ID Map).
            let arm = !params::armed_track(ps, time);
            unsafe { (crate::param().paramEditBegin)(ps, b"cwSegment pick objects\0".as_ptr() as *const c_char) };
            params::set_int(ps, params::P_CLICK_MODE, if arm { params::CLICK_TRACK } else { 0 });
            let list = params::get_string(ps, params::P_TRACK_LIST);
            if !arm && params::get_int(ps, params::P_TRACK, time) == crate::track::DIR_OFF && !list.trim().is_empty() {
                params::set_int(ps, params::P_TRACK, crate::track::DIR_BOTH);
            }
            fill_range(effect_handle, ps, time, false);
            let anchor = params::get_int(ps, params::P_TRACK_ANCHOR, time) as f64;
            track_info(ps, if arm {
                "picking: click objects (Pick Objects again or any other change ends it)".to_string()
            } else {
                format!("picked {} on frame {}", list.trim(), flame_frame(effect_handle, ps, anchor))
            }.as_str());
            unsafe { (crate::param().paramEditEnd)(ps) };
            crate::log::log(if arm { "track: picking objects" } else { "track: picking done" });
            crate::log::log(&format!("track: time props{}",
                cwtools_core::image::time_probe(effect_handle, kOfxImageEffectSimpleSourceClipName, in_args)));
            kOfxStatOK
        }
        "trackClear" => {
            unsafe { (crate::param().paramEditBegin)(ps, b"cwSegment clear picks\0".as_ptr() as *const c_char) };
            params::set_string(ps, params::P_TRACK_LIST, "");
            track_info(ps, "picks cleared");
            unsafe { (crate::param().paramEditEnd)(ps) };
            crate::track::reset(node);
            crate::set_node_message(effect_handle, None);
            kOfxStatOK
        }
        "trackRun" => {
            // Track Range: the next render of this node tracks the whole range (OFX gives
            // images only inside a render); the status line change makes Flame render.
            // Flame pauses while the render reads the frames (no OFX progress bar in
            // Flame): say so first, unless this press stops a running job.
            if !crate::track::job(node).map_or(false, |j| j.running) {
                let n = frames_to_fetch(effect_handle, ps, time, node);
                if n >= 10 {
                    let ms = match crate::track::fetch_ms() { 0 => 250, m => m };
                    let secs = (n as u64 * ms).div_ceil(1000).max(1);
                    cwtools_core::message::dialog(effect_handle, &format!(
                        "cwSegment Track Range: Flame first reads {n} frames and pauses for about {secs} s \
                         (longer with heavy nodes upstream) - it is not frozen.\n\nThen the tracking runs in the \
                         background: scrub or step to see the tracked frames; the node message counts them."));
                }
            }
            crate::track::request_run(node);
            unsafe { (crate::param().paramEditBegin)(ps, b"cwSegment track range\0".as_ptr() as *const c_char) };
            if params::get_int(ps, params::P_TRACK, time) == crate::track::DIR_OFF {
                params::set_int(ps, params::P_TRACK, crate::track::DIR_BOTH);
            }
            fill_range(effect_handle, ps, time, false);
            track_info(ps, "Track Range pressed");
            unsafe { (crate::param().paramEditEnd)(ps) };
            crate::log::log("track: Track Range requested");
            kOfxStatOK
        }
        "trackFirstFrame" => {
            unsafe { (crate::param().paramEditBegin)(ps, b"cwSegment batch start\0".as_ptr() as *const c_char) };
            fill_range(effect_handle, ps, time, true);
            unsafe { (crate::param().paramEditEnd)(ps) };
            kOfxStatOK
        }
        "trackReset" => {
            crate::track::reset(node);
            unsafe { (crate::param().paramEditBegin)(ps, b"cwSegment reset track\0".as_ptr() as *const c_char) };
            track_info(ps, "reset: tracked frames forgotten");
            unsafe { (crate::param().paramEditEnd)(ps) };
            kOfxStatOK
        }
        "trackHere" => {
            unsafe { (crate::param().paramEditBegin)(ps, b"cwSegment anchor here\0".as_ptr() as *const c_char) };
            params::set_int(ps, params::P_TRACK_ANCHOR, time.round() as i32);
            fill_range(effect_handle, ps, time, false);
            track_info(ps, &format!("anchor = frame {}", flame_frame(effect_handle, ps, time.round())));
            unsafe { (crate::param().paramEditEnd)(ps) };
            kOfxStatOK
        }
        "clearPoints" => {
            clear_points(ps);
            kOfxStatOK
        }
        "clearCache" => {
            crate::render::clear_caches();
            crate::vlm::clear_cache();
            crate::track::reset_all();
            crate::set_node_message(effect_handle, None);
            crate::log::log("cache cleared");
            kOfxStatOK
        }
        "refreshList" => {
            let txt = crate::render::last_list(effect_handle as usize)
                .unwrap_or_else(|| "(render a frame with several concepts / Object Channels / ID Map first)".into());
            unsafe {
                (crate::param().paramEditBegin)(ps, b"cwSegment object list\0".as_ptr() as *const c_char);
            }
            params::set_string(ps, params::P_OBJ_LIST, &txt);
            unsafe {
                (crate::param().paramEditEnd)(ps);
            }
            kOfxStatOK
        }
        "vlmCopy" => {
            match crate::vlm::last(effect_handle as usize) {
                Some(c) if !c.is_empty() => {
                    let t = prop_get_double_or(in_args, kOfxPropTime, 0.0);
                    let cur = params::get_string(ps, params::P_TEXT);
                    let mode = params::get_int(ps, params::P_VLM_AUTO, t);
                    let list = if mode == 2 {
                        crate::objects::merge_concepts(&crate::objects::parse_concepts(&cur), &c)
                    } else {
                        c
                    };
                    unsafe {
                        (crate::param().paramEditBegin)(ps, b"cwSegment copy VLM concepts\0".as_ptr() as *const c_char);
                    }
                    params::set_string(ps, params::P_TEXT, &list.join(", "));
                    params::set_int(ps, params::P_VLM_AUTO, 0);
                    unsafe {
                        (crate::param().paramEditEnd)(ps);
                    }
                    crate::set_node_message(effect_handle, Some(&format!("Text Prompt = {}", list.join(", "))));
                }
                _ => crate::set_node_message(effect_handle, Some("no VLM result yet for this node - render a frame with Auto Concepts on")),
            }
            kOfxStatOK
        }
        "releaseGpu" => {
            // Same path as cwRefine / cwMatte: drop the shared sessions (the CUDA
            // keeper stays) + the result caches (masks, SAM embeddings).
            let n = crate::local::release_gpu();
            crate::render::clear_caches();
            let mode = params::get_int(ps, params::P_MODE, prop_get_double_or(in_args, kOfxPropTime, 0.0));
            let msg = if mode == 0 && n == 0 {
                "cwSegment: Mode = Remote - SAM3 runs on the server, no local model to release".to_string()
            } else if mode == 0 {
                format!("cwSegment: Mode = Remote (SAM3 on the server); released {n} local model session(s) from earlier Local renders")
            } else {
                format!("cwSegment: released {n} model session(s) - VRAM freed, next render reloads")
            };
            crate::log::log(&msg);
            crate::set_node_message(effect_handle, Some(&msg));
            kOfxStatOK
        }
        "testServer" => {
            let v = params::read_values(ps, prop_get_double_or(in_args, kOfxPropTime, 0.0));
            let url = v.server.clone();
            match crate::sam_remote::test_server(&url, &crate::local::model_folder(&v)) {
                Ok(info) => crate::set_node_message(effect_handle, Some(&format!("Server OK: {info}"))),
                Err(e) => crate::set_node_message(effect_handle, Some(&format!("Server FAILED ({url}): {e}"))),
            }
            kOfxStatOK
        }
        _ => kOfxStatReplyDefault,
    }
}
