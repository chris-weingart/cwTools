//! Overlay interact: draws the prompt points/box in the viewer and turns pen
//! clicks into point/box params. Whether Flame calls OFX overlay interacts at
//! all is UNKNOWN — everything here is a harmless no-op if it doesn't. The
//! first Draw / PenDown received is logged so a test can tell (see README).
//!
//! OpenGL is resolved at runtime with dlsym(RTLD_DEFAULT) — no link-time GL
//! dependency; if the symbols are missing, drawing is skipped.

use crate::actions::effect_param_set;
use crate::ofx_api::*;
use crate::params;
use crate::{guarded, prop_get_pointer, prop_try_double, suites};
use std::collections::HashMap;
use std::ffi::CStr;
use std::os::raw::{c_char, c_int, c_void};
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
use std::sync::{Mutex, OnceLock};

extern "C" {
    fn dlsym(handle: *mut c_void, symbol: *const c_char) -> *mut c_void;
}
#[cfg(target_os = "macos")]
const RTLD_DEFAULT: *mut c_void = -2isize as *mut c_void;
#[cfg(not(target_os = "macos"))]
const RTLD_DEFAULT: *mut c_void = std::ptr::null_mut();

const GL_LINES: u32 = 0x0001;
const GL_LINE_LOOP: u32 = 0x0002;

struct Gl {
    begin: unsafe extern "C" fn(u32),
    end: unsafe extern "C" fn(),
    vertex2d: unsafe extern "C" fn(f64, f64),
    color3f: unsafe extern "C" fn(f32, f32, f32),
    line_width: unsafe extern "C" fn(f32),
}
unsafe impl Send for Gl {}
unsafe impl Sync for Gl {}

fn gl() -> Option<&'static Gl> {
    static GL: OnceLock<Option<Gl>> = OnceLock::new();
    GL.get_or_init(|| unsafe {
        let s = |n: &[u8]| dlsym(RTLD_DEFAULT, n.as_ptr() as *const c_char);
        let (b, e, v, c, l) = (s(b"glBegin\0"), s(b"glEnd\0"), s(b"glVertex2d\0"), s(b"glColor3f\0"), s(b"glLineWidth\0"));
        if b.is_null() || e.is_null() || v.is_null() || c.is_null() || l.is_null() {
            crate::log::log("interact: OpenGL symbols not found, overlay drawing disabled");
            return None;
        }
        Some(Gl {
            begin: std::mem::transmute::<*mut c_void, unsafe extern "C" fn(u32)>(b),
            end: std::mem::transmute::<*mut c_void, unsafe extern "C" fn()>(e),
            vertex2d: std::mem::transmute::<*mut c_void, unsafe extern "C" fn(f64, f64)>(v),
            color3f: std::mem::transmute::<*mut c_void, unsafe extern "C" fn(f32, f32, f32)>(c),
            line_width: std::mem::transmute::<*mut c_void, unsafe extern "C" fn(f32)>(l),
        })
    })
    .as_ref()
}

static SEEN_DRAW: AtomicBool = AtomicBool::new(false);
static SEEN_PEN: AtomicBool = AtomicBool::new(false);
static NEXT_SLOT: AtomicUsize = AtomicUsize::new(0);
/// Interacts currently dragging a box (keyed by interact handle address).
static DRAGGING: Mutex<Option<HashMap<usize, bool>>> = Mutex::new(None);

fn set_dragging(h: usize, on: bool) {
    if let Ok(mut g) = DRAGGING.lock() {
        g.get_or_insert_with(HashMap::new).insert(h, on);
    }
}
fn is_dragging(h: usize) -> bool {
    DRAGGING.lock().ok().and_then(|g| g.as_ref().and_then(|m| m.get(&h).copied())).unwrap_or(false)
}

fn effect_of(interact: OfxInteractHandle) -> Option<OfxImageEffectHandle> {
    let s = suites();
    if s.interact.is_null() {
        return None;
    }
    let mut props: OfxPropertySetHandle = std::ptr::null_mut();
    unsafe { ((*s.interact).interactGetPropertySet)(interact, &mut props) };
    if props.is_null() {
        return None;
    }
    let e = prop_get_pointer(props, kOfxPropEffectInstance, 0);
    if e.is_null() { None } else { Some(e) }
}

fn redraw(interact: OfxInteractHandle) {
    let s = suites();
    if !s.interact.is_null() {
        unsafe { ((*s.interact).interactRedraw)(interact) };
    }
}

pub unsafe extern "C" fn overlay_main(
    action: *const c_char,
    handle: *const c_void,
    in_args: OfxPropertySetHandle,
    _out_args: OfxPropertySetHandle,
) -> OfxStatus {
    if action.is_null() {
        return kOfxStatReplyDefault;
    }
    let act = CStr::from_ptr(action);
    let name = act.to_string_lossy().into_owned();
    guarded(&format!("interact {name}"), || {
        let a = act.to_bytes_with_nul();
        let ih = handle as OfxInteractHandle;
        if a == kOfxActionDescribe || a == kOfxActionCreateInstance || a == kOfxActionDestroyInstance {
            return kOfxStatOK;
        }
        if a == kOfxInteractActionDraw {
            if !SEEN_DRAW.swap(true, Ordering::Relaxed) {
                crate::log::log("interact: host called overlay Draw (overlays ARE supported)");
            }
            return draw(ih, in_args);
        }
        if a == kOfxInteractActionPenDown {
            return pen(ih, in_args, 0);
        }
        if a == kOfxInteractActionPenMotion {
            return pen(ih, in_args, 1);
        }
        if a == kOfxInteractActionPenUp {
            return pen(ih, in_args, 2);
        }
        kOfxStatReplyDefault
    })
}

fn draw(ih: OfxInteractHandle, in_args: OfxPropertySetHandle) -> OfxStatus {
    let Some(gl) = gl() else { return kOfxStatReplyDefault };
    let Some(eh) = effect_of(ih) else { return kOfxStatReplyDefault };
    let t = prop_try_double(in_args, kOfxPropTime, 0).unwrap_or(0.0);
    let ps = effect_param_set(eh);
    let v = params::read_values(ps, t);
    let sx = prop_try_double(in_args, kOfxInteractPropPixelScale, 0).unwrap_or(1.0);
    let sy = prop_try_double(in_args, kOfxInteractPropPixelScale, 1).unwrap_or(1.0);
    let (rx, ry) = (8.0 * sx, 8.0 * sy);
    // Normalised (<=1) values → canonical via the project size (see render.rs).
    let mut ep: OfxPropertySetHandle = std::ptr::null_mut();
    unsafe { (crate::effect().getPropertySet)(eh, &mut ep) };
    let pw = prop_try_double(ep, kOfxImageEffectPropProjectSize, 0).unwrap_or(1920.0);
    let ph = prop_try_double(ep, kOfxImageEffectPropProjectSize, 1).unwrap_or(1080.0);
    let n = |x: f64, y: f64| if x.abs() <= 1.0 && y.abs() <= 1.0 { (x * pw, y * ph) } else { (x, y) };
    unsafe {
        (gl.line_width)(2.0);
        for p in &v.points {
            let (px, py) = n(p.x, p.y);
            let p = crate::params::UserPoint { x: px, y: py, fg: p.fg };
            if p.fg {
                (gl.color3f)(0.1, 1.0, 0.1);
            } else {
                (gl.color3f)(0.2, 0.4, 1.0);
            }
            (gl.begin)(GL_LINES);
            (gl.vertex2d)(p.x - rx, p.y);
            (gl.vertex2d)(p.x + rx, p.y);
            (gl.vertex2d)(p.x, p.y - ry);
            (gl.vertex2d)(p.x, p.y + ry);
            (gl.end)();
            (gl.begin)(GL_LINE_LOOP);
            (gl.vertex2d)(p.x - rx * 0.6, p.y - ry * 0.6);
            (gl.vertex2d)(p.x + rx * 0.6, p.y - ry * 0.6);
            (gl.vertex2d)(p.x + rx * 0.6, p.y + ry * 0.6);
            (gl.vertex2d)(p.x - rx * 0.6, p.y + ry * 0.6);
            (gl.end)();
        }
        if let Some((x0, y0, x1, y1)) = v.box_canon {
            let (x0, y0) = n(x0, y0);
            let (x1, y1) = n(x1, y1);
            (gl.color3f)(1.0, 1.0, 0.0);
            (gl.begin)(GL_LINE_LOOP);
            (gl.vertex2d)(x0, y0);
            (gl.vertex2d)(x1, y0);
            (gl.vertex2d)(x1, y1);
            (gl.vertex2d)(x0, y1);
            (gl.end)();
        }
        // Picked slots: a ring in the slot colour (Matte = white) at the pick point.
        const SLOT_RGB: [(f32, f32, f32); 4] = [(1.0, 0.2, 0.2), (0.2, 1.0, 0.2), (0.3, 0.5, 1.0), (1.0, 1.0, 1.0)];
        // Pick armed: a frame in the slot colour around the image (instead of a popup).
        if let Some(k) = params::armed_pick(ps, t) {
            let (r, g, b) = SLOT_RGB[k];
            (gl.color3f)(r, g, b);
            (gl.line_width)(4.0);
            for inset in [0.0, 6.0] {
                let (ix, iy) = (inset * sx, inset * sy);
                (gl.begin)(GL_LINE_LOOP);
                (gl.vertex2d)(ix, iy);
                (gl.vertex2d)(pw - ix, iy);
                (gl.vertex2d)(pw - ix, ph - iy);
                (gl.vertex2d)(ix, ph - iy);
                (gl.end)();
            }
            (gl.line_width)(2.0);
        }
        // Picking objects to track: an orange frame until Pick Objects is pressed again.
        if params::armed_track(ps, t) {
            (gl.color3f)(1.0, 0.55, 0.0);
            (gl.line_width)(4.0);
            for inset in [0.0, 6.0, 12.0] {
                let (ix, iy) = (inset * sx, inset * sy);
                (gl.begin)(GL_LINE_LOOP);
                (gl.vertex2d)(ix, iy);
                (gl.vertex2d)(pw - ix, iy);
                (gl.vertex2d)(pw - ix, ph - iy);
                (gl.vertex2d)(ix, ph - iy);
                (gl.end)();
            }
            (gl.line_width)(2.0);
        }
        for (k, p) in v.picks.iter().enumerate().take(4) {
            let Some((px, py)) = *p else { continue };
            let (px, py) = n(px, py);
            let (r, g, b) = SLOT_RGB[k];
            (gl.color3f)(r, g, b);
            (gl.begin)(GL_LINE_LOOP);
            for i in 0..24 {
                let a = i as f64 * std::f64::consts::TAU / 24.0;
                (gl.vertex2d)(px + rx * 1.4 * a.cos(), py + ry * 1.4 * a.sin());
            }
            (gl.end)();
            (gl.begin)(GL_LINES);
            (gl.vertex2d)(px - rx * 0.5, py);
            (gl.vertex2d)(px + rx * 0.5, py);
            (gl.vertex2d)(px, py - ry * 0.5);
            (gl.vertex2d)(px, py + ry * 0.5);
            (gl.end)();
        }
        (gl.line_width)(1.0);
    }
    kOfxStatOK
}

/// phase 0 = down, 1 = motion, 2 = up
fn pen(ih: OfxInteractHandle, in_args: OfxPropertySetHandle, phase: c_int) -> OfxStatus {
    let (Some(x), Some(y)) = (
        prop_try_double(in_args, kOfxInteractPropPenPosition, 0),
        prop_try_double(in_args, kOfxInteractPropPenPosition, 1),
    ) else {
        return kOfxStatReplyDefault;
    };
    if phase == 0 && !SEEN_PEN.swap(true, Ordering::Relaxed) {
        crate::log::log(&format!("interact: host delivered PenDown at canonical ({x:.1},{y:.1}) (clicks ARE supported)"));
    }
    let Some(eh) = effect_of(ih) else { return kOfxStatReplyDefault };
    let t = prop_try_double(in_args, kOfxPropTime, 0).unwrap_or(0.0);
    let ps = effect_param_set(eh);
    let mode = params::get_int(ps, params::P_CLICK_MODE, t);
    let key = ih as usize;
    match (mode, phase) {
        (1 | 2, 0) => {
            // Next free slot, else round-robin.
            let mut slot = None;
            for i in 0..params::MAX_POINTS {
                if !params::get_bool(ps, &params::point_on_name(i), t) {
                    slot = Some(i);
                    break;
                }
            }
            let i = slot.unwrap_or_else(|| NEXT_SLOT.fetch_add(1, Ordering::Relaxed) % params::MAX_POINTS);
            unsafe { (crate::param().paramEditBegin)(ps, b"cwSegment add point\0".as_ptr() as *const c_char) };
            params::set_xy(ps, &params::point_name(i), x, y);
            params::set_bool(ps, &params::point_bg_name(i), mode == 2);
            params::set_bool(ps, &params::point_on_name(i), true);
            unsafe { (crate::param().paramEditEnd)(ps) };
            redraw(ih);
            kOfxStatOK
        }
        (3, 0) => {
            unsafe { (crate::param().paramEditBegin)(ps, b"cwSegment box\0".as_ptr() as *const c_char) };
            params::set_xy(ps, params::P_BOX_A, x, y);
            params::set_xy(ps, params::P_BOX_B, x, y);
            params::set_bool(ps, params::P_USE_BOX, true);
            unsafe { (crate::param().paramEditEnd)(ps) };
            set_dragging(key, true);
            redraw(ih);
            kOfxStatOK
        }
        (3, 1) if is_dragging(key) => {
            params::set_xy(ps, params::P_BOX_B, x, y);
            redraw(ih);
            kOfxStatOK
        }
        (3, 2) if is_dragging(key) => {
            params::set_xy(ps, params::P_BOX_B, x, y);
            set_dragging(key, false);
            redraw(ih);
            kOfxStatOK
        }
        (m, 0) if (params::CLICK_PICK0..params::CLICK_PICK0 + 4).contains(&m) => {
            // Pick for slot R/G/B/A: store the point, slot -> Picked, Output -> Object
            // Channels (the picks only show there), clicks back to Off.
            let k = (m - params::CLICK_PICK0) as usize;
            unsafe { (crate::param().paramEditBegin)(ps, b"cwSegment pick object\0".as_ptr() as *const c_char) };
            params::set_xy(ps, &params::slot_pick_name(k), x, y);
            // Off then Picked: a slot that was already Picked must still change a
            // visible param, or Flame skips the re-render (pick point and click mode
            // are hidden) and the armed ID Map stays up (seen 2026-10-22, slot B).
            params::set_int(ps, &params::slot_sel_name(k), crate::objects::SEL_OFF);
            params::set_int(ps, &params::slot_sel_name(k), crate::objects::SEL_PICK);
            if params::get_int(ps, params::P_OUTPUT, t) < params::OUT_CHANNELS {
                params::set_int(ps, params::P_OUTPUT, params::OUT_CHANNELS);
            }
            params::set_int(ps, params::P_CLICK_MODE, 0);
            unsafe { (crate::param().paramEditEnd)(ps) };
            crate::log::log(&format!("pick: slot {} at canonical ({x:.1},{y:.1})", params::SLOTS[k]));
            redraw(ih);
            kOfxStatOK
        }
        (params::CLICK_TRACK, 0) => {
            // Add the object under the click (numbered as in the ID Map shown while
            // picking) to Tracked Objects; a pick on another frame starts a new list there.
            match crate::multi::object_at(eh as usize, x, y) {
                Some((fr, Some(n))) => {
                    let mut list = crate::track::parse_list(&params::get_string(ps, params::P_TRACK_LIST));
                    let anchor = params::get_int(ps, params::P_TRACK_ANCHOR, t) as i64;
                    unsafe { (crate::param().paramEditBegin)(ps, b"cwSegment pick to track\0".as_ptr() as *const c_char) };
                    if anchor != fr {
                        list.clear();
                        params::set_int(ps, params::P_TRACK_ANCHOR, fr as i32);
                    }
                    if !list.contains(&n) {
                        list.push(n);
                    }
                    params::set_string(ps, params::P_TRACK_LIST, &crate::track::format_list(&list));
                    unsafe { (crate::param().paramEditEnd)(ps) };
                    crate::log::log(&format!("track pick: object {n} on frame {fr}, list {}", crate::track::format_list(&list)));
                }
                Some((_, None)) => {
                    crate::set_node_message(eh, Some("cwSegment: no object under the click"));
                }
                None => {
                    crate::set_node_message(eh, Some("cwSegment: render a frame first (the click picks from what the viewer shows)"));
                }
            }
            redraw(ih);
            kOfxStatOK
        }
        _ => kOfxStatReplyDefault,
    }
}
