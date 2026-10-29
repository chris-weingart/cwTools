//! Spline-trimap overlay interact.
//!
//! Draw: the geometry the last render of this frame stored (spline.rs) —
//! inner (green), middle (white), outer (yellow), outer 2 (orange), point
//! handles (cyan anchor on the edge, magenta Out handle, blue In handle). In
//! Edit mode also the source frame border and the seed bbox (grey) so a test
//! can see whether overlay and image line up.
//!
//! Pen (Edit Splines on only), all in canonical coordinates:
//!   - on a point's anchor  → slide it along the edge (re-snap, new U/V)
//!   - on a point's Out/In handle → radial drag = that point's Out / In
//!   - on the outer line    → drag = global Dilate Out (Kernel Size if symmetric)
//!   - on the inner line    → drag = global Erode In  (Kernel Size if symmetric)
//!   - anywhere else        → new point in the next free slot, anchored at
//!     the nearest edge point, offset so the band reaches the click; the
//!     drag continues on its handle.
//! Every parameter written is a plain animatable param, so an external
//! controller (Meow) can drive the same edits.
//!
//! Probe logging (Flame visibility unproven): one log line per Draw call
//! (first 200, then every 50th; all with CWREFINE_VERBOSE), with the GL
//! error after drawing; every PenDown / PenUp, every 10th PenMotion.
//! CWREFINE_DRAW_DUMP=<file> also records each Draw's primitives as text.

use crate::params::{self, cp_name};
use crate::spline::{Geometry, MAX_POINTS};
use cwtools_core::log;
use cwtools_core::ofx_api::*;
use cwtools_core::overlay_gl::{gl, Canvas, GlCanvas, RecordCanvas, GL_LINES, GL_LINE_LOOP, GL_LINE_STRIP};
use cwtools_core::params::{get_double, get_int, n, set_bool};
use cwtools_core::props::{prop_get_pointer, prop_try_double};
use cwtools_core::suites::{effect_param_set, suites};
use cwtools_core::{guarded, param};
use std::collections::HashMap;
use std::ffi::CStr;
use std::os::raw::{c_char, c_int, c_uint, c_void};
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::{Arc, Mutex};

static DRAWS: AtomicU64 = AtomicU64::new(0);
static MOTIONS: AtomicU64 = AtomicU64::new(0);
static GL_MISSING_LOGGED: AtomicBool = AtomicBool::new(false);

#[derive(Clone)]
enum Drag {
    /// Slide point `slot` along the edge.
    Anchor { slot: usize },
    /// Radial drag of a point's Out (outer = true) or In handle.
    Handle { slot: usize, outer: bool, anchor: (f64, f64), normal: (f64, f64), r_c: f64 },
    /// Global kernel drag: param name, start value, start distance to middle.
    Global { param: &'static str, k0: f64, d0: f64 },
}

struct DragState {
    drag: Drag,
    geo: Arc<Geometry>,
}

static DRAGGING: Mutex<Option<HashMap<usize, DragState>>> = Mutex::new(None);

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
    if e.is_null() {
        None
    } else {
        Some(e)
    }
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
        if a == kOfxActionDescribe || a == kOfxActionCreateInstance {
            return kOfxStatOK;
        }
        if a == kOfxActionDestroyInstance {
            if let Ok(mut g) = DRAGGING.lock() {
                if let Some(m) = g.as_mut() {
                    m.remove(&(ih as usize));
                }
            }
            return kOfxStatOK;
        }
        if a == kOfxInteractActionDraw {
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

fn flags(eh: OfxImageEffectHandle, t: f64) -> (bool, bool, bool) {
    let ps = effect_param_set(eh);
    let b = |nm: &str| get_int(ps, &n(nm), t) != 0;
    (b(params::P_SPLINE_ON), b(params::P_SPLINE_EDIT), b(params::P_SPLINE_SHOW))
}

fn square(c: &mut dyn Canvas, p: (f64, f64), r: f64) {
    c.prim(GL_LINE_LOOP, &[(p.0 - r, p.1 - r), (p.0 + r, p.1 - r), (p.0 + r, p.1 + r), (p.0 - r, p.1 + r)]);
}

/// Draw `g` onto any canvas. `px` = canonical units per screen pixel.
pub fn draw_geometry(c: &mut dyn Canvas, g: &Geometry, px: f64, edit: bool, stale: bool) {
    let dim = if stale { 0.5 } else { 1.0 };
    if edit {
        // alignment probe: frame border + seed bbox
        let (x1, y1, x2, y2) = g.frame_c;
        c.color(0.5 * dim, 0.5 * dim, 0.5 * dim);
        c.width(1.0);
        c.prim(GL_LINE_LOOP, &[(x1, y1), (x2, y1), (x2, y2), (x1, y2)]);
        let (bx0, by0, bx1, by1) = g.bbox_c;
        c.prim(GL_LINE_LOOP, &[(bx0, by0), (bx1, by0), (bx1, by1), (bx0, by1)]);
    }
    let lines: [(&Vec<Vec<(f64, f64)>>, (f32, f32, f32), f32); 4] = [
        (&g.outer2, (1.0, 0.45, 0.1), 1.5),
        (&g.outer, (1.0, 0.85, 0.1), 2.0),
        (&g.middle, (0.9, 0.9, 0.9), 1.0),
        (&g.inner, (0.2, 1.0, 0.35), 2.0),
    ];
    for (pls, (r, gg, b), w) in lines {
        c.color(r * dim, gg * dim, b * dim);
        c.width(w);
        for pl in pls.iter() {
            c.prim(GL_LINE_STRIP, pl);
        }
    }
    let hr = 5.0 * px;
    c.width(2.0);
    for p in &g.points {
        c.color(0.4 * dim, 0.6 * dim, 1.0 * dim);
        c.prim(GL_LINES, &[p.anchor, p.in_handle]);
        c.color(1.0 * dim, 0.3 * dim, 1.0 * dim);
        c.prim(GL_LINES, &[p.anchor, p.out_handle]);
        square(c, p.out_handle, hr);
        c.color(0.4 * dim, 0.6 * dim, 1.0 * dim);
        square(c, p.in_handle, hr * 0.8);
        c.color(0.1 * dim, 1.0 * dim, 1.0 * dim);
        square(c, p.anchor, hr * 0.7);
    }
    c.width(1.0);
}

fn draw(ih: OfxInteractHandle, in_args: OfxPropertySetHandle) -> OfxStatus {
    let k = DRAWS.fetch_add(1, Ordering::Relaxed) + 1;
    let t = prop_try_double(in_args, kOfxPropTime, 0).unwrap_or(0.0);
    let px = prop_try_double(in_args, kOfxInteractPropPixelScale, 0).unwrap_or(1.0);
    let logit = log::verbose() || k <= 200 || k % 50 == 0;
    let Some(eh) = effect_of(ih) else {
        if logit {
            log::log(&format!("draw #{k}: t={t} no effect instance on the interact"));
        }
        return kOfxStatReplyDefault;
    };
    let (on, edit, show) = flags(eh, t);
    if !(on && (edit || show)) {
        if logit && k <= 3 {
            log::log(&format!("draw #{k}: t={t} splines hidden (Spline Trimap {on}, Edit {edit}, Show {show})"));
        }
        return kOfxStatReplyDefault;
    }
    let Some((g, stale)) = crate::spline::geometry_for(eh as usize, t) else {
        if logit {
            log::log(&format!("draw #{k}: t={t} no spline geometry yet (render this frame once)"));
        }
        return kOfxStatReplyDefault;
    };
    if let Ok(path) = std::env::var("CWREFINE_DRAW_DUMP") {
        let mut rc = RecordCanvas::new();
        draw_geometry(&mut rc, &g, px, edit, stale);
        if let Err(e) = rc.save(&path) {
            log::log(&format!("draw dump {path} failed: {e}"));
        }
    }
    let Some(glf) = gl() else {
        if !GL_MISSING_LOGGED.swap(true, Ordering::Relaxed) {
            log::log("interact: OpenGL symbols not found in this process - overlay not drawn");
        }
        if logit {
            log::log(&format!("draw #{k}: t={t} (no GL) geometry {} vertices stale={stale}", g.vertex_count()));
        }
        return kOfxStatOK;
    };
    let mut c = GlCanvas::new(glf);
    draw_geometry(&mut c, &g, px, edit, stale);
    let err = c.error();
    if logit {
        log::log(&format!(
            "draw #{k}: t={t} pixelScale={px:.4} prims={} vertices={} stale={stale} edit={edit} glGetError={}",
            c.prims,
            c.vertices,
            err.map(|e| format!("0x{e:04x}{}", if e == 0 { " (ok)" } else { " (GL REJECTED the draw - core profile?)" }))
                .unwrap_or_else(|| "n/a".into())
        ));
    }
    kOfxStatOK
}

// ---------------------------------------------------------------------------
// Param writes (keyframe-aware)
// ---------------------------------------------------------------------------

fn handle(ps: OfxParamSetHandle, name: &str) -> OfxParamHandle {
    cwtools_core::params::handle(ps, &n(name))
}

/// Set a Double: at time `t` if it is animated (has keys), else its value.
pub fn set_double(ps: OfxParamSetHandle, name: &str, t: f64, v: f64) {
    let h = handle(ps, name);
    if h.is_null() || !v.is_finite() {
        return;
    }
    let mut keys: c_uint = 0;
    let st = unsafe { (param().paramGetNumKeys)(h, &mut keys as *mut c_uint as *mut _) };
    unsafe {
        if st == kOfxStatOK && keys > 0 {
            cw_param_set_double_at(param().paramSetValueAtTime, h, t, v);
        } else {
            cw_param_set_double(param().paramSetValue, h, v);
        }
    }
}

fn dist(a: (f64, f64), b: (f64, f64)) -> f64 {
    ((a.0 - b.0).powi(2) + (a.1 - b.1).powi(2)).sqrt()
}

fn edit_begin(ps: OfxParamSetHandle, what: &[u8]) {
    unsafe { (param().paramEditBegin)(ps, what.as_ptr() as *const c_char) };
}
fn edit_end(ps: OfxParamSetHandle) {
    unsafe { (param().paramEditEnd)(ps) };
}

/// Kernel param a global drag of the outer / inner line drives.
fn kernel_param(_ps: OfxParamSetHandle, _t: f64, outer: bool) -> &'static str {
    if outer {
        params::P_DILATE_OUT
    } else {
        params::P_ERODE_IN
    }
}

/// Pure decision for a PenDown (testable): what the click grabs.
pub enum Pick {
    Anchor(usize),
    Handle(usize, bool),
    Outer,
    Inner,
    New,
}

pub fn pick(g: &Geometry, p: (f64, f64), radius: f64) -> Pick {
    let mut best: Option<(f64, Pick)> = None;
    let mut consider = |d: f64, pk: Pick| {
        if d <= radius && best.as_ref().map(|b| d < b.0).unwrap_or(true) {
            best = Some((d, pk));
        }
    };
    for gp in &g.points {
        consider(dist(p, gp.anchor) * 0.9, Pick::Anchor(gp.slot)); // anchor wins ties
        consider(dist(p, gp.out_handle), Pick::Handle(gp.slot, true));
        consider(dist(p, gp.in_handle), Pick::Handle(gp.slot, false));
    }
    if let Some((_, pk)) = best {
        return pk;
    }
    let dout = Geometry::dist_to(&g.outer, p);
    let din = Geometry::dist_to(&g.inner, p);
    if dout <= radius && dout <= din {
        Pick::Outer
    } else if din <= radius {
        Pick::Inner
    } else {
        Pick::New
    }
}

/// phase 0 = down, 1 = motion, 2 = up
fn pen(ih: OfxInteractHandle, in_args: OfxPropertySetHandle, phase: c_int) -> OfxStatus {
    let (Some(x), Some(y)) =
        (prop_try_double(in_args, kOfxInteractPropPenPosition, 0), prop_try_double(in_args, kOfxInteractPropPenPosition, 1))
    else {
        return kOfxStatReplyDefault;
    };
    let p = (x, y);
    let t = prop_try_double(in_args, kOfxPropTime, 0).unwrap_or(0.0);
    let px = prop_try_double(in_args, kOfxInteractPropPixelScale, 0).unwrap_or(1.0);
    let key = ih as usize;
    match phase {
        0 => log::log(&format!("interact: PenDown canonical ({x:.1},{y:.1}) t={t} pixelScale={px:.4}")),
        2 => log::log(&format!("interact: PenUp canonical ({x:.1},{y:.1})")),
        _ => {
            if MOTIONS.fetch_add(1, Ordering::Relaxed) % 10 == 0 {
                log::log(&format!("interact: PenMotion canonical ({x:.1},{y:.1}) (every 10th logged)"));
            }
        }
    }
    let Some(eh) = effect_of(ih) else { return kOfxStatReplyDefault };
    let (on, edit, _) = flags(eh, t);
    if !(on && edit) {
        return kOfxStatReplyDefault;
    }
    let ps = effect_param_set(eh);
    let mut lock = DRAGGING.lock().unwrap_or_else(|e| e.into_inner());
    let drags = lock.get_or_insert_with(HashMap::new);

    if phase == 0 {
        let Some((g, stale)) = crate::spline::geometry_for(eh as usize, t) else {
            log::log("interact: no spline geometry for this frame yet - render it once");
            return kOfxStatReplyDefault;
        };
        if stale {
            log::log("interact: using geometry of another frame (this frame not rendered yet)");
        }
        let radius = 8.0 * px.abs().max(1e-6);
        let drag = match pick(&g, p, radius) {
            Pick::Anchor(slot) => Drag::Anchor { slot },
            Pick::Handle(slot, outer) => {
                let gp = g.points.iter().find(|q| q.slot == slot).copied().unwrap();
                let r_c = if outer { g.r_out_c } else { g.r_in_c };
                Drag::Handle { slot, outer, anchor: gp.anchor, normal: gp.normal, r_c }
            }
            Pick::Outer | Pick::Inner => {
                let outer = matches!(pick(&g, p, radius), Pick::Outer);
                let pname = kernel_param(ps, t, outer);
                let k0 = get_double(ps, &n(pname), t);
                let d0 = g.nearest_middle(p).map(|m| m.2).unwrap_or(0.0);
                Drag::Global { param: pname, k0, d0 }
            }
            Pick::New => {
                let Some(slot) = (0..MAX_POINTS).find(|&i| get_int(ps, &n(&cp_name(i, "On")), t) == 0) else {
                    log::log("interact: all 8 control points are in use - turn one off or Clear Points");
                    return kOfxStatOK;
                };
                let Some((a, nrm, _)) = g.nearest_middle(p) else { return kOfxStatReplyDefault };
                let along = (p.0 - a.0) * nrm.0 + (p.1 - a.1) * nrm.1;
                let outer = along >= 0.0;
                let (u, v) = g.uv(a);
                edit_begin(ps, b"cwRefine spline point\0");
                set_double(ps, &cp_name(slot, "U"), t, u);
                set_double(ps, &cp_name(slot, "V"), t, v);
                if outer {
                    set_double(ps, &cp_name(slot, "Out"), t, along - g.r_out_c);
                    set_double(ps, &cp_name(slot, "In"), t, 0.0);
                } else {
                    set_double(ps, &cp_name(slot, "In"), t, -along - g.r_in_c);
                    set_double(ps, &cp_name(slot, "Out"), t, 0.0);
                }
                set_bool(ps, &n(&cp_name(slot, "On")), true);
                log::log(&format!(
                    "interact: new point {} at U={u:.3} V={v:.3} {} {:.1} px",
                    slot + 1,
                    if outer { "Out" } else { "In" },
                    if outer { along - g.r_out_c } else { -along - g.r_in_c }
                ));
                drags.insert(key, DragState {
                    drag: Drag::Handle { slot, outer, anchor: a, normal: nrm, r_c: if outer { g.r_out_c } else { g.r_in_c } },
                    geo: g,
                });
                redraw(ih);
                return kOfxStatOK;
            }
        };
        edit_begin(ps, b"cwRefine spline drag\0");
        drags.insert(key, DragState { drag, geo: g });
    }

    let Some(st) = drags.get(&key) else { return kOfxStatReplyDefault };
    let g = st.geo.clone();
    match st.drag.clone() {
        Drag::Anchor { slot } => {
            if let Some((a, _, _)) = g.nearest_middle(p) {
                let (u, v) = g.uv(a);
                set_double(ps, &cp_name(slot, "U"), t, u);
                set_double(ps, &cp_name(slot, "V"), t, v);
            }
        }
        Drag::Handle { slot, outer, anchor, normal, r_c } => {
            let along = (p.0 - anchor.0) * normal.0 + (p.1 - anchor.1) * normal.1;
            if outer {
                set_double(ps, &cp_name(slot, "Out"), t, along - r_c);
            } else {
                set_double(ps, &cp_name(slot, "In"), t, -along - r_c);
            }
        }
        Drag::Global { param: pname, k0, d0 } => {
            let d = g.nearest_middle(p).map(|m| m.2).unwrap_or(d0);
            let k = (k0 + 2.0 * (d - d0) / g.fpp.max(1e-9)).max(1.0).round();
            set_double(ps, pname, t, k);
        }
    }
    if phase == 2 {
        drags.remove(&key);
        edit_end(ps);
    }
    redraw(ih);
    kOfxStatOK
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::spline::GeoPoint;
    fn geo() -> Geometry {
        let circle = |r: f64| (0..=64).map(|i| {
            let a = -(i as f64) / 64.0 * std::f64::consts::TAU; // clockwise (as rendered geometry)
            (100.0 + r * a.cos(), 100.0 + r * a.sin())
        }).collect::<Vec<_>>();
        Geometry {
            inner: vec![circle(48.0)],
            middle: vec![circle(50.0)],
            outer: vec![circle(65.0)],
            bbox_c: (50.0, 50.0, 150.0, 150.0),
            fpp: 1.0,
            r_out_c: 15.0,
            r_in_c: 2.0,
            points: vec![GeoPoint { slot: 2, anchor: (150.0, 100.0), normal: (1.0, 0.0), out_handle: (175.0, 100.0), in_handle: (145.0, 100.0) }],
            ..Default::default()
        }
    }
    #[test]
    fn pick_priorities() {
        let g = geo();
        assert!(matches!(pick(&g, (151.0, 100.0), 8.0), Pick::Anchor(2)));
        assert!(matches!(pick(&g, (176.0, 101.0), 8.0), Pick::Handle(2, true)));
        assert!(matches!(pick(&g, (100.0, 165.0), 8.0), Pick::Outer));
        assert!(matches!(pick(&g, (100.0, 147.0), 4.0), Pick::Inner));
        assert!(matches!(pick(&g, (100.0, 190.0), 8.0), Pick::New));
        // nearest middle point + outward normal (canonical, y-up)
        let (a, nrm, d) = g.nearest_middle((100.0, 190.0)).unwrap();
        assert!((a.1 - 150.0).abs() < 1e-6 && (d - 40.0).abs() < 1e-6, "{a:?} {d}");
        assert!(nrm.1 > 0.99, "outward normal {nrm:?}");
        let (u, v) = g.uv(a);
        assert!((u - 0.5).abs() < 1e-9 && (v - 1.0).abs() < 1e-9);
    }
    #[test]
    fn draw_records_primitives() {
        let g = geo();
        let mut rc = RecordCanvas::new();
        draw_geometry(&mut rc, &g, 1.0, true, false);
        assert!(rc.text.lines().count() >= 8);
        assert!(rc.text.contains("strip 1.000 0.850 0.100")); // outer = yellow
    }
}
