//! Render: Source (+ optional prompt matte) → model image + prompts → backend
//! → soft mask → upscale → Output.
//!
//! Failure policy: never hang, never crash. Any backend error passes the plate
//! through with an EMPTY matte (alpha 0; Matte mode = black), posts a node
//! message and logs to /tmp/cwSegment.log. Render still returns OK so Flame
//! keeps working.

use crate::actions::effect_param_set;
use crate::imgutil::{self, Gray, Rgb8};
use crate::ofx_api::*;
use crate::params::{self, Values};
use crate::prompt::{self, Prompts};
use crate::{effect, prop_get_double, prop_get_rect_i, prop_try_double};
use cwtools_core::image::{downscale_ofx, fetch_image, release, Image};
use std::sync::Mutex;

pub const LOCAL_SIZE: usize = 1008;

// ---------------------------------------------------------------------------
// Mask cache (u8, model-image resolution).
// ---------------------------------------------------------------------------
struct CacheEntry {
    key: String,
    w: usize,
    h: usize,
    data: Vec<u8>,
}
static MASK_CACHE: Mutex<Vec<CacheEntry>> = Mutex::new(Vec::new());
const MASK_CACHE_CAP: usize = 48;

fn cache_get(key: &str) -> Option<Gray> {
    let mut g = MASK_CACHE.lock().ok()?;
    let i = g.iter().position(|e| e.key == key)?;
    let e = g.remove(i);
    let out = Gray { w: e.w, h: e.h, data: e.data.iter().map(|v| *v as f32 / 255.0).collect() };
    g.push(e); // most-recent last
    Some(out)
}
fn cache_put(key: String, m: &Gray) {
    if let Ok(mut g) = MASK_CACHE.lock() {
        g.retain(|e| e.key != key);
        if g.len() >= MASK_CACHE_CAP {
            g.remove(0);
        }
        g.push(CacheEntry {
            key,
            w: m.w,
            h: m.h,
            data: m.data.iter().map(|v| (v.clamp(0.0, 1.0) * 255.0 + 0.5) as u8).collect(),
        });
    }
}
pub fn clear_caches() {
    if let Ok(mut g) = MASK_CACHE.lock() {
        g.clear();
    }
    crate::multi::clear_cache();
    crate::local::clear_cache();
}

// ---------------------------------------------------------------------------

#[derive(Clone)]
pub(crate) struct Geo {
    /// full-frame size (source bounds) at render scale
    pub fw: usize,
    pub fh: usize,
    pub b: OfxRectI,
    pub rs: (f64, f64),
    /// model image size
    pub mw: usize,
    pub mh: usize,
}

impl Geo {
    /// Canonical coords → model-image coords (top-down). Coordinates with
    /// both |x|,|y| <= 1 are taken as NORMALISED (0..1 of the frame) — this
    /// covers hosts that ignore kOfxParamCoordinatesNormalised defaults and
    /// lets users type 0..1 values; anything larger is canonical pixels.
    pub(crate) fn canon_to_model(&self, x: f64, y: f64) -> (f32, f32) {
        let (px, py) = cwtools_core::image::canon_to_frame(x, y, self.fw, self.fh, self.b, self.rs);
        (
            (px * self.mw as f64 / self.fw as f64) as f32,
            (py * self.mh as f64 / self.fh as f64) as f32,
        )
    }
    /// Canonical coords → OFX pixel coords at render scale.
    fn canon_to_pixel(&self, x: f64, y: f64) -> (i32, i32) {
        ((x * self.rs.0).round() as i32, (y * self.rs.1).round() as i32)
    }
}

fn build_prompts(_eh: OfxImageEffectHandle, src: &Image, v: &Values, g: &Geo, _time: f64) -> Prompts {
    let mut p = Prompts { text: v.text.clone(), ..Default::default() };
    for up in &v.points {
        let (x, y) = g.canon_to_model(up.x, up.y);
        if x >= 0.0 && y >= 0.0 && x <= g.mw as f32 && y <= g.mh as f32 {
            p.points.push((x, y, up.fg));
        }
    }
    if let Some((x0, y0, x1, y1)) = v.box_canon {
        let (ax, ay) = g.canon_to_model(x0, y0);
        let (bx, by) = g.canon_to_model(x1, y1);
        let (lx, hx) = (ax.min(bx).max(0.0), ax.max(bx).min(g.mw as f32));
        let (ly, hy) = (ay.min(by).max(0.0), ay.max(by).min(g.mh as f32));
        if hx - lx >= 2.0 && hy - ly >= 2.0 {
            p.bbox = Some([lx, ly, hx, hy]);
        }
    }
    if v.use_matte {
        // The prompt mask is the Source Alpha socket (the Prompt clip was removed).
        let matte = if src.n_comps == 4 {
            Some(downscale_ofx(src, &[3], g.mw, g.mh))
        } else {
            None
        };
        if let Some(m) = matte {
            let gm = Gray { w: g.mw, h: g.mh, data: m };
            if let Some((bbox, pts)) = prompt::from_matte(&gm) {
                if v.matte_as != 2 && p.bbox.is_none() {
                    p.bbox = Some(bbox);
                }
                if v.matte_as != 1 {
                    for (x, y) in pts {
                        p.points.push((x, y, true));
                    }
                }
            }
        }
    }
    p
}

enum Outcome {
    Mask(Gray), // soft, model size
    NoPrompt,
    Error(String),
}

pub use crate::multi::last_list;

pub fn render(eh: OfxImageEffectHandle, in_args: OfxPropertySetHandle) -> OfxStatus {
    let time = prop_get_double(in_args, kOfxPropTime, 0);
    let win = prop_get_rect_i(in_args, kOfxImageEffectPropRenderWindow);
    let rs = (
        prop_try_double(in_args, kOfxImageEffectPropRenderScale, 0).unwrap_or(1.0),
        prop_try_double(in_args, kOfxImageEffectPropRenderScale, 1).unwrap_or(1.0),
    );
    let ps = effect_param_set(eh);
    let mut v = params::read_values(ps, time);
    // A pick is armed: show every object (ID Map) so the next one can be chosen;
    // the click sets the slot and the panel's Output comes back.
    if params::armed_pick(ps, time).is_some() {
        v.output = params::OUT_IDMAP;
    }
    // Picking objects to track: the detections, numbered, not the tracked result.
    if params::armed_track(ps, time) {
        v.output = params::OUT_IDMAP;
        v.track = crate::track::DIR_OFF;
        v.picking = true; // every detection, the picked ones highlighted
    }

    let Some(src) = fetch_image(eh, kOfxImageEffectSimpleSourceClipName, time) else {
        return kOfxStatFailed;
    };
    let Some(out) = fetch_image(eh, kOfxImageEffectOutputClipName, time) else {
        release(&src);
        return kOfxStatFailed;
    };
    if src.n_comps < 3 || out.n_comps < 3 || src.w() == 0 || src.h() == 0 {
        release(&src);
        release(&out);
        return kOfxStatFailed;
    }

    let (fw, fh) = (src.w(), src.h());
    // Local and Remote run the same graphs on the same 1008 px image
    let (mw, mh) = imgutil::fit_size(fw, fh, LOCAL_SIZE);
    let geo = Geo { fw, fh, b: src.bounds, rs, mw, mh };

    let prompts = build_prompts(eh, &src, &v, &geo, time);

    // Multi-object path: several concepts, an object view, or VLM Auto.
    // Everything else (single concept / geometric prompts, Output 0-2) runs
    // the original single-matte code below, unchanged.
    let typed = crate::objects::parse_concepts(&v.text);
    let tracking = v.track != crate::track::DIR_OFF && !v.track_list.is_empty();
    if v.output >= params::OUT_CHANNELS || typed.len() > 1 || v.vlm_auto != 0 || tracking {
        let st = crate::multi::render_multi(eh, &src, &out, &win, &v, &geo, &prompts, typed, time);
        release(&src);
        release(&out);
        return st;
    }

    let outcome = if prompts.is_empty() {
        Outcome::NoPrompt
    } else {
        // Model image (display-referred RGB8, top-down).
        let rgb = downscale_ofx(&src, &[0, 1, 2], mw, mh);
        let img8 = imgutil::rgbf_to_display8(&imgutil::RgbF { w: mw, h: mh, data: rgb }, v.colorspace);
        let fk = format!("cwseg_{}_{}x{}", imgutil::frame_key(&img8), mw, mh);
        let ckey = format!(
            "{}|{}|{}|{}|{:.2}|{}",
            v.mode, fk, prompts.key(), v.best_only, v.score_threshold, v.model_dir
        );
        match if v.cache { cache_get(&ckey) } else { None } {
            Some(m) => Outcome::Mask(m),
            None => {
                if unsafe { (effect().abort)(eh) } != 0 {
                    release(&src);
                    release(&out);
                    return kOfxStatOK;
                }
                let t0 = std::time::Instant::now();
                // text only -> detector; points / box -> tracker (text ignored); Local or Remote
                let r = crate::local::segment(&fk, &img8, &prompts, &v);
                match r {
                    Ok(m) => {
                        let m = if m.w != mw || m.h != mh { imgutil::bilinear_gray(&m, mw, mh) } else { m };
                        if v.cache {
                            cache_put(ckey, &m);
                        }
                        if crate::log::verbose() {
                            crate::log::log(&format!(
                                "t={time} mode={} {}x{} -> {}x{} {:.0}ms prompts={}",
                                v.mode, fw, fh, mw, mh, t0.elapsed().as_secs_f64() * 1000.0, prompts.key()
                            ));
                        }
                        Outcome::Mask(m)
                    }
                    Err(e) => Outcome::Error(e),
                }
            }
        }
    };

    let full = match &outcome {
        Outcome::Mask(m) => {
            crate::set_node_message(eh, None);
            let mut f = imgutil::bilinear_gray(m, fw, fh);
            if !v.soft {
                for x in f.data.iter_mut() {
                    *x = if *x >= 0.5 { 1.0 } else { 0.0 };
                }
            }
            Some(f)
        }
        Outcome::NoPrompt => {
            crate::set_node_message(eh, Some("cwSegment: no prompt - set Text, Points, Box or Use Prompt Mask"));
            None
        }
        Outcome::Error(e) => {
            let mode = if v.mode == 1 { "Local" } else { "Remote" };
            crate::set_node_message(eh, Some(&format!("cwSegment {mode} failed: {e}")));
            None
        }
    };

    write_output(&src, &out, &win, full.as_ref(), &v, &geo, &prompts);
    release(&src);
    release(&out);
    kOfxStatOK
}

pub(crate) fn write_output(src: &Image, out: &Image, win: &OfxRectI, mask: Option<&Gray>, v: &Values, g: &Geo, p: &Prompts) {
    let sb = src.bounds;
    let x1 = win.x1.max(out.bounds.x1);
    let x2 = win.x2.min(out.bounds.x2);
    let y1 = win.y1.max(out.bounds.y1);
    let y2 = win.y2.min(out.bounds.y2);
    for y in y1..y2 {
        for x in x1..x2 {
            let inside = x >= sb.x1 && x < sb.x2 && y >= sb.y1 && y < sb.y2;
            let (r, gg, b) = if inside {
                let sp = unsafe { src.px(x, y) };
                unsafe { (*sp, *sp.add(1), *sp.add(2)) }
            } else {
                (0.0, 0.0, 0.0)
            };
            let mut m = match (mask, inside) {
                (Some(mk), true) => {
                    let col = (x - sb.x1) as usize;
                    let row = (sb.y2 - 1 - y) as usize;
                    mk.data[row * mk.w + col]
                }
                _ => 0.0,
            };
            if v.invert && mask.is_some() {
                m = 1.0 - m;
            }
            let (or, og, ob) = match v.output {
                0 => (m, m, m),
                1 => (r, gg, b),
                _ => cwtools_core::color::overlay_red(r, gg, b, m),
            };
            let op = unsafe { out.px(x, y) };
            unsafe {
                *op = or;
                *op.add(1) = og;
                *op.add(2) = ob;
                if out.n_comps == 4 {
                    *op.add(3) = m;
                }
            }
        }
    }
    if v.output == 2 {
        draw_prompt_marks(out, win, v, g, p);
    }
}

/// Overlay mode: stamp prompt markers into the image (works even if the host
/// never calls the overlay interact). FG = green, BG = blue, box = yellow.
fn draw_prompt_marks(out: &Image, win: &OfxRectI, v: &Values, g: &Geo, p: &Prompts) {
    let put = |x: i32, y: i32, c: [f32; 3]| {
        if x >= win.x1 && x < win.x2 && y >= win.y1 && y < win.y2 {
            let op = unsafe { out.px(x, y) };
            unsafe {
                *op = c[0];
                *op.add(1) = c[1];
                *op.add(2) = c[2];
            }
        }
    };
    let rad = ((g.fw.max(g.fh) as f32 / 400.0).ceil() as i32).max(3);
    // Model-space prompts (includes matte-derived ones) → OFX pixels.
    let to_px = |mx: f32, my: f32| -> (i32, i32) {
        let px = mx as f64 * g.fw as f64 / g.mw as f64 + g.b.x1 as f64;
        let py = g.b.y2 as f64 - my as f64 * g.fh as f64 / g.mh as f64;
        (px.round() as i32, py.round() as i32)
    };
    for &(mx, my, fg) in &p.points {
        let (cx, cy) = to_px(mx, my);
        let c = if fg { [0.1, 1.0, 0.1] } else { [0.1, 0.3, 1.0] };
        for dy in -rad..=rad {
            for dx in -rad..=rad {
                put(cx + dx, cy + dy, c);
            }
        }
    }
    if let Some(b) = p.bbox {
        let (x0, y0) = to_px(b[0], b[1]);
        let (x1, y1) = to_px(b[2], b[3]);
        let (lx, hx, ly, hy) = (x0.min(x1), x0.max(x1), y0.min(y1), y0.max(y1));
        let t = (rad / 2).max(1);
        for x in lx..=hx {
            for k in 0..t {
                put(x, ly + k, [1.0, 1.0, 0.0]);
                put(x, hy - k, [1.0, 1.0, 0.0]);
            }
        }
        for y in ly..=hy {
            for k in 0..t {
                put(lx + k, y, [1.0, 1.0, 0.0]);
                put(hx - k, y, [1.0, 1.0, 0.0]);
            }
        }
    }
    let _ = (v, g.canon_to_pixel(0.0, 0.0));
}

#[allow(dead_code)]
fn _unused(_: &Rgb8) {}
