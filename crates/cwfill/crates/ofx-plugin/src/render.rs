//! Render action: pull the Source image, run the foreground solve over the
//! render window, write the chosen output.
//!
//! Matte conditioning (soft-band gating, band dilation, RoI-padded solve crop),
//! residual despill, and the input colour-space handling are wired here. The
//! solve always runs in a perceptual space: ACEScg input is encoded to ACEScct
//! and decoded back; ACEScct / Display input is already perceptual and solved
//! as-is. Scene-linear is never solved raw (sparse bright pixels corrupt the
//! multilevel solve's mean-based seeding — see tests). Querying the clip's
//! *tagged* space via the 2026.1 OFX Colour Management suite is a later
//! refinement.
//!
//! v0.3.0: every per-pixel stage is row-parallel, the solve uses the pipelined
//! multi-threaded sweep, and solved F/B are cached between renders (see
//! `cache.rs`) so output-only edits (despill, gating, Output mode) skip the
//! solve. All of it is bit-identical to v0.2.0 (`examples/parity.rs`).

use crate::cache;
use crate::ofx_api::*;
use crate::par::{self, SendPtr};
use crate::params::{self, Values};
use crate::solver::{self, FgParams, Plane};
use crate::{
    effect, param, prop_get_double, prop_get_int, prop_get_pointer, prop_get_rect_i,
};
use std::os::raw::c_char;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Mutex;
use std::time::Instant;

/// A host image buffer: OFX layout (rows bottom-up, `row_bytes` stride,
/// interleaved float components). Host-agnostic so the render core can be
/// driven outside an OFX host (`examples/parity.rs`).
#[derive(Clone, Copy)]
pub struct View {
    pub data: *mut u8,
    pub bounds: OfxRectI,
    pub row_bytes: i32,
    pub n_comps: i32, // 4 = RGBA, 3 = RGB
}
// Render threads only touch disjoint rows of a View (output) or read it.
unsafe impl Send for View {}
unsafe impl Sync for View {}

struct Image {
    props: OfxPropertySetHandle,
    view: View,
    half: bool, // OfxBitDepthHalf: 16-bit components, converted around the f32 core
}

fn fetch_image(clip_name: &[u8], effect_handle: OfxImageEffectHandle, time: f64) -> Option<Image> {
    let mut clip: OfxImageClipHandle = std::ptr::null_mut();
    let mut clip_props: OfxPropertySetHandle = std::ptr::null_mut();
    unsafe {
        if (effect().clipGetHandle)(effect_handle, clip_name.as_ptr() as *const c_char, &mut clip, &mut clip_props)
            != kOfxStatOK
        {
            return None;
        }
    }
    let mut img_props: OfxPropertySetHandle = std::ptr::null_mut();
    unsafe {
        if (effect().clipGetImage)(clip, time, std::ptr::null(), &mut img_props) != kOfxStatOK
            || img_props.is_null()
        {
            return None;
        }
    }
    let data = prop_get_pointer(img_props, kOfxImagePropData, 0) as *mut u8;
    let bounds = prop_get_rect_i(img_props, kOfxImagePropBounds);
    let row_bytes = prop_get_int(img_props, kOfxImagePropRowBytes, 0);

    // Components string → channel count.
    let mut comp_str: *mut c_char = std::ptr::null_mut();
    unsafe {
        (crate::prop().propGetString)(
            img_props,
            kOfxImageEffectPropComponents.as_ptr() as *const c_char,
            0,
            &mut comp_str,
        );
    }
    let n_comps = components_count(comp_str);
    let mut depth: *mut c_char = std::ptr::null_mut();
    unsafe {
        (crate::prop().propGetString)(
            img_props,
            kOfxImageEffectPropPixelDepth.as_ptr() as *const c_char,
            0,
            &mut depth,
        );
    }
    let half = !depth.is_null() && unsafe { std::ffi::CStr::from_ptr(depth) }.to_bytes() == b"OfxBitDepthHalf";

    Some(Image { props: img_props, view: View { data, bounds, row_bytes, n_comps }, half })
}

fn components_count(s: *mut c_char) -> i32 {
    if s.is_null() {
        return 0;
    }
    let cs = unsafe { std::ffi::CStr::from_ptr(s) };
    match cs.to_bytes() {
        b"OfxImageComponentRGBA" => 4,
        b"OfxImageComponentRGB" => 3,
        b"OfxImageComponentAlpha" => 1,
        _ => 0,
    }
}

#[inline]
unsafe fn px(img: &View, x: i32, y: i32) -> *mut f32 {
    let off = (y - img.bounds.y1) as isize * img.row_bytes as isize
        + (x - img.bounds.x1) as isize * (img.n_comps as isize) * 4;
    img.data.offset(off) as *mut f32
}

pub fn render(effect_handle: OfxImageEffectHandle, in_args: OfxPropertySetHandle) -> OfxStatus {
    let time = prop_get_double(in_args, kOfxPropTime, 0);
    let win = prop_get_rect_i(in_args, kOfxImageEffectPropRenderWindow);

    let mut param_set: OfxParamSetHandle = std::ptr::null_mut();
    unsafe {
        let _ = (effect().getParamSet)(effect_handle, &mut param_set);
    }
    let v = params::read_all(param_set);
    let _ = param(); // ensure suite present

    let src = match fetch_image(kOfxImageEffectSimpleSourceClipName, effect_handle, time) {
        Some(i) => i,
        None => return kOfxStatFailed,
    };
    let out = match fetch_image(kOfxImageEffectOutputClipName, effect_handle, time) {
        Some(i) => i,
        None => {
            release(&src);
            return kOfxStatFailed;
        }
    };

    // Half buffers: the core works on f32 copies; the output is converted
    // back over the render window.
    let threads = par::default_threads();
    let (_src_buf, src_v) = if src.half { half_to_f32(&src.view, threads) } else { (Vec::new(), src.view) };
    let (_out_buf, out_v) = if out.half { f32_scratch(&out.view) } else { (Vec::new(), out.view) };

    // Only the RGBA path carries a matte (alpha). Without alpha there is no
    // matte to solve against → passthrough copy.
    let (status, info) = if src_v.n_comps < 4 {
        copy_through(&src_v, &out_v, &win);
        (kOfxStatOK, RenderInfo::default())
    } else {
        let ctx = CacheCtx { instance: effect_handle as usize, time };
        solve_window(&src_v, &out_v, &win, &v, Some(ctx), threads)
    };
    if out.half {
        f32_to_half_window(&out_v, &out.view, &win, threads);
    }
    if let Some(path) = debug_log_path() {
        debug_log(path, effect_handle as usize, time, &win, &src_v, &v, &info, src.half, out.half);
    }
    release(&src);
    release(&out);
    status
}

/// `CWFILL_DEBUG=1` (→ /tmp/cwFill.log) or `CWFILL_DEBUG=/some/file.log`:
/// append one line per render with the values the plugin actually read
/// (params incl. the Screen Colour RGB, Source alpha stats, timings). For
/// diagnosing host behaviour in Flame without a debugger. Off by default.
fn debug_log_path() -> Option<&'static str> {
    static P: std::sync::OnceLock<Option<String>> = std::sync::OnceLock::new();
    P.get_or_init(|| match std::env::var("CWFILL_DEBUG") {
        Ok(s) if s == "1" || s.eq_ignore_ascii_case("true") => Some("/tmp/cwFill.log".to_string()),
        Ok(s) if s.contains('/') => Some(s),
        _ => None,
    })
    .as_deref()
}

#[allow(clippy::too_many_arguments)]
fn debug_log(path: &str, instance: usize, time: f64, win: &OfxRectI, src: &View, v: &Values, info: &RenderInfo, src_half: bool, out_half: bool) {
    use std::io::Write;
    let (mut amin, mut amax, mut asum, mut n, mut n_band) = (f32::MAX, f32::MIN, 0f64, 0u64, 0u64);
    for y in win.y1..win.y2 {
        for x in win.x1..win.x2 {
            let a = unsafe { *px(src, x, y).add(3) };
            amin = amin.min(a);
            amax = amax.max(a);
            asum += a as f64;
            n += 1;
            if a > v.pin_low && a < v.pin_high {
                n_band += 1;
            }
        }
    }
    let line = format!(
        "[cwFill {}] inst={:#x} t={} win=({},{})-({},{}) srcComps={} alpha min={:.4} max={:.4} mean={:.4} band_px={} | \
         depth in={} out={} softBand={} out={} cs={} edgeTone={} despill: screen={} colour=[{:.6},{:.6},{:.6}] strength={} range={} soft={} limitSoft={} thresh={} pinLow={} pinHigh={} | \
         crop={}x{} cacheHit={} solve={:.1}ms total={:.1}ms\n",
        env!("CARGO_PKG_VERSION"), instance, time, win.x1, win.y1, win.x2, win.y2, src.n_comps,
        amin, amax, if n > 0 { asum / n as f64 } else { 0.0 }, n_band,
        if src_half { "half" } else { "float" }, if out_half { "half" } else { "float" },
        v.soft_band_only as u8, v.output_mode, v.input_colorspace, v.edge_tone, v.despill_screen,
        v.screen_colour[0], v.screen_colour[1], v.screen_colour[2], v.residual_despill,
        v.despill_range, v.despill_softness, v.despill_soft_only as u8, v.despill_matte_thresh,
        v.pin_low, v.pin_high, info.crop.0, info.crop.1, info.cache_hit as u8,
        info.solve * 1e3, info.total * 1e3,
    );
    // Best effort: a logging failure must never fail the render.
    if let Ok(mut f) = std::fs::OpenOptions::new().create(true).append(true).open(path) {
        let _ = f.write_all(line.as_bytes());
    }
}

/// f32 copy of a half image (same bounds, tightly packed rows). The View
/// points into the returned Vec, which must outlive it.
fn half_to_f32(img: &View, threads: usize) -> (Vec<f32>, View) {
    let (w, h, n) = view_dims(img);
    let mut buf = vec![0f32; w * h * n];
    let pb = SendPtr(buf.as_mut_ptr());
    let src = SendPtr(img.data);
    par::row_chunks(h, par::threads_for(w * h, threads), |y0, y1| {
        let (pb, src) = (pb, src);
        for y in y0..y1 {
            let s = unsafe { src.0.offset(y as isize * img.row_bytes as isize) } as *const u16;
            for i in 0..w * n {
                unsafe { *pb.0.add(y * w * n + i) = f16_to_f32(s.add(i).read_unaligned()) };
            }
        }
    });
    let v = View { data: buf.as_mut_ptr() as *mut u8, bounds: img.bounds, row_bytes: (w * n * 4) as i32, n_comps: img.n_comps };
    (buf, v)
}

/// Zeroed f32 stand-in for a half output image (same bounds).
fn f32_scratch(img: &View) -> (Vec<f32>, View) {
    let (w, h, n) = view_dims(img);
    let mut buf = vec![0f32; w * h * n];
    let v = View { data: buf.as_mut_ptr() as *mut u8, bounds: img.bounds, row_bytes: (w * n * 4) as i32, n_comps: img.n_comps };
    (buf, v)
}

/// Write the render window of an f32 view into a half image.
fn f32_to_half_window(f: &View, out: &View, win: &OfxRectI, threads: usize) {
    let n = out.n_comps.min(f.n_comps).max(0) as usize;
    let (x1, x2) = (win.x1.max(out.bounds.x1), win.x2.min(out.bounds.x2));
    let (y1, y2) = (win.y1.max(out.bounds.y1), win.y2.min(out.bounds.y2));
    if x2 <= x1 || y2 <= y1 {
        return;
    }
    let rows = (y2 - y1) as usize;
    let dst = SendPtr(out.data);
    par::row_chunks(rows, par::threads_for(rows * (x2 - x1) as usize, threads), |r0, r1| {
        let dst = dst;
        for r in r0..r1 {
            let y = y1 + r as i32;
            for x in x1..x2 {
                unsafe {
                    let s = px(f, x, y);
                    let off = (y - out.bounds.y1) as isize * out.row_bytes as isize
                        + (x - out.bounds.x1) as isize * out.n_comps as isize * 2;
                    let d = dst.0.offset(off) as *mut u16;
                    for c in 0..n {
                        d.add(c).write_unaligned(f32_to_f16(*s.add(c)));
                    }
                }
            }
        }
    });
}

fn view_dims(img: &View) -> (usize, usize, usize) {
    let w = (img.bounds.x2 - img.bounds.x1).max(0) as usize;
    let h = (img.bounds.y2 - img.bounds.y1).max(0) as usize;
    (w, h, img.n_comps.max(0) as usize)
}

/// IEEE half → f32 (exact).
pub fn f16_to_f32(h: u16) -> f32 {
    let sign = ((h as u32) & 0x8000) << 16;
    let e = ((h >> 10) & 0x1f) as u32;
    let m = (h & 0x3ff) as u32;
    let bits = if e == 0 {
        if m == 0 {
            sign
        } else {
            // subnormal: normalise the mantissa
            let (mut e2, mut m2) = (113u32, m);
            while m2 & 0x400 == 0 {
                m2 <<= 1;
                e2 -= 1;
            }
            sign | (e2 << 23) | ((m2 & 0x3ff) << 13)
        }
    } else if e == 31 {
        sign | 0x7f80_0000 | (m << 13)
    } else {
        sign | ((e + 112) << 23) | (m << 13)
    };
    f32::from_bits(bits)
}

/// f32 → IEEE half, round to nearest even; overflow → inf, NaN stays NaN.
pub fn f32_to_f16(f: f32) -> u16 {
    let x = f.to_bits();
    let sign = ((x >> 16) & 0x8000) as u16;
    let e = ((x >> 23) & 0xff) as i32;
    let m = x & 0x7f_ffff;
    if e == 255 {
        return sign | 0x7c00 | if m != 0 { 0x200 | (m >> 13) as u16 } else { 0 };
    }
    let e16 = e - 127 + 15;
    if e16 >= 31 {
        return sign | 0x7c00;
    }
    if e16 <= 0 {
        if e16 < -10 {
            return sign;
        }
        let m = m | 0x80_0000;
        let shift = (14 - e16) as u32;
        let (q, rem, half) = (m >> shift, m & ((1 << shift) - 1), 1u32 << (shift - 1));
        let r = q + (rem > half || (rem == half && q & 1 == 1)) as u32;
        return sign | r as u16;
    }
    let q = ((e16 as u32) << 10) | (m >> 13);
    let rem = m & 0x1fff;
    let r = q + (rem > 0x1000 || (rem == 0x1000 && q & 1 == 1)) as u32; // a carry rolls into the exponent (to inf at the top)
    sign | r as u16
}

fn release(img: &Image) {
    unsafe {
        let _ = (effect().clipReleaseImage)(img.props);
    }
}

fn copy_through(src: &View, out: &View, win: &OfxRectI) {
    let n = out.n_comps.min(src.n_comps);
    for y in win.y1..win.y2 {
        for x in win.x1..win.x2 {
            unsafe {
                let s = px(src, x, y);
                let d = px(out, x, y);
                for c in 0..n {
                    *d.add(c as usize) = *s.add(c as usize);
                }
                // RGBA out from an RGB source: no matte to carry — write opaque
                // (otherwise the output alpha plane is left uninitialised).
                if out.n_comps == 4 && src.n_comps < 4 {
                    *d.add(3) = 1.0;
                }
            }
        }
    }
}

/// Identifies the render for the F/B cache. `None` = bypass the cache.
#[derive(Clone, Copy)]
pub struct CacheCtx {
    pub instance: usize,
    pub time: f64,
}

/// Per-stage wall times (seconds) + whether the solve came from the cache.
#[derive(Default, Clone, Debug)]
pub struct RenderInfo {
    pub gather: f64,
    pub band: f64,
    pub crop_encode: f64,
    pub solve: f64,
    pub output: f64,
    pub total: f64,
    pub cache_hit: bool,
    pub crop: (usize, usize),
}

/// Every parameter that feeds the solve (anything else is output-only and
/// may change without invalidating cached F/B).
fn solve_param_bits(v: &Values) -> Vec<u32> {
    vec![
        v.soft_band_only as u32,
        v.band_dilation as u32,
        v.pin_low.to_bits(),
        v.pin_high.to_bits(),
        v.roi_padding as u32,
        v.self_mul_strength.to_bits(),
        v.self_mul_mode as u32,
        v.quality as u32,
        v.regularization.to_bits(),
        v.gradient_weight.to_bits(),
        v.image_affinity.to_bits(),
        v.auto_solve as u32,
        v.input_colorspace as u32,
    ]
}

/// Render the window: `src` must be RGBA. Host-agnostic core of `render`.
pub fn solve_window(
    src: &View,
    out: &View,
    win: &OfxRectI,
    v: &Values,
    ctx: Option<CacheCtx>,
    threads: usize,
) -> (OfxStatus, RenderInfo) {
    let mut info = RenderInfo::default();
    let t_all = Instant::now();
    let w = (win.x2 - win.x1).max(0) as usize;
    let h = (win.y2 - win.y1).max(0) as usize;
    if w == 0 || h == 0 {
        return (kOfxStatOK, info);
    }
    let t_px = par::threads_for(w * h, threads);
    let use_cache = ctx.is_some() && cache::enabled();

    // ── Gather RGB + alpha over the full window (row-parallel), hashing the
    // raw Source bits per row for the cache key. We keep the whole window for
    // the output pass; the multigrid solve itself runs on a (possibly smaller)
    // RoI crop — see below.
    let t0 = Instant::now();
    let mut alpha = vec![0.0f32; h * w];
    let mut rgb_orig = vec![0.0f32; h * w * 3];
    let mut row_hash = vec![[0u64; 2]; if use_cache { h } else { 0 }];
    {
        let (pa, pr, ph) = (SendPtr(alpha.as_mut_ptr()), SendPtr(rgb_orig.as_mut_ptr()), SendPtr(row_hash.as_mut_ptr()));
        par::row_chunks(h, t_px, |y0, y1| {
            let (pa, pr, ph) = (pa, pr, ph);
            for yy in y0..y1 {
                let mut hs = cache::Hasher::new(yy as u64);
                let y = win.y1 + yy as i32;
                for xx in 0..w {
                    let x = win.x1 + xx as i32;
                    unsafe {
                        let s = px(src, x, y);
                        let i = yy * w + xx;
                        for c in 0..3 {
                            *pr.0.add(i * 3 + c) = *s.add(c);
                        }
                        *pa.0.add(i) = (*s.add(3)).clamp(0.0, 1.0);
                        if use_cache {
                            let b = |c: usize| (*s.add(c)).to_bits() as u64;
                            hs.word(b(0) | (b(1) << 32));
                            hs.word(b(2) | (b(3) << 32));
                        }
                    }
                }
                if use_cache {
                    unsafe { *ph.0.add(yy) = hs.finish() };
                }
            }
        });
    }
    let key = ctx.filter(|_| use_cache).map(|c| {
        let mut hs = cache::Hasher::new(0xC0FFEE);
        for rh in &row_hash {
            hs.word(rh[0]);
            hs.word(rh[1]);
        }
        cache::Key {
            instance: c.instance,
            time_bits: c.time.to_bits(),
            win: [win.x1, win.y1, win.x2, win.y2],
            content: hs.finish(),
            params: solve_param_bits(v),
        }
    });
    info.gather = t0.elapsed().as_secs_f64();

    let cached = key.as_ref().and_then(cache::get);
    let entry: std::sync::Arc<cache::Entry> = match cached {
        Some(e) => {
            info.cache_hit = true;
            e
        }
        None => match solve(&rgb_orig, &alpha, w, h, v, threads, &mut info) {
            Some((mask, rect, fg, bg)) => {
                let fresh = key.is_some();
                let e = std::sync::Arc::new(cache::Entry {
                    // Uncached renders carry a placeholder key (never stored).
                    key: key.clone().unwrap_or_else(cache::Key::uncached),
                    mask,
                    rect,
                    fg,
                    bg,
                });
                if fresh {
                    cache::put_arc(e.clone());
                }
                e
            }
            None => {
                // No soft band (or empty crop): nothing to solve, every pixel
                // is a hard region. It must still go through the hard-region
                // output (body despill, premult, contamination) — v0.2.0 wrote
                // the raw plate here, so despill silently did nothing on any
                // frame whose matte had no soft band: an all-solid matte, or
                // Flame's constant alpha when the Matte socket is unconnected.
                std::sync::Arc::new(cache::Entry {
                    key: cache::Key::uncached(),
                    mask: None,
                    rect: (0, 0, 0, 0), // empty crop → no pixel is "solved"
                    fg: Plane::new(0, 0, 3),
                    bg: Plane::new(0, 0, 3),
                })
            }
        },
    };

    let t0 = Instant::now();
    let body = (v.edge_tone != 0.0 && entry.rect.2 > entry.rect.0).then(|| {
        let mut b = body_reference(&rgb_orig, &alpha, w, h, v, t_px);
        tone_field(&entry, &mut b, &alpha, w, h, v);
        b
    });
    write_output(&entry, body.as_ref(), &rgb_orig, &alpha, out, win, w, h, v, t_px);
    info.output = t0.elapsed().as_secs_f64();
    info.crop = ((entry.rect.2 - entry.rect.0) as usize, (entry.rect.3 - entry.rect.1) as usize);
    info.total = t_all.elapsed().as_secs_f64();
    (kOfxStatOK, info)
}

type Solved = (Option<Vec<u8>>, (i32, i32, i32, i32), Plane, Plane);

/// Band mask + crop + encode + solve. None = nothing to solve (all hard).
fn solve(
    rgb_orig: &[f32],
    alpha: &[f32],
    w: usize,
    h: usize,
    v: &Values,
    threads: usize,
    info: &mut RenderInfo,
) -> Option<Solved> {
    let t_px = par::threads_for(w * h, threads);
    // Determine the output mask + solve crop.
    //  - Soft-Band Only ON  (default): recover FG only in the soft band
    //    (pin_low < alpha < pin_high), dilated by `band_dilation`. The solve
    //    runs on that mask's bbox padded by `roi_padding` so the multigrid keeps
    //    the surrounding hard-alpha anchors as boundary conditions. Pixels
    //    outside the mask pass the plate through. `pin_hard` is redundant here
    //    (plate passthrough already pins hard regions) so it is ignored.
    //  - Soft-Band Only OFF: full-window solve, FG everywhere, `pin_hard`
    //    forces alpha<=low / >=high pixels to plate (legacy behaviour).
    // mask[i] != 0 → pixel receives a solved value; otherwise plate.
    let t0 = Instant::now();
    let (mask, rx1, ry1, rx2, ry2) = if v.soft_band_only {
        let mut band = vec![0u8; h * w];
        let any = AtomicBool::new(false);
        {
            let pb = SendPtr(band.as_mut_ptr());
            par::row_chunks(h, t_px, |y0, y1| {
                let pb = pb;
                let mut hit = false;
                for i in y0 * w..y1 * w {
                    let a = alpha[i];
                    if a > v.pin_low && a < v.pin_high {
                        unsafe { *pb.0.add(i) = 1 };
                        hit = true;
                    }
                }
                if hit {
                    any.store(true, Ordering::Relaxed);
                }
            });
        }
        if !any.load(Ordering::Relaxed) {
            info.band = t0.elapsed().as_secs_f64();
            return None;
        }
        let mask = if v.band_dilation > 0 {
            dilate(&band, w, h, v.band_dilation as usize, t_px)
        } else {
            band
        };
        let (bx1, by1, bx2, by2) = bbox(&mask, w, h, t_px).expect("band non-empty");
        let pad = v.roi_padding.max(0);
        (
            Some(mask),
            (bx1 - pad).max(0),
            (by1 - pad).max(0),
            (bx2 + pad + 1).min(w as i32),
            (by2 + pad + 1).min(h as i32),
        )
    } else {
        (None, 0i32, 0i32, w as i32, h as i32)
    };
    info.band = t0.elapsed().as_secs_f64();

    let cw = (rx2 - rx1).max(0) as usize;
    let ch = (ry2 - ry1).max(0) as usize;
    if cw == 0 || ch == 0 {
        return None;
    }

    // Build the solver planes from the crop, encoding into the solver's
    // perceptual working space clamped to [0,1] (the solver also clamps F/B
    // internally). ACEScg input gets a fixed lin→ACEScct transfer (temporally
    // stable, HDR-graceful, spreads shadow detail so the gradient weighting
    // conditions well in darks); already-perceptual input (ACEScct / sRGB)
    // passes through. Optional pre-solve matte self-multiplication
    // (soft-band tightening) is applied per pixel.
    //
    // The solver's in-place Gauss-Seidel sweep must run top-down in FRAME
    // terms to match pymatting (numpy row 0 = top of frame). OFX buffers put
    // row 0 at the BOTTOM, so solving them as-is runs the sweep vertically
    // flipped: identical math, opposite propagation bias, visibly different
    // extrapolation in the under-constrained faint-alpha band. So crop row cy
    // (OFX order) lands in solver row ch-1-cy (frame order); the output pass
    // reads it back the same way.
    let t0 = Instant::now();
    let mut image = Plane::new(ch, cw, 3);
    let mut alpha_solve = vec![0.0f32; ch * cw];
    {
        let t_crop = par::threads_for(cw * ch, threads);
        let (pi, pa) = (SendPtr(image.data.as_mut_ptr()), SendPtr(alpha_solve.as_mut_ptr()));
        par::row_chunks(ch, t_crop, |c0, c1| {
            let (pi, pa) = (pi, pa);
            for cy in c0..c1 {
                let sr = ch - 1 - cy;
                for cx in 0..cw {
                    let gi = (ry1 as usize + cy) * w + (rx1 as usize + cx);
                    let si = sr * cw + cx;
                    let a = alpha[gi];
                    unsafe {
                        *pa.0.add(si) = if v.self_mul_strength > 0.0 {
                            self_mul_value(a, v.self_mul_strength, v.self_mul_mode)
                        } else {
                            a
                        };
                        for c in 0..3 {
                            *pi.0.add(si * 3 + c) =
                                encode(rgb_orig[gi * 3 + c], v.input_colorspace).clamp(0.0, 1.0);
                        }
                    }
                }
            }
        });
    }

    let (n_small, n_big) = match v.quality {
        0 => (4, 1),
        1 => (8, 2),
        _ => (10, 2),
    };
    // Auto-Tune Solver (experimental, opt-in): a single-pass heuristic that derives
    // regularization + gradient weight from the band's measured edge detail. This is
    // NOT a full metric-driven parameter search (that needs many solves — an offline
    // job); it just measures once and sets smart defaults. Off → the manual sliders.
    // (Kept sequential: its f64 sums must accumulate in v0.2.0 order.)
    let (reg, gw) = if v.auto_solve {
        let detail = measure_band_detail(rgb_orig, alpha, w, h, v.pin_low, v.pin_high);
        auto_solve_params(detail)
    } else {
        (v.regularization.max(1e-8), v.gradient_weight)
    };
    let fp = FgParams {
        regularization: reg,
        n_small_iterations: n_small,
        n_big_iterations: n_big,
        small_size: 32,
        gradient_weight: gw,
        // Slider 0..1 → exponent 0..200: the bars-lab study saturates near
        // s=100, so 0.5 is the working value and headroom tops out at 200.
        image_affinity: v.image_affinity * 200.0,
    };
    info.crop_encode = t0.elapsed().as_secs_f64();

    let t0 = Instant::now();
    let (fg, bg) = solver::estimate_fb_ml_mt(&image, &alpha_solve, &fp, threads);
    info.solve = t0.elapsed().as_secs_f64();
    Some((mask, (rx1, ry1, rx2, ry2), fg, bg))
}

/// Write the chosen output over the full window, decoding back to scene
/// values. Pixels outside the solved mask (or outside the crop) get plate.
#[allow(clippy::too_many_arguments)]
fn write_output(
    e: &cache::Entry,
    body: Option<&BodyRef>,
    rgb_orig: &[f32],
    alpha: &[f32],
    out: &View,
    win: &OfxRectI,
    w: usize,
    h: usize,
    v: &Values,
    threads: usize,
) {
    let (rx1, ry1, rx2, ry2) = e.rect;
    let cw = (rx2 - rx1) as usize;
    let ch = (ry2 - ry1) as usize;
    let (fg, bg) = (&e.fg, &e.bg);
    par::row_chunks(h, threads, |y0, y1| {
        for yy in y0..y1 {
            for xx in 0..w {
                let gi = yy * w + xx;
                let a = alpha[gi];
                // Optionally gate despill by the matte: when "Limit to Soft Matte" is on,
                // ramp despill down to zero at/above the threshold (spare the solid body),
                // with a small feather so there's no hard chroma seam at the contour.
                let despill_str = if v.despill_soft_only {
                    let t = v.despill_matte_thresh;
                    v.residual_despill * (1.0 - smoothstep(t - 0.05, t, a))
                } else {
                    v.residual_despill
                };
                let (x, y) = (win.x1 + xx as i32, win.y1 + yy as i32);
                let in_crop = (xx as i32) >= rx1
                    && (xx as i32) < rx2
                    && (yy as i32) >= ry1
                    && (yy as i32) < ry2;
                let solved = e.mask.as_ref().map_or(true, |m| m[gi] != 0) && in_crop;
                unsafe {
                    let d = px(out, x, y);
                    if !solved {
                        // Hard region outside the solved band (alpha ~0 or ~1). The
                        // value is closed-form per output mode, using the plate as
                        // the foreground estimate (exact where alpha=1). Mirrors the
                        // full-frame solve so premult/contamination/background stay
                        // consistent across the band boundary instead of showing the
                        // raw plate everywhere.
                        let mut col = [
                            rgb_orig[gi * 3],
                            rgb_orig[gi * 3 + 1],
                            rgb_orig[gi * 3 + 2],
                        ];
                        // Despill the solid foreground body too (where alpha is
                        // foreground) — green bounce lives all over the subject, not
                        // just the soft edge. Skip pure background (alpha<=pin_low).
                        if (v.output_mode == 0 || v.output_mode == 1) && a > v.pin_low {
                            apply_despill(&mut col, v.despill_screen, v.screen_colour, despill_str, v.despill_range, v.despill_softness);
                        }
                        for c in 0..3 {
                            *d.add(c) = match v.output_mode {
                                1 => col[c] * a, // premult: black where clear, plate where solid
                                2 => 0.0,        // contamination: nothing removed in hard regions
                                _ => col[c],     // foreground / background: plate passthrough
                            };
                        }
                    } else {
                        // Solver planes are frame order (top row first): OFX crop
                        // row cy lives in solver row ch-1-cy.
                        let cy = (yy as i32 - ry1) as usize;
                        let ci = (ch - 1 - cy) * cw + (xx as i32 - rx1) as usize;
                        let dec = |p: &Plane| {
                            [
                                decode(p.data[ci * 3], v.input_colorspace),
                                decode(p.data[ci * 3 + 1], v.input_colorspace),
                                decode(p.data[ci * 3 + 2], v.input_colorspace),
                            ]
                        };
                        // Solved foreground, Edge Tone applied in the solve space
                        // (an equal shift of the encoded channels = an exposure
                        // change for ACEScct, so hue and saturation stay).
                        // soft band only: hard pixels (pinned bg / solid body) are never toned
                        let fg_dec = || match body.filter(|_| a > v.pin_low && a < v.pin_high) {
                            Some(b) => {
                                let f = [fg.data[ci * 3], fg.data[ci * 3 + 1], fg.data[ci * 3 + 2]];
                                let k = b.shift[gi];
                                [
                                    decode(f[0] + k, v.input_colorspace),
                                    decode(f[1] + k, v.input_colorspace),
                                    decode(f[2] + k, v.input_colorspace),
                                ]
                            }
                            None => dec(fg),
                        };
                        // pin_hard only applies in full-solve mode; in soft-band
                        // mode the mask already gates hard regions to plate.
                        let pinned =
                            !v.soft_band_only && v.pin_hard && (a <= v.pin_low || a >= v.pin_high);
                        match v.output_mode {
                            0 | 1 => {
                                let mut col = if pinned {
                                    // Pinned = exact plate, but the solid body still
                                    // gets despill (as in the soft-band hard-region
                                    // path) — spill lives on the body too. Skip pure
                                    // background (alpha <= pin_low).
                                    let mut col = [
                                        rgb_orig[gi * 3],
                                        rgb_orig[gi * 3 + 1],
                                        rgb_orig[gi * 3 + 2],
                                    ];
                                    if a > v.pin_low {
                                        apply_despill(&mut col, v.despill_screen, v.screen_colour, despill_str, v.despill_range, v.despill_softness);
                                    }
                                    col
                                } else {
                                    let mut fg_c = fg_dec();
                                    apply_despill(&mut fg_c, v.despill_screen, v.screen_colour, despill_str, v.despill_range, v.despill_softness);
                                    fg_c
                                };
                                if v.output_mode == 1 {
                                    for c in col.iter_mut() {
                                        *c *= a;
                                    }
                                }
                                for c in 0..3 {
                                    *d.add(c) = col[c];
                                }
                            }
                            2 => {
                                // Contamination = the background colour that bled into
                                // the edge: source - foreground = (1-a)(B - F).
                                let fg_c = fg_dec();
                                for c in 0..3 {
                                    *d.add(c) = rgb_orig[gi * 3 + c] - fg_c[c];
                                }
                            }
                            _ => {
                                // Background — the only mode that needs B decoded.
                                let bg_c = dec(bg);
                                for c in 0..3 {
                                    *d.add(c) = bg_c[c];
                                }
                            }
                        }
                    }
                    if out.n_comps == 4 {
                        *d.add(3) = a; // carry matte through
                    }
                }
            }
        }
    });
}

// ---------------------------------------------------------------------------
// Edge Tone. Over a bright background the solve leaves soft-band fill brighter
// than the foreground really is (clipping, glare: light the linear mix can't
// explain); over a dark one, darker. The reference is the nearby solid body
// (alpha >= Pin High): a blurred luma of the plate there, in the solve space.
// Fill on the background's side of the body by more than a small margin is
// pulled back to that margin (slider > 0): rolled off over a bright
// background, lifted over a dark one. Slider < 0 pushes that same deviation
// further (more glow / deeper edge). Output-only (no re-solve).
// ---------------------------------------------------------------------------

/// Encoded-space margin the edge may sit above / below the body untouched
/// (0.03 ACEScct = about half a stop): real rim light keeps some brightness.
const EDGE_TONE_MARGIN: f32 = 0.03;

/// Window-sized (OFX row order) body reference: `lum` = blurred encoded luma
/// of the solid body, `cov` = the gate (0..1: solid body in reach).
pub struct BodyRef {
    lum: Vec<f32>,
    cov: Vec<f32>,
    /// Window-sized Edge Tone shift (encoded units), 0 outside the solved soft
    /// band; filled by [`tone_field`] (carried outward by Tone Spread).
    shift: Vec<f32>,
}

/// Body luma at two reaches: near (frame/64 a pass, the local hair level) and
/// wide (frame/8 a pass, ~1/4 of the frame) so flyaway wisps far from the core
/// still find a knee. The near level wins where enough body is close.
fn body_levels(m: &[f32], ml: &[f32], w: usize, h: usize, r: usize) -> (Vec<f32>, Vec<f32>) {
    let (mut m, mut ml) = (m.to_vec(), ml.to_vec());
    for _ in 0..2 {
        box_blur(&mut m, w, h, r);
        box_blur(&mut ml, w, h, r);
    }
    let lum = m.iter().zip(&ml).map(|(&c, &l)| if c > 1e-6 { l / c } else { 0.0 }).collect();
    (lum, m)
}

fn body_reference(rgb_orig: &[f32], alpha: &[f32], w: usize, h: usize, v: &Values, threads: usize) -> BodyRef {
    let n = w * h;
    let mut m = vec![0f32; n];
    let mut ml = vec![0f32; n];
    {
        let (pm, pl) = (SendPtr(m.as_mut_ptr()), SendPtr(ml.as_mut_ptr()));
        par::row_chunks(h, threads, |y0, y1| {
            let (pm, pl) = (pm, pl);
            for i in y0 * w..y1 * w {
                if alpha[i] >= v.pin_high {
                    let c = [0, 1, 2].map(|k| encode(rgb_orig[i * 3 + k], v.input_colorspace).clamp(0.0, 1.0));
                    unsafe {
                        *pm.0.add(i) = 1.0;
                        *pl.0.add(i) = luma_ap1(&c);
                    }
                }
            }
        });
    }
    // Two box passes ~ a soft kernel; radii scale with the frame so a proxy
    // render matches the full one (2K: near 32 px a pass, wide 256 px).
    let (near_l, near_c) = body_levels(&m, &ml, w, h, (w.max(h) / 64).max(4));
    let (wide_l, wide_c) = body_levels(&m, &ml, w, h, (w.max(h) / 8).max(16));
    let mut lum = Vec::with_capacity(n);
    let mut cov = Vec::with_capacity(n);
    for i in 0..n {
        let k = smoothstep(0.02, 0.2, near_c[i]);
        lum.push(wide_l[i] + (near_l[i] - wide_l[i]) * k);
        cov.push(smoothstep(0.0005, 0.01, wide_c[i]));
    }
    BodyRef { lum, cov, shift: Vec::new() }
}

/// Edge Tone shift per window pixel of the solved soft band (pin_low < alpha <
/// pin_high). Each pixel's own correction (the excess beyond the margin, see
/// [`edge_tone_shift`]) is then carried outward with Tone Spread: a max-
/// propagation that decays as exp(-distance / spread), so the strong pull at
/// the dense edge reaches the wisps beyond it instead of stopping in a ring
/// (an averaging blur thinned it out instead). A carried correction only
/// applies where the pixel itself sits on the background's side of the body
/// (never the opposite way) and never moves it past the body level.
fn tone_field(e: &cache::Entry, b: &mut BodyRef, alpha: &[f32], w: usize, h: usize, v: &Values) {
    let (rx1, ry1, rx2, ry2) = e.rect;
    let (cw, ch) = ((rx2 - rx1) as usize, (ry2 - ry1) as usize);
    let n = w * h;
    let mut own = vec![0f32; n];
    // per band pixel: distance to the body level (|e|) and which way it may move (+1 / -1, 0 = none)
    let mut room = vec![0f32; n];
    let mut dir = vec![0i8; n];
    let s = v.edge_tone;
    for yy in ry1.max(0) as usize..(ry2.max(0) as usize).min(h) {
        for xx in rx1.max(0) as usize..(rx2.max(0) as usize).min(w) {
            let gi = yy * w + xx;
            let a = alpha[gi];
            if !(a > v.pin_low && a < v.pin_high) || !e.mask.as_ref().map_or(true, |m| m[gi] != 0) {
                continue;
            }
            let ci = (ch - 1 - (yy - ry1 as usize)) * cw + (xx - rx1 as usize);
            let fl = luma_ap1(&[e.fg.data[ci * 3], e.fg.data[ci * 3 + 1], e.fg.data[ci * 3 + 2]]);
            let bl = luma_ap1(&[e.bg.data[ci * 3], e.bg.data[ci * 3 + 1], e.bg.data[ci * 3 + 2]]);
            own[gi] = edge_tone_shift(fl, bl, b.lum[gi], b.cov[gi], a, s);
            let (ex, d) = (fl - b.lum[gi], bl - b.lum[gi]);
            // roll-off case (brighter edge over a brighter bg) moves down for s > 0, lift case up
            let case = if ex > 0.0 && d > 0.0 { -1i8 } else if ex < 0.0 && d < 0.0 { 1 } else { 0 };
            dir[gi] = if s >= 0.0 { case } else { -case };
            room[gi] = ex.abs() * b.cov[gi].clamp(0.0, 1.0);
        }
    }
    let spread = v.edge_tone_spread.max(0.0);
    if spread < 0.5 {
        b.shift = own;
        return;
    }
    // carried magnitudes, one field per direction
    let mut down: Vec<f32> = own.iter().map(|&k| (-k).max(0.0)).collect();
    let mut up: Vec<f32> = own.iter().map(|&k| k.max(0.0)).collect();
    chamfer_max(&mut down, w, h, spread);
    chamfer_max(&mut up, w, h, spread);
    let mut k = vec![0f32; n];
    for i in 0..n {
        let carried = match dir[i] {
            -1 => -down[i].min(room[i]),
            1 => up[i].min(room[i]),
            _ => 0.0,
        };
        // keep the pixel's own correction where it is larger; fade toward the solid body
        let o = own[i];
        let best = if carried.abs() > o.abs() { carried } else { o };
        k[i] = best * (1.0 - smoothstep(0.95, 1.0, alpha[i]));
    }
    b.shift = k;
}

/// In-place max-propagation with exponential decay (two raster passes, 8-
/// neighbour chamfer): p = max over q of p(q) * exp(-|p - q| / len).
fn chamfer_max(f: &mut [f32], w: usize, h: usize, len: f32) {
    let (f1, f2) = ((-1.0 / len).exp(), (-std::f32::consts::SQRT_2 / len).exp());
    for y in 0..h {
        for x in 0..w {
            let i = y * w + x;
            let mut m = f[i];
            if x > 0 {
                m = m.max(f[i - 1] * f1);
            }
            if y > 0 {
                m = m.max(f[i - w] * f1);
                if x > 0 {
                    m = m.max(f[i - w - 1] * f2);
                }
                if x + 1 < w {
                    m = m.max(f[i - w + 1] * f2);
                }
            }
            f[i] = m;
        }
    }
    for y in (0..h).rev() {
        for x in (0..w).rev() {
            let i = y * w + x;
            let mut m = f[i];
            if x + 1 < w {
                m = m.max(f[i + 1] * f1);
            }
            if y + 1 < h {
                m = m.max(f[i + w] * f1);
                if x + 1 < w {
                    m = m.max(f[i + w + 1] * f2);
                }
                if x > 0 {
                    m = m.max(f[i + w - 1] * f2);
                }
            }
            f[i] = m;
        }
    }
}

/// In-place separable box blur (radius `r`, edge-normalised), f64 running sums.
fn box_blur(p: &mut [f32], w: usize, h: usize, r: usize) {
    let mut tmp = vec![0f32; w.max(h)];
    for y in 0..h {
        let row = &mut p[y * w..(y + 1) * w];
        blur_line(row, &mut tmp[..w], r);
    }
    let mut col = vec![0f32; h];
    for x in 0..w {
        for y in 0..h {
            col[y] = p[y * w + x];
        }
        blur_line(&mut col, &mut tmp[..h], r);
        for y in 0..h {
            p[y * w + x] = col[y];
        }
    }
}

fn blur_line(line: &mut [f32], tmp: &mut [f32], r: usize) {
    let n = line.len();
    let mut pre = 0f64;
    let mut sums = Vec::with_capacity(n + 1);
    sums.push(0f64);
    for &x in line.iter() {
        pre += x as f64;
        sums.push(pre);
    }
    for i in 0..n {
        let (lo, hi) = (i.saturating_sub(r), (i + r + 1).min(n));
        tmp[i] = ((sums[hi] - sums[lo]) / (hi - lo) as f64) as f32;
    }
    line.copy_from_slice(&tmp[..n]);
}

/// Shift (encoded units, add to every channel) that Edge Tone applies to a
/// solved foreground of luma `f_l` over a background of luma `b_l`, with body
/// reference `body` / gate `cov` (0..1), matte `a`, slider `s` (-1..1). Full
/// strength up to alpha 0.95, fading to 0 at 1 (the solid body is the reference).
fn edge_tone_shift(f_l: f32, b_l: f32, body: f32, cov: f32, a: f32, s: f32) -> f32 {
    let g = s.abs() * cov.clamp(0.0, 1.0) * (1.0 - smoothstep(0.95, 1.0, a));
    if g <= 0.0 {
        return 0.0;
    }
    let (e, d) = (f_l - body, b_l - body);
    // the excess beyond the margin on the background's side (signed like d)
    let excess = if e > EDGE_TONE_MARGIN && d > 0.0 {
        (e - EDGE_TONE_MARGIN) * smoothstep(0.0, 0.1, d)
    } else if e < -EDGE_TONE_MARGIN && d < 0.0 {
        (e + EDGE_TONE_MARGIN) * smoothstep(0.0, 0.1, -d)
    } else {
        0.0
    };
    // s > 0: pull the excess back; s < 0: push it out as far again (at -1)
    -excess * g * s.signum()
}

/// Square (box) dilation of a 0/1 mask by `r` pixels, done separably
/// (horizontal max then vertical max), both passes row-parallel and row-major
/// (the vertical pass ORs the 2r+1 source rows into each output row — same
/// set as the column walk, cache-friendly). O(w*h*r) — fine for the small
/// radii used for band growth (default 2).
fn dilate(mask: &[u8], w: usize, h: usize, r: usize, threads: usize) -> Vec<u8> {
    if r == 0 {
        return mask.to_vec();
    }
    let mut tmp = vec![0u8; w * h];
    par::chunks_mut(&mut tmp, par::rows_per_job(h, threads) * w, threads, |ci, chunk| {
        let y_base = ci * par::rows_per_job(h, threads);
        for (ry, trow) in chunk.chunks_mut(w).enumerate() {
            let y = y_base + ry;
            let mrow = &mask[y * w..(y + 1) * w];
            for x in 0..w {
                let x0 = x.saturating_sub(r);
                let x1 = (x + r).min(w - 1);
                trow[x] = mrow[x0..=x1].iter().any(|&m| m != 0) as u8;
            }
        }
    });
    let mut outm = vec![0u8; w * h];
    par::chunks_mut(&mut outm, par::rows_per_job(h, threads) * w, threads, |ci, chunk| {
        let y_base = ci * par::rows_per_job(h, threads);
        for (ry, orow) in chunk.chunks_mut(w).enumerate() {
            let y = y_base + ry;
            let y0 = y.saturating_sub(r);
            let y1 = (y + r).min(h - 1);
            for yy in y0..=y1 {
                let trow = &tmp[yy * w..(yy + 1) * w];
                for x in 0..w {
                    orow[x] |= trow[x];
                }
            }
        }
    });
    outm
}

/// Inclusive bounding box (x1,y1,x2,y2) of the set pixels, or None if empty.
fn bbox(mask: &[u8], w: usize, h: usize, threads: usize) -> Option<(i32, i32, i32, i32)> {
    let acc = Mutex::new((i32::MAX, i32::MAX, i32::MIN, i32::MIN));
    par::row_chunks(h, threads, |y0, y1| {
        let (mut x1, mut ya, mut x2, mut yb) = (i32::MAX, i32::MAX, i32::MIN, i32::MIN);
        for y in y0..y1 {
            let row = &mask[y * w..(y + 1) * w];
            if let Some(first) = row.iter().position(|&m| m != 0) {
                let last = row.iter().rposition(|&m| m != 0).unwrap_or(first);
                x1 = x1.min(first as i32);
                x2 = x2.max(last as i32);
                ya = ya.min(y as i32);
                yb = yb.max(y as i32);
            }
        }
        let mut g = acc.lock().unwrap_or_else(|e| e.into_inner());
        *g = (g.0.min(x1), g.1.min(ya), g.2.max(x2), g.3.max(yb));
    });
    let (x1, y1, x2, y2) = acc.into_inner().unwrap_or_else(|e| e.into_inner());
    if x2 < x1 {
        None
    } else {
        Some((x1, y1, x2, y2))
    }
}

// ACEScg (AP1) luminance weights. The chroma axis is built to have zero luma
// under these weights, so the despill subtraction preserves luminance exactly.
const LUMA_AP1: [f32; 3] = [0.272_229, 0.674_082, 0.053_689_5];

#[inline]
fn luma_ap1(c: &[f32; 3]) -> f32 {
    LUMA_AP1[0] * c[0] + LUMA_AP1[1] * c[1] + LUMA_AP1[2] * c[2]
}

/// Chroma (colour minus its luminance-grey). Has zero luma under LUMA_AP1.
#[inline]
fn chroma_ap1(c: &[f32; 3]) -> [f32; 3] {
    let y = luma_ap1(c);
    [c[0] - y, c[1] - y, c[2] - y]
}

#[inline]
fn smoothstep(e0: f32, e1: f32, x: f32) -> f32 {
    let t = ((x - e0) / (e1 - e0)).clamp(0.0, 1.0);
    t * t * (3.0 - 2.0 * t)
}

#[inline]
fn lerp(a: f32, b: f32, t: f32) -> f32 {
    a + (b - a) * t
}

/// Single-pass "edge detail" measure over the soft band (`lo < alpha < hi`): the
/// mean local image-gradient magnitude relative to mean band luma (scale-invariant).
/// Intricate edges (hair) → high; smooth transitions → low. EXPERIMENTAL — the
/// scale constant is a first guess to tune against real shots.
fn measure_band_detail(rgb: &[f32], alpha: &[f32], w: usize, h: usize, lo: f32, hi: f32) -> f32 {
    let lum = |i: usize| luma_ap1(&[rgb[i * 3], rgb[i * 3 + 1], rgb[i * 3 + 2]]) as f64;
    let (mut sum_l, mut sum_g, mut n) = (0.0f64, 0.0f64, 0u64);
    for y in 0..h {
        for x in 0..w {
            let i = y * w + x;
            let a = alpha[i];
            if a > lo && a < hi {
                let l = lum(i);
                let lx = if x + 1 < w { lum(i + 1) } else { l };
                let ly = if y + 1 < h { lum(i + w) } else { l };
                sum_l += l;
                sum_g += (lx - l).abs() + (ly - l).abs();
                n += 1;
            }
        }
    }
    if n == 0 {
        return 0.0;
    }
    let mean_l = (sum_l / n as f64).max(1e-4);
    let mean_g = sum_g / n as f64;
    ((mean_g / mean_l) * 4.0).clamp(0.0, 1.0) as f32 // SCALE=4.0, experimental
}

/// Map band edge-detail → (regularization, gradient_weight). More detail = follow
/// the image (lower reg, higher gradient weight); smoother = more regularization.
/// Conservative nudges around the pymatting defaults. EXPERIMENTAL.
fn auto_solve_params(detail: f32) -> (f32, f32) {
    let d = detail.clamp(0.0, 1.0);
    (lerp(3e-5, 3e-6, d), lerp(0.7, 1.6, d))
}

/// Hue-targeted despill by chroma-axis projection. `screen`: 0 = none, 1 = green,
/// 2 = blue, 3 = red. Builds the screen hue's *chroma axis* (a zero-luminance
/// direction in RGB), measures how much the pixel's chroma points along it
/// (`proj`) and how close the pixel's hue is to the screen hue (`w`, gated by
/// `range`/`softness`), and removes `strength*w*proj` along that axis. Because the
/// axis has zero luma, **luminance is preserved by construction** — warm/blonde
/// hair keeps its brightness; and off-hue colours (skin) get a low `w` so they're
/// spared. No channel clamp.
fn apply_despill(
    rgb: &mut [f32; 3],
    screen: i32,
    custom: [f32; 3],
    strength: f32,
    range: f32,
    softness: f32,
) {
    if strength <= 0.0 || screen == 0 {
        return;
    }
    let screen_rgb = match screen {
        2 => [0.0, 0.0, 1.0],
        3 => [1.0, 0.0, 0.0],
        4 => custom, // exact picked screen colour
        _ => [0.0, 1.0, 0.0],
    };
    let sc = chroma_ap1(&screen_rgb);
    let sc_len = (sc[0] * sc[0] + sc[1] * sc[1] + sc[2] * sc[2]).sqrt();
    // A (near-)neutral pick has no hue to target. Relative test: the AP1 luma
    // weights sum to 1 only to ~5e-7, so a grey/white pick leaves float-noise
    // chroma that an absolute 1e-8 floor let through as a random axis.
    let peak = screen_rgb.iter().fold(0.0f32, |m, c| m.max(c.abs()));
    if sc_len < 1e-8 || sc_len < 1e-3 * peak {
        return;
    }
    let axis = [sc[0] / sc_len, sc[1] / sc_len, sc[2] / sc_len]; // unit, zero-luma

    let pc = chroma_ap1(rgb);
    let pc_len = (pc[0] * pc[0] + pc[1] * pc[1] + pc[2] * pc[2]).sqrt();
    if pc_len < 1e-5 {
        return; // neutral pixel — no hue to despill
    }
    let proj = pc[0] * axis[0] + pc[1] * axis[1] + pc[2] * axis[2];
    if proj <= 0.0 {
        return; // chroma points away from the screen hue — not spill
    }
    // Hue-band membership: cos angle between pixel chroma and the screen axis.
    let cos = proj / pc_len; // [-1, 1]; 1 = exactly the screen hue
    let hi = 1.0 - range.clamp(0.0, 1.0); // range 0 → only the exact hue; 1 → wide band
    let lo = (hi - softness.clamp(0.0, 1.0).max(1e-3)).max(-1.0);
    let w = smoothstep(lo, hi, cos);
    if w <= 0.0 {
        return;
    }
    let k = strength.clamp(0.0, 1.0) * w * proj;
    rgb[0] -= k * axis[0];
    rgb[1] -= k * axis[1];
    rgb[2] -= k * axis[2];
}

// ---------------------------------------------------------------------------
// Input colour space → solver working space. `mode`: 0 = ACEScg (scene-linear,
// encoded to ACEScct for the solve and decoded back), 1 = ACEScct, 2 = Display
// (sRGB) — the latter two are already perceptual and pass through unchanged.
// ---------------------------------------------------------------------------
fn encode(v: f32, mode: i32) -> f32 {
    match mode {
        0 => lin_to_acescct(v),
        _ => v,
    }
}

fn decode(v: f32, mode: i32) -> f32 {
    match mode {
        0 => acescct_to_lin(v),
        _ => v,
    }
}

/// ACES ACEScct transfer (ACES spec). Linear segment below the breakpoint keeps
/// it finite for 0/negative AP1 values.
fn lin_to_acescct(x: f32) -> f32 {
    if x <= 0.0078125 {
        10.540_238 * x + 0.072_905_53
    } else {
        (x.log2() + 9.72) / 17.52
    }
}
fn acescct_to_lin(y: f32) -> f32 {
    if y <= 0.155_251_14 {
        (y - 0.072_905_53) / 10.540_238
    } else {
        2.0_f32.powf(y * 17.52 - 9.72)
    }
}

/// Pre-solve alpha self-multiplication of one value. Mirrors `_alpha_mul_self`
/// (vitmatte_server.py): mode 0 = linear, mode 1 = body-protect. A `strength`
/// of <=1 is a single pass (exact parity with the offline tool); above 1 the
/// pass repeats (the integer part as full self-multiplies, the remainder as a
/// final partial pass), i.e. "keep multiplying alpha by itself" for a stronger
/// edge choke. (Element-wise, so per-value == v0.2.0's per-array passes.)
fn self_mul_value(alpha: f32, strength: f32, mode: i32) -> f32 {
    let mut v = alpha;
    let mut remaining = strength;
    while remaining > 0.0 {
        let step = remaining.min(1.0);
        let next = if mode == 1 {
            v * (1.0 - step * (1.0 - v).powi(2))
        } else {
            v * (1.0 - step) + (v * v) * step
        };
        v = next.clamp(0.0, 1.0);
        remaining -= 1.0;
    }
    v
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn acescct_roundtrip() {
        // covers shadow, mid (18% grey), white, and HDR highlight
        for &x in &[0.0f32, 0.001, 0.0078125, 0.18, 1.0, 8.0] {
            let r = acescct_to_lin(lin_to_acescct(x));
            assert!((r - x).abs() <= 1e-4 + x.abs() * 1e-3, "acescct rt {x} -> {r}");
        }
        // 18% grey lands mid-range (well inside [0,1]) — the point of solving in log
        let g = lin_to_acescct(0.18);
        assert!(g > 0.39 && g < 0.43, "acescct(0.18)={g}");
        // linear 1.0 stays < 1 in ACEScct (no clip of normal range)
        assert!(lin_to_acescct(1.0) < 1.0);
    }

    #[test]
    fn despill_preserves_luma_and_removes_green() {
        let mut p = [0.3f32, 0.6, 0.3]; // pure-green-hue spill
        let y0 = luma_ap1(&p);
        apply_despill(&mut p, 1, [0.0,1.0,0.0], 1.0, 0.8, 0.5);
        assert!(p[1] < 0.6, "green should drop: {p:?}");
        // the headline invariant: luminance unchanged at any strength
        assert!((luma_ap1(&p) - y0).abs() < 1e-4, "luma preserved: {} vs {y0}", luma_ap1(&p));
    }

    #[test]
    fn auto_solve_params_monotonic_bounded() {
        let (r0, g0) = auto_solve_params(0.0);
        let (r1, g1) = auto_solve_params(1.0);
        assert!(r0 > r1, "more detail -> less regularization");
        assert!(g1 > g0, "more detail -> more gradient weight");
        for &d in &[0.0f32, 0.5, 1.0] {
            let (r, g) = auto_solve_params(d);
            assert!((3e-6..=3e-5).contains(&r), "reg bounded: {r}");
            assert!((0.7..=1.6).contains(&g), "gw bounded: {g}");
        }
    }

    #[test]
    fn acescg_path_beats_raw_linear_solve() {
        // The Flare finding: solving raw scene-linear data corrupts
        // the multilevel solve — sparse bright pixels (hair rims / speculars)
        // dominate its mean-based seeds and coarse levels, producing large
        // blotchy FG errors in weakly-constrained regions. The ACEScg path
        // (lin→ACEScct encode) must recover the dark foreground where the raw
        // solve does not. Scenario: near-black subject, green backing, wide
        // blur band, sparse 2.5–4.5 rim pixels, faint AI-matte noise in the BG.
        use crate::solver::{estimate_fb_ml, FgParams, Plane};
        let (h, w) = (256usize, 512usize);
        let f_true = [0.03f32, 0.02, 0.025];
        let b_base = [0.35f32, 0.62, 0.33];

        let mut seed = 12345u32;
        let mut rnd = move || {
            seed = seed.wrapping_mul(1664525).wrapping_add(1013904223);
            (seed >> 16) as f32 / 65536.0
        };

        let mut lin = vec![0.0f32; h * w * 3];
        let mut alpha = vec![0.0f32; h * w];
        for y in 0..h {
            for x in 0..w {
                let a = if x < 60 {
                    1.0
                } else if x >= 260 {
                    0.0
                } else {
                    1.0 - (x - 60) as f32 / 200.0
                };
                let i = y * w + x;
                alpha[i] = if a > 0.0 { a } else { 0.003 + 0.009 * rnd() };
                let tex = 0.97 + 0.06 * rnd();
                for c in 0..3 {
                    lin[i * 3 + c] = a * f_true[c] + (1.0 - a) * b_base[c] * tex;
                }
                if x < 60 && rnd() < 0.02 {
                    for c in 0..3 {
                        lin[i * 3 + c] = 2.5 + 2.0 * rnd(); // rim-light outliers
                    }
                }
            }
        }

        let norm = lin.iter().cloned().fold(1.0f32, f32::max);
        let band_err = |use_acescg_path: bool| -> f32 {
            let mut image = Plane::new(h, w, 3);
            for i in 0..(h * w) {
                for c in 0..3 {
                    let v = lin[i * 3 + c];
                    image.data[i * 3 + c] = if use_acescg_path {
                        encode(v, 0).clamp(0.0, 1.0)
                    } else {
                        (v / norm).clamp(0.0, 1.0) // the removed max-normalise mode
                    };
                }
            }
            let (fg, _bg) = estimate_fb_ml(&image, &alpha, &FgParams::default());
            let mut max_err = 0.0f32;
            for y in (20..h - 20).step_by(8) {
                for x in 80..240 {
                    let i = y * w + x;
                    for c in 0..3 {
                        let f = if use_acescg_path {
                            decode(fg.data[i * 3 + c], 0)
                        } else {
                            fg.data[i * 3 + c] * norm
                        };
                        max_err = max_err.max((f - f_true[c]).abs());
                    }
                }
            }
            max_err
        };

        let err_acescg = band_err(true);
        let err_raw = band_err(false);
        assert!(
            err_acescg < err_raw,
            "ACEScg path ({err_acescg}) must beat raw linear ({err_raw})"
        );
        // Loose absolute bound: the band's low-alpha end is weakly constrained,
        // so max error stays well above the solid-region tolerance.
        assert!(
            err_acescg < 0.1,
            "ACEScg-path band recovery too loose: {err_acescg} (f_true ~0.02)"
        );
    }

    // v0.2.0's column-major dilate, kept as the oracle for the row-wise one.
    fn dilate_ref(mask: &[u8], w: usize, h: usize, r: usize) -> Vec<u8> {
        let mut tmp = vec![0u8; w * h];
        for y in 0..h {
            for x in 0..w {
                let (x0, x1) = (x.saturating_sub(r), (x + r).min(w - 1));
                tmp[y * w + x] = (x0..=x1).any(|xx| mask[y * w + xx] != 0) as u8;
            }
        }
        let mut outm = vec![0u8; w * h];
        for x in 0..w {
            for y in 0..h {
                let (y0, y1) = (y.saturating_sub(r), (y + r).min(h - 1));
                outm[y * w + x] = (y0..=y1).any(|yy| tmp[yy * w + x] != 0) as u8;
            }
        }
        outm
    }

    #[test]
    fn dilate_and_bbox_match_reference() {
        let (w, h) = (97usize, 61usize);
        let mut seed = 7u32;
        let mask: Vec<u8> = (0..w * h)
            .map(|_| {
                seed = seed.wrapping_mul(1664525).wrapping_add(1013904223);
                ((seed >> 24) < 6) as u8
            })
            .collect();
        for &r in &[1usize, 2, 5, 40] {
            for &t in &[1usize, 3, 8] {
                assert_eq!(dilate(&mask, w, h, r, t), dilate_ref(&mask, w, h, r), "r={r} t={t}");
            }
        }
        let mut one = vec![0u8; w * h];
        one[13 * w + 42] = 1;
        one[50 * w + 3] = 1;
        assert_eq!(bbox(&one, w, h, 4), Some((3, 13, 42, 50)));
        assert_eq!(bbox(&vec![0u8; w * h], w, h, 4), None);
    }

    fn test_values() -> Values {
        Values {
            self_mul_strength: 0.0,
            self_mul_mode: 1,
            soft_band_only: true,
            band_dilation: 2,
            pin_hard: true,
            pin_low: 0.002,
            pin_high: 0.998,
            regularization: 1e-5,
            quality: 2,
            gradient_weight: 1.0,
            image_affinity: 0.0,
            auto_solve: false,
            input_colorspace: 0,
            roi_padding: 64,
            output_mode: 0,
            despill_screen: 1,
            residual_despill: 0.0,
            despill_range: 0.5,
            despill_softness: 0.5,
            screen_colour: [0.0, 1.0, 0.0],
            despill_soft_only: false,
            despill_matte_thresh: 1.0,
            edge_tone: 0.0,
            edge_tone_spread: 0.0,
        }
    }

    /// Synthetic RGBA source (OFX layout) with a soft band, plus a renderer.
    struct Rig {
        w: usize,
        h: usize,
        src: Vec<f32>,
    }
    impl Rig {
        fn new(w: usize, h: usize) -> Self {
            let mut src = vec![0f32; w * h * 4];
            let mut seed = 99u32;
            for y in 0..h {
                for x in 0..w {
                    seed = seed.wrapping_mul(1664525).wrapping_add(1013904223);
                    let n = (seed >> 16) as f32 / 65536.0;
                    let r = (((x as f32 - w as f32 * 0.5).powi(2) + (y as f32 - h as f32 * 0.4).powi(2)).sqrt())
                        / (w as f32 * 0.3);
                    let a = (1.5 - r).clamp(0.0, 1.0);
                    let i = (y * w + x) * 4;
                    src[i] = a * 0.8 + (1.0 - a) * 0.1 + 0.02 * n;
                    src[i + 1] = a * 0.3 + (1.0 - a) * 0.7;
                    src[i + 2] = a * 0.2 + (1.0 - a) * 0.15 * n;
                    src[i + 3] = a;
                }
            }
            Rig { w, h, src }
        }
        fn render(&mut self, v: &Values, ctx: Option<CacheCtx>, threads: usize) -> (Vec<u32>, RenderInfo) {
            let win = OfxRectI { x1: 0, y1: 0, x2: self.w as i32, y2: self.h as i32 };
            let mut out = vec![0f32; self.w * self.h * 4];
            let mk = |p: *mut f32| View { data: p as *mut u8, bounds: win, row_bytes: (self.w * 16) as i32, n_comps: 4 };
            let (sv, ov) = (mk(self.src.as_mut_ptr()), mk(out.as_mut_ptr()));
            let (_, info) = solve_window(&sv, &ov, &win, v, ctx, threads);
            (out.iter().map(|x| x.to_bits()).collect(), info)
        }
    }

    #[test]
    fn render_is_thread_count_invariant() {
        let mut rig = Rig::new(400, 260);
        for sb in [true, false] {
            for om in 0..4 {
                let v = Values { soft_band_only: sb, output_mode: om, residual_despill: 0.6, ..test_values() };
                let (base, _) = rig.render(&v, None, 1);
                for t in [2usize, 4, 16] {
                    assert!(rig.render(&v, None, t).0 == base, "sb={sb} om={om} threads={t}");
                }
            }
        }
    }

    #[test]
    fn cache_reuses_solve_only_for_output_param_changes() {
        let mut rig = Rig::new(400, 260);
        let ctx = Some(CacheCtx { instance: 0xC0DE_0001, time: 12.0 });
        let v = test_values();
        let (_, i0) = rig.render(&v, ctx, 4);
        assert!(!i0.cache_hit);
        // Output-only edits: hit, and identical to a fresh uncached render.
        let edits = [
            Values { residual_despill: 0.8, ..v },
            Values { residual_despill: 0.8, despill_soft_only: true, despill_matte_thresh: 0.7, ..v },
            Values { output_mode: 3, ..v },
            Values { output_mode: 2, ..v },
            Values { despill_screen: 4, screen_colour: [0.1, 0.8, 0.3], residual_despill: 1.0, ..v },
        ];
        for e in &edits {
            let (hit, ih) = rig.render(e, ctx, 4);
            assert!(ih.cache_hit, "output-only change must hit");
            assert!(hit == rig.render(e, None, 4).0, "cache hit must equal a fresh render");
        }
        // Solve-affecting edits miss.
        for e in [Values { pin_low: 0.01, ..v }, Values { band_dilation: 3, ..v }, Values { input_colorspace: 1, ..v }] {
            assert!(!rig.render(&e, ctx, 4).1.cache_hit, "solve param change must miss");
        }
        // Other frame time misses; changed upstream pixels miss.
        assert!(!rig.render(&v, Some(CacheCtx { instance: 0xC0DE_0001, time: 13.0 }), 4).1.cache_hit);
        rig.src[(130 * 400 + 200) * 4] += 0.001;
        assert!(!rig.render(&v, ctx, 4).1.cache_hit, "content change must miss");
        cache::purge_instance(0xC0DE_0001);
    }

    // Regression (Flame 2027.2, Linux): Custom Colour despill at
    // strength 1 "did nothing" on a cwFill node with the Matte socket
    // unconnected. A matte with no soft band (constant / all-solid) took a
    // plate-passthrough shortcut that skipped the hard-region output, so the
    // body was never despilled. Must despill exactly like any solid pixel.
    #[test]
    fn despill_applies_when_matte_has_no_soft_band() {
        let (w, h) = (64usize, 48usize);
        let blue = [0.249f32, 0.538, 0.720]; // the picked colour from the live setup
        let mut rig = Rig::new(w, h);
        for i in 0..w * h {
            let n = (i % 7) as f32 * 0.01;
            rig.src[i * 4..i * 4 + 4].copy_from_slice(&[blue[0] + n, blue[1], blue[2] - n, 1.0]);
        }
        let v = Values { despill_screen: 4, screen_colour: blue, residual_despill: 1.0, ..test_values() };
        for sb in [true, false] {
            let v = Values { soft_band_only: sb, ..v };
            let (out, _) = rig.render(&v, None, 1);
            for i in 0..w * h {
                let mut want = [rig.src[i * 4], rig.src[i * 4 + 1], rig.src[i * 4 + 2]];
                apply_despill(&mut want, 4, blue, 1.0, v.despill_range, v.despill_softness);
                let got = [f32::from_bits(out[i * 4]), f32::from_bits(out[i * 4 + 1]), f32::from_bits(out[i * 4 + 2])];
                assert_eq!(got, want, "sb={sb} px {i}");
                assert!((got[2] - rig.src[i * 4 + 2]).abs() > 0.05, "screen hue must be pulled");
            }
            // Contamination of an all-hard frame is zero, not the plate.
            let (c, _) = rig.render(&Values { output_mode: 2, ..v }, None, 1);
            if sb {
                assert!(c.chunks(4).all(|p| p[..3].iter().all(|&b| f32::from_bits(b) == 0.0)));
            }
        }
    }

    // ---- half-float I/O (16-bit in = 16-bit out) ----

    #[test]
    fn half_roundtrip_is_exact_for_every_finite_half() {
        for h in 0..=u16::MAX {
            let f = f16_to_f32(h);
            if f.is_nan() {
                assert!(f32_to_f16(f) & 0x7c00 == 0x7c00 && f32_to_f16(f) & 0x3ff != 0, "{h:#06x} NaN");
                continue;
            }
            assert_eq!(f32_to_f16(f), h, "{h:#06x} -> {f}");
        }
    }

    #[test]
    fn f32_to_half_rounds_to_nearest_even() {
        for (f, h) in [
            (1.0f32, 0x3c00u16), (-2.0, 0xc000), (0.1, 0x2e66), (65504.0, 0x7bff), (65519.0, 0x7bff),
            (65520.0, 0x7c00), (1e9, 0x7c00), (5.960_464_5e-8, 0x0001), (2.0e-8, 0x0000), (0.0, 0x0000),
            (-0.0, 0x8000), (f32::INFINITY, 0x7c00), (f32::NEG_INFINITY, 0xfc00),
        ] {
            assert_eq!(f32_to_f16(f), h, "{f}");
        }
        // Midpoints between neighbouring halves (exact in f32) go to the even one,
        // normal and subnormal ranges alike.
        for h in (1u16..0x7bff).step_by(7) {
            let mid = (f16_to_f32(h) + f16_to_f32(h + 1)) * 0.5;
            assert_eq!(f32_to_f16(mid), if h & 1 == 0 { h } else { h + 1 }, "mid {h:#06x}");
        }
    }

    #[test]
    fn half_buffers_render_like_float_rounded_once() {
        // The same frame as half in / half out must equal the float render of the
        // half-decoded source, rounded to half once (no f32 read of half bits).
        let (w, h) = (160usize, 120usize);
        let mut rig = Rig::new(w, h);
        for x in rig.src.iter_mut() {
            *x = f16_to_f32(f32_to_f16(*x));
        }
        let v = Values { residual_despill: 0.5, edge_tone: 0.7, ..test_values() };
        let (want, _) = rig.render(&v, None, 2);
        let win = OfxRectI { x1: 0, y1: 0, x2: w as i32, y2: h as i32 };
        let mut src16: Vec<u16> = rig.src.iter().map(|&x| f32_to_f16(x)).collect();
        let mut out16 = vec![0u16; w * h * 4];
        let mk = |p: *mut u16| View { data: p as *mut u8, bounds: win, row_bytes: (w * 8) as i32, n_comps: 4 };
        let (sv, ov) = (mk(src16.as_mut_ptr()), mk(out16.as_mut_ptr()));
        let (_sb, s32) = half_to_f32(&sv, 3);
        let (_ob, o32) = f32_scratch(&ov);
        solve_window(&s32, &o32, &win, &v, None, 2);
        f32_to_half_window(&o32, &ov, &win, 3);
        for i in 0..w * h * 4 {
            assert_eq!(out16[i], f32_to_f16(f32::from_bits(want[i])), "component {i}");
        }
    }

    // ---- Edge Tone ----

    #[test]
    fn edge_tone_rolls_off_over_bright_and_lifts_over_dark_only() {
        let m = EDGE_TONE_MARGIN;
        // bright background, edge brighter than the body: pulled down to the margin at s=1
        let k = edge_tone_shift(0.70, 0.95, 0.40, 1.0, 0.5, 1.0);
        assert!((0.70 + k - (0.40 + m)).abs() < 1e-6, "{k}");
        // dark background, edge darker than the body: lifted to the margin
        let k = edge_tone_shift(0.20, 0.05, 0.50, 1.0, 0.5, 1.0);
        assert!((0.20 + k - (0.50 - m)).abs() < 1e-6, "{k}");
        // never the other way: dark edge over a bright background, bright edge over a dark one
        assert_eq!(edge_tone_shift(0.20, 0.95, 0.50, 1.0, 0.5, 1.0), 0.0);
        assert_eq!(edge_tone_shift(0.70, 0.05, 0.40, 1.0, 0.5, 1.0), 0.0);
        // inside the margin, without a body in reach, on solid pixels, or at 0: untouched
        assert_eq!(edge_tone_shift(0.40 + m * 0.9, 0.95, 0.40, 1.0, 0.5, 1.0), 0.0);
        assert_eq!(edge_tone_shift(0.70, 0.95, 0.40, 0.0, 0.5, 1.0), 0.0);
        assert_eq!(edge_tone_shift(0.70, 0.95, 0.40, 1.0, 1.0, 1.0), 0.0);
        assert_eq!(edge_tone_shift(0.70, 0.95, 0.40, 1.0, 0.5, 0.0), 0.0);
        // half strength = half the pull
        let full = edge_tone_shift(0.70, 0.95, 0.40, 1.0, 0.5, 1.0);
        assert!((edge_tone_shift(0.70, 0.95, 0.40, 1.0, 0.5, 0.5) - full * 0.5).abs() < 1e-6);
        // negative slider: the same excess pushed the other way, as far at -1
        assert!((edge_tone_shift(0.70, 0.95, 0.40, 1.0, 0.5, -1.0) + full).abs() < 1e-6);
        assert!(edge_tone_shift(0.20, 0.05, 0.50, 1.0, 0.5, -1.0) < 0.0); // darker edge over dark: deeper
        assert_eq!(edge_tone_shift(0.20, 0.95, 0.50, 1.0, 0.5, -1.0), 0.0); // never the opposite case
        // full strength up to alpha 0.95 (near-opaque edges included)
        assert_eq!(edge_tone_shift(0.70, 0.95, 0.40, 1.0, 0.95, 1.0), full);
    }

    #[test]
    fn tone_spread_carries_with_exponential_decay() {
        let (w, h) = (101usize, 61usize);
        let mut f = vec![0f32; w * h];
        f[30 * w + 50] = 1.0;
        chamfer_max(&mut f, w, h, 10.0);
        let at = |x: usize, y: usize| f[y * w + x];
        assert_eq!(at(50, 30), 1.0);
        assert!((at(60, 30) - (-1.0f32).exp()).abs() < 1e-5); // 10 px straight: e^-1
        assert!((at(80, 30) - (-3.0f32).exp()).abs() < 1e-5);
        assert!((at(60, 40) - (-std::f32::consts::SQRT_2).exp()).abs() < 1e-5); // diagonal
        assert!(at(0, 0) > 0.0 && at(0, 0) < at(40, 30)); // monotone with distance
    }

    #[test]
    fn edge_tone_body_reaches_far_wisps() {
        // 2K frame, solid body on the left half: a wisp 150 px out still gets a
        // full gate and the body's level as its knee (the old 64 px reach gave 0).
        let (w, h) = (2048usize, 1080usize);
        let mut rgb = vec![0.0f32; w * h * 3];
        let mut alpha = vec![0.0f32; w * h];
        for y in 0..h {
            for x in 0..1000 {
                alpha[y * w + x] = 1.0;
                rgb[(y * w + x) * 3..(y * w + x) * 3 + 3].copy_from_slice(&[0.05, 0.05, 0.05]);
            }
        }
        let v = test_values();
        let b = body_reference(&rgb, &alpha, w, h, &v, 4);
        let body_l = luma_ap1(&[0.05f32, 0.05, 0.05].map(|c| encode(c, v.input_colorspace).clamp(0.0, 1.0)));
        let i = 540 * w + 1150;
        assert!(b.cov[i] > 0.99, "gate {}", b.cov[i]);
        assert!((b.lum[i] - body_l).abs() < 1e-4, "knee {} vs body {body_l}", b.lum[i]);
        assert!(b.cov[540 * w + 1450] > 0.0); // 450 px out: still in the ~512 px (1/4 frame) reach
        assert_eq!(b.cov[540 * w + 2047], 0.0); // beyond it: untouched
    }

    #[test]
    fn edge_tone_is_output_only_and_darkens_fill_over_a_bright_background() {
        // Dark body over a near-white background: fill in the band reads brighter
        // than the body; Edge Tone must bring it down, hit the cache, leave the
        // hard regions and alpha alone, and be thread-count invariant.
        let (w, h) = (200usize, 140usize);
        let mut rig = Rig::new(w, h);
        for y in 0..h {
            for x in 0..w {
                let i = (y * w + x) * 4;
                // 12 px soft band around a solid disc (r = 40 px)
                let d = ((x as f32 - 100.0).powi(2) + (y as f32 - 70.0).powi(2)).sqrt();
                let a = ((46.0 - d) / 12.0).clamp(0.0, 1.0);
                rig.src[i + 3] = a;
                let (f, b) = (0.05f32, 4.0f32);
                for c in 0..3 {
                    rig.src[i + c] = a * f + (1.0 - a) * b + if a > 0.0 && a < 1.0 { 0.6 * a * (1.0 - a) } else { 0.0 };
                }
            }
        }
        let ctx = Some(CacheCtx { instance: 0xC0DE_0002, time: 1.0 });
        let v0 = test_values();
        let v1 = Values { edge_tone: 1.0, ..v0 };
        let (o0, i0) = rig.render(&v0, ctx, 4);
        let (o1, i1) = rig.render(&v1, ctx, 4);
        assert!(!i0.cache_hit && i1.cache_hit, "Edge Tone must not re-solve");
        assert!(o1 == rig.render(&v1, None, 1).0, "thread invariant + cache hit == fresh");
        let (mut darker, mut band) = (0, 0);
        for p in 0..w * h {
            let a = rig.src[p * 4 + 3];
            let g = |o: &Vec<u32>, c: usize| f32::from_bits(o[p * 4 + c]);
            assert_eq!(o0[p * 4 + 3], o1[p * 4 + 3], "alpha carried");
            if a <= v0.pin_low || a >= v0.pin_high {
                assert_eq!(&o0[p * 4..p * 4 + 3], &o1[p * 4..p * 4 + 3], "hard pixel {p} touched");
            } else if a > 0.2 && a < 0.8 {
                band += 1;
                assert!(g(&o1, 1) <= g(&o0, 1) + 1e-6, "edge tone brightened px {p}");
                darker += (g(&o1, 1) < g(&o0, 1) * 0.98) as usize;
            }
        }
        assert!(band > 0 && darker * 2 > band, "most of the band darker: {darker}/{band}");
        // Tone Spread carries the correction outward (more of the band touched)
        // without raising its peak, still output-only and soft-band only.
        let v2 = Values { edge_tone_spread: 32.0, ..v1 };
        let (o2, i2) = rig.render(&v2, ctx, 4);
        assert!(i2.cache_hit, "Tone Spread must not re-solve");
        let (mut touched1, mut touched2, mut peak1, mut peak2) = (0, 0, 0f32, 0f32);
        for p in 0..w * h {
            let a = rig.src[p * 4 + 3];
            let g = |o: &Vec<u32>| f32::from_bits(o[p * 4 + 1]);
            let (d1, d2) = ((g(&o1) - g(&o0)).abs(), (g(&o2) - g(&o0)).abs());
            if a <= v0.pin_low || a >= v0.pin_high {
                assert_eq!(&o0[p * 4..p * 4 + 3], &o2[p * 4..p * 4 + 3], "hard pixel {p} touched (spread)");
                continue;
            }
            touched1 += (d1 > 1e-6) as usize;
            touched2 += (d2 > 1e-6) as usize;
            peak1 = peak1.max(d1);
            peak2 = peak2.max(d2);
        }
        assert!(touched2 >= touched1 && peak2 <= peak1 + 1e-6, "spread {touched2} px peak {peak2} vs {touched1} px peak {peak1}");
        cache::purge_instance(0xC0DE_0002);
    }

    #[test]
    fn neutral_custom_pick_is_a_no_op() {
        for pick in [[1.0f32, 1.0, 1.0], [0.18, 0.18, 0.18], [0.0, 0.0, 0.0]] {
            let mut p = [0.3f32, 0.6, 0.3];
            apply_despill(&mut p, 4, pick, 1.0, 1.0, 1.0);
            assert_eq!(p, [0.3, 0.6, 0.3], "pick {pick:?}");
        }
    }

    #[test]
    fn despill_spares_offhue_and_neutral() {
        // warm skin-ish vs green screen → chroma points away → untouched
        let mut skin = [0.6f32, 0.35, 0.25];
        let before = skin;
        apply_despill(&mut skin, 1, [0.0,1.0,0.0], 1.0, 0.5, 0.5);
        let dz: f32 = (0..3).map(|i| (skin[i] - before[i]).abs()).sum();
        assert!(dz < 1e-4, "skin spared: {before:?} -> {skin:?}");
        // neutral grey has no hue → untouched even at full range/strength
        let mut grey = [0.5f32, 0.5, 0.5];
        apply_despill(&mut grey, 1, [0.0,1.0,0.0], 1.0, 1.0, 1.0);
        assert_eq!(grey, [0.5, 0.5, 0.5]);
    }
}
