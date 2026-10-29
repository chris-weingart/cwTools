//! Render: Source (+ Seed, + Trimap) → engine → Output, per the Output view.
//!
//! Failure policy: never hang, never crash. Any model/remote error passes the
//! plate through with the UNREFINED seed as alpha, posts a node message and
//! logs to /tmp/cwRefine.log. Render returns OK so Flame keeps working.

use crate::engine::{self, Prepared};
use crate::params::{self, Values, VIEW_CHANNELS, VIEW_CROP, VIEW_DIFF, VIEW_EDIT, VIEW_MATTE, VIEW_PREMULT, VIEW_TRIMAP};
use crate::spline;
use crate::recipe::CropPlan;
use crate::TRIMAP_CLIP_NAME;
use cwtools_core::color::to_u8;
use cwtools_core::image::{clip_connected, fetch_image, release, Image};
use cwtools_core::ofx_api::*;
use cwtools_core::props::{prop_get_double, prop_get_rect_i, prop_try_double};
use cwtools_core::suites::{aborted, effect_param_set};
use cwtools_core::{log, set_node_message};

/// Full-frame (source bounds) matte as u8, top-down. Reads `ch` of `img`.
fn matte_u8(img: &Image, ch: usize, sb: OfxRectI) -> Vec<u8> {
    img.read_channel(ch, sb.x1, sb.y1, sb.x2, sb.y2, 0.0).into_iter().map(|x| to_u8(x.clamp(0.0, 1.0))).collect()
}

enum Refined {
    /// Full-frame refined alpha (float, 0..1).
    Alpha(Vec<f32>),
    /// Nothing to refine (empty seed): alpha 0.
    Empty,
    /// Failure: message; output falls back to the seed.
    Failed(String),
}

pub fn render(eh: OfxImageEffectHandle, in_args: OfxPropertySetHandle) -> OfxStatus {
    let time = prop_get_double(in_args, kOfxPropTime, 0);
    let win = prop_get_rect_i(in_args, kOfxImageEffectPropRenderWindow);
    let rs = prop_try_double(in_args, kOfxImageEffectPropRenderScale, 0).unwrap_or(1.0);
    let ps = effect_param_set(eh);
    let mut v = params::read_values(ps, time).scaled(rs);
    // Edit Splines: fast preview (plate + bands), never inference.
    if v.spline_edit && v.spline.enabled {
        v.view = VIEW_EDIT;
    }
    // Result RGB = Channel Pickers: RGB from the pickers, Matte Out = refined.
    let channels = v.view == VIEW_CHANNELS;

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
    let sb = src.bounds;
    let (w, h) = (src.w(), src.h());

    // ── input mask: always the Source alpha socket ──────────────────────
    let mut note: Option<String> = None;
    let mut seed: Vec<u8> = if src.n_comps == 4 {
        matte_u8(&src, 3, sb)
    } else {
        note = Some("cwRefine: Source has no alpha - connect the input mask to the Source Alpha socket".into());
        vec![0u8; w * h]
    };
    let seed_in = seed.clone(); // as fed by the user (diff view, failure fallback)

    // ── trimap clip ─────────────────────────────────────────────────────
    let trimap: Option<Vec<u8>> = if v.use_trimap_clip && clip_connected(eh, TRIMAP_CLIP_NAME) {
        fetch_image(eh, TRIMAP_CLIP_NAME, time).and_then(|img| {
            let t = matte_u8(&img, 0, sb);
            release(&img);
            engine::trimap_nonempty(&t).then_some(t)
        })
    } else {
        None
    };

    engine::prep_seed(&mut seed, w, h, &v);
    let planned = engine::plan(&seed, trimap.as_deref(), w, h, &v);

    let rgb_crop = |x0: usize, y0: usize, x1: usize, y1: usize| -> Vec<u8> {
        cwtools_core::image::display_rgb_u8(&src, x0, y0, x1, y1, v.colorspace)
    };

    // Only the Output clip is ever written (every other OFX clip is an input).
    let aux_sel: Vec<i32> = if channels { v.aux.to_vec() } else { Vec::new() };
    let aux_needs = |k: i32| aux_sel.contains(&k);
    let want_geom = v.spline.enabled && (v.spline_edit || v.spline_show || aux_needs(params::AUX_CLASS));

    let mut plan: Option<CropPlan> = None;
    let mut aux: Option<Vec<u8>> = None; // trimap view (full frame)
    let mut class_full: Option<Vec<u8>> = None; // spline band class (full frame, 0..3)
    let refined = match planned {
        None => Refined::Empty,
        Some((p, srcty)) => {
            plan = Some(p);
            // Matte Out always carries the refined matte, so every Result RGB view
            // runs the model (cached); only the Edit Splines preview skips it.
            let need_model = v.view != VIEW_EDIT;
            let need_prep = need_model
                || v.view == VIEW_TRIMAP
                || v.view == VIEW_EDIT
                || want_geom
                || aux_needs(params::AUX_TRIMAP);
            if !need_prep {
                Refined::Empty
            } else {
                let t0 = std::time::Instant::now();
                let (prep, sout) = engine::prepare_ex(&seed, trimap.as_deref(), w, p, srcty, &v, want_geom, rgb_crop);
                let prep_ms = t0.elapsed().as_secs_f64() * 1000.0;
                if v.view == VIEW_TRIMAP || v.view == VIEW_EDIT || aux_needs(params::AUX_TRIMAP) {
                    aux = Some(engine::paste(w, h, &p, &engine::trimap_view(&prep)));
                }
                if sout.is_none() && (v.spline_edit || v.spline_show) {
                    spline::clear_geometry(eh as usize); // trimap clip / nothing to draw
                }
                if let Some(so) = &sout {
                    if v.view == VIEW_EDIT || aux_needs(params::AUX_CLASS) {
                        class_full = Some(engine::paste(w, h, &p, &engine::proc_to_crop(&prep, &so.applied.class)));
                    }
                    if v.spline_edit || v.spline_show {
                        let cn = spline::Canon { bx1: sb.x1 as f64, by1: sb.y1 as f64, h: h as f64, rs };
                        let fc = (sb.x1 as f64 / rs, sb.y1 as f64 / rs, sb.x2 as f64 / rs, sb.y2 as f64 / rs);
                        let g = spline::build_geometry(
                            &so.applied, p.padw, p.pw, p.ph, &so.map, &cn, so.bbox, &v.spline_full, v.spline.outer2_on, time, fc,
                        );
                        if log::verbose() {
                            log::log(&format!(
                                "t={time} spline geometry: {} vertices (inner {} / middle {} / outer {} / outer2 {} lines), {} points, r_out {:.1} r_in {:.1} canon",
                                g.vertex_count(), g.inner.len(), g.middle.len(), g.outer.len(), g.outer2.len(), g.points.len(), g.r_out_c, g.r_in_c
                            ));
                        }
                        spline::store_geometry(eh as usize, g);
                    }
                }
                if !need_model {
                    Refined::Empty
                } else if aborted(eh) {
                    release(&src);
                    release(&out);
                    return kOfxStatOK;
                } else {
                    run_model(&v, &prep, w, h, prep_ms, time)
                }
            }
        }
    };

    // ── messages ────────────────────────────────────────────────────────
    match (&refined, &planned) {
        (Refined::Failed(e), _) => set_node_message(eh, Some(&format!("cwRefine: {e} (input mask passed through)"))),
        (_, None) => set_node_message(
            eh,
            Some(if trimap.is_some() { "cwRefine: trimap is empty" } else { "cwRefine: empty input mask (nothing above 0.5) - output alpha is black" }),
        ),
        _ if v.view == VIEW_EDIT => set_node_message(
            eh,
            Some("cwRefine: Edit Splines - preview only (no ViTMatte). Drag the yellow/green line, click to add a point, drag its handle. Edit Splines off = refine."),
        ),
        _ => set_node_message(eh, note.as_deref()),
    }

    let have_refined = matches!(refined, Refined::Alpha(_));
    let alpha: Vec<f32> = match refined {
        Refined::Alpha(a) => a,
        Refined::Empty => vec![0.0; w * h],
        Refined::Failed(_) => seed_in.iter().map(|&s| s as f32 / 255.0).collect(),
    };
    if channels {
        let refined_or_none = if have_refined { Some(alpha.as_slice()) } else { None };
        write_channels(&src, &out, &win, &v.aux, &alpha, v.matte_inverted, refined_or_none, &seed_in, aux.as_deref(), plan.as_ref(), class_full.as_deref());
    } else {
        write_output(&src, &out, &win, &v, &alpha, &seed_in, aux.as_deref(), plan.as_ref(), rs, class_full.as_deref());
    }
    release(&src);
    release(&out);
    kOfxStatOK
}

/// Result RGB = Channel Pickers: each of R/G/B carries the picked buffer
/// (Plate = that plate channel), alpha = `alpha` (refined, or the input mask
/// when refinement failed, like the default view).
#[allow(clippy::too_many_arguments)]
fn write_channels(
    src: &Image,
    out: &Image,
    win: &OfxRectI,
    sel: &[i32; 3],
    alpha: &[f32],
    inverted: bool,
    refined: Option<&[f32]>,
    seed: &[u8],
    trimap_full: Option<&[u8]>,
    plan: Option<&CropPlan>,
    class_full: Option<&[u8]>,
) {
    let sb = src.bounds;
    let w = src.w();
    let x1 = win.x1.max(out.bounds.x1);
    let x2 = win.x2.min(out.bounds.x2);
    let y1 = win.y1.max(out.bounds.y1);
    let y2 = win.y2.min(out.bounds.y2);
    let nc = (out.n_comps.max(1) as usize).min(4);
    let val = |k: i32, i: Option<usize>, col: i64, row: i64| -> f32 {
        let Some(i) = i else { return 0.0 };
        match k {
            params::AUX_TRIMAP => trimap_full.map(|t| t[i] as f32 / 255.0).unwrap_or(0.0),
            params::AUX_SEED => seed[i] as f32 / 255.0,
            params::AUX_REFINED => refined.map(|a| a[i]).unwrap_or(0.0),
            params::AUX_DIFF => refined.map(|a| (a[i] - seed[i] as f32 / 255.0).abs()).unwrap_or(0.0),
            params::AUX_CROP => plan
                .map(|p| (col >= p.x0 as i64 && col < p.x1 as i64 && row >= p.y0 as i64 && row < p.y1 as i64) as i32 as f32)
                .unwrap_or(0.0),
            params::AUX_CLASS => class_full.map(|c| c[i] as f32 / 3.0).unwrap_or(0.0),
            _ => 0.0,
        }
    };
    for y in y1..y2 {
        for x in x1..x2 {
            let inside = src.contains(x, y);
            let (col, row) = ((x - sb.x1) as i64, (sb.y2 - 1 - y) as i64);
            let idx = if inside { Some(row as usize * w + col as usize) } else { None };
            let sp = if inside { Some(unsafe { src.px(x, y) }) } else { None };
            let ch = |c: usize| -> f32 {
                if sel[c] == params::AUX_OFF {
                    sp.map(|p| unsafe { *p.add(c) }).unwrap_or(0.0)
                } else {
                    val(sel[c], idx, col, row)
                }
            };
            let a = idx.map(|i| alpha[i]).unwrap_or(0.0);
            let px = [ch(0), ch(1), ch(2), if inverted { 1.0 - a } else { a }];
            let op = unsafe { out.px(x, y) };
            if nc == 1 {
                unsafe { *op = px[3] };
                continue;
            }
            for (c, &pv) in px.iter().enumerate().take(nc) {
                unsafe { *op.add(c) = pv };
            }
        }
    }
}

fn run_model(v: &Values, prep: &Prepared, w: usize, h: usize, prep_ms: f64, time: f64) -> Refined {
    let p = prep.plan;
    let rkey = crate::local::result_key(v, prep.key);
    if v.cache {
        if let Some(a) = crate::local::result_get(&rkey) {
            if log::verbose() {
                log::log(&format!("t={time} cache hit {}x{} crop", p.cw(), p.ch()));
            }
            return Refined::Alpha(engine::paste(w, h, &p, &a));
        }
    }
    let t0 = std::time::Instant::now();
    let r = if v.remote {
        crate::remote::infer(v, prep, time)
    } else {
        crate::local::infer(v, prep.pixel_values(), p.padh, p.padw).and_then(|(alphas, info)| {
            let crop = engine::finish_f32(&p, &alphas)?;
            if log::verbose() || !info.on_gpu {
                log::log(&format!(
                    "t={time} frame {w}x{h} crop {}x{} @({},{}) proc {}x{} pad {}x{} trimap={:?} prep {prep_ms:.0} ms model {:.0} ms total {:.0} ms gpu={}",
                    p.cw(), p.ch(), p.x0, p.y0, p.pw, p.ph, p.padw, p.padh, prep.source, info.ms,
                    t0.elapsed().as_secs_f64() * 1000.0 + prep_ms, info.on_gpu
                ));
            }
            Ok(crop)
        })
    };
    match r {
        Ok(crop) => {
            if v.cache {
                crate::local::result_put(rkey, &crop);
            }
            Refined::Alpha(engine::paste(w, h, &p, &crop))
        }
        Err(e) => Refined::Failed(e),
    }
}

#[allow(clippy::too_many_arguments)]
fn write_output(
    src: &Image,
    out: &Image,
    win: &OfxRectI,
    v: &Values,
    alpha: &[f32],
    seed: &[u8],
    trimap_full: Option<&[u8]>,
    plan: Option<&CropPlan>,
    rs: f64,
    class_full: Option<&[u8]>,
) {
    let sb = src.bounds;
    let w = src.w();
    let x1 = win.x1.max(out.bounds.x1);
    let x2 = win.x2.min(out.bounds.x2);
    let y1 = win.y1.max(out.bounds.y1);
    let y2 = win.y2.min(out.bounds.y2);
    let edge = ((2.0 * rs).round() as i64).max(1);
    for y in y1..y2 {
        for x in x1..x2 {
            let inside = src.contains(x, y);
            let (r, g, b) = if inside {
                let sp = unsafe { src.px(x, y) };
                unsafe { (*sp, *sp.add(1), *sp.add(2)) }
            } else {
                (0.0, 0.0, 0.0)
            };
            let (col, row) = ((x - sb.x1) as usize, (sb.y2 - 1 - y) as usize);
            let idx = if inside { Some(row * w + col) } else { None };
            let a = idx.map(|i| alpha[i]).unwrap_or(0.0);
            let mo = if v.matte_inverted { 1.0 - a } else { a }; // Matte Out
            let (or, og, ob, oa) = match v.view {
                VIEW_MATTE => (mo, mo, mo, mo),
                VIEW_PREMULT => (r * mo, g * mo, b * mo, mo),
                VIEW_TRIMAP => {
                    let t = idx.and_then(|i| trimap_full.map(|t| t[i] as f32 / 255.0)).unwrap_or(0.0);
                    (t, t, t, mo)
                }
                VIEW_CROP => {
                    let (inroi, border) = match (plan, idx) {
                        (Some(p), Some(_)) => {
                            let (c, rr) = (col as i64, row as i64);
                            let inr = c >= p.x0 as i64 && c < p.x1 as i64 && rr >= p.y0 as i64 && rr < p.y1 as i64;
                            let bd = inr
                                && (c < p.x0 as i64 + edge || c >= p.x1 as i64 - edge || rr < p.y0 as i64 + edge || rr >= p.y1 as i64 - edge);
                            (inr, bd)
                        }
                        _ => (false, false),
                    };
                    if border {
                        (1.0, 0.85, 0.0, mo)
                    } else if inroi {
                        (r, g, b, mo)
                    } else {
                        (r * 0.3, g * 0.3, b * 0.3, mo)
                    }
                }
                VIEW_EDIT => {
                    // plate dimmed, bands tinted; alpha = the trimap ViTMatte would get
                    let c = idx.and_then(|i| class_full.map(|m| m[i])).unwrap_or(0);
                    let t = idx.and_then(|i| trimap_full.map(|t| t[i] as f32 / 255.0)).unwrap_or(0.0);
                    let (k, tr, tg, tb) = match c {
                        spline::C_CORE => (0.55, 0.0, 0.22, 0.05),
                        spline::C_BAND => (0.5, 0.32, 0.28, 0.0),
                        spline::C_OUTER2 => (0.45, 0.30, 0.10, 0.0),
                        _ => (0.3, 0.0, 0.0, 0.0),
                    };
                    (r * k + tr, g * k + tg, b * k + tb, t)
                }
                VIEW_DIFF => {
                    let s = idx.map(|i| seed[i] as f32 / 255.0).unwrap_or(0.0);
                    let base = 0.35 * a;
                    (base + (s - a).max(0.0), base + (a - s).max(0.0), base, mo)
                }
                _ => (r, g, b, mo),
            };
            let op = unsafe { out.px(x, y) };
            unsafe {
                *op = or;
                *op.add(1) = og;
                *op.add(2) = ob;
                if out.n_comps == 4 {
                    *op.add(3) = oa;
                }
            }
        }
    }
}
