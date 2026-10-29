//! Render: Source → (ROI) → BiRefNet → matte ops → Output, per the Output view.
//!
//! Failure policy: never hang, never crash. Any model / remote error passes
//! the plate through (alpha = 0 for Matte / Overlay, black for Matte Only),
//! posts a node message and logs to /tmp/cwMatte.log. Render returns OK so
//! Flame keeps working.

use crate::engine::{self, Roi};
use crate::params::{self, Values, ROI_ALPHA, ROI_MANUAL, VIEW_MATTE, VIEW_OVERLAY, VIEW_PREMULT};
use cwtools_core::color::overlay_red;
use cwtools_core::image::{canon_to_frame, display_rgb_u8, fetch_image, release, Image};
use cwtools_core::ofx_api::*;
use cwtools_core::props::{prop_get_double, prop_get_rect_i, prop_try_double};
use cwtools_core::suites::{aborted, effect_param_set};
use cwtools_core::{log, set_node_message};
use std::sync::Arc;

pub fn render(eh: OfxImageEffectHandle, in_args: OfxPropertySetHandle) -> OfxStatus {
    let time = prop_get_double(in_args, kOfxPropTime, 0);
    let win = prop_get_rect_i(in_args, kOfxImageEffectPropRenderWindow);
    let rs = prop_try_double(in_args, kOfxImageEffectPropRenderScale, 0).unwrap_or(1.0);
    let rs_y = prop_try_double(in_args, kOfxImageEffectPropRenderScale, 1).unwrap_or(rs);
    let ps = effect_param_set(eh);
    let v = params::read_values(ps, time).scaled(rs);

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
    let (w, h) = (src.w(), src.h());
    let sb = src.bounds;

    // Source alpha (top-down) only when something uses it.
    let need_alpha = v.garbage || v.roi_mode == ROI_ALPHA;
    let alpha: Option<Vec<f32>> = (need_alpha && src.n_comps == 4)
        .then(|| src.read_channel(3, sb.x1, sb.y1, sb.x2, sb.y2, 0.0).into_iter().map(|a| a.clamp(0.0, 1.0)).collect());

    let mut notes: Vec<String> = Vec::new();
    // ── ROI ─────────────────────────────────────────────────────────────
    let roi = match v.roi_mode {
        ROI_ALPHA => match alpha.as_deref().and_then(|a| engine::bbox_gt(a, w, h, 0.0)) {
            Some((x0, y0, x1, y1)) => engine::pad_box((x0 as f64, y0 as f64, x1 as f64, y1 as f64), v.roi_pad, w, h),
            None => {
                notes.push("ROI: Source alpha is empty - using the full frame".into());
                None
            }
        },
        ROI_MANUAL => {
            let (ax, ay) = canon_to_frame(v.roi_a.0, v.roi_a.1, w, h, sb, (rs, rs_y));
            let (bx, by) = canon_to_frame(v.roi_b.0, v.roi_b.1, w, h, sb, (rs, rs_y));
            let r = engine::pad_box((ax, ay, bx, by), v.roi_pad, w, h);
            if r.is_none() {
                notes.push("ROI: manual box is outside the frame / too small - using the full frame".into());
            }
            r
        }
        _ => None,
    }
    .unwrap_or(Roi::full(w, h));

    if aborted(eh) {
        release(&src);
        release(&out);
        return kOfxStatOK;
    }

    // ── model ───────────────────────────────────────────────────────────
    let matte: Result<Vec<f32>, String> = run_model(&v, &src, roi, w, h, time, &mut notes);
    let mut m = match matte {
        Ok(mut m) => {
            engine::floor_haze(&mut m);
            engine::apply_ops(&mut m, &v.ops);
            Some(m)
        }
        Err(e) => {
            notes.insert(0, format!("{e} (plate passed through)"));
            None
        }
    };
    if v.garbage {
        match (&mut m, alpha.as_deref()) {
            (Some(m), Some(a)) => m.iter_mut().zip(a).for_each(|(x, g)| *x *= g),
            (Some(_), None) => notes.push("garbage mask: Source has no alpha - ignored".into()),
            _ => {}
        }
    }
    if notes.is_empty() {
        set_node_message(eh, None);
    } else {
        set_node_message(eh, Some(&format!("cwMatte: {}", notes.join("; "))));
    }

    write_output(&src, &out, &win, &v, m.as_deref(), roi, rs);
    release(&src);
    release(&out);
    kOfxStatOK
}

/// Full-frame matte (before matte ops) for the ROI, or an artist-facing error.
fn run_model(v: &Values, src: &Image, roi: Roi, w: usize, h: usize, time: f64, notes: &mut Vec<String>) -> Result<Vec<f32>, String> {
    let t0 = std::time::Instant::now();
    // Local: the shared session (+ its metadata). Remote: metadata from the
    // server's /models, the model runs there.
    let (loaded, remote_meta) = if v.remote {
        (None, Some(crate::remote::meta(v)?))
    } else {
        let l = crate::local::load(v)?;
        if v.use_gpu && !l.shared.on_gpu {
            notes.push("running on the CPU (the CUDA session could not be created, see /tmp/cwMatte.log) - very slow".into());
        }
        (Some(l), None)
    };
    let meta = match (&loaded, &remote_meta) {
        (Some(l), _) => &l.meta,
        (None, Some((m, _, _))) => m,
        _ => unreachable!(),
    };
    let rgb = display_rgb_u8(src, roi.x0, roi.y0, roi.x1, roi.y1, v.colorspace);
    let prep = engine::prepare(&rgb, roi, meta.sw, meta.sh);
    let t_prep = t0.elapsed().as_secs_f64() * 1000.0;
    let rkey = crate::local::result_key(v, prep.key);
    let sq: Arc<Vec<f32>> = match v.cache.then(|| crate::local::result_get(&rkey)).flatten() {
        Some(m) => {
            if log::verbose() {
                log::log(&format!("t={time} cache hit roi {}x{}@({},{})", roi.w(), roi.h(), roi.x0, roi.y0));
            }
            m
        }
        None => {
            let input = engine::normalise(&prep.square, meta.sw, meta.sh, meta.mean, meta.std);
            let raw = match (&loaded, &remote_meta) {
                (Some(l), _) => {
                    let (raw, info) = crate::local::infer(v, l, input)?;
                    if log::verbose() || !info.on_gpu {
                        log::log(&format!(
                            "t={time} frame {w}x{h} roi {}x{}@({},{}) model {}x{} prep {t_prep:.0} ms model {:.0} ms gpu={}",
                            roi.w(), roi.h(), roi.x0, roi.y0, meta.sw, meta.sh, info.ms, info.on_gpu
                        ));
                    }
                    raw
                }
                (None, Some((m, name, sha))) => crate::remote::infer(v, m, name, sha, input, &prep.square, time)?,
                _ => unreachable!(),
            };
            let m = Arc::new(engine::finish(&raw, meta.sigmoid)?);
            if v.cache {
                crate::local::result_put(rkey, m.clone());
            }
            m
        }
    };
    let full = engine::to_frame(&sq, meta.sw, meta.sh, roi, w, h);
    if log::verbose() {
        log::log(&format!("t={time} total {:.0} ms", t0.elapsed().as_secs_f64() * 1000.0));
    }
    Ok(full)
}

fn write_output(src: &Image, out: &Image, win: &OfxRectI, v: &Values, matte: Option<&[f32]>, roi: Roi, rs: f64) {
    let sb = src.bounds;
    let w = src.w();
    let x1 = win.x1.max(out.bounds.x1);
    let x2 = win.x2.min(out.bounds.x2);
    let y1 = win.y1.max(out.bounds.y1);
    let y2 = win.y2.min(out.bounds.y2);
    let edge = ((2.0 * rs).round() as i64).max(1);
    let show_box = !roi.is_full(src.w(), src.h());
    for y in y1..y2 {
        for x in x1..x2 {
            let inside = src.contains(x, y);
            let (r, g, b) = if inside {
                let sp = unsafe { src.px(x, y) };
                unsafe { (*sp, *sp.add(1), *sp.add(2)) }
            } else {
                (0.0, 0.0, 0.0)
            };
            let (col, row) = ((x - sb.x1) as i64, (sb.y2 - 1 - y) as i64);
            let a = match (inside, matte) {
                (true, Some(m)) => m[row as usize * w + col as usize],
                _ => 0.0,
            };
            // a = Matte Out (inverted if asked); the overlay always tints the subject
            let (or, og, ob) = match v.view {
                VIEW_MATTE => (a, a, a),
                VIEW_PREMULT => (r * a, g * a, b * a),
                VIEW_OVERLAY => {
                    let (x0, y0, x1, y1) = (roi.x0 as i64, roi.y0 as i64, roi.x1 as i64, roi.y1 as i64);
                    let in_roi = col >= x0 && col < x1 && row >= y0 && row < y1;
                    let border = show_box
                        && in_roi
                        && (col < x0 + edge || col >= x1 - edge || row < y0 + edge || row >= y1 - edge);
                    if border {
                        (1.0, 0.85, 0.0)
                    } else {
                        overlay_red(r, g, b, if v.ops.invert { 1.0 - a } else { a })
                    }
                }
                _ => (r, g, b),
            };
            let op = unsafe { out.px(x, y) };
            unsafe {
                *op = or;
                *op.add(1) = og;
                *op.add(2) = ob;
                if out.n_comps == 4 {
                    *op.add(3) = a;
                }
            }
        }
    }
}
