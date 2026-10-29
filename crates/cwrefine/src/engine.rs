//! Frame-level refine pipeline, independent of OFX (testable on buffers):
//!
//! seed (u8, full frame) [+ trimap clip] → invert → pre-dilate → bbox →
//! crop plan → crop + Lanczos ≤ max crop → /32 edge pad → trimap (generated
//! or supplied) → HF pixel_values → [core model] → uint8 truncation → unpad →
//! Lanczos back → paste (zero outside the ROI).
//!
//! All frame buffers are top-down u8, frame size `w × h`.

use cwtools_core::resample::{resize_f32, Filter};
use crate::recipe::{self, CropPlan};
use crate::params::Values;
use crate::spline;
use cwtools_core::pixels::{fnv1a, fnv1a_continue};

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum TrimapSource {
    /// Generated from the seed (production `_gen_trimap_production`).
    Seed,
    /// Supplied through the Trimap clip.
    Clip,
}

pub struct Prepared {
    pub plan: CropPlan,
    pub source: TrimapSource,
    /// Display RGB u8, padded size (padw × padh × 3).
    pub img_pad: Vec<u8>,
    /// Trimap u8 (0 / unknown / 255), padded size.
    pub tri_pad: Vec<u8>,
    /// Hash of the exact model input (img + trimap + size).
    pub key: u64,
}

impl Prepared {
    pub fn pixel_values(&self) -> Vec<f32> {
        recipe::hf_pixel_values(&self.img_pad, &self.tri_pad, self.plan.padw, self.plan.padh)
    }
}

/// Is a supplied trimap "connected and non-empty" (any pixel above the BG level)?
pub fn trimap_nonempty(tri: &[u8]) -> bool {
    tri.iter().any(|&v| v > recipe::TRI_LO_U8)
}

/// Seed preprocessing shared by every path: invert, pre-dilate.
pub fn prep_seed(seed: &mut Vec<u8>, w: usize, h: usize, v: &Values) {
    if v.invert_seed {
        seed.iter_mut().for_each(|s| *s = 255 - *s);
    }
    if v.black_clip > 0 {
        let t = v.black_clip;
        for x in seed.iter_mut() {
            if *x <= t {
                *x = 0;
            }
        }
    }
    if v.pre_dilate > 0 {
        *seed = recipe::dilate_rect(seed, w, h, v.pre_dilate);
    }
}

/// Plan the crop. Production: seed > 128 bbox + padding. With a trimap clip
/// (same rule as the Flame Inference ViTMatte trimap ONNX): that seed crop
/// whenever the seed has content, so a trimap derived from the mask (cwTrimap)
/// gets production's crop, processing size and context and stray trimap pixels
/// outside it are ignored; only an empty seed uses the trimap's own non-BG bbox
/// + padding. 0.1 always used the trimap bbox; 0.2.0 used it whenever the
/// trimap reached outside the seed crop. None = nothing to refine.
pub fn plan(seed: &[u8], trimap: Option<&[u8]>, w: usize, h: usize, v: &Values) -> Option<(CropPlan, TrimapSource)> {
    let seed_plan = |extra: usize| {
        recipe::bbox_gt(seed, w, h, 128)
            .and_then(|b| recipe::crop_plan(w, h, b, v.padding + extra, v.max_crop))
            .and_then(|p| cover_faint(p, seed, w, h, v))
    };
    match trimap {
        None => seed_plan(v.spline.extra_reach()).map(|p| (p, TrimapSource::Seed)),
        // A connected trimap uses the input mask's crop whenever the mask has
        // content; anything the trimap marks outside it is ignored (a stray
        // faint pixel used to blow the crop up to the full frame). Only an empty
        // mask falls back to the trimap's own non-BG bbox + Padding.
        Some(t) => match seed_plan(0) {
            Some(p) => Some((p, TrimapSource::Clip)),
            None => {
                let b = recipe::bbox_gt(t, w, h, recipe::TRI_LO_U8)?;
                recipe::crop_plan(w, h, b, v.padding, v.max_crop).map(|p| (p, TrimapSource::Clip))
            }
        },
    }
}

/// Faint mask levels at or below this (unless Mask Black Clip is higher) may
/// fall outside the crop: ~1.6 %, so a near-black haze cannot blow it up.
pub const FAINT_U8: u8 = 4;
/// Frame px kept past the outer edge band, so the band is not cut by the crop.
const FAINT_MARGIN: usize = 16;

/// Grow the core crop (seed > 0.5 + Padding) so it also holds every faint
/// mask pixel the edge band refines (Mask Edges: all > 0; Mid Point: >= the
/// level) plus the band's outward reach. Faint hair beyond Padding used to be
/// cut off at the crop edge. An unchanged box keeps the plan as it was.
fn cover_faint(core: CropPlan, seed: &[u8], w: usize, h: usize, v: &Values) -> Option<CropPlan> {
    let lo = match v.trimap.mid_point {
        Some(m) => (((m * 255.0).round() as i32).clamp(1, 255) - 1) as u8,
        None => v.black_clip.max(FAINT_U8),
    };
    let Some((fx0, fx1, fy0, fy1)) = recipe::bbox_gt(seed, w, h, lo) else { return Some(core) };
    let reach = v.trimap.asym.map_or(v.trimap.kernel_size, |(_, kout)| kout).div_ceil(2) + FAINT_MARGIN;
    let x0 = core.x0.min(fx0.saturating_sub(reach));
    let y0 = core.y0.min(fy0.saturating_sub(reach));
    let x1 = core.x1.max((fx1 + 1 + reach).min(w));
    let y1 = core.y1.max((fy1 + 1 + reach).min(h));
    if (x0, y0, x1, y1) == (core.x0, core.y0, core.x1, core.y1) {
        return Some(core);
    }
    // padding 0: crop_plan's box is then exactly [x0, x1) x [y0, y1)
    recipe::crop_plan(w, h, (x0, x1, y0, y1), 0, v.max_crop)
}

/// Build the model input for `plan`. `rgb_crop(x0, y0, x1, y1)` returns the
/// display-referred RGB u8 of that frame rectangle (top-down).
pub fn prepare(
    seed: &[u8],
    trimap: Option<&[u8]>,
    w: usize,
    plan: CropPlan,
    source: TrimapSource,
    v: &Values,
    rgb_crop: impl FnOnce(usize, usize, usize, usize) -> Vec<u8>,
) -> Prepared {
    prepare_ex(seed, trimap, w, plan, source, v, false, rgb_crop).0
}

/// Spline context of a prepared frame (for views / overlay geometry).
pub struct SplineOut {
    pub applied: spline::Applied,
    pub map: spline::CropMap,
    /// Seed bbox (> 0.5) in frame px, inclusive, top-down.
    pub bbox: (usize, usize, usize, usize),
}

/// [`prepare`] plus the spline trimap edits; `want_geometry` also returns the
/// class map + contours when nothing is edited (overlay / aux views).
#[allow(clippy::too_many_arguments)]
pub fn prepare_ex(
    seed: &[u8],
    trimap: Option<&[u8]>,
    w: usize,
    plan: CropPlan,
    source: TrimapSource,
    v: &Values,
    want_geometry: bool,
    rgb_crop: impl FnOnce(usize, usize, usize, usize) -> Vec<u8>,
) -> (Prepared, Option<SplineOut>) {
    let (cw, ch) = (plan.cw(), plan.ch());
    let (pw, ph) = (plan.pw, plan.ph);
    let img_c = rgb_crop(plan.x0, plan.y0, plan.x1, plan.y1);
    let img_r = if plan.scaled() { recipe::lanczos_u8(&img_c, cw, ch, 3, pw, ph) } else { img_c };
    let img_pad = recipe::edge_pad(&img_r, pw, ph, 3, plan.padw, plan.padh);

    let mut sout: Option<SplineOut> = None;
    let tri_pad = match (source, trimap) {
        (TrimapSource::Clip, Some(t)) => {
            // Supplied trimap: crop at the same box, nearest-resample (keeps the
            // classes), classify, edge-pad like the image.
            let t_c = recipe::crop(t, w, 1, plan.x0, plan.y0, plan.x1, plan.y1);
            let t_r = recipe::nearest_u8(&t_c, cw, ch, pw, ph);
            let mut t_e: Vec<u8> = t_r.into_iter().map(recipe::classify_trimap).collect();
            if v.trimap.mid_bias.is_some() {
                let u = v.trimap.unknown_level();
                t_e.iter_mut().filter(|x| **x != 0 && **x != 255).for_each(|x| *x = u);
            }
            recipe::edge_pad(&t_e, pw, ph, 1, plan.padw, plan.padh)
        }
        _ => {
            // Production: seed crop → (LANCZOS if scaled) → edge pad → trimap.
            let s_c = recipe::crop(seed, w, 1, plan.x0, plan.y0, plan.x1, plan.y1);
            let s_r = if plan.scaled() { recipe::lanczos_u8(&s_c, cw, ch, 1, pw, ph) } else { s_c };
            let s_p = recipe::edge_pad(&s_r, pw, ph, 1, plan.padw, plan.padh);
            let mut tri = recipe::gen_trimap(&s_p, plan.padw, plan.padh, &v.trimap);
            let spline_on = v.spline.enabled && (want_geometry || !v.spline.is_identity());
            if spline_on {
                let h = seed.len() / w.max(1);
                if let Some(bbox) = recipe::bbox_gt(seed, w, h, 128) {
                    let map = spline::CropMap {
                        x0: plan.x0 as f64,
                        y0: plan.y0 as f64,
                        sx: pw as f64 / cw as f64,
                        sy: ph as f64 / ch as f64,
                    };
                    if let Some(ap) = spline::apply(
                        &mut tri, &s_p, plan.padw, plan.padh, pw, ph, &v.trimap, &v.spline, &map, bbox, want_geometry,
                    ) {
                        sout = Some(SplineOut { applied: ap, map, bbox });
                    }
                }
            }
            tri
        }
    };
    let mut key = fnv1a(&img_pad);
    key = fnv1a_continue(key, &tri_pad);
    key = fnv1a_continue(key, &[plan.padw as u64, plan.padh as u64].map(|x| x.to_le_bytes()).concat());
    (Prepared { plan, source, img_pad, tri_pad, key }, sout)
}

/// A padded proc-size u8 buffer back to crop size (nearest), like [`trimap_view`].
pub fn proc_to_crop(p: &Prepared, buf: &[u8]) -> Vec<u8> {
    let unpad = recipe::crop(buf, p.plan.padw, 1, 0, 0, p.plan.pw, p.plan.ph);
    recipe::nearest_u8(&unpad, p.plan.pw, p.plan.ph, p.plan.cw(), p.plan.ch())
}

/// Model output (padded size, f32) → crop-size alpha u8 (production post).
pub fn finish(plan: &CropPlan, alphas: &[f32]) -> Result<Vec<u8>, String> {
    if alphas.len() != plan.padw * plan.padh {
        return Err(format!("model output has {} values, expected {}x{}", alphas.len(), plan.padw, plan.padh));
    }
    let a8 = recipe::alpha_to_u8(alphas).ok_or("model output contains NaN/Inf (corrupt weights or GPU fault)")?;
    finish_u8(plan, &a8)
}

/// Padded uint8 alpha (= `recipe::alpha_to_u8` of the model output, e.g. the
/// Remote server's `u8_trunc255`) → crop-size alpha u8 (unpad, Lanczos back).
pub fn finish_u8(plan: &CropPlan, a8: &[u8]) -> Result<Vec<u8>, String> {
    if a8.len() != plan.padw * plan.padh {
        return Err(format!("alpha has {} values, expected {}x{}", a8.len(), plan.padw, plan.padh));
    }
    let unpad = recipe::crop(a8, plan.padw, 1, 0, 0, plan.pw, plan.ph);
    Ok(if plan.scaled() { recipe::lanczos_u8(&unpad, plan.pw, plan.ph, 1, plan.cw(), plan.ch()) } else { unpad })
}

/// Padded float alpha (the model output) → crop-size float alpha (unpad,
/// Lanczos back, clamped 0..1). The node's matte path: no 8-bit step, so faint
/// hair keeps its sub-1/255 levels. [`finish`] (uint8, the production recipe
/// bit for bit) stays for parity tooling.
pub fn finish_f32(plan: &CropPlan, alphas: &[f32]) -> Result<Vec<f32>, String> {
    if alphas.len() != plan.padw * plan.padh {
        return Err(format!("model output has {} values, expected {}x{}", alphas.len(), plan.padw, plan.padh));
    }
    if alphas.iter().any(|a| !a.is_finite()) {
        return Err("model output contains NaN/Inf (corrupt weights or GPU fault)".into());
    }
    let unpad: Vec<f32> = recipe::crop(alphas, plan.padw, 1, 0, 0, plan.pw, plan.ph).into_iter().map(|a| a.clamp(0.0, 1.0)).collect();
    Ok(if plan.scaled() {
        resize_f32(&unpad, plan.pw, plan.ph, plan.cw(), plan.ch(), Filter::Lanczos).into_iter().map(|a| a.clamp(0.0, 1.0)).collect()
    } else {
        unpad
    })
}

/// The trimap as seen by the model, mapped back to crop size (nearest) for the Trimap view.
pub fn trimap_view(p: &Prepared) -> Vec<u8> {
    let unpad = recipe::crop(&p.tri_pad, p.plan.padw, 1, 0, 0, p.plan.pw, p.plan.ph);
    recipe::nearest_u8(&unpad, p.plan.pw, p.plan.ph, p.plan.cw(), p.plan.ch())
}

/// Paste a crop-size buffer into a zero frame.
pub fn paste<T: Copy + Default>(w: usize, h: usize, plan: &CropPlan, crop: &[T]) -> Vec<T> {
    let mut out = vec![T::default(); w * h];
    let cw = plan.cw();
    for (r, y) in (plan.y0..plan.y1).enumerate() {
        out[y * w + plan.x0..y * w + plan.x1].copy_from_slice(&crop[r * cw..(r + 1) * cw]);
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn mask_black_clip() {
        let mut v = Values::production();
        let orig: Vec<u8> = (0..=255).collect();
        let mut m = orig.clone();
        prep_seed(&mut m, 256, 1, &v);
        assert_eq!(m, orig, "black clip 0 = unchanged");
        v.black_clip = 20;
        prep_seed(&mut m, 256, 1, &v);
        assert!(m[..=20].iter().all(|&x| x == 0));
        assert_eq!(&m[21..], &orig[21..]);
    }
    #[test]
    fn crop_covers_faint_hair_beyond_padding() {
        let v = Values::production();
        let (w, h) = (1000usize, 1000usize);
        let mut m = vec![0u8; w * h];
        for y in 500..700 {
            m[y * w + 400..y * w + 600].fill(255); // solid core
        }
        let core = plan(&m, None, w, h, &v).unwrap().0;
        assert_eq!((core.x0, core.y0, core.x1, core.y1), (272, 372, 727, 827));
        for y in 150..500 {
            m[y * w + 480] = 60; // faint strand 350 px up, far past Padding 128
        }
        let p = plan(&m, None, w, h, &v).unwrap().0;
        assert_eq!((p.x0, p.y0, p.x1, p.y1), (272, 150 - 31, 727, 827), "15 px band + 16 margin above the strand");
        m[5 * w + 5] = FAINT_U8; // near-black haze does not grow the crop
        assert_eq!(plan(&m, None, w, h, &v).unwrap().0.y0, 119);
        let mut v2 = v.clone();
        v2.black_clip = 80; // Mask Black Clip removes the strand -> core crop
        let mut mc = m.clone();
        prep_seed(&mut mc, w, h, &v2);
        assert_eq!(plan(&mc, None, w, h, &v2).unwrap().0.y0, 372);
    }
    #[test]
    fn empty_seed_has_no_plan() {
        let v = Values::production();
        assert!(plan(&vec![0u8; 64 * 32], None, 64, 32, &v).is_none());
        assert!(plan(&vec![128u8; 64 * 32], None, 64, 32, &v).is_none()); // > 128 needed
    }
    #[test]
    fn trimap_clip_sets_roi() {
        let (w, h) = (300, 200);
        let mut t = vec![0u8; w * h];
        for y in 50..60 {
            for x in 100..140 {
                t[y * w + x] = 128;
            }
        }
        let mut v = Values::production();
        v.padding = 10;
        let (p, src) = plan(&vec![0u8; w * h], Some(&t), w, h, &v).unwrap();
        assert_eq!(src, TrimapSource::Clip);
        assert_eq!((p.x0, p.y0, p.x1, p.y1), (90, 40, 149, 69));
        let prep = prepare(&vec![0u8; w * h], Some(&t), w, p, src, &v, |x0, y0, x1, y1| vec![100u8; (x1 - x0) * (y1 - y0) * 3]);
        assert_eq!(prep.tri_pad.len(), p.padw * p.padh);
        assert!(prep.tri_pad.iter().all(|&x| x == 0 || x == 128));
        let tv = trimap_view(&prep);
        assert_eq!(tv.len(), p.cw() * p.ch());
    }
    #[test]
    fn trimap_inside_seed_crop_keeps_seed_crop() {
        // seed block + a trimap band 20 px around it (well inside padding 128)
        let (w, h) = (1000, 800);
        let mut seed = vec![0u8; w * h];
        let mut t = vec![0u8; w * h];
        for y in 280..520 {
            for x in 380..620 {
                t[y * w + x] = 128;
                if (300..500).contains(&y) && (400..600).contains(&x) {
                    seed[y * w + x] = 255;
                    t[y * w + x] = 255;
                }
            }
        }
        let v = Values::production();
        let (auto, _) = plan(&seed, None, w, h, &v).unwrap();
        let (p, src) = plan(&seed, Some(&t), w, h, &v).unwrap();
        assert_eq!(src, TrimapSource::Clip);
        assert_eq!((p.x0, p.y0, p.x1, p.y1, p.pw, p.ph), (auto.x0, auto.y0, auto.x1, auto.y1, auto.pw, auto.ph));
        // same with a Max Crop that forces a downscale: still the seed plan
        let mut v2 = Values::production();
        v2.max_crop = 256;
        let (a2, _) = plan(&seed, None, w, h, &v2).unwrap();
        let (p2, _) = plan(&seed, Some(&t), w, h, &v2).unwrap();
        assert_eq!((p2.x0, p2.x1, p2.pw, p2.ph), (a2.x0, a2.x1, a2.pw, a2.ph));
        // a trimap blob (or one stray faint pixel) outside the seed crop is
        // ignored: still the seed plan
        for y in 700..720 {
            for x in 50..70 {
                t[y * w + x] = 128;
            }
        }
        t[w - 1] = 128;
        let (p3, _) = plan(&seed, Some(&t), w, h, &v).unwrap();
        assert_eq!((p3.x0, p3.y0, p3.x1, p3.y1), (auto.x0, auto.y0, auto.x1, auto.y1));
        // empty mask -> the trimap's own non-BG bbox + padding
        let empty = vec![0u8; w * h];
        let (p4, s4) = plan(&empty, Some(&t), w, h, &v).unwrap();
        assert_eq!(s4, TrimapSource::Clip);
        assert_eq!((p4.x0, p4.y0, p4.x1, p4.y1), (0, 0, w, h));
    }
    #[test]
    fn finish_rejects_nan_and_truncates() {
        let p = recipe::crop_plan(40, 40, (10, 20, 10, 20), 2, 2048).unwrap();
        let mut a = vec![0.999f32; p.padw * p.padh];
        let o = finish(&p, &a).unwrap();
        assert!(o.iter().all(|&x| x == 254)); // 0.999*255 = 254.7 -> 254
        a[3] = f32::NAN;
        assert!(finish(&p, &a).is_err());
    }
    #[test]
    fn finish_f32_keeps_faint_levels() {
        // unscaled crop: the float matte comes back exactly (clamped), faint levels intact
        let p = recipe::crop_plan(40, 40, (10, 20, 10, 20), 2, 2048).unwrap();
        assert!(!p.scaled());
        let mut a: Vec<f32> = (0..p.padw * p.padh).map(|i| (i % 97) as f32 * 1e-4).collect();
        a[0] = 1.2;
        a[1] = -0.1;
        let o = finish_f32(&p, &a).unwrap();
        assert_eq!(o.len(), p.cw() * p.ch());
        let unpad: Vec<f32> = recipe::crop(&a, p.padw, 1, 0, 0, p.pw, p.ph).into_iter().map(|x| x.clamp(0.0, 1.0)).collect();
        assert_eq!(o, unpad);
        assert!(o.iter().any(|&x| x > 0.0 && x < 1.0 / 255.0)); // below one 8-bit step, still there
        // scaled crop: Lanczos in float, within one 8-bit step of the u8 production recipe
        let p = recipe::crop_plan(3000, 3000, (100, 2900, 100, 2900), 0, 1024).unwrap();
        assert!(p.scaled());
        let a: Vec<f32> = (0..p.padw * p.padh).map(|i| ((i % p.padw) as f32 / p.padw as f32)).collect();
        let f = finish_f32(&p, &a).unwrap();
        let u = finish(&p, &a).unwrap();
        let worst = f.iter().zip(&u).map(|(x, &y)| (x * 255.0 - y as f32).abs()).fold(0.0f32, f32::max);
        assert!(worst <= 2.0, "worst {worst}");
        let mut bad = a.clone();
        bad[5] = f32::INFINITY;
        assert!(finish_f32(&p, &bad).is_err());
    }
}

#[cfg(test)]
mod bench {
    use super::*;
    /// `cargo test --release -p cwrefine bench_prepare -- --ignored --nocapture`
    #[test]
    #[ignore]
    fn bench_prepare() {
        let (w, h) = (2048usize, 1080usize);
        let mut seed = vec![0u8; w * h];
        for y in 441..1080 {
            for x in 527..1023 {
                seed[y * w + x] = if (x + y) % 17 == 0 { 200 } else { 252 };
            }
        }
        let v = Values::production();
        let t = std::time::Instant::now();
        let (p, s) = plan(&seed, None, w, h, &v).unwrap();
        let t_plan = t.elapsed();
        let t = std::time::Instant::now();
        let prep = prepare(&seed, None, w, p, s, &v, |x0, y0, x1, y1| vec![90u8; (x1 - x0) * (y1 - y0) * 3]);
        let t_prep = t.elapsed();
        let t = std::time::Instant::now();
        let tri = recipe::gen_trimap(&prep.tri_pad, p.padw, p.padh, &v.trimap);
        let t_tri = t.elapsed();
        let t = std::time::Instant::now();
        let pv = prep.pixel_values();
        let t_pv = t.elapsed();
        let t = std::time::Instant::now();
        let full = paste(w, h, &p, &trimap_view(&prep));
        let t_paste = t.elapsed();
        let t = std::time::Instant::now();
        let big = recipe::lanczos_u8(&vec![7u8; 2239 * 767 * 3], 2239, 767, 3, 2048, 702);
        let t_lz = t.elapsed();
        eprintln!("plan {t_plan:?} prepare {t_prep:?} (trimap alone {t_tri:?}) pixel_values {t_pv:?} paste {t_paste:?} lanczos 2239x767->2048x702 rgb {t_lz:?} [{} {} {}]", tri.len(), pv.len(), full.len() + big.len());
    }
}
