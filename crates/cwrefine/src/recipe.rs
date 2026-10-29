//! The production cropped-ViTMatte recipe on the CPU, bit-exact where
//! production is integer (scripts/vitmatte_server.py `_vitmatte_one_region` +
//! `_gen_trimap_production` of the reference server):
//!
//! - [`lanczos_u8`]: Pillow `Image.resize(LANCZOS)` on 8-bit images (Resample.c:
//!   precompute_coeffs + 22-bit fixed-point normalize_coeffs_8bpc, horizontal
//!   pass then vertical, each rounded to uint8).
//! - [`ellipse_rows`] / [`dilate_ellipse`] / [`erode_ellipse`]: OpenCV
//!   `getStructuringElement(MORPH_ELLIPSE)` + `cv2.dilate/erode` (default
//!   anchor, border never wins).
//! - [`dilate_rect`]: production's `pre_dilate_px` (MORPH_RECT 2p+1).
//! - [`gen_trimap`]: `_gen_trimap_production` (seed clamp, asymmetric or
//!   symmetric kernels, mid-bias).
//! - [`crop_plan`]: bbox(seed > 128) + padding + max-crop proc size (Python
//!   `round` = ties-to-even) + /32 padding.
//! - [`hf_pixel_values`]: HF VitMatteImageProcessor normalisation.
//!
//! All buffers are top-down, row-major u8, `c` interleaved channels.

// ---------------------------------------------------------------------------
// Pillow LANCZOS (shared implementation: cwtools_core::resample)
// ---------------------------------------------------------------------------
pub use cwtools_core::resample::{par_rows, FORCE_SERIAL};
use cwtools_core::resample::{resize_u8, Filter};

/// Pillow `Image.resize((dw, dh), Image.LANCZOS)` for 8-bit L (c=1) / RGB (c=3).
pub fn lanczos_u8(src: &[u8], sw: usize, sh: usize, c: usize, dw: usize, dh: usize) -> Vec<u8> {
    resize_u8(src, sw, sh, c, dw, dh, Filter::Lanczos)
}

// ---------------------------------------------------------------------------
// OpenCV morphology
// ---------------------------------------------------------------------------

/// `cv2.getStructuringElement(MORPH_ELLIPSE, (k, k))` as rows of
/// (dy, dx0, dx1) relative to the default anchor (k/2, k/2). `cv2.dilate`
/// reads src(x + dx, y + dy) for every set element.
pub fn ellipse_rows(k: usize) -> Vec<(i32, i32, i32)> {
    let k = k.max(1) as i32;
    let r = k / 2;
    let c = k / 2;
    let inv_r2 = if r != 0 { 1.0 / (r as f64 * r as f64) } else { 0.0 };
    let mut rows = Vec::new();
    for i in 0..k {
        let dy = i - r;
        let (j1, j2) = if dy.abs() <= r {
            // saturate_cast<int>(double) = round to nearest, ties to even.
            let dx = (c as f64 * (((r * r - dy * dy) as f64) * inv_r2).sqrt()).round_ties_even() as i32;
            ((c - dx).max(0), (c + dx + 1).min(k))
        } else {
            (0, 0)
        };
        if j2 > j1 {
            rows.push((dy, j1 - c, j2 - 1 - c));
        }
    }
    rows
}

/// Sliding max over the window [x+a, x+b] of each row (b >= a), out-of-row
/// samples = `pad`. van Herk / Gil-Werman (block prefix/suffix maxima), O(1)
/// per pixel; rows in parallel.
fn row_window_max(src: &[u8], w: usize, h: usize, a: i32, b: i32, pad: u8) -> Vec<u8> {
    let l = (b - a + 1) as usize;
    let mut out = vec![0u8; w * h];
    let _ = h;
    let n = w + l - 1;
    par_rows(&mut out, w, |y, orow| {
        let row = &src[y * w..(y + 1) * w];
        // ext[i] = row[i + a] (pad outside)
        let mut ext = vec![pad; n];
        let lo = (-a).max(0) as usize; // first ext index with an in-row sample
        let s0 = a.max(0) as usize; // first row sample used
        if s0 < w {
            let cnt = (w - s0).min(n - lo);
            ext[lo..lo + cnt].copy_from_slice(&row[s0..s0 + cnt]);
        }
        if l == 1 {
            orow.copy_from_slice(&ext[..w]);
            return;
        }
        let mut g = vec![0u8; n];
        let mut hh = vec![0u8; n];
        let mut bs = 0;
        while bs < n {
            let be = (bs + l).min(n);
            let mut m = ext[bs];
            g[bs] = m;
            for i in bs + 1..be {
                m = m.max(ext[i]);
                g[i] = m;
            }
            let mut m = ext[be - 1];
            hh[be - 1] = m;
            for i in (bs..be - 1).rev() {
                m = m.max(ext[i]);
                hh[i] = m;
            }
            bs = be;
        }
        for (x, o) in orow.iter_mut().enumerate() {
            *o = hh[x].max(g[x + l - 1]);
        }
    });
    out
}

/// `cv2.dilate(src, MORPH_ELLIPSE k)`.
pub fn dilate_ellipse(src: &[u8], w: usize, h: usize, k: usize) -> Vec<u8> {
    let rows = ellipse_rows(k);
    let mut keys: Vec<(i32, i32)> = Vec::new();
    for &(_, a, b) in &rows {
        if !keys.contains(&(a, b)) {
            keys.push((a, b));
        }
    }
    let spans: Vec<Vec<u8>> = keys.iter().map(|&(a, b)| row_window_max(src, w, h, a, b, 0)).collect();
    let taps: Vec<(i32, &Vec<u8>)> =
        rows.iter().map(|&(dy, a, b)| (dy, &spans[keys.iter().position(|k| *k == (a, b)).unwrap()])).collect();
    let mut out = vec![0u8; w * h];
    par_rows(&mut out, w, |y, orow| {
        for &(dy, hm) in &taps {
            let sy = y as i32 + dy;
            if sy < 0 || sy as usize >= h {
                continue;
            }
            let srow = &hm[sy as usize * w..(sy as usize + 1) * w];
            for (o, &v) in orow.iter_mut().zip(srow) {
                *o = (*o).max(v);
            }
        }
    });
    out
}

/// `cv2.erode(src, MORPH_ELLIPSE k)` (= 255 - dilate(255 - src); the border
/// never wins in either).
pub fn erode_ellipse(src: &[u8], w: usize, h: usize, k: usize) -> Vec<u8> {
    let inv: Vec<u8> = src.iter().map(|v| 255 - v).collect();
    dilate_ellipse(&inv, w, h, k).into_iter().map(|v| 255 - v).collect()
}

/// `cv2.dilate(src, MORPH_RECT (2p+1, 2p+1))` — production's pre-dilate.
pub fn dilate_rect(src: &[u8], w: usize, h: usize, p: usize) -> Vec<u8> {
    if p == 0 {
        return src.to_vec();
    }
    let p = p as i32;
    let hm = row_window_max(src, w, h, -p, p, 0);
    // vertical pass = horizontal pass on the transpose
    let mut t = vec![0u8; w * h];
    for y in 0..h {
        for x in 0..w {
            t[x * h + y] = hm[y * w + x];
        }
    }
    let tv = row_window_max(&t, h, w, -p, p, 0);
    let mut out = vec![0u8; w * h];
    for y in 0..h {
        for x in 0..w {
            out[y * w + x] = tv[x * h + y];
        }
    }
    out
}

// ---------------------------------------------------------------------------
// Trimap
// ---------------------------------------------------------------------------
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct TrimapParams {
    pub kernel_size: usize,
    /// (erode in, dilate out); None = symmetric `kernel_size`.
    pub asym: Option<(usize, usize)>,
    /// Unknown-band level 0..1; None = 128.
    pub mid_bias: Option<f64>,
    /// Band From = Mid Point: cut the seed at this level (0..1) before the
    /// erode / dilate, so the band is centred on that contour whatever the
    /// mask's softness. None = Mask Edges (production: from the solid core in,
    /// from the outermost non-zero pixel out).
    pub mid_point: Option<f64>,
    /// uint8 seed clamp; >= 255 disables.
    pub seed_clamp: i32,
}

impl TrimapParams {
    /// The production hair recipe.
    pub const HAIR: TrimapParams = TrimapParams { kernel_size: 15, asym: Some((4, 30)), mid_bias: None, mid_point: None, seed_clamp: 250 };

    pub fn unknown_level(&self) -> u8 {
        match self.mid_bias {
            None => 128,
            Some(m) => ((m * 255.0).round_ties_even() as i64).clamp(1, 254) as u8,
        }
    }
}

/// Production seed clamp: values >= thr (clamped to 1..254) become 255.
pub fn seed_clamp(seed: &mut [u8], thr: i32) {
    if thr < 255 {
        let t = thr.clamp(1, 254) as u8;
        for v in seed.iter_mut() {
            if *v >= t {
                *v = 255;
            }
        }
    }
}

/// Band From = Mid Point: values >= the level become 255, the rest 0.
pub fn mid_point_cut(seed: &mut [u8], level: f64) {
    let t = ((level * 255.0).round() as i32).clamp(1, 255) as u8;
    for v in seed.iter_mut() {
        *v = if *v >= t { 255 } else { 0 };
    }
}

/// `_gen_trimap_production(alpha_u8, ...)` (seed clamp included).
pub fn gen_trimap(seed: &[u8], w: usize, h: usize, p: &TrimapParams) -> Vec<u8> {
    let mut s = seed.to_vec();
    seed_clamp(&mut s, p.seed_clamp);
    if let Some(m) = p.mid_point {
        mid_point_cut(&mut s, m);
    }
    let (ke, kd) = match p.asym {
        Some((kin, kout)) => (kin.max(1), kout.max(1)),
        None => (p.kernel_size.max(1), p.kernel_size.max(1)),
    };
    let dil = dilate_ellipse(&s, w, h, kd);
    let ero = erode_ellipse(&s, w, h, ke);
    let unk = p.unknown_level();
    dil.iter()
        .zip(&ero)
        .map(|(&d, &e)| if d == 0 { 0 } else if e == 255 { 255 } else { unk })
        .collect()
}

/// Supplied-trimap classification (as the trimap ONNX variant): <= 25 = BG,
/// > 229 = FG, anything in between keeps its level (0.5 → 128 = neutral).
pub const TRI_LO_U8: u8 = 25; // 0.1 * 255 = 25.5 → "> 25.5" ⇔ ">= 26"
pub const TRI_HI_U8: u8 = 229; // 0.9 * 255 = 229.5 → "> 229.5" ⇔ ">= 230"
#[inline]
pub fn classify_trimap(v: u8) -> u8 {
    if v > TRI_HI_U8 {
        255
    } else if v > TRI_LO_U8 {
        v
    } else {
        0
    }
}

/// Source indices of a nearest-neighbour resample n_in → n_out
/// (pixel-centre: floor((2i+1) * n_in / (2 n_out))).
pub fn nearest_index(n_in: usize, n_out: usize) -> Vec<usize> {
    (0..n_out).map(|i| ((2 * i + 1) * n_in / (2 * n_out)).min(n_in.saturating_sub(1))).collect()
}

pub fn nearest_u8(src: &[u8], sw: usize, sh: usize, dw: usize, dh: usize) -> Vec<u8> {
    if sw == dw && sh == dh {
        return src.to_vec();
    }
    let xi = nearest_index(sw, dw);
    let yi = nearest_index(sh, dh);
    let mut out = vec![0u8; dw * dh];
    for (y, &sy) in yi.iter().enumerate() {
        for (x, &sx) in xi.iter().enumerate() {
            out[y * dw + x] = src[sy * sw + sx];
        }
    }
    out
}

// ---------------------------------------------------------------------------
// Crop plan
// ---------------------------------------------------------------------------
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct CropPlan {
    /// Crop box in frame pixels, top-down, [x0, x1) x [y0, y1).
    pub x0: usize,
    pub y0: usize,
    pub x1: usize,
    pub y1: usize,
    /// Processing (model) size before /32 padding.
    pub pw: usize,
    pub ph: usize,
    /// Padded size fed to the model (multiples of 32).
    pub padw: usize,
    pub padh: usize,
}

impl CropPlan {
    pub fn cw(&self) -> usize {
        self.x1 - self.x0
    }
    pub fn ch(&self) -> usize {
        self.y1 - self.y0
    }
    pub fn scaled(&self) -> bool {
        self.pw != self.cw() || self.ph != self.ch()
    }
}

/// Inclusive bbox (xmin, xmax, ymin, ymax) of pixels with value > `thr`.
pub fn bbox_gt(m: &[u8], w: usize, h: usize, thr: u8) -> Option<(usize, usize, usize, usize)> {
    let (mut x0, mut x1, mut y0, mut y1) = (usize::MAX, 0usize, usize::MAX, 0usize);
    let mut any = false;
    for y in 0..h {
        let row = &m[y * w..(y + 1) * w];
        let first = row.iter().position(|&v| v > thr);
        if let Some(f) = first {
            let l = row.iter().rposition(|&v| v > thr).unwrap();
            any = true;
            x0 = x0.min(f);
            x1 = x1.max(l);
            y0 = y0.min(y);
            y1 = y1.max(y);
        }
    }
    any.then_some((x0, x1, y0, y1))
}

/// Production crop arithmetic. `bbox` inclusive (xmin, xmax, ymin, ymax).
/// Note production's right/bottom edge: `min(W, xmax + padding)` (exclusive),
/// i.e. one pixel less margin than left/top — reproduced on purpose.
pub fn crop_plan(fw: usize, fh: usize, bbox: (usize, usize, usize, usize), padding: usize, max_crop: usize) -> Option<CropPlan> {
    let (xmin, xmax, ymin, ymax) = bbox;
    let x0 = xmin.saturating_sub(padding);
    let y0 = ymin.saturating_sub(padding);
    let x1 = fw.min(xmax + padding);
    let y1 = fh.min(ymax + padding);
    if x1 <= x0 || y1 <= y0 {
        return None;
    }
    let (cw, ch) = (x1 - x0, y1 - y0);
    let mc = max_crop.max(32);
    let (pw, ph) = if cw.max(ch) > mc {
        if cw >= ch {
            (mc, (((ch * mc) as f64 / cw as f64).round_ties_even() as usize).max(1))
        } else {
            ((((cw * mc) as f64 / ch as f64).round_ties_even() as usize).max(1), mc)
        }
    } else {
        (cw, ch)
    };
    Some(CropPlan { x0, y0, x1, y1, pw, ph, padw: pw.div_ceil(32) * 32, padh: ph.div_ceil(32) * 32 })
}

/// Crop `[x0,x1) x [y0,y1)` of a `c`-channel top-down image.
pub fn crop<T: Copy>(src: &[T], w: usize, c: usize, x0: usize, y0: usize, x1: usize, y1: usize) -> Vec<T> {
    let cw = x1 - x0;
    let mut out = Vec::with_capacity(cw * (y1 - y0) * c);
    for y in y0..y1 {
        out.extend_from_slice(&src[(y * w + x0) * c..(y * w + x1) * c]);
    }
    out
}

/// `np.pad(..., ((0, ph-h), (0, pw-w)), mode="edge")`.
pub fn edge_pad(src: &[u8], w: usize, h: usize, c: usize, pw: usize, ph: usize) -> Vec<u8> {
    if pw == w && ph == h {
        return src.to_vec();
    }
    let mut out = vec![0u8; pw * ph * c];
    for y in 0..ph {
        let sy = y.min(h - 1);
        let srow = &src[sy * w * c..(sy + 1) * w * c];
        let orow = &mut out[y * pw * c..(y + 1) * pw * c];
        orow[..w * c].copy_from_slice(srow);
        for x in w..pw {
            orow[x * c..(x + 1) * c].copy_from_slice(&srow[(w - 1) * c..w * c]);
        }
    }
    out
}

/// HF VitMatteImageProcessor: image `(x * (1/255) - 0.5) / 0.5` (rescale in
/// f64 → f32, normalise in f32), trimap `x * (1/255)`; NCHW [1, 4, h, w].
pub fn hf_pixel_values(img_rgb: &[u8], tri: &[u8], w: usize, h: usize) -> Vec<f32> {
    let n = w * h;
    let mut pv = vec![0f32; 4 * n];
    let s = 1.0f64 / 255.0;
    for i in 0..n {
        for ch in 0..3 {
            let r = (img_rgb[i * 3 + ch] as f64 * s) as f32;
            pv[ch * n + i] = (r - 0.5f32) / 0.5f32;
        }
        pv[3 * n + i] = (tri[i] as f64 * s) as f32;
    }
    pv
}

/// Production `(alpha * 255).clip(0, 255).astype(uint8)` (truncation).
/// Returns None if any value is non-finite (corrupt model output).
pub fn alpha_to_u8(alpha: &[f32]) -> Option<Vec<u8>> {
    let mut out = Vec::with_capacity(alpha.len());
    for &a in alpha {
        if !a.is_finite() {
            return None;
        }
        out.push((a * 255.0f32).clamp(0.0, 255.0) as u8);
    }
    Some(out)
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn ellipse_k4_k30_shapes() {
        // k=4: rows dy -2..1; centre row full width offsets -2..1.
        let r = ellipse_rows(4);
        assert_eq!(r[0], (-2, 0, 0));
        assert!(r.contains(&(0, -2, 1)));
        let r = ellipse_rows(30);
        assert_eq!(r.len(), 30);
        assert!(r.contains(&(0, -15, 14)));
    }
    #[test]
    fn window_max_matches_naive() {
        let (w, h) = (37, 3);
        let src: Vec<u8> = (0..w * h).map(|i| ((i * 7919) % 251) as u8).collect();
        for (a, b) in [(-3, 3), (-5, 0), (0, 4), (-15, 14), (0, 0)] {
            let got = row_window_max(&src, w, h, a, b, 0);
            for y in 0..h {
                for x in 0..w {
                    let mut m = 0u8;
                    for d in a..=b {
                        let sx = x as i32 + d;
                        if sx >= 0 && (sx as usize) < w {
                            m = m.max(src[y * w + sx as usize]);
                        }
                    }
                    assert_eq!(got[y * w + x], m, "a={a} b={b} x={x}");
                }
            }
        }
    }
    #[test]
    fn crop_plan_production_cases() {
        // 104_1030 f1030 (ref meta: bbox [399,313,1150,1080], proc 751x767)
        let p = crop_plan(2048, 1080, (527, 1022, 441, 1079), 128, 2048).unwrap();
        assert_eq!((p.x0, p.y0, p.x1, p.y1), (399, 313, 1150, 1080));
        assert_eq!((p.pw, p.ph, p.padw, p.padh), (751, 767, 768, 768));
        // stretched 4x: crop 2239x767 -> 2048x702
        let p = crop_plan(8192, 1080, (2107, 4090, 441, 1079), 128, 2048).unwrap();
        assert_eq!((p.cw(), p.ch(), p.pw, p.ph), (2239, 767, 2048, 702));
    }
    #[test]
    fn hf_norm_values() {
        let pv = hf_pixel_values(&[0, 255, 128], &[128], 1, 1);
        assert_eq!(pv[0], -1.0);
        assert_eq!(pv[1], 1.0);
        assert!((pv[2] - 0.003_921_628).abs() < 1e-6);
        assert!((pv[3] - 0.501_960_8).abs() < 1e-6);
    }
    /// Threaded paths (used on real crops) == serial path == brute force.
    #[test]
    fn threaded_matches_serial_and_bruteforce() {
        use std::sync::atomic::Ordering;
        let (w, h) = (403usize, 287usize);
        let mut s = 12345u64;
        let mut img: Vec<u8> = (0..w * h * 3)
            .map(|_| {
                s = s.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
                (s >> 56) as u8
            })
            .collect();
        for (i, v) in img.iter_mut().enumerate() {
            if (i / 3 / w) % 40 < 20 {
                *v = (*v / 4).saturating_add(180); // structure, not only noise
            }
        }
        let gray: Vec<u8> = img.iter().step_by(3).copied().collect();
        let run = || {
            (
                lanczos_u8(&img, w, h, 3, 311, 199),
                lanczos_u8(&gray, w, h, 1, 517, 401),
                dilate_ellipse(&gray, w, h, 30),
                erode_ellipse(&gray, w, h, 4),
                dilate_rect(&gray, w, h, 7),
            )
        };
        let par = run();
        FORCE_SERIAL.store(true, Ordering::Relaxed);
        let ser = run();
        FORCE_SERIAL.store(false, Ordering::Relaxed);
        assert!(par.0 == ser.0 && par.1 == ser.1 && par.2 == ser.2 && par.3 == ser.3 && par.4 == ser.4);
        // brute-force ellipse dilate (k=30) from the structuring element definition
        let rows = ellipse_rows(30);
        for y in (0..h).step_by(7) {
            for x in 0..w {
                let mut m = 0u8;
                for &(dy, a, b) in &rows {
                    for dx in a..=b {
                        let (sx, sy) = (x as i32 + dx, y as i32 + dy);
                        if sx >= 0 && sy >= 0 && (sx as usize) < w && (sy as usize) < h {
                            m = m.max(gray[sy as usize * w + sx as usize]);
                        }
                    }
                }
                assert_eq!(par.2[y * w + x], m, "({x},{y})");
            }
        }
    }
    #[test]
    fn lanczos_identity_and_constant() {
        let src = vec![77u8; 50 * 20];
        assert_eq!(lanczos_u8(&src, 50, 20, 1, 50, 20), src);
        let d = lanczos_u8(&src, 50, 20, 1, 31, 13);
        assert!(d.iter().all(|&v| v == 77));
    }

    /// Band From = Mid Point: the band no longer depends on the mask softness,
    /// and the level shifts it.
    #[test]
    fn mid_point_band_ignores_softness_and_moves_with_the_level() {
        let (w, h) = (200usize, 9usize);
        // vertical edge at x = 100: hard step vs a 60 px linear ramp with the same 0.5 crossing
        let seed = |ramp: f64| -> Vec<u8> {
            (0..w * h).map(|i| {
                let x = (i % w) as f64 + 0.5;
                let v = if ramp <= 0.0 { if x < 100.0 { 255.0 } else { 0.0 } } else { ((100.0 - x) / ramp + 0.5).clamp(0.0, 1.0) * 255.0 };
                v.round() as u8
            }).collect()
        };
        let row = |t: &[u8]| t[4 * w..5 * w].to_vec();
        let band = |t: &[u8]| { let r = row(t); (r.iter().position(|v| *v != 255).unwrap(), r.iter().position(|v| *v == 0).unwrap()) };
        let mut p = TrimapParams { kernel_size: 15, asym: Some((9, 21)), mid_bias: None, mid_point: Some(0.5), seed_clamp: 250 };
        let hard = gen_trimap(&seed(0.0), w, h, &p);
        let soft = gen_trimap(&seed(60.0), w, h, &p);
        assert_eq!(row(&hard), row(&soft));
        let (in0, out0) = band(&soft);
        // Mask Edges (production): the soft ramp widens the band
        p.mid_point = None;
        let (in1, out1) = band(&gen_trimap(&seed(60.0), w, h, &p));
        assert!(out1 - in1 > out0 - in0 + 40, "edges {in1}..{out1} vs mid {in0}..{out0}");
        // a lower level moves the whole band out (toward the background)
        p.mid_point = Some(0.2);
        let (in2, out2) = band(&gen_trimap(&seed(60.0), w, h, &p));
        assert!(in2 > in0 && out2 > out0 && out2 - in2 == out0 - in0, "{in2}..{out2} vs {in0}..{out0}");
    }
}
