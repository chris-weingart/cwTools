//! Pillow-exact `Image.resize` (libImaging/Resample.c) for the cwTools models,
//! which were all trained / validated on PIL-resized inputs:
//!
//! - [`resize_u8`]: 8-bit L / RGB images (`precompute_coeffs` + 22-bit
//!   fixed-point `normalize_coeffs_8bpc`; horizontal pass, then vertical,
//!   each rounded to uint8). Bit-exact to Pillow.
//! - [`resize_f32`]: mode "F" images (double coefficients, double
//!   accumulation, float32 store). Matches Pillow to float rounding.
//!
//! Filters: [`Filter::Bilinear`] (support 1) and [`Filter::Lanczos`]
//! (support 3). Like Pillow, downscaling widens the filter by the scale
//! factor (antialiasing); upscaling is plain interpolation.
//!
//! All buffers are top-down, row-major, `c` interleaved channels.
//! (Factored out of cwRefine's recipe, where Lanczos is production-pinned.)

use std::sync::atomic::{AtomicBool, Ordering};

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Filter {
    /// `Image.BILINEAR` (triangle, support 1).
    Bilinear,
    /// `Image.LANCZOS` (Lanczos-3).
    Lanczos,
}

impl Filter {
    fn support(self) -> f64 {
        match self {
            Filter::Bilinear => 1.0,
            Filter::Lanczos => 3.0,
        }
    }
    #[inline]
    fn eval(self, x: f64) -> f64 {
        match self {
            Filter::Bilinear => {
                let x = x.abs();
                if x < 1.0 {
                    1.0 - x
                } else {
                    0.0
                }
            }
            Filter::Lanczos => {
                if (-3.0..3.0).contains(&x) {
                    sinc(x) * sinc(x / 3.0)
                } else {
                    0.0
                }
            }
        }
    }
}

const PRECISION_BITS: u32 = 32 - 8 - 2;

#[inline]
fn sinc(x: f64) -> f64 {
    if x == 0.0 {
        return 1.0;
    }
    let x = x * std::f64::consts::PI;
    x.sin() / x
}

/// Per-output-sample (first tap, tap count) + normalised double weights
/// (`ksize` slots per sample) — Pillow `precompute_coeffs`.
struct Coeffs {
    ksize: usize,
    bounds: Vec<(usize, usize)>,
    k: Vec<f64>,
}

fn precompute_coeffs(in_size: usize, out_size: usize, f: Filter) -> Coeffs {
    // Pillow's box is float[4]: (double)(in1 - in0) with float operands.
    let scale = in_size as f32 as f64 / out_size as f64;
    let filterscale = scale.max(1.0);
    let support = f.support() * filterscale;
    let ksize = support.ceil() as usize * 2 + 1;
    let mut bounds = Vec::with_capacity(out_size);
    let mut k = vec![0f64; out_size * ksize];
    for xx in 0..out_size {
        let center = 0.0 + (xx as f64 + 0.5) * scale;
        let ss = 1.0 / filterscale;
        // C `(int)` truncates toward zero.
        let mut xmin = (center - support + 0.5) as i64;
        if xmin < 0 {
            xmin = 0;
        }
        let mut xmax = (center + support + 0.5) as i64;
        if xmax > in_size as i64 {
            xmax = in_size as i64;
        }
        let n = (xmax - xmin).max(0) as usize;
        let kk = &mut k[xx * ksize..xx * ksize + n];
        let mut ww = 0.0;
        for (x, slot) in kk.iter_mut().enumerate() {
            let w = f.eval(((x as i64 + xmin) as f64 - center + 0.5) * ss);
            *slot = w;
            ww += w;
        }
        if ww != 0.0 {
            kk.iter_mut().for_each(|w| *w /= ww);
        }
        bounds.push((xmin as usize, n));
    }
    Coeffs { ksize, bounds, k }
}

/// Pillow `normalize_coeffs_8bpc`.
fn fixed_point(co: &Coeffs) -> Vec<i32> {
    co.k
        .iter()
        .map(|&p| {
            let v = p * (1u32 << PRECISION_BITS) as f64;
            if p < 0.0 {
                (-0.5 + v) as i32
            } else {
                (0.5 + v) as i32
            }
        })
        .collect()
}

#[inline]
fn clip8(v: i64) -> u8 {
    let s = v >> PRECISION_BITS;
    s.clamp(0, 255) as u8
}

/// Tests: force the single-threaded path (to compare against the threaded one).
pub static FORCE_SERIAL: AtomicBool = AtomicBool::new(false);

/// Run `f(row_index, row)` over the `row_len`-sized rows of `out`, split into
/// contiguous bands across threads (plain std::thread::scope, no pool). Small
/// jobs run inline.
pub fn par_rows<T: Send>(out: &mut [T], row_len: usize, f: impl Fn(usize, &mut [T]) + Sync) {
    let rows = if row_len == 0 { 0 } else { out.len() / row_len };
    let threads = std::thread::available_parallelism().map(|n| n.get()).unwrap_or(1).min(16);
    if rows < 64 || threads <= 1 || out.len() < 1 << 16 || FORCE_SERIAL.load(Ordering::Relaxed) {
        for (i, r) in out.chunks_mut(row_len.max(1)).enumerate() {
            f(i, r);
        }
        return;
    }
    let per = rows.div_ceil(threads);
    std::thread::scope(|sc| {
        for (bi, band) in out.chunks_mut(per * row_len).enumerate() {
            let f = &f;
            sc.spawn(move || {
                for (i, r) in band.chunks_mut(row_len).enumerate() {
                    f(bi * per + i, r);
                }
            });
        }
    });
}

fn h_u8(src: &[u8], sw: usize, sh: usize, c: usize, dw: usize, f: Filter) -> Vec<u8> {
    let co = precompute_coeffs(sw, dw, f);
    let kk = fixed_point(&co);
    let mut out = vec![0u8; dw * sh * c];
    par_rows(&mut out, dw * c, |y, orow| {
        let row = &src[y * sw * c..(y + 1) * sw * c];
        for xx in 0..dw {
            let (xmin, n) = co.bounds[xx];
            let k = &kk[xx * co.ksize..xx * co.ksize + n];
            for ch in 0..c {
                let mut ss: i64 = 1 << (PRECISION_BITS - 1);
                for (x, &kv) in k.iter().enumerate() {
                    ss += row[(x + xmin) * c + ch] as i64 * kv as i64;
                }
                orow[xx * c + ch] = clip8(ss);
            }
        }
    });
    out
}

fn v_u8(src: &[u8], sw: usize, sh: usize, c: usize, dh: usize, f: Filter) -> Vec<u8> {
    let co = precompute_coeffs(sh, dh, f);
    let kk = fixed_point(&co);
    let stride = sw * c;
    let mut out = vec![0u8; stride * dh];
    par_rows(&mut out, stride, |yy, orow| {
        let (ymin, n) = co.bounds[yy];
        let k = &kk[yy * co.ksize..yy * co.ksize + n];
        let mut acc = vec![1i64 << (PRECISION_BITS - 1); stride];
        for (y, &kv) in k.iter().enumerate() {
            let row = &src[(y + ymin) * stride..(y + ymin + 1) * stride];
            for (a, &p) in acc.iter_mut().zip(row) {
                *a += p as i64 * kv as i64;
            }
        }
        for (o, &a) in orow.iter_mut().zip(&acc) {
            *o = clip8(a);
        }
    });
    out
}

/// Pillow `Image.resize((dw, dh), filter)` for 8-bit L (c=1) / RGB (c=3).
pub fn resize_u8(src: &[u8], sw: usize, sh: usize, c: usize, dw: usize, dh: usize, f: Filter) -> Vec<u8> {
    assert_eq!(src.len(), sw * sh * c);
    if sw == dw && sh == dh {
        return src.to_vec();
    }
    let mut cur = src.to_vec();
    let mut w = sw;
    if dw != sw {
        // Pillow only resamples the rows the vertical pass reads; computing
        // all rows gives identical values.
        cur = h_u8(&cur, sw, sh, c, dw, f);
        w = dw;
    }
    if dh != sh {
        cur = v_u8(&cur, w, sh, c, dh, f);
    }
    cur
}

/// Pillow `Image.resize((dw, dh), filter)` for a mode "F" (float32) image.
pub fn resize_f32(src: &[f32], sw: usize, sh: usize, dw: usize, dh: usize, f: Filter) -> Vec<f32> {
    assert_eq!(src.len(), sw * sh);
    if sw == dw && sh == dh {
        return src.to_vec();
    }
    let mut cur = src.to_vec();
    let mut w = sw;
    if dw != sw {
        let co = precompute_coeffs(sw, dw, f);
        let mut out = vec![0f32; dw * sh];
        par_rows(&mut out, dw, |y, orow| {
            let row = &cur[y * sw..(y + 1) * sw];
            for (xx, o) in orow.iter_mut().enumerate() {
                let (xmin, n) = co.bounds[xx];
                let k = &co.k[xx * co.ksize..xx * co.ksize + n];
                let mut ss = 0.0f64;
                for (x, &kv) in k.iter().enumerate() {
                    ss += row[x + xmin] as f64 * kv;
                }
                *o = ss as f32;
            }
        });
        cur = out;
        w = dw;
    }
    if dh != sh {
        let co = precompute_coeffs(sh, dh, f);
        let mut out = vec![0f32; w * dh];
        par_rows(&mut out, w, |yy, orow| {
            let (ymin, n) = co.bounds[yy];
            let k = &co.k[yy * co.ksize..yy * co.ksize + n];
            let mut acc = vec![0f64; w];
            for (y, &kv) in k.iter().enumerate() {
                let row = &cur[(y + ymin) * w..(y + ymin + 1) * w];
                for (a, &p) in acc.iter_mut().zip(row) {
                    *a += p as f64 * kv;
                }
            }
            for (o, &a) in orow.iter_mut().zip(&acc) {
                *o = a as f32;
            }
        });
        cur = out;
    }
    cur
}

#[cfg(test)]
mod tests {
    use super::*;

    fn noise(n: usize, seed: u64) -> Vec<u8> {
        let mut s = seed;
        (0..n)
            .map(|_| {
                s = s.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
                (s >> 56) as u8
            })
            .collect()
    }

    #[test]
    fn identity_and_constant() {
        for f in [Filter::Bilinear, Filter::Lanczos] {
            let src = vec![77u8; 50 * 20];
            assert_eq!(resize_u8(&src, 50, 20, 1, 50, 20, f), src);
            assert!(resize_u8(&src, 50, 20, 1, 31, 13, f).iter().all(|&v| v == 77));
            assert!(resize_u8(&src, 50, 20, 1, 97, 64, f).iter().all(|&v| v == 77));
            let fs = vec![0.25f32; 50 * 20];
            assert!(resize_f32(&fs, 50, 20, 31, 77, f).iter().all(|&v| (v - 0.25).abs() < 1e-7));
        }
    }

    #[test]
    fn bilinear_upscale_2x_is_quarter_weights() {
        // Pillow BILINEAR 2 -> 4 (upscale): centres at 0.25, 0.75, 1.25, 1.75
        // in input space; edge samples clamp to the outermost pixel.
        let o = resize_f32(&[0.0, 1.0], 2, 1, 4, 1, Filter::Bilinear);
        assert_eq!(o, vec![0.0, 0.25, 0.75, 1.0]);
    }

    #[test]
    fn threaded_matches_serial() {
        let (w, h) = (403usize, 287usize);
        let img = noise(w * h * 3, 7);
        let fimg: Vec<f32> = img.iter().step_by(3).map(|&v| v as f32 / 255.0).collect();
        let run = || {
            (
                resize_u8(&img, w, h, 3, 1024, 1024, Filter::Bilinear),
                resize_u8(&img, w, h, 3, 211, 99, Filter::Bilinear),
                resize_f32(&fimg, w, h, 777, 133, Filter::Bilinear),
            )
        };
        let par = run();
        FORCE_SERIAL.store(true, Ordering::Relaxed);
        let ser = run();
        FORCE_SERIAL.store(false, Ordering::Relaxed);
        assert!(par.0 == ser.0 && par.1 == ser.1 && par.2 == ser.2);
    }
}
