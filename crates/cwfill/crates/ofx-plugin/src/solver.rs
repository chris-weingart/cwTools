//! Faithful Rust port of pymatting's `estimate_foreground_ml`
//! (Germer et al. 2020, "Fast Multi-Level Foreground Estimation").
//!
//! Ported verbatim from the numba kernel `_estimate_fb_ml`
//! (pymatting/foreground/estimate_foreground_ml.py, master). Same multilevel
//! pyramid, same per-pixel 2x2 Gauss-Seidel update, same alpha-gradient
//! smoothness weight. Sequential sweep (the upstream `@njit` has no
//! `parallel=True`, so `prange` is plain `range` → in-place Gauss-Seidel).
//!
//! Contract: `image` channels and `alpha` must be in [0, 1] (the kernel clamps
//! F/B to [0, 1] each iteration). Colour-space encoding / HDR normalisation is
//! the caller's job — see the Flame OFX colour-management path. This module is
//! pure math and host-agnostic, so it is unit-testable without Flame.
//!
//! Row order matters: the in-place Gauss-Seidel sweep runs row 0 first, and
//! pymatting's row 0 is the TOP of the frame. Callers holding bottom-up
//! buffers (OFX) must flip rows before and after the solve — the result is
//! NOT flip-equivariant in under-constrained (faint-alpha) regions (see
//! `sweep_order_is_direction_sensitive`).

/// Image stored row-major as (h, w, depth), alpha as (h, w).
pub struct Plane {
    pub h: usize,
    pub w: usize,
    pub depth: usize,
    pub data: Vec<f32>,
}

impl Plane {
    pub fn new(h: usize, w: usize, depth: usize) -> Self {
        Plane { h, w, depth, data: vec![0.0; h * w * depth] }
    }
    #[inline]
    fn at(&self, y: usize, x: usize, c: usize) -> f32 {
        self.data[(y * self.w + x) * self.depth + c]
    }
    #[inline]
    fn set(&mut self, y: usize, x: usize, c: usize, v: f32) {
        self.data[(y * self.w + x) * self.depth + c] = v;
    }
}

/// Python's `round()` — round half to even (banker's rounding) — so pyramid
/// level sizes match pymatting exactly.
fn py_round(x: f64) -> usize {
    let f = x.floor();
    let diff = x - f;
    let r = if diff > 0.5 {
        f + 1.0
    } else if diff < 0.5 {
        f
    } else {
        // exactly .5 → round to even
        if (f as i64) % 2 == 0 { f } else { f + 1.0 }
    };
    r.max(1.0) as usize
}

/// Nearest-neighbour resize of a multichannel plane into a (h_dst, w_dst) grid,
/// matching pymatting's `x_src = x_dst * w_src // w_dst` integer mapping.
fn resize_nearest(src: &Plane, h_dst: usize, w_dst: usize) -> Plane {
    let mut dst = Plane::new(h_dst, w_dst, src.depth);
    for y in 0..h_dst {
        let y_src = ((y * src.h) / h_dst).min(src.h - 1);
        for x in 0..w_dst {
            let x_src = ((x * src.w) / w_dst).min(src.w - 1);
            for c in 0..src.depth {
                dst.set(y, x, c, src.at(y_src, x_src, c));
            }
        }
    }
    dst
}

fn resize_nearest_alpha(src: &[f32], sh: usize, sw: usize, h_dst: usize, w_dst: usize) -> Vec<f32> {
    let mut dst = vec![0.0f32; h_dst * w_dst];
    for y in 0..h_dst {
        let y_src = ((y * sh) / h_dst).min(sh - 1);
        for x in 0..w_dst {
            let x_src = ((x * sw) / w_dst).min(sw - 1);
            dst[y * w_dst + x] = src[y_src * sw + x_src];
        }
    }
    dst
}

pub struct FgParams {
    pub regularization: f32,
    pub n_small_iterations: i32,
    pub n_big_iterations: i32,
    pub small_size: i32,
    pub gradient_weight: f32,
    /// Image-aware BACKGROUND affinity (0 = off = exact pymatting parity).
    ///
    /// pymatting's smoothness weight sees only the ALPHA gradient, never image
    /// edges — so the estimated background is forced continuous across real
    /// background discontinuities (a seam, a pattern edge), and the residual
    /// is dumped into F: coloured chips in the fill wherever the soft band
    /// crosses background structure. When > 0, the background smoothness
    /// coupling to each neighbour is attenuated by `exp(-s · dB²)`, where
    /// `dB² = |ΔI|² / max((1-ᾱ)², 0.01)` estimates the TRUE background edge
    /// strength (under the mix model I = αF + (1-α)B, background edges appear
    /// in the composite attenuated by (1-α)). The estimated background may
    /// then break at background edges instead of contaminating F.
    ///
    /// Deliberate departure from pymatting (2026-07-29 bars-lab study,
    /// examples/bars_lab.rs): B-only attenuation cut band error ~10x on a
    /// seam fixture with zero regression on flat or grainy backgrounds.
    /// FOREGROUND coupling is intentionally untouched — F must keep
    /// propagating across image edges (attenuating F measured ~40% WORSE
    /// than baseline). Useful range ~30–300; saturates around 100.
    pub image_affinity: f32,
}

impl Default for FgParams {
    fn default() -> Self {
        FgParams {
            regularization: 1e-5,
            n_small_iterations: 10,
            n_big_iterations: 2,
            small_size: 32,
            gradient_weight: 1.0,
            image_affinity: 0.0,
        }
    }
}

/// Estimate foreground (and background) colour from `image` + `alpha`.
/// Returns (F, B) as full-resolution planes. Single-threaded; see
/// [`estimate_fb_ml_mt`].
pub fn estimate_fb_ml(image: &Plane, alpha: &[f32], p: &FgParams) -> (Plane, Plane) {
    estimate_fb_ml_mt(image, alpha, p, 1)
}

/// The v0.2.0 solver, kept VERBATIM as the parity oracle: a direct port of
/// `_estimate_fb_ml` with generic channel depth and bounds-checked access.
/// The fast path ([`estimate_fb_ml_mt`]) must reproduce it bit-for-bit — the
/// tests below and `examples/parity.rs` enforce that. Also the fallback for
/// depth != 3.
pub fn estimate_fb_ml_reference(
    image: &Plane,
    alpha: &[f32],
    p: &FgParams,
) -> (Plane, Plane) {
    let (h0, w0, depth) = (image.h, image.w, image.depth);
    assert_eq!(alpha.len(), h0 * w0, "alpha must be h0*w0");

    // Average FG / BG colour (alpha > 0.9 / < 0.1).
    let mut f_mean = vec![0.0f32; depth];
    let mut b_mean = vec![0.0f32; depth];
    let (mut f_count, mut b_count) = (0.0f32, 0.0f32);
    for y in 0..h0 {
        for x in 0..w0 {
            let a = alpha[y * w0 + x];
            if a > 0.9 {
                for c in 0..depth {
                    f_mean[c] += image.at(y, x, c);
                }
                f_count += 1.0;
            }
            if a < 0.1 {
                for c in 0..depth {
                    b_mean[c] += image.at(y, x, c);
                }
                b_count += 1.0;
            }
        }
    }
    for c in 0..depth {
        f_mean[c] /= f_count + 1e-5;
        b_mean[c] /= b_count + 1e-5;
    }

    // 1x1 seed planes filled with the mean colours.
    let mut f_prev = Plane::new(1, 1, depth);
    let mut b_prev = Plane::new(1, 1, depth);
    for c in 0..depth {
        f_prev.set(0, 0, c, f_mean[c]);
        b_prev.set(0, 0, c, b_mean[c]);
    }

    let n_levels = (((w0.max(h0)) as f64).log2()).ceil() as i32;
    let n_levels = n_levels.max(0);

    let dx: [isize; 4] = [-1, 1, 0, 0];
    let dy: [isize; 4] = [0, 0, -1, 1];

    let mut f = f_prev;
    let mut b = b_prev;

    for i_level in 0..=n_levels {
        let frac = i_level as f64 / n_levels.max(1) as f64;
        let w = py_round((w0 as f64).powf(frac));
        let h = py_round((h0 as f64).powf(frac));

        let image_l = resize_nearest(image, h, w);
        let alpha_l = resize_nearest_alpha(alpha, h0, w0, h, w);

        f = resize_nearest(&f, h, w);
        b = resize_nearest(&b, h, w);

        let n_iter = if (w as i32) <= p.small_size && (h as i32) <= p.small_size {
            p.n_small_iterations
        } else {
            p.n_big_iterations
        };

        for _ in 0..n_iter {
            for y in 0..h {
                for x in 0..w {
                    let a0 = alpha_l[y * w + x];
                    let a1 = 1.0 - a0;

                    let mut a00 = a0 * a0;
                    let a01 = a0 * a1;
                    let mut a11 = a1 * a1;

                    let mut b0 = [0.0f32; 8];
                    let mut b1 = [0.0f32; 8];
                    for c in 0..depth {
                        b0[c] = a0 * image_l.at(y, x, c);
                        b1[c] = a1 * image_l.at(y, x, c);
                    }

                    // Smoothness sums, tracked separately so the image-aware
                    // path can compute the determinant in its cancellation-free
                    // form (see below).
                    let mut s_f = 0.0f32;
                    let mut s_b = 0.0f32;

                    for d in 0..4 {
                        let x2 = (x as isize + dx[d]).clamp(0, w as isize - 1) as usize;
                        let y2 = (y as isize + dy[d]).clamp(0, h as isize - 1) as usize;

                        let gradient = (a0 - alpha_l[y2 * w + x2]).abs();
                        let da = p.regularization + p.gradient_weight * gradient;

                        // Image-aware background affinity (see FgParams doc):
                        // let estimated B break at background edges. 0 = off,
                        // and da_b == da bit-exactly (pymatting parity).
                        let da_b = if p.image_affinity > 0.0 {
                            let mut di2 = 0.0f32;
                            for c in 0..depth {
                                let dc = image_l.at(y, x, c) - image_l.at(y2, x2, c);
                                di2 += dc * dc;
                            }
                            let a_mid = 0.5 * (a0 + alpha_l[y2 * w + x2]);
                            let inv = 1.0 - a_mid;
                            let mut db2 = di2 / (inv * inv).max(0.01);
                            // Observability gate: fade the attenuation out as
                            // alpha -> 1. Near-solid pixels carry no data
                            // constraint on B — smoothness is the only thing
                            // holding B there, and the (1-a)² normalisation
                            // would otherwise amplify NOISE differences x100
                            // (worst in cct, which expands dark-tone noise)
                            // and cut B's last anchor: B degenerates and the
                            // data term drags F dark (black chips).
                            db2 *= ((0.97 - a_mid) / 0.07).clamp(0.0, 1.0);
                            // Floor keeps the 2x2 system conditioned where the
                            // data term vanishes (a1 -> 0 with all couplings cut).
                            (da * (-p.image_affinity * db2).exp()).max(1e-8)
                        } else {
                            da
                        };

                        a00 += da;
                        a11 += da_b;
                        s_f += da;
                        s_b += da_b;

                        for c in 0..depth {
                            b0[c] += da * f.at(y2, x2, c);
                            b1[c] += da_b * b.at(y2, x2, c);
                        }
                    }

                    // Determinant of the 2x2 system. The textbook form
                    // a00*a11 - a01² subtracts two near-equal ~a0²a1² products;
                    // when image-aware attenuation cuts s_b to the 1e-8 floor
                    // the true value is ~a0²·s_b and f32 cancellation error can
                    // flip it NEGATIVE -> garbage F/B (clamped to black).
                    // Expanding symbolically: det = a0²·s_b + a1²·s_f + s_f·s_b
                    // (the a0²a1² terms cancel exactly) — every term positive,
                    // no cancellation. Only used on the image-aware path to
                    // keep the off path bit-identical to pymatting.
                    let determinant = if p.image_affinity > 0.0 {
                        a0 * a0 * s_b + a1 * a1 * s_f + s_f * s_b
                    } else {
                        a00 * a11 - a01 * a01
                    };
                    let inv_det = 1.0 / determinant;
                    let b00 = inv_det * a11;
                    let b01 = inv_det * -a01;
                    let b11 = inv_det * a00;

                    for c in 0..depth {
                        let f_c = (b00 * b0[c] + b01 * b1[c]).clamp(0.0, 1.0);
                        let b_c = (b01 * b0[c] + b11 * b1[c]).clamp(0.0, 1.0);
                        f.set(y, x, c, f_c);
                        b.set(y, x, c, b_c);
                    }
                }
            }
        }
    }

    (f, b)
}

// ---------------------------------------------------------------------------
// Fast path (v0.3.0). Same arithmetic in the same operation order as the
// reference — f32 ops are never reordered or fused by rustc, so identical
// source-level order means identical bits. Differences are purely mechanical:
// fixed 3 channels, unchecked indexing, index-table resizes, and a pipelined
// multi-threaded sweep that preserves the exact Gauss-Seidel read set.
// ---------------------------------------------------------------------------

use crate::par::{self, SendPtr};
use std::borrow::Cow;
use std::sync::atomic::{AtomicU32, Ordering};

/// Column block for the pipelined sweep (pixels per progress publish).
const SWEEP_BLOCK: usize = 64;

/// Levels smaller than this run sequentially (thread hand-off would dominate).
const PAR_MIN_PIXELS: usize = 65_536;

/// Nearest-neighbour resize with a precomputed column table (no per-pixel
/// 64-bit divide), rows split across threads. Same `src * dst_idx / dst_len`
/// mapping as `resize_nearest`, so bit-identical. Identity → borrowed.
fn resize_tab<'a>(
    src: &'a [f32],
    sh: usize,
    sw: usize,
    depth: usize,
    h_dst: usize,
    w_dst: usize,
    threads: usize,
) -> Cow<'a, [f32]> {
    if sh == h_dst && sw == w_dst {
        // x_src == x and y_src == y: a plain copy — skip it.
        return Cow::Borrowed(src);
    }
    let xs: Vec<usize> = (0..w_dst).map(|x| ((x * sw) / w_dst).min(sw - 1)).collect();
    let mut dst = vec![0f32; h_dst * w_dst * depth];
    let row = w_dst * depth;
    let t = par::threads_for(h_dst * w_dst, threads);
    let rpj = par::rows_per_job(h_dst, t);
    par::chunks_mut(&mut dst, rpj * row, t, |ci, chunk| {
        for (ry, drow) in chunk.chunks_mut(row).enumerate() {
            let y = ci * rpj + ry;
            let y_src = ((y * sh) / h_dst).min(sh - 1);
            let srow = &src[y_src * sw * depth..(y_src + 1) * sw * depth];
            for (x, &xsrc) in xs.iter().enumerate() {
                let xi = xsrc * depth;
                for c in 0..depth {
                    drow[x * depth + c] = srow[xi + c];
                }
            }
        }
    });
    Cow::Owned(dst)
}

#[derive(Clone, Copy)]
struct Kern {
    w: usize,
    h: usize,
    alpha: *const f32,
    img: *const f32,
    f: SendPtr<f32>,
    b: SendPtr<f32>,
    reg: f32,
    gw: f32,
    aff: f32,
}
unsafe impl Send for Kern {}
unsafe impl Sync for Kern {}

/// One Gauss-Seidel pixel update, depth 3. Mirrors the reference loop body
/// statement for statement (neighbour order left, right, up, down; clamped).
#[inline(always)]
unsafe fn pixel<const AFF: bool>(k: &Kern, y: usize, x: usize) {
    let (w, h) = (k.w, k.h);
    let i = y * w + x;
    let a0 = *k.alpha.add(i);
    let a1 = 1.0 - a0;
    let mut a00 = a0 * a0;
    let a01 = a0 * a1;
    let mut a11 = a1 * a1;
    let ii = i * 3;
    let im = [*k.img.add(ii), *k.img.add(ii + 1), *k.img.add(ii + 2)];
    let mut b0 = [a0 * im[0], a0 * im[1], a0 * im[2]];
    let mut b1 = [a1 * im[0], a1 * im[1], a1 * im[2]];
    let mut s_f = 0.0f32;
    let mut s_b = 0.0f32;
    let xl = if x == 0 { 0 } else { x - 1 };
    let xr = if x + 1 >= w { w - 1 } else { x + 1 };
    let yu = if y == 0 { 0 } else { y - 1 };
    let yd = if y + 1 >= h { h - 1 } else { y + 1 };
    let ns = [y * w + xl, y * w + xr, yu * w + x, yd * w + x];
    let (f, b) = (k.f.0, k.b.0);
    for &n in ns.iter() {
        let an = *k.alpha.add(n);
        let gradient = (a0 - an).abs();
        let da = k.reg + k.gw * gradient;
        let da_b = if AFF {
            let nn = n * 3;
            let mut di2 = 0.0f32;
            for c in 0..3 {
                let dc = im[c] - *k.img.add(nn + c);
                di2 += dc * dc;
            }
            let a_mid = 0.5 * (a0 + an);
            let inv = 1.0 - a_mid;
            let mut db2 = di2 / (inv * inv).max(0.01);
            db2 *= ((0.97 - a_mid) / 0.07).clamp(0.0, 1.0);
            (da * (-k.aff * db2).exp()).max(1e-8)
        } else {
            da
        };
        a00 += da;
        a11 += da_b;
        s_f += da;
        s_b += da_b;
        let nn = n * 3;
        for c in 0..3 {
            b0[c] += da * *f.add(nn + c);
            b1[c] += da_b * *b.add(nn + c);
        }
    }
    let determinant = if AFF {
        a0 * a0 * s_b + a1 * a1 * s_f + s_f * s_b
    } else {
        a00 * a11 - a01 * a01
    };
    let inv_det = 1.0 / determinant;
    let b00 = inv_det * a11;
    let b01 = inv_det * -a01;
    let b11 = inv_det * a00;
    for c in 0..3 {
        let f_c = (b00 * b0[c] + b01 * b1[c]).clamp(0.0, 1.0);
        let b_c = (b01 * b0[c] + b11 * b1[c]).clamp(0.0, 1.0);
        *f.add(ii + c) = f_c;
        *b.add(ii + c) = b_c;
    }
}

#[inline(always)]
unsafe fn span<const AFF: bool>(k: &Kern, y: usize, x0: usize, x1: usize) {
    for x in x0..x1 {
        pixel::<AFF>(k, y, x);
    }
}

/// `n_iter` in-place Gauss-Seidel sweeps over one level.
///
/// Threaded = pipelined rows ("doacross" wavefront): the sweep is unrolled
/// into `n_iter * h` row-jobs in sequential order; row job r may update
/// column block j only once row job r-1 has finished block j. A pixel then
/// reads exactly what the sequential sweep reads — up/left already updated
/// this pass, down/right not yet — because (a) the row above finished this
/// block (Acquire on its progress counter), (b) the row below cannot start
/// this block until we publish it, and (c) by transitivity every earlier row
/// job has finished the block too (needs h >= 2, checked). So output is
/// bit-identical to `threads == 1` for any thread count. Rows are claimed in
/// increasing order from a shared counter, so the chain always has a running
/// head and fewer threads than requested (a failed spawn) only costs speed.
unsafe fn sweep<const AFF: bool>(k: &Kern, n_iter: usize, threads: usize) {
    let (w, h) = (k.w, k.h);
    let threads = threads.min(h / 2).max(1);
    if threads <= 1 || w * h < PAR_MIN_PIXELS || h < 2 {
        for _ in 0..n_iter {
            for y in 0..h {
                span::<AFF>(k, y, 0, w);
            }
        }
        return;
    }
    let nblk = (w + SWEEP_BLOCK - 1) / SWEEP_BLOCK;
    let nrows = n_iter * h;
    let progress: Vec<AtomicU32> = (0..nrows).map(|_| AtomicU32::new(0)).collect();
    let prog = &progress[..];
    let k = *k;
    par::run_jobs(nrows, threads, move |r| {
        let y = r % h;
        let mut j = 0usize;
        while j < nblk {
            // Blocks the previous row job has finished (all of them for r == 0).
            let ready = if r == 0 {
                nblk
            } else {
                let mut spins = 0u32;
                loop {
                    let p = prog[r - 1].load(Ordering::Acquire) as usize;
                    if p > j {
                        break p;
                    }
                    spins += 1;
                    if spins < 1 << 10 {
                        std::hint::spin_loop();
                    } else {
                        std::thread::yield_now();
                    }
                }
            };
            while j < ready {
                let x1 = ((j + 1) * SWEEP_BLOCK).min(w);
                span::<AFF>(&k, y, j * SWEEP_BLOCK, x1);
                j += 1;
                prog[r].store(j as u32, Ordering::Release);
            }
        }
    });
}

/// Estimate F/B on up to `threads` threads. Bit-identical to
/// [`estimate_fb_ml_reference`] for every thread count (tests +
/// `examples/parity.rs`). Depth != 3 falls back to the reference.
pub fn estimate_fb_ml_mt(image: &Plane, alpha: &[f32], p: &FgParams, threads: usize) -> (Plane, Plane) {
    if image.depth != 3 {
        return estimate_fb_ml_reference(image, alpha, p);
    }
    let (h0, w0) = (image.h, image.w);
    assert_eq!(alpha.len(), h0 * w0, "alpha must be h0*w0");

    // Seeds: identical accumulation order to the reference.
    let mut f_mean = [0.0f32; 3];
    let mut b_mean = [0.0f32; 3];
    let (mut f_count, mut b_count) = (0.0f32, 0.0f32);
    for i in 0..h0 * w0 {
        let a = alpha[i];
        if a > 0.9 {
            for c in 0..3 {
                f_mean[c] += image.data[i * 3 + c];
            }
            f_count += 1.0;
        }
        if a < 0.1 {
            for c in 0..3 {
                b_mean[c] += image.data[i * 3 + c];
            }
            b_count += 1.0;
        }
    }
    for c in 0..3 {
        f_mean[c] /= f_count + 1e-5;
        b_mean[c] /= b_count + 1e-5;
    }
    let mut f = Plane { h: 1, w: 1, depth: 3, data: f_mean.to_vec() };
    let mut b = Plane { h: 1, w: 1, depth: 3, data: b_mean.to_vec() };

    let n_levels = ((((w0.max(h0)) as f64).log2()).ceil() as i32).max(0);
    for i_level in 0..=n_levels {
        let frac = i_level as f64 / n_levels.max(1) as f64;
        let w = py_round((w0 as f64).powf(frac));
        let h = py_round((h0 as f64).powf(frac));

        let image_l = resize_tab(&image.data, h0, w0, 3, h, w, threads);
        let alpha_l = resize_tab(alpha, h0, w0, 1, h, w, threads);
        if f.h != h || f.w != w {
            f = Plane { h, w, depth: 3, data: resize_tab(&f.data, f.h, f.w, 3, h, w, threads).into_owned() };
            b = Plane { h, w, depth: 3, data: resize_tab(&b.data, b.h, b.w, 3, h, w, threads).into_owned() };
        }

        let n_iter = if (w as i32) <= p.small_size && (h as i32) <= p.small_size {
            p.n_small_iterations
        } else {
            p.n_big_iterations
        }
        .max(0) as usize;

        let k = Kern {
            w,
            h,
            alpha: alpha_l.as_ptr(),
            img: image_l.as_ptr(),
            f: SendPtr(f.data.as_mut_ptr()),
            b: SendPtr(b.data.as_mut_ptr()),
            reg: p.regularization,
            gw: p.gradient_weight,
            aff: p.image_affinity,
        };
        unsafe {
            if p.image_affinity > 0.0 {
                sweep::<true>(&k, n_iter, threads);
            } else {
                sweep::<false>(&k, n_iter, threads);
            }
        }
    }
    (f, b)
}

#[cfg(test)]
mod tests {
    use super::*;

    // A known composite: I = a*F_true + (1-a)*B_true with constant F_true,
    // B_true. The smoothest data-consistent solution is F == F_true, so the
    // solver must recover it across the soft band.
    #[test]
    fn recovers_constant_foreground() {
        let (h, w, depth) = (24usize, 24usize, 3usize);
        let f_true = [0.82f32, 0.21, 0.11];
        let b_true = [0.07f32, 0.09, 0.55];

        let mut image = Plane::new(h, w, depth);
        let mut alpha = vec![0.0f32; h * w];
        for y in 0..h {
            for x in 0..w {
                // alpha ramps 1 -> 0 left to right, with hard anchors at edges.
                let a = if x < 4 {
                    1.0
                } else if x >= w - 4 {
                    0.0
                } else {
                    1.0 - (x - 4) as f32 / (w - 8) as f32
                };
                alpha[y * w + x] = a;
                for c in 0..depth {
                    image.set(y, x, c, a * f_true[c] + (1.0 - a) * b_true[c]);
                }
            }
        }

        let (f, _b) = estimate_fb_ml(&image, &alpha, &FgParams::default());

        // Across the unknown band, recovered F should be near F_true.
        let mut max_err = 0.0f32;
        for y in 0..h {
            for x in 5..(w - 5) {
                for c in 0..depth {
                    max_err = max_err.max((f.at(y, x, c) - f_true[c]).abs());
                }
            }
        }
        assert!(max_err < 0.05, "FG recovery error too high: {max_err}");
    }

    // The reason callers must orient rows top-down: on data with no vertical
    // symmetry, solving a flipped copy and flipping back does NOT reproduce
    // the direct solve in under-constrained (faint-alpha) pixels — the
    // in-place sweep's propagation bias follows row order. If this test ever
    // starts failing (the two runs agreeing), the sweep is no longer
    // sequential Gauss-Seidel and parity with pymatting must be re-validated.
    #[test]
    fn sweep_order_is_direction_sensitive() {
        let (h, w, depth) = (48usize, 48usize, 3usize);
        let mut image = Plane::new(h, w, depth);
        let mut alpha = vec![0.0f32; h * w];
        // Deterministic LCG noise; solid FG block top-left, faint band below.
        let mut state = 12345u64;
        let mut rnd = move || {
            state = state.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
            ((state >> 33) as f32) / (u32::MAX >> 1) as f32
        };
        for y in 0..h {
            for x in 0..w {
                let a = if y < 8 && x < 20 {
                    1.0
                } else if y < 40 {
                    0.02 + 0.05 * rnd()
                } else {
                    0.0
                };
                alpha[y * w + x] = a;
                for c in 0..depth {
                    image.set(y, x, c, (0.2 + 0.6 * rnd()) * a + 0.1 * (1.0 - a));
                }
            }
        }
        let p = FgParams::default();
        let (f_direct, _) = estimate_fb_ml(&image, &alpha, &p);

        // Flip rows, solve, flip back.
        let mut image_f = Plane::new(h, w, depth);
        let mut alpha_f = vec![0.0f32; h * w];
        for y in 0..h {
            for x in 0..w {
                alpha_f[y * w + x] = alpha[(h - 1 - y) * w + x];
                for c in 0..depth {
                    image_f.set(y, x, c, image.at(h - 1 - y, x, c));
                }
            }
        }
        let (f_flip, _) = estimate_fb_ml(&image_f, &alpha_f, &p);

        let mut max_diff = 0.0f32;
        for y in 0..h {
            for x in 0..w {
                let a = alpha[y * w + x];
                if a > 0.0 && a < 0.1 {
                    for c in 0..depth {
                        let d = (f_direct.at(y, x, c) - f_flip.at(h - 1 - y, x, c)).abs();
                        max_diff = max_diff.max(d);
                    }
                }
            }
        }
        assert!(
            max_diff > 1e-3,
            "solve became flip-equivariant (max_diff={max_diff}) — sweep order changed?"
        );
    }

    // Image-aware background affinity: on a composite whose soft band crosses
    // hard BACKGROUND edges (colour bars), plain pymatting forces estimated B
    // continuous across the seams and dumps the residual into F (coloured
    // chips). With image_affinity on, B may break at the seams and F error
    // must drop sharply. On a FLAT background the feature must stay harmless.
    // (Full study: examples/bars_lab.rs, 2026-07-29.)
    #[test]
    fn image_affinity_cuts_background_seam_error() {
        let (h, w, depth) = (128usize, 128usize, 3usize);
        let f_true = [0.80f32, 0.08, 0.06];
        let bars: [[f32; 3]; 4] = [
            [0.75, 0.75, 0.00],
            [0.00, 0.75, 0.75],
            [0.75, 0.00, 0.75],
            [0.00, 0.00, 0.75],
        ];

        // Soft band: alpha ramps 1 -> 0 top to bottom, crossing all vertical
        // bar seams; hard anchors at top/bottom.
        let mut alpha = vec![0.0f32; h * w];
        for y in 0..h {
            let a = if y < 16 {
                1.0
            } else if y >= h - 16 {
                0.0
            } else {
                1.0 - (y - 16) as f32 / (h - 32) as f32
            };
            for x in 0..w {
                alpha[y * w + x] = a;
            }
        }

        let band_err = |image: &Plane, p: &FgParams| -> f32 {
            let (f, _b) = estimate_fb_ml(image, &alpha, p);
            let (mut sum, mut n) = (0f64, 0u64);
            for y in 0..h {
                for x in 0..w {
                    let a = alpha[y * w + x];
                    if a <= 0.005 || a >= 0.995 {
                        continue;
                    }
                    for c in 0..depth {
                        let e = (f.at(y, x, c) - f_true[c]) as f64;
                        sum += e * e;
                        n += 1;
                    }
                }
            }
            ((sum / n as f64) as f32).sqrt()
        };

        let compose = |bg: &dyn Fn(usize) -> [f32; 3]| -> Plane {
            let mut image = Plane::new(h, w, depth);
            for y in 0..h {
                for x in 0..w {
                    let a = alpha[y * w + x];
                    let b = bg(x);
                    for c in 0..depth {
                        image.set(y, x, c, a * f_true[c] + (1.0 - a) * b[c]);
                    }
                }
            }
            image
        };

        let seamed = compose(&|x| bars[(x * 4 / w).min(3)]);
        let flat = compose(&|_| [0.10, 0.45, 0.20]);

        let base = FgParams::default();
        let aware = FgParams { image_affinity: 100.0, ..FgParams::default() };

        let e_base = band_err(&seamed, &base);
        let e_aware = band_err(&seamed, &aware);
        assert!(
            e_aware < 0.5 * e_base,
            "image_affinity should cut seam error at least in half: base {e_base}, aware {e_aware}"
        );

        let e_flat = band_err(&flat, &aware);
        assert!(e_flat < 0.02, "flat background must stay clean with image_affinity on: {e_flat}");
    }

    // Class guard for the v0.3.0 fast path: the threaded, unchecked kernel
    // must reproduce the v0.2.0 reference BIT-FOR-BIT for any thread count,
    // with and without image affinity, on sizes that exercise the pipelined
    // sweep (>= 64K px), odd widths (partial column blocks) and tiny levels.
    #[test]
    fn fast_path_is_bit_identical_to_reference() {
        let mut state = 0x5eedu64;
        let mut rnd = move || {
            state = state.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
            ((state >> 33) as f32) / (u32::MAX >> 1) as f32
        };
        for &(h, w) in &[(257usize, 301usize), (300, 260), (7, 5), (1, 64), (2, 40000)] {
            let mut image = Plane::new(h, w, 3);
            let mut alpha = vec![0.0f32; h * w];
            for y in 0..h {
                for x in 0..w {
                    let r = ((x as f32 / w as f32) - 0.5).abs() * 2.0;
                    let a = if r < 0.3 { 1.0 } else if r > 0.8 { 0.0 } else { (0.8 - r) * 2.0 * rnd() };
                    alpha[y * w + x] = a;
                    for c in 0..3 {
                        image.set(y, x, c, rnd());
                    }
                }
            }
            for &aff in &[0.0f32, 100.0] {
                let p = FgParams { image_affinity: aff, ..FgParams::default() };
                let (fr, br) = estimate_fb_ml_reference(&image, &alpha, &p);
                for &t in &[1usize, 2, 4, 16] {
                    let (ff, bf) = estimate_fb_ml_mt(&image, &alpha, &p, t);
                    let same = |a: &Plane, b: &Plane| {
                        a.data.len() == b.data.len()
                            && a.data.iter().zip(&b.data).all(|(x, y)| x.to_bits() == y.to_bits())
                    };
                    assert!(same(&fr, &ff) && same(&br, &bf), "mismatch {h}x{w} aff={aff} threads={t}");
                }
            }
        }
    }

    #[test]
    fn py_round_is_banker() {
        assert_eq!(py_round(2.5), 2);
        assert_eq!(py_round(3.5), 4);
        assert_eq!(py_round(2.4), 2);
        assert_eq!(py_round(2.6), 3);
        assert_eq!(py_round(0.4), 1); // clamped to >=1
    }
}
