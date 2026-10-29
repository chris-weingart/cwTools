//! bars_lab — experiment harness for image-aware affinities.
//!
//! Reproduces the SMPTE-bars stress test: a red "rock" with
//! horizontal motion blur composited over saturated vertical bars. Because the
//! composite is synthesized as I = a*F_true + (1-a)*B_bars with a CONSTANT
//! F_true, ground truth is known exactly and every solver variant can be
//! scored numerically (band RMSE, seam-local RMSE) instead of eyeballed.
//!
//! Run: cargo run --release --example bars_lab
//! Outputs: dist/lab/*.ppm (convert with oiiotool for viewing) + a metrics
//! table on stdout.
//!
//! The instrumented solver below is a copy of solver::estimate_fb_ml with
//! variant hooks (image-aware affinity attenuation, big-iteration override,
//! per-level dumps). The `baseline` variant is asserted bit-identical to the
//! library solver so the copy can't silently drift.

use cwfill::solver::{estimate_fb_ml, FgParams, Plane};
use std::fs;
use std::io::Write;

// Bump for high-res runs (e.g. 2048) — geometry scales with W below.
const W: usize = 512;
const H: usize = W;
const DEPTH: usize = 3;
const F_TRUE: [f32; 3] = [0.80, 0.08, 0.06];
const N_BARS: usize = 7;

// ---------------------------------------------------------------------------
// Fixture
// ---------------------------------------------------------------------------

fn bar_color(x: usize) -> [f32; 3] {
    // 75% SMPTE bars: gray, yellow, cyan, green, magenta, red, blue.
    // Includes a BLACK bar (real SMPTE patterns have black/pluge segments) —
    // the trigger for the ACEScg-path sub-black investigation below.
    const BARS: [[f32; 3]; N_BARS] = [
        [0.75, 0.75, 0.75],
        [0.75, 0.75, 0.00],
        [0.00, 0.75, 0.75],
        [0.00, 0.00, 0.00],
        [0.75, 0.00, 0.75],
        [0.75, 0.00, 0.00],
        [0.00, 0.00, 0.75],
    ];
    BARS[(x * N_BARS / W).min(N_BARS - 1)]
}

fn seam_xs() -> Vec<usize> {
    (1..N_BARS).map(|i| i * W / N_BARS).collect()
}

fn box_blur_h(a: &mut [f32], radius: usize) {
    let mut out = vec![0f32; W * H];
    let win = (2 * radius + 1) as f32;
    for y in 0..H {
        let row = &a[y * W..(y + 1) * W];
        let mut prefix = vec![0f32; W + 1];
        for x in 0..W {
            prefix[x + 1] = prefix[x] + row[x];
        }
        for x in 0..W {
            let lo = x.saturating_sub(radius);
            let hi = (x + radius + 1).min(W);
            // Edge handling irrelevant: the disc is far from frame borders.
            out[y * W + x] = (prefix[hi] - prefix[lo]) / win;
        }
    }
    a.copy_from_slice(&out);
}

fn soften_v(a: &mut [f32]) {
    let mut out = vec![0f32; W * H];
    for y in 0..H {
        let ym = y.saturating_sub(1);
        let yp = (y + 1).min(H - 1);
        for x in 0..W {
            out[y * W + x] =
                0.25 * a[ym * W + x] + 0.5 * a[y * W + x] + 0.25 * a[yp * W + x];
        }
    }
    a.copy_from_slice(&out);
}

fn build_alpha() -> Vec<f32> {
    // Binary disc, then horizontal box blur = rigid-object motion smear.
    let mut alpha = vec![0f32; W * H];
    let (cx, cy, r) = (
        W as f32 / 2.0 - 0.5,
        H as f32 / 2.0 - 0.5,
        W as f32 * 100.0 / 512.0,
    );
    for y in 0..H {
        for x in 0..W {
            let d = ((x as f32 - cx).powi(2) + (y as f32 - cy).powi(2)).sqrt();
            alpha[y * W + x] = if d <= r { 1.0 } else { 0.0 };
        }
    }
    box_blur_h(&mut alpha, W * 60 / 512);
    soften_v(&mut alpha);
    soften_v(&mut alpha);
    alpha
}

fn composite(alpha: &[f32], bg: impl Fn(usize, usize) -> [f32; 3]) -> Plane {
    let mut image = Plane::new(H, W, DEPTH);
    for y in 0..H {
        for x in 0..W {
            let a = alpha[y * W + x];
            let b = bg(x, y);
            for c in 0..DEPTH {
                image.data[(y * W + x) * DEPTH + c] = a * F_TRUE[c] + (1.0 - a) * b[c];
            }
        }
    }
    image
}

// ---------------------------------------------------------------------------
// Instrumented solver (copy of solver::estimate_fb_ml + variant hooks)
// ---------------------------------------------------------------------------

#[derive(Clone, Default)]
struct Variant {
    name: &'static str,
    /// Attenuate the BACKGROUND smoothness coupling by exp(-s * |dI|^2).
    img_aff_b: f32,
    /// Attenuate the FOREGROUND smoothness coupling by exp(-s * |dI|^2).
    img_aff_f: f32,
    /// Override n_big_iterations (0 = keep FgParams default).
    n_big_override: i32,
    /// Normalize the image distance by (1-alpha)^2: in the band the composite
    /// attenuates background edges by (1-a), so dI/(1-a) estimates the TRUE
    /// background edge strength independent of alpha.
    normalize: bool,
    /// Observability gate: fade the attenuation out as alpha -> 1. Near-solid
    /// pixels have no data constraint on B — smoothness is the ONLY thing
    /// holding B there, so cutting it lets B degenerate and the data term
    /// drags F dark (the ACEScg-path "black values" failure).
    gate: bool,
    /// Dump per-level F snapshots.
    dump_levels: bool,
}

fn py_round(x: f64) -> usize {
    let f = x.floor();
    let diff = x - f;
    let r = if diff > 0.5 {
        f + 1.0
    } else if diff < 0.5 {
        f
    } else if (f as i64) % 2 == 0 {
        f
    } else {
        f + 1.0
    };
    r.max(1.0) as usize
}

fn resize_nearest(src: &Plane, h_dst: usize, w_dst: usize) -> Plane {
    let mut dst = Plane::new(h_dst, w_dst, src.depth);
    for y in 0..h_dst {
        let y_src = ((y * src.h) / h_dst).min(src.h - 1);
        for x in 0..w_dst {
            let x_src = ((x * src.w) / w_dst).min(src.w - 1);
            for c in 0..src.depth {
                dst.data[(y * w_dst + x) * src.depth + c] =
                    src.data[(y_src * src.w + x_src) * src.depth + c];
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

fn fb_ml_lab(image: &Plane, alpha: &[f32], p: &FgParams, v: &Variant) -> (Plane, Plane) {
    let (h0, w0, depth) = (image.h, image.w, image.depth);

    let mut f_mean = vec![0.0f32; depth];
    let mut b_mean = vec![0.0f32; depth];
    let (mut f_count, mut b_count) = (0.0f32, 0.0f32);
    for y in 0..h0 {
        for x in 0..w0 {
            let a = alpha[y * w0 + x];
            if a > 0.9 {
                for c in 0..depth {
                    f_mean[c] += image.data[(y * w0 + x) * depth + c];
                }
                f_count += 1.0;
            }
            if a < 0.1 {
                for c in 0..depth {
                    b_mean[c] += image.data[(y * w0 + x) * depth + c];
                }
                b_count += 1.0;
            }
        }
    }
    for c in 0..depth {
        f_mean[c] /= f_count + 1e-5;
        b_mean[c] /= b_count + 1e-5;
    }

    let mut f = Plane::new(1, 1, depth);
    let mut b = Plane::new(1, 1, depth);
    for c in 0..depth {
        f.data[c] = f_mean[c];
        b.data[c] = b_mean[c];
    }

    let n_levels = (((w0.max(h0)) as f64).log2()).ceil() as i32;
    let n_levels = n_levels.max(0);

    let dx: [isize; 4] = [-1, 1, 0, 0];
    let dy: [isize; 4] = [0, 0, -1, 1];

    let n_big = if v.n_big_override > 0 {
        v.n_big_override
    } else {
        p.n_big_iterations
    };

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
            n_big
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
                        b0[c] = a0 * image_l.data[(y * w + x) * depth + c];
                        b1[c] = a1 * image_l.data[(y * w + x) * depth + c];
                    }

                    for d in 0..4 {
                        let x2 = (x as isize + dx[d]).clamp(0, w as isize - 1) as usize;
                        let y2 = (y as isize + dy[d]).clamp(0, h as isize - 1) as usize;

                        let gradient = (a0 - alpha_l[y2 * w + x2]).abs();
                        let da = p.regularization + p.gradient_weight * gradient;

                        let (mut da_f, mut da_b) = (da, da);
                        if v.img_aff_b > 0.0 || v.img_aff_f > 0.0 {
                            let mut di2 = 0f32;
                            for c in 0..depth {
                                let dc = image_l.data[(y * w + x) * depth + c]
                                    - image_l.data[(y2 * w + x2) * depth + c];
                                di2 += dc * dc;
                            }
                            let a_mid = 0.5 * (a0 + alpha_l[y2 * w + x2]);
                            if v.normalize {
                                let inv = 1.0 - a_mid;
                                di2 /= (inv * inv).max(0.01);
                            }
                            if v.gate {
                                di2 *= ((0.97 - a_mid) / 0.07).clamp(0.0, 1.0);
                            }
                            if v.img_aff_b > 0.0 {
                                da_b = (da * (-v.img_aff_b * di2).exp()).max(1e-8);
                            }
                            if v.img_aff_f > 0.0 {
                                da_f = (da * (-v.img_aff_f * di2).exp()).max(1e-8);
                            }
                        }

                        a00 += da_f;
                        a11 += da_b;

                        for c in 0..depth {
                            b0[c] += da_f * f.data[(y2 * w + x2) * depth + c];
                            b1[c] += da_b * b.data[(y2 * w + x2) * depth + c];
                        }
                    }

                    let determinant = a00 * a11 - a01 * a01;
                    let inv_det = 1.0 / determinant;
                    let b00 = inv_det * a11;
                    let b01 = inv_det * -a01;
                    let b11 = inv_det * a00;

                    for c in 0..depth {
                        let f_c = (b00 * b0[c] + b01 * b1[c]).clamp(0.0, 1.0);
                        let b_c = (b01 * b0[c] + b11 * b1[c]).clamp(0.0, 1.0);
                        f.data[(y * w + x) * depth + c] = f_c;
                        b.data[(y * w + x) * depth + c] = b_c;
                    }
                }
            }
        }

        if v.dump_levels && w >= 64 {
            write_ppm(&format!("dist/lab/level_{i_level:02}_{w}x{h}_F.ppm"), &f);
        }
    }

    (f, b)
}

// ---------------------------------------------------------------------------
// Metrics & output
// ---------------------------------------------------------------------------

struct Score {
    band_rmse: f32,
    band_max: f32,
    seam_rmse: f32,
    seam_max: f32,
}

fn score(f: &Plane, alpha: &[f32]) -> Score {
    let seams = seam_xs();
    let near_seam = |x: usize| seams.iter().any(|&s| (x as isize - s as isize).abs() <= 3);
    let (mut bs, mut bn, mut bmax) = (0f64, 0u64, 0f32);
    let (mut ss, mut sn, mut smax) = (0f64, 0u64, 0f32);
    for y in 0..H {
        for x in 0..W {
            let a = alpha[y * W + x];
            if a <= 0.005 || a >= 0.995 {
                continue;
            }
            let mut e2 = 0f32;
            let mut emax = 0f32;
            for c in 0..DEPTH {
                let e = (f.data[(y * W + x) * DEPTH + c] - F_TRUE[c]).abs();
                e2 += e * e;
                emax = emax.max(e);
            }
            bs += e2 as f64;
            bn += DEPTH as u64;
            bmax = bmax.max(emax);
            if near_seam(x) {
                ss += e2 as f64;
                sn += DEPTH as u64;
                smax = smax.max(emax);
            }
        }
    }
    Score {
        band_rmse: ((bs / bn as f64) as f32).sqrt(),
        band_max: bmax,
        seam_rmse: ((ss / sn.max(1) as f64) as f32).sqrt(),
        seam_max: smax,
    }
}

fn write_ppm(path: &str, p: &Plane) {
    let mut buf = Vec::with_capacity(p.w * p.h * 3);
    for y in 0..p.h {
        for x in 0..p.w {
            for c in 0..3 {
                let v = p.data[(y * p.w + x) * p.depth + c.min(p.depth - 1)];
                // 1/2.2 gamma for viewing.
                buf.push((v.clamp(0.0, 1.0).powf(1.0 / 2.2) * 255.0 + 0.5) as u8);
            }
        }
    }
    let mut f = fs::File::create(path).expect("create ppm");
    write!(f, "P6\n{} {}\n255\n", p.w, p.h).unwrap();
    f.write_all(&buf).unwrap();
}

fn write_ppm_scalar(path: &str, a: &[f32]) {
    let mut p = Plane::new(H, W, 3);
    for i in 0..W * H {
        for c in 0..3 {
            p.data[i * 3 + c] = a[i];
        }
    }
    write_ppm(path, &p);
}

fn write_err_ppm(path: &str, f: &Plane, alpha: &[f32]) {
    // |F - F_true| x4, band only (zero elsewhere) — the chip map.
    let mut p = Plane::new(H, W, 3);
    for y in 0..H {
        for x in 0..W {
            let a = alpha[y * W + x];
            if a <= 0.005 || a >= 0.995 {
                continue;
            }
            for c in 0..DEPTH {
                p.data[(y * W + x) * 3 + c] =
                    ((f.data[(y * W + x) * DEPTH + c] - F_TRUE[c]).abs() * 4.0).min(1.0);
            }
        }
    }
    write_ppm(path, &p);
}

fn main() {
    fs::create_dir_all("dist/lab").unwrap();
    let alpha = build_alpha();
    let bars = composite(&alpha, |x, _| bar_color(x));
    // Control fixture: same alpha over a FLAT background — the case the solver
    // already handles well. A variant must not regress here.
    let flat = composite(&alpha, |_, _| [0.10, 0.45, 0.20]);
    // Grain control: flat background + per-pixel noise (sigma ~0.03) on the
    // whole composite — image-aware attenuation must not fall apart on grain.
    let grain = {
        let mut g = composite(&alpha, |_, _| [0.10, 0.45, 0.20]);
        let mut state = 987654321u64;
        let mut rnd = move || {
            state = state
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            ((state >> 33) as f32) / (u32::MAX >> 1) as f32 - 0.5
        };
        for v in g.data.iter_mut() {
            *v = (*v + 0.06 * rnd()).clamp(0.0, 1.0);
        }
        g
    };
    write_ppm("dist/lab/00_composite.ppm", &bars);
    write_ppm_scalar("dist/lab/00_alpha.ppm", &alpha);

    let p = FgParams::default();

    // Parity check: instrumented copy with everything off must match the
    // library solver bit-for-bit.
    let (f_lib, _) = estimate_fb_ml(&bars, &alpha, &p);
    let (f_copy, _) = fb_ml_lab(&bars, &alpha, &p, &Variant { name: "parity", ..Default::default() });
    assert_eq!(f_lib.data, f_copy.data, "instrumented copy drifted from library solver");

    let variants = [
        Variant { name: "baseline", dump_levels: true, ..Default::default() },
        Variant { name: "big8", n_big_override: 8, ..Default::default() },
        Variant { name: "big16", n_big_override: 16, ..Default::default() },
        Variant { name: "imgB_s3", img_aff_b: 3.0, ..Default::default() },
        Variant { name: "imgB_s10", img_aff_b: 10.0, ..Default::default() },
        Variant { name: "imgB_s30", img_aff_b: 30.0, ..Default::default() },
        Variant { name: "imgB_s100", img_aff_b: 100.0, ..Default::default() },
        Variant { name: "imgB_s300", img_aff_b: 300.0, ..Default::default() },
        Variant { name: "imgF_s10", img_aff_f: 10.0, ..Default::default() },
        Variant { name: "imgFB_s10", img_aff_b: 10.0, img_aff_f: 10.0, ..Default::default() },
        Variant { name: "imgB_s10_big8", img_aff_b: 10.0, n_big_override: 8, ..Default::default() },
        Variant { name: "imgB_s100_big8", img_aff_b: 100.0, n_big_override: 8, ..Default::default() },
        Variant { name: "imgBn_s10", img_aff_b: 10.0, normalize: true, ..Default::default() },
        Variant { name: "imgBn_s30", img_aff_b: 30.0, normalize: true, ..Default::default() },
        Variant { name: "imgBn_s100", img_aff_b: 100.0, normalize: true, ..Default::default() },
        Variant { name: "imgBn_s30_big8", img_aff_b: 30.0, normalize: true, n_big_override: 8, ..Default::default() },
        Variant { name: "imgBn_s100_big8", img_aff_b: 100.0, normalize: true, n_big_override: 8, ..Default::default() },
        Variant { name: "imgB_s100_big4", img_aff_b: 100.0, n_big_override: 4, ..Default::default() },
        Variant { name: "imgBn_s100_big4", img_aff_b: 100.0, normalize: true, n_big_override: 4, ..Default::default() },
    ];

    // LAB_SKIP_TABLE=1 skips the (slow) linear-path variant table when
    // iterating on the cct-path analysis below.
    let skip_table = std::env::var("LAB_SKIP_TABLE").is_ok();
    println!(
        "{:<16} | {:>9} {:>9} {:>9} {:>9} | {:>9} {:>9} | {:>9} {:>9}",
        "variant", "band_rmse", "band_max", "seam_rmse", "seam_max", "flat_rmse", "flat_max",
        "grainrmse", "grain_max"
    );
    for v in variants.iter().take(if skip_table { 0 } else { variants.len() }) {
        let (f, _b) = fb_ml_lab(&bars, &alpha, &p, v);
        let s = score(&f, &alpha);
        let quiet = Variant { dump_levels: false, ..v.clone() };
        let (ff, _b) = fb_ml_lab(&flat, &alpha, &p, &quiet);
        let sf = score(&ff, &alpha);
        let (fg, _b) = fb_ml_lab(&grain, &alpha, &p, &quiet);
        let sg = score(&fg, &alpha);
        println!(
            "{:<16} | {:>9.4} {:>9.4} {:>9.4} {:>9.4} | {:>9.4} {:>9.4} | {:>9.4} {:>9.4}",
            v.name, s.band_rmse, s.band_max, s.seam_rmse, s.seam_max,
            sf.band_rmse, sf.band_max, sg.band_rmse, sg.band_max
        );
        write_ppm(&format!("dist/lab/F_{}.ppm", v.name), &f);
        write_err_ppm(&format!("dist/lab/err_{}.ppm", v.name), &f, &alpha);
    }

    // ---- ACEScg-path black-values investigation --------------
    // The OFX ACEScg mode solves encode(I) in ACEScct and DECODES the result
    // back to linear. Anything the solver leaves below encode(0) = 0.0729
    // decodes NEGATIVE (Flame: black / blacker-than-black). The raw path
    // clamps F to [0,1] linear and cannot go sub-zero. Measure how often the
    // solve dips below the black point, with and without image-aware.
    fn cct(x: f32) -> f32 {
        if x <= 0.0078125 { 10.5402377 * x + 0.0729055 } else { (x.log2() + 9.72) / 17.52 }
    }
    fn cct_inv(y: f32) -> f32 {
        if y <= 0.155_251_14 { (y - 0.0729055) / 10.5402377 } else { 2.0_f32.powf(y * 17.52 - 9.72) }
    }
    // Two sources: the clean bars comp, and a grainy one (real renders carry
    // grain/AA noise — di2 is never exactly zero inside flat regions).
    let grainy_bars = {
        let mut g = Plane::new(H, W, DEPTH);
        let mut state = 24681357u64;
        let mut rnd = move || {
            state = state
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            ((state >> 33) as f32) / (u32::MAX >> 1) as f32 - 0.5
        };
        for i in 0..W * H * DEPTH {
            g.data[i] = (bars.data[i] + 0.02 * rnd()).clamp(0.0, 1.0);
        }
        g
    };
    // Grainy source also gets band-only ALPHA noise (rendered mattes are not
    // analytically smooth).
    let alpha_noisy = {
        let mut an = alpha.clone();
        let mut state = 1122334455u64;
        let mut rnd = move || {
            state = state
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            ((state >> 33) as f32) / (u32::MAX >> 1) as f32 - 0.5
        };
        for v in an.iter_mut() {
            if *v > 0.0 && *v < 1.0 {
                *v = (*v + 0.02 * rnd()).clamp(0.0, 1.0);
            }
        }
        an
    };
    for (src_name, src, alpha_src) in
        [("clean", &bars, &alpha), ("grainy", &grainy_bars, &alpha_noisy)]
    {
        let alpha = alpha_src;
        let mut enc = Plane::new(H, W, DEPTH);
        for i in 0..W * H * DEPTH {
            enc.data[i] = cct(src.data[i]).clamp(0.0, 1.0);
        }
        let cct_variants = [
            Variant { name: "off", ..Default::default() },
            Variant { name: "norm_s100", img_aff_b: 100.0, normalize: true, ..Default::default() },
            Variant { name: "norm_s200", img_aff_b: 200.0, normalize: true, ..Default::default() },
            Variant { name: "norm_gate_s100", img_aff_b: 100.0, normalize: true, gate: true, ..Default::default() },
            Variant { name: "norm_gate_s200", img_aff_b: 200.0, normalize: true, gate: true, ..Default::default() },
            Variant { name: "plain_gate_s200", img_aff_b: 200.0, gate: true, ..Default::default() },
        ];
        println!("\nACEScg-path ({src_name} source, solve in cct, decode to linear) — band pixels:");
        println!(
            "{:<16} {:>8} {:>8} {:>8} {:>12} {:>10}",
            "variant", "neg_px", "nan_px", "dark_px", "min_linear", "band_rmse"
        );
        for v in &cct_variants {
            let (f, _b) = fb_ml_lab(&enc, alpha, &FgParams::default(), v);
            let (mut neg, mut nan, mut dark) = (0u32, 0u32, 0u32);
            let mut min_lin = f32::MAX;
            let (mut es, mut en) = (0f64, 0u64);
            for y in 0..H {
                for x in 0..W {
                    let a = alpha[y * W + x];
                    if a <= 0.005 || a >= 0.995 {
                        continue;
                    }
                    let (mut px_neg, mut px_nan) = (false, false);
                    let mut px_max_lin = 0f32;
                    for c in 0..DEPTH {
                        let v = f.data[(y * W + x) * DEPTH + c];
                        if v.is_nan() {
                            px_nan = true;
                            continue;
                        }
                        let lin = cct_inv(v);
                        min_lin = min_lin.min(lin);
                        px_max_lin = px_max_lin.max(lin);
                        if lin < -0.0005 {
                            px_neg = true;
                        }
                        let e = (lin - F_TRUE[c]) as f64;
                        es += e * e;
                        en += 1;
                    }
                    neg += px_neg as u32;
                    nan += px_nan as u32;
                    // "dark chip": brightest channel of the fill below 0.25
                    // linear where the true fill peaks at 0.8 — reads black.
                    dark += (px_max_lin < 0.25) as u32;
                }
            }
            println!(
                "{:<16} {:>8} {:>8} {:>8} {:>12.4} {:>10.4}",
                v.name, neg, nan, dark, min_lin, ((es / en.max(1) as f64) as f32).sqrt()
            );
            let mut dec = Plane::new(H, W, DEPTH);
            for i in 0..W * H * DEPTH {
                dec.data[i] = cct_inv(f.data[i]).clamp(0.0, 1.0);
            }
            write_ppm(&format!("dist/lab/cctpath_{src_name}_{}_F.ppm", v.name), &dec);
        }
    }
}
