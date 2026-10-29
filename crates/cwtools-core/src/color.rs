//! Input colour-space handling. The ML models in cwTools expect
//! DISPLAY-referred RGB (what production fed them as 8-bit PNGs), so scene-
//! linear and log plates get a view transform first.
//!
//! Since 0.2.0 that view is the real ACES 2.0 SDR one the reference pipeline uses
//! (`Rec.1886 Rec.709 - Display` / `ACES 2.0 - SDR 100 nits (Rec.709)` from
//! Flame's aces2.0_config), baked by `tools/bake_aces2_lut.py` into a 65^3
//! LUT over ACEScct code values (`data/aces2_sdr709_cct65.*`). Measured on a
//! UHD ACEScg plate: mean 0.09 / max 1.3 8-bit levels from OCIO. The 0.1.x
//! approximation (AP1->709, soft shoulder, sRGB OETF) was ~20 levels off on
//! average: lifted shadows, flat mids, clipped highlights.
//! Values match the `Input Colour` choice order used by all plugins.

use crate::pixels::{Rgb8, RgbF};

pub const CS_ACESCG: i32 = 0;
pub const CS_LIN709: i32 = 1;
pub const CS_VIDEO: i32 = 2;
pub const CS_ACESCCT: i32 = 3;

/// Choice labels in param order (NUL-terminated for the OFX API).
pub const CS_OPTIONS: [&[u8]; 4] = [b"ACEScg (linear)\0", b"Linear Rec.709\0", b"Video / sRGB\0", b"ACEScct (log)\0"];

#[inline]
pub fn acescct_to_lin(y: f32) -> f32 {
    if y <= 0.155_251_14 {
        (y - 0.072_905_53) / 10.540_237
    } else {
        (2f32).powf(y * 17.52 - 9.72)
    }
}

const LUT_N: usize = 65;
static LUT_U16: &[u8] = include_bytes!("../data/aces2_sdr709_cct65.u16");
static LUT: std::sync::OnceLock<Vec<f32>> = std::sync::OnceLock::new();

fn lut() -> &'static [f32] {
    LUT.get_or_init(|| LUT_U16.chunks_exact(2).map(|b| u16::from_le_bytes([b[0], b[1]]) as f32 / 65535.0).collect())
}

/// ACEScg linear -> ACEScct code value (the LUT's domain).
#[inline]
pub fn lin_to_acescct(x: f32) -> f32 {
    if x <= 0.007_812_5 {
        10.540_237_7 * x + 0.072_905_53
    } else {
        (x.log2() + 9.72) / 17.52
    }
}

/// ACEScct code values -> display (ACES 2.0 SDR Rec.709, Rec.1886 encoded),
/// trilinear over the baked LUT. Non-finite / out-of-domain inputs clamp.
#[inline]
pub fn aces2_display_cct(r: f32, g: f32, b: f32) -> [f32; 3] {
    let t = lut();
    let n1 = (LUT_N - 1) as f32;
    let ax = |v: f32| -> (usize, f32) {
        let c = if v.is_finite() { v.clamp(0.0, 1.0) * n1 } else { 0.0 };
        let i = (c.floor() as usize).min(LUT_N - 2);
        (i, c - i as f32)
    };
    let ((r0, fr), (g0, fg), (b0, fb)) = (ax(r), ax(g), ax(b));
    let at = |bi: usize, gi: usize, ri: usize, k: usize| t[((bi * LUT_N + gi) * LUT_N + ri) * 3 + k];
    let mut out = [0f32; 3];
    for (k, o) in out.iter_mut().enumerate() {
        let c00 = at(b0, g0, r0, k) * (1.0 - fr) + at(b0, g0, r0 + 1, k) * fr;
        let c01 = at(b0, g0 + 1, r0, k) * (1.0 - fr) + at(b0, g0 + 1, r0 + 1, k) * fr;
        let c10 = at(b0 + 1, g0, r0, k) * (1.0 - fr) + at(b0 + 1, g0, r0 + 1, k) * fr;
        let c11 = at(b0 + 1, g0 + 1, r0, k) * (1.0 - fr) + at(b0 + 1, g0 + 1, r0 + 1, k) * fr;
        let c0 = c00 * (1.0 - fg) + c01 * fg;
        let c1 = c10 * (1.0 - fg) + c11 * fg;
        *o = c0 * (1.0 - fb) + c1 * fb;
    }
    out
}

/// Convert one pixel to display-referred [0,1] (what production's 8-bit PNGs held).
#[inline]
pub fn to_display(r: f32, g: f32, b: f32, cs: i32) -> [f32; 3] {
    match cs {
        CS_VIDEO => [r.clamp(0.0, 1.0), g.clamp(0.0, 1.0), b.clamp(0.0, 1.0)],
        CS_ACESCCT => aces2_display_cct(r, g, b),
        CS_LIN709 => {
            // Linear Rec.709 -> ACEScg (OCIO "Linear Rec.709 (sRGB)" -> ACEScg, Bradford D65->D60)
            let r2 = 0.613_097 * r + 0.339_523 * g + 0.047_379 * b;
            let g2 = 0.070_194 * r + 0.916_354 * g + 0.013_452 * b;
            let b2 = 0.020_616 * r + 0.109_570 * g + 0.869_815 * b;
            aces2_display_cct(lin_to_acescct(r2), lin_to_acescct(g2), lin_to_acescct(b2))
        }
        _ => aces2_display_cct(lin_to_acescct(r), lin_to_acescct(g), lin_to_acescct(b)),
    }
}

/// Display value [0,1] → 8-bit level (round half up; non-finite → 0).
#[inline]
pub fn to_u8(v: f32) -> u8 {
    if v.is_finite() {
        (v * 255.0 + 0.5).clamp(0.0, 255.0) as u8
    } else {
        0
    }
}

/// Matte overlay tint (cwSegment / cwMatte "Overlay" view): the subject is
/// pushed towards red, keeping its luminance readable.
#[inline]
pub fn overlay_red(r: f32, g: f32, b: f32, m: f32) -> (f32, f32, f32) {
    let k = 0.55 * m;
    let lum = (0.2126 * r + 0.7152 * g + 0.0722 * b).max(0.05);
    (r * (1.0 - k) + k * lum * 3.0, g * (1.0 - k), b * (1.0 - k))
}

pub fn rgbf_to_display8(img: &RgbF, cs: i32) -> Rgb8 {
    let mut data = vec![0u8; img.w * img.h * 3];
    for (i, px) in img.data.chunks_exact(3).enumerate() {
        let d = to_display(px[0], px[1], px[2], cs);
        for k in 0..3 {
            data[i * 3 + k] = to_u8(d[k]);
        }
    }
    Rgb8 { w: img.w, h: img.h, data }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn display_transform_ranges() {
        // ACES 2.0 SDR: 18% grey -> 0.383 (OCIO), not the old approximation's 0.461
        let d = to_display(0.18, 0.18, 0.18, CS_ACESCG);
        assert!((d[0] - 0.383).abs() < 0.003, "{d:?}");
        let d = to_display(100.0, 100.0, 100.0, CS_ACESCG);
        assert!(d.iter().all(|v| *v <= 1.0));
    }
    #[test]
    fn video_roundtrips_8bit_levels() {
        for k in 0..=255u32 {
            let v = k as f32 / 255.0;
            let d = to_display(v, v, v, CS_VIDEO);
            assert_eq!(to_u8(d[0]) as u32, k);
        }
    }
    #[test]
    fn lut_matches_bake_vectors_and_ocio_ramp() {
        let meta: serde_json::Value = serde_json::from_str(include_str!("../data/aces2_sdr709_cct65.json")).unwrap();
        let tv = &meta["test_vectors"];
        for (i, o) in tv["acescg"].as_array().unwrap().iter().zip(tv["display"].as_array().unwrap()) {
            let i: Vec<f32> = i.as_array().unwrap().iter().map(|x| x.as_f64().unwrap() as f32).collect();
            let o: Vec<f32> = o.as_array().unwrap().iter().map(|x| x.as_f64().unwrap() as f32).collect();
            let d = to_display(i[0], i[1], i[2], CS_ACESCG);
            for k in 0..3 {
                assert!((d[k] - o[k]).abs() < 2e-5, "{i:?} -> {d:?} vs bake {o:?}");
            }
        }
        for (x, v) in meta["grey_ramp"].as_object().unwrap() {
            let x: f32 = x.parse().unwrap();
            let ocio = v["ocio"].as_f64().unwrap() as f32;
            let d = to_display(x, x, x, CS_ACESCG);
            assert!((d[0] - ocio).abs() * 255.0 < 0.5, "grey {x}: {} vs ocio {ocio}", d[0]);
        }
    }
    #[test]
    fn acescct_and_lin709_routes() {
        let g = to_display(0.18, 0.18, 0.18, CS_ACESCG);
        let c = lin_to_acescct(0.18);
        let d = to_display(c, c, c, CS_ACESCCT);
        assert!((d[0] - g[0]).abs() < 1e-6);
        // neutral grey is neutral in both primaries
        let l = to_display(0.18, 0.18, 0.18, CS_LIN709);
        assert!((l[0] - g[0]).abs() < 2e-3 && (l[2] - g[2]).abs() < 2e-3, "{l:?} vs {g:?}");
        assert_eq!(to_display(f32::NAN, 0.0, 1e9, CS_ACESCG).iter().all(|v| v.is_finite()), true);
    }
}
