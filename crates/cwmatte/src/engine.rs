//! Frame-level BiRefNet pipeline, independent of OFX (testable on buffers).
//! Mirrors the validation recipe (the BiRefNet ONNX validation scripts, 
//! birefnet_onnx/validate_birefnet_onnx.py, = official BiRefNet inference):
//!
//! display RGB u8 (ROI crop) → Pillow BILINEAR stretch to the model's S×S
//! (no aspect preservation) → /255 → ImageNet mean/std → [model] → sigmoid
//! (graph outputs logits) → Pillow BILINEAR (mode "F") back to the ROI size →
//! paste into a zero frame → clip / gamma / contrast / invert → × garbage.
//!
//! Frame buffers are top-down, `w × h`.

use crate::params::MatteOps;
use cwtools_core::pixels::{fnv1a, fnv1a_continue};
use cwtools_core::resample::{par_rows, resize_f32, resize_u8, Filter};

pub const IMAGENET_MEAN: [f32; 3] = [0.485, 0.456, 0.406];
pub const IMAGENET_STD: [f32; 3] = [0.229, 0.224, 0.225];

/// Region of the frame fed to the model, top-down pixel coords, `[x0,x1) × [y0,y1)`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Roi {
    pub x0: usize,
    pub y0: usize,
    pub x1: usize,
    pub y1: usize,
}

impl Roi {
    pub fn full(w: usize, h: usize) -> Roi {
        Roi { x0: 0, y0: 0, x1: w, y1: h }
    }
    pub fn w(&self) -> usize {
        self.x1 - self.x0
    }
    pub fn h(&self) -> usize {
        self.y1 - self.y0
    }
    pub fn is_full(&self, w: usize, h: usize) -> bool {
        *self == Roi::full(w, h)
    }
}

/// Bbox of `m > thr` (top-down, w×h), exclusive max.
pub fn bbox_gt(m: &[f32], w: usize, h: usize, thr: f32) -> Option<(usize, usize, usize, usize)> {
    let (mut x0, mut y0, mut x1, mut y1) = (usize::MAX, usize::MAX, 0usize, 0usize);
    for y in 0..h {
        let row = &m[y * w..(y + 1) * w];
        let (Some(a), Some(b)) = (row.iter().position(|&v| v > thr), row.iter().rposition(|&v| v > thr)) else {
            continue;
        };
        y0 = y0.min(y);
        y1 = y + 1;
        x0 = x0.min(a);
        x1 = x1.max(b + 1);
    }
    (x0 != usize::MAX).then_some((x0, y0, x1, y1))
}

/// Grow a box by `pad` and clip it to the frame. None if it ends up < 2 px.
pub fn pad_box(b: (f64, f64, f64, f64), pad: usize, w: usize, h: usize) -> Option<Roi> {
    let p = pad as f64;
    let x0 = (b.0.min(b.2) - p).floor().max(0.0) as usize;
    let y0 = (b.1.min(b.3) - p).floor().max(0.0) as usize;
    let x1 = ((b.0.max(b.2) + p).ceil().max(0.0) as usize).min(w);
    let y1 = ((b.1.max(b.3) + p).ceil().max(0.0) as usize).min(h);
    (x1 >= x0 + 2 && y1 >= y0 + 2).then_some(Roi { x0, y0, x1, y1 })
}

/// ImageNet-normalised NCHW tensor of an S×S display RGB u8 image, exactly as
/// numpy computes it: `(u8 / 255.0f32 - mean) / std` in float32.
pub fn normalise(img: &[u8], sw: usize, sh: usize, mean: [f32; 3], std: [f32; 3]) -> Vec<f32> {
    let plane = sw * sh;
    let mut t = vec![0f32; 3 * plane];
    // Per-channel LUT (256 levels) = the same float32 ops per pixel.
    let lut: Vec<[f32; 256]> = (0..3)
        .map(|c| {
            let mut l = [0f32; 256];
            for (k, v) in l.iter_mut().enumerate() {
                *v = (k as f32 / 255.0 - mean[c]) / std[c];
            }
            l
        })
        .collect();
    for (c, pl) in t.chunks_exact_mut(plane).enumerate() {
        let l = &lut[c];
        par_rows(pl, sw, |y, row| {
            let src = &img[y * sw * 3..(y + 1) * sw * 3];
            for (o, p) in row.iter_mut().zip(src.chunks_exact(3)) {
                *o = l[p[c] as usize];
            }
        });
    }
    t
}

/// The model input for one frame: the Pillow-stretched u8 square and its hash.
pub struct Prepared {
    pub roi: Roi,
    pub sw: usize,
    pub sh: usize,
    pub square: Vec<u8>,
    pub key: u64,
}

/// `rgb` = display RGB u8 of the ROI (roi.w × roi.h × 3).
pub fn prepare(rgb: &[u8], roi: Roi, sw: usize, sh: usize) -> Prepared {
    let square = resize_u8(rgb, roi.w(), roi.h(), 3, sw, sh, Filter::Bilinear);
    let mut key = fnv1a(&square);
    key = fnv1a_continue(key, &[sw as u64, sh as u64].map(|x| x.to_le_bytes()).concat());
    Prepared { roi, sw, sh, square, key }
}

#[inline]
pub fn sigmoid(x: f32) -> f32 {
    1.0 / (1.0 + (-x).exp())
}

/// Model output → S×S matte in [0,1]. `sigmoid`: Some(true/false) from the
/// sidecar, None = auto (apply when the output is outside [0,1], i.e. logits).
pub fn finish(out: &[f32], sigmoid_mode: Option<bool>) -> Result<Vec<f32>, String> {
    if out.iter().any(|v| !v.is_finite()) {
        return Err("model output contains NaN/Inf (corrupt weights or GPU fault)".into());
    }
    let apply = sigmoid_mode.unwrap_or_else(|| out.iter().any(|&v| !(0.0..=1.0).contains(&v)));
    Ok(if apply { out.iter().map(|&v| sigmoid(v)).collect() } else { out.iter().map(|&v| v.clamp(0.0, 1.0)).collect() })
}

/// S×S matte → full-frame matte (Pillow BILINEAR "F" to the ROI size, 0 outside).
pub fn to_frame(m: &[f32], sw: usize, sh: usize, roi: Roi, w: usize, h: usize) -> Vec<f32> {
    let crop = resize_f32(m, sw, sh, roi.w(), roi.h(), Filter::Bilinear);
    if roi.is_full(w, h) {
        return crop.into_iter().map(|v| v.clamp(0.0, 1.0)).collect();
    }
    let mut out = vec![0f32; w * h];
    let cw = roi.w();
    for (r, y) in (roi.y0..roi.y1).enumerate() {
        for (o, &v) in out[y * w + roi.x0..y * w + roi.x1].iter_mut().zip(&crop[r * cw..(r + 1) * cw]) {
            *o = v.clamp(0.0, 1.0);
        }
    }
    out
}

/// Clip black/white → gamma → contrast → invert, in place.
/// The model's near-zero / near-one floor: a sigmoid never reaches 0 or 1, so
/// the background carries a 1e-7..1e-4 haze (and solid cores a matching
/// dent) that gamma / grading lifts into view. Values within 1/4096 of 0 or 1
/// (below a 12-bit code value) snap to exactly 0 / 1; nothing in between moves.
pub const HAZE_FLOOR: f32 = 1.0 / 4096.0;

pub fn floor_haze(m: &mut [f32]) {
    for v in m.iter_mut() {
        if *v < HAZE_FLOOR {
            *v = 0.0;
        } else if *v > 1.0 - HAZE_FLOOR {
            *v = 1.0;
        }
    }
}

pub fn apply_ops(m: &mut [f32], o: &MatteOps) {
    if *o == MatteOps::IDENTITY {
        return;
    }
    let (b, wt) = (o.clip_black, o.clip_white.max(o.clip_black + 1e-6));
    let inv_g = 1.0 / o.gamma.max(1e-3);
    for v in m.iter_mut() {
        let mut x = ((*v - b) / (wt - b)).clamp(0.0, 1.0);
        if inv_g != 1.0 {
            x = x.powf(inv_g);
        }
        if o.contrast != 1.0 {
            x = ((x - 0.5) * o.contrast + 0.5).clamp(0.0, 1.0);
        }
        if o.invert {
            x = 1.0 - x;
        }
        *v = x;
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn haze_floor_snaps_only_the_ends() {
        let mut m = vec![1e-7, 1e-4, 2.4e-4, 3e-4, 0.5, 0.9997, 0.99999, 1.0];
        floor_haze(&mut m);
        assert_eq!(m, vec![0.0, 0.0, 0.0, 3e-4, 0.5, 0.9997, 1.0, 1.0]);
    }
    #[test]
    fn bbox_and_pad() {
        let (w, h) = (100, 50);
        let mut m = vec![0f32; w * h];
        for y in 10..20 {
            for x in 30..41 {
                m[y * w + x] = 0.5;
            }
        }
        assert_eq!(bbox_gt(&m, w, h, 0.0), Some((30, 10, 41, 20)));
        assert_eq!(bbox_gt(&vec![0f32; w * h], w, h, 0.0), None);
        let r = pad_box((30.0, 10.0, 41.0, 20.0), 15, w, h).unwrap();
        assert_eq!(r, Roi { x0: 15, y0: 0, x1: 56, y1: 35 });
        assert_eq!(pad_box((0.0, 0.0, 1000.0, 1000.0), 0, w, h).unwrap(), Roi::full(w, h));
        assert!(pad_box((5.0, 5.0, 5.5, 5.5), 0, w, h).is_none());
    }
    #[test]
    fn finish_sigmoid_modes() {
        let o = finish(&[-20.0, 0.0, 20.0], None).unwrap();
        assert!(o[0] < 1e-8 && (o[1] - 0.5).abs() < 1e-7 && o[2] > 0.999_999);
        let o = finish(&[0.0, 0.25, 1.0], None).unwrap(); // already a matte: untouched
        assert_eq!(o, vec![0.0, 0.25, 1.0]);
        assert_eq!(finish(&[0.25], Some(true)).unwrap()[0], sigmoid(0.25));
        assert!(finish(&[f32::NAN], None).is_err());
    }
    #[test]
    fn ops_order_and_identity() {
        let mut m = vec![0.0, 0.1, 0.5, 0.9, 1.0];
        let orig = m.clone();
        apply_ops(&mut m, &MatteOps::IDENTITY);
        assert_eq!(m, orig);
        let mut m = vec![0.1, 0.55, 0.9];
        apply_ops(&mut m, &MatteOps { clip_black: 0.1, clip_white: 0.9, gamma: 1.0, contrast: 1.0, invert: true });
        assert!((m[0] - 1.0).abs() < 1e-6 && (m[1] - 0.4375).abs() < 1e-6 && m[2].abs() < 1e-6);
        let mut m = vec![0.25];
        apply_ops(&mut m, &MatteOps { gamma: 2.0, ..MatteOps::IDENTITY });
        assert!((m[0] - 0.5).abs() < 1e-6);
        let mut m = vec![0.25, 0.75];
        apply_ops(&mut m, &MatteOps { contrast: 2.0, ..MatteOps::IDENTITY });
        assert!(m[0].abs() < 1e-6 && (m[1] - 1.0).abs() < 1e-6);
    }
    #[test]
    fn to_frame_pastes_roi() {
        let m = vec![1.0f32; 16 * 16];
        let roi = Roi { x0: 3, y0: 2, x1: 13, y1: 7 };
        let f = to_frame(&m, 16, 16, roi, 20, 10);
        assert_eq!(f.iter().filter(|&&v| v == 1.0).count(), 50);
        assert_eq!(f[2 * 20 + 3], 1.0);
        assert_eq!(f[2 * 20 + 2], 0.0);
        assert_eq!(f[7 * 20 + 3], 0.0);
    }
}
