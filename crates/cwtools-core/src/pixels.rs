//! Top-down, row-major pixel buffers and simple resamplers / hashes.
//! (OFX images are bottom-up; [`crate::image`] flips at the boundary.)

/// Interleaved RGB f32 image, top-down.
#[derive(Clone)]
pub struct RgbF {
    pub w: usize,
    pub h: usize,
    pub data: Vec<f32>, // w*h*3
}

/// Interleaved RGB u8 image, top-down.
#[derive(Clone)]
pub struct Rgb8 {
    pub w: usize,
    pub h: usize,
    pub data: Vec<u8>,
}

/// Single-channel f32 image, top-down (masks, mattes).
#[derive(Clone)]
pub struct Gray {
    pub w: usize,
    pub h: usize,
    pub data: Vec<f32>,
}

/// Target size with the longest edge clamped to `max_edge` (never upscales).
pub fn fit_size(w: usize, h: usize, max_edge: usize) -> (usize, usize) {
    let long = w.max(h);
    if long <= max_edge || long == 0 {
        return (w.max(1), h.max(1));
    }
    let s = max_edge as f64 / long as f64;
    (((w as f64 * s).round() as usize).max(1), ((h as f64 * s).round() as usize).max(1))
}

/// Area-average (box) resample of an interleaved image with `c` channels.
/// Exact for integer factors, reasonable for fractional ones; downscale only.
pub fn area_resize(src: &[f32], sw: usize, sh: usize, c: usize, dw: usize, dh: usize) -> Vec<f32> {
    if sw == dw && sh == dh {
        return src.to_vec();
    }
    let mut out = vec![0f32; dw * dh * c];
    let fx = sw as f64 / dw as f64;
    let fy = sh as f64 / dh as f64;
    for y in 0..dh {
        let y0 = (y as f64 * fy).floor() as usize;
        let y1 = (((y + 1) as f64 * fy).ceil() as usize).clamp(y0 + 1, sh);
        for x in 0..dw {
            let x0 = (x as f64 * fx).floor() as usize;
            let x1 = (((x + 1) as f64 * fx).ceil() as usize).clamp(x0 + 1, sw);
            let mut acc = [0f64; 4];
            for yy in y0..y1 {
                let row = yy * sw;
                for xx in x0..x1 {
                    let i = (row + xx) * c;
                    for k in 0..c {
                        acc[k] += src[i + k] as f64;
                    }
                }
            }
            let n = ((y1 - y0) * (x1 - x0)) as f64;
            let o = (y * dw + x) * c;
            for k in 0..c {
                out[o + k] = (acc[k] / n) as f32;
            }
        }
    }
    out
}

/// Bilinear resample of a single-channel image (used to upscale masks).
/// Pixel-centre aligned (same convention as cv2.resize INTER_LINEAR).
pub fn bilinear_gray(src: &Gray, dw: usize, dh: usize) -> Gray {
    if src.w == dw && src.h == dh {
        return src.clone();
    }
    let mut out = vec![0f32; dw * dh];
    let sx = src.w as f32 / dw as f32;
    let sy = src.h as f32 / dh as f32;
    let maxx = src.w as f32 - 1.0;
    let maxy = src.h as f32 - 1.0;
    // Precompute x taps.
    let xt: Vec<(usize, usize, f32)> = (0..dw)
        .map(|x| {
            let fx = ((x as f32 + 0.5) * sx - 0.5).clamp(0.0, maxx);
            let x0 = fx.floor() as usize;
            let x1 = (x0 + 1).min(src.w - 1);
            (x0, x1, fx - x0 as f32)
        })
        .collect();
    for y in 0..dh {
        let fy = ((y as f32 + 0.5) * sy - 0.5).clamp(0.0, maxy);
        let y0 = fy.floor() as usize;
        let y1 = (y0 + 1).min(src.h - 1);
        let ty = fy - y0 as f32;
        let r0 = &src.data[y0 * src.w..(y0 + 1) * src.w];
        let r1 = &src.data[y1 * src.w..(y1 + 1) * src.w];
        let o = &mut out[y * dw..(y + 1) * dw];
        for (x, &(x0, x1, tx)) in xt.iter().enumerate() {
            let a = r0[x0] + (r0[x1] - r0[x0]) * tx;
            let b = r1[x0] + (r1[x1] - r1[x0]) * tx;
            o[x] = a + (b - a) * ty;
        }
    }
    Gray { w: dw, h: dh, data: out }
}

/// 64-bit FNV-1a.
pub fn fnv1a(bytes: &[u8]) -> u64 {
    fnv1a_continue(0xcbf2_9ce4_8422_2325, bytes)
}

/// Continue an FNV-1a hash over more bytes.
pub fn fnv1a_continue(mut h: u64, bytes: &[u8]) -> u64 {
    for &b in bytes {
        h ^= b as u64;
        h = h.wrapping_mul(0x0000_0100_0000_01b3);
    }
    h
}

pub fn frame_key(img: &Rgb8) -> String {
    let mut h = fnv1a(&img.data);
    h ^= (img.w as u64) << 32 | img.h as u64;
    format!("{h:016x}")
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn fit_size_clamps_long_edge() {
        assert_eq!(fit_size(3840, 2160, 1280), (1280, 720));
        assert_eq!(fit_size(800, 600, 1280), (800, 600));
    }
    #[test]
    fn area_resize_averages() {
        let src = vec![0.0, 1.0, 0.0, 1.0]; // 2x2 single channel
        let o = area_resize(&src, 2, 2, 1, 1, 1);
        assert!((o[0] - 0.5).abs() < 1e-6);
    }
    #[test]
    fn bilinear_upscale_constant() {
        let m = Gray { w: 2, h: 2, data: vec![1.0; 4] };
        let u = bilinear_gray(&m, 7, 5);
        assert!(u.data.iter().all(|v| (v - 1.0).abs() < 1e-6));
    }
    #[test]
    fn fnv_continue_matches_one_shot() {
        assert_eq!(fnv1a(b"abcdef"), fnv1a_continue(fnv1a(b"abc"), b"def"));
    }
}
