//! Prompt model shared by both backends. Coordinates are in the pixel space of
//! the *model input image* (the downscaled frame), top-down, x right.

use crate::imgutil::Gray;

#[derive(Clone, Debug, Default)]
pub struct Prompts {
    pub text: String,
    /// (x, y, positive)
    pub points: Vec<(f32, f32, bool)>,
    /// x0, y0, x1, y1
    pub bbox: Option<[f32; 4]>,
}

impl Prompts {
    pub fn is_empty(&self) -> bool {
        self.text.is_empty() && self.points.is_empty() && self.bbox.is_none()
    }
    /// Stable textual descriptor (cache key component).
    pub fn key(&self) -> String {
        let mut s = format!("t={}|", self.text);
        for p in &self.points {
            s += &format!("p{:.1},{:.1},{}|", p.0, p.1, p.2 as u8);
        }
        if let Some(b) = self.bbox {
            s += &format!("b{:.1},{:.1},{:.1},{:.1}", b[0], b[1], b[2], b[3]);
        }
        s
    }
}

/// Derive a box and up to 3 interior points from a prompt matte (white =
/// object, threshold 0.5). Returns None when the matte is empty or full-frame
/// (an unconnected alpha socket is typically all 0 or all 1).
pub fn from_matte(m: &Gray) -> Option<([f32; 4], Vec<(f32, f32)>)> {
    let (w, h) = (m.w, m.h);
    let (mut x0, mut y0, mut x1, mut y1) = (usize::MAX, usize::MAX, 0usize, 0usize);
    let mut n = 0usize;
    for y in 0..h {
        for x in 0..w {
            if m.data[y * w + x] > 0.5 {
                n += 1;
                x0 = x0.min(x);
                y0 = y0.min(y);
                x1 = x1.max(x);
                y1 = y1.max(y);
            }
        }
    }
    if n == 0 || n as f64 > 0.98 * (w * h) as f64 {
        return None;
    }
    let bbox = [x0 as f32, y0 as f32, (x1 + 1) as f32, (y1 + 1) as f32];

    // Interior points: the "deepest" pixel (max Chebyshev distance to the
    // outside, via two-pass chamfer) of the whole matte and of each half of
    // the bbox along its long axis.
    let dist = chamfer(m);
    let deepest = |xa: usize, xb: usize, ya: usize, yb: usize| -> Option<(f32, f32)> {
        let mut best = 0f32;
        let mut at = None;
        for y in ya..yb {
            for x in xa..xb {
                let d = dist[y * w + x];
                if d > best {
                    best = d;
                    at = Some((x as f32 + 0.5, y as f32 + 0.5));
                }
            }
        }
        at
    };
    let mut pts = Vec::new();
    if let Some(p) = deepest(x0, x1 + 1, y0, y1 + 1) {
        pts.push(p);
    }
    let (bw, bh) = (x1 + 1 - x0, y1 + 1 - y0);
    let halves = if bw >= bh {
        let xm = x0 + bw / 2;
        [(x0, xm, y0, y1 + 1), (xm, x1 + 1, y0, y1 + 1)]
    } else {
        let ym = y0 + bh / 2;
        [(x0, x1 + 1, y0, ym), (x0, x1 + 1, ym, y1 + 1)]
    };
    let min_sep = (bw.max(bh) as f32 * 0.15).max(4.0);
    for (xa, xb, ya, yb) in halves {
        if let Some(p) = deepest(xa, xb, ya, yb) {
            if pts.iter().all(|q: &(f32, f32)| ((q.0 - p.0).powi(2) + (q.1 - p.1).powi(2)).sqrt() > min_sep) {
                pts.push(p);
            }
        }
    }
    Some((bbox, pts))
}

fn chamfer(m: &Gray) -> Vec<f32> {
    let (w, h) = (m.w, m.h);
    let big = (w + h) as f32;
    let mut d: Vec<f32> = m.data.iter().map(|v| if *v > 0.5 { big } else { 0.0 }).collect();
    for y in 0..h {
        for x in 0..w {
            let i = y * w + x;
            if d[i] == 0.0 {
                continue;
            }
            let mut v = d[i];
            if x == 0 || y == 0 {
                v = v.min(1.0);
            } else {
                v = v.min(d[i - 1] + 1.0).min(d[i - w] + 1.0).min(d[i - w - 1] + 1.0);
                if x + 1 < w {
                    v = v.min(d[i - w + 1] + 1.0);
                }
            }
            d[i] = v;
        }
    }
    for y in (0..h).rev() {
        for x in (0..w).rev() {
            let i = y * w + x;
            if d[i] == 0.0 {
                continue;
            }
            let mut v = d[i];
            if x + 1 == w || y + 1 == h {
                v = v.min(1.0);
            } else {
                v = v.min(d[i + 1] + 1.0).min(d[i + w] + 1.0).min(d[i + w + 1] + 1.0);
                if x > 0 {
                    v = v.min(d[i + w - 1] + 1.0);
                }
            }
            d[i] = v;
        }
    }
    d
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn matte_box_and_points() {
        let (w, h) = (40, 20);
        let mut m = Gray { w, h, data: vec![0.0; w * h] };
        for y in 5..15 {
            for x in 10..30 {
                m.data[y * w + x] = 1.0;
            }
        }
        let (b, pts) = from_matte(&m).unwrap();
        assert_eq!(b, [10.0, 5.0, 30.0, 15.0]);
        assert!(!pts.is_empty());
        for p in &pts {
            assert!(p.0 > 10.0 && p.0 < 30.0 && p.1 > 5.0 && p.1 < 15.0);
        }
    }
    #[test]
    fn empty_and_full_matte_ignored() {
        let m = Gray { w: 4, h: 4, data: vec![0.0; 16] };
        assert!(from_matte(&m).is_none());
        let m = Gray { w: 4, h: 4, data: vec![1.0; 16] };
        assert!(from_matte(&m).is_none());
    }
}
