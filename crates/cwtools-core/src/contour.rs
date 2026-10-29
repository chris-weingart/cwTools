//! Contours and distance fields on binary masks (first used by the cwRefine
//! spline trimap):
//!
//! - [`marching_squares`]: every boundary of a binary mask as closed, ordered
//!   loops (outer boundaries and holes alike). Vertices sit on the midpoints
//!   between a pixel inside and its 4-neighbour outside, in pixel-centre
//!   coordinates (pixel (x, y) has its centre at (x, y)), top-down. Saddles
//!   resolve to the 4-connected reading of the foreground. The inside is
//!   always on the same side of the travel direction, so the outward normal
//!   of a segment with direction (dx, dy) is (dy, -dx) — see [`Loop::normal`].
//! - [`edt`]: exact Euclidean distance transform with nearest-feature index
//!   (Felzenszwalb & Huttenlocher, separable; rows in parallel).
//!
//! Everything is deterministic (no hash-order dependence).

use crate::resample::par_rows;
use std::collections::HashMap;

/// One closed boundary loop. `pts[i] -> pts[i + 1] -> ... -> pts[0]`.
#[derive(Clone, Debug, PartialEq)]
pub struct Loop {
    pub pts: Vec<(f32, f32)>,
}

impl Loop {
    pub fn len(&self) -> usize {
        self.pts.len()
    }
    pub fn is_empty(&self) -> bool {
        self.pts.is_empty()
    }
    /// Unit outward normal at vertex `i` (central difference of neighbours).
    pub fn normal(&self, i: usize) -> (f64, f64) {
        let n = self.pts.len();
        if n < 2 {
            return (0.0, -1.0);
        }
        let a = self.pts[(i + n - 1) % n];
        let b = self.pts[(i + 1) % n];
        let (dx, dy) = ((b.0 - a.0) as f64, (b.1 - a.1) as f64);
        let l = (dx * dx + dy * dy).sqrt();
        if l < 1e-12 {
            return (0.0, -1.0);
        }
        (dy / l, -dx / l)
    }
    /// Distance between vertex `i` and `i + 1` (wrapping).
    pub fn seg_len(&self, i: usize) -> f64 {
        let n = self.pts.len();
        let a = self.pts[i % n];
        let b = self.pts[(i + 1) % n];
        (((b.0 - a.0) as f64).powi(2) + ((b.1 - a.1) as f64).powi(2)).sqrt()
    }
    /// Signed area (shoelace, top-down coordinates).
    pub fn signed_area(&self) -> f64 {
        let n = self.pts.len();
        let mut s = 0.0;
        for i in 0..n {
            let (x0, y0) = self.pts[i];
            let (x1, y1) = self.pts[(i + 1) % n];
            s += x0 as f64 * y1 as f64 - x1 as f64 * y0 as f64;
        }
        0.5 * s
    }
}

/// All boundary loops of `mask` (w × h, row-major, top-down). Outside the
/// image counts as "not in the mask".
pub fn marching_squares(mask: &[bool], w: usize, h: usize) -> Vec<Loop> {
    assert_eq!(mask.len(), w * h);
    let inside = |x: i64, y: i64| x >= 0 && y >= 0 && (x as usize) < w && (y as usize) < h && mask[y as usize * w + x as usize];
    let ww = w as i64 + 2;
    // Edge keys: kind 0 = H (pixel (x,y)-(x+1,y)), kind 1 = V ((x,y)-(x,y+1)).
    let key = |kind: i64, x: i64, y: i64| -> u64 { (((y + 1) * ww + (x + 1)) * 2 + kind) as u64 };
    let point = |k: u64| -> (f32, f32) {
        let kind = (k % 2) as i64;
        let c = (k / 2) as i64;
        let (x, y) = (c % ww - 1, c / ww - 1);
        if kind == 0 {
            (x as f32 + 0.5, y as f32)
        } else {
            (x as f32, y as f32 + 0.5)
        }
    };
    let mut next: HashMap<u64, u64> = HashMap::new();
    let mut order: Vec<u64> = Vec::new();
    for cy in -1..h as i64 {
        for cx in -1..w as i64 {
            // corners clockwise (screen): a TL, b TR, c BR, d BL
            let cin = [inside(cx, cy), inside(cx + 1, cy), inside(cx + 1, cy + 1), inside(cx, cy + 1)];
            let cnt = cin.iter().filter(|&&b| b).count();
            if cnt == 0 || cnt == 4 {
                continue;
            }
            // edge i between corner i and corner i+1
            let ek = [key(0, cx, cy), key(1, cx + 1, cy), key(0, cx, cy + 1), key(1, cx, cy)];
            for i in 0..4 {
                if cin[i] && !cin[(i + 1) % 4] {
                    // exit edge i; walk back over the inside run to its entry edge
                    let mut j = i;
                    while cin[j] {
                        j = (j + 3) % 4;
                    }
                    next.insert(ek[i], ek[j]);
                    order.push(ek[i]);
                }
            }
        }
    }
    let mut done: HashMap<u64, ()> = HashMap::with_capacity(order.len());
    let mut loops = Vec::new();
    for &start in &order {
        if done.contains_key(&start) {
            continue;
        }
        let mut pts = Vec::new();
        let mut k = start;
        loop {
            if done.insert(k, ()).is_some() {
                break;
            }
            pts.push(point(k));
            match next.get(&k) {
                Some(&n) => k = n,
                None => break,
            }
            if k == start {
                break;
            }
        }
        if pts.len() >= 2 {
            loops.push(Loop { pts });
        }
    }
    loops
}

/// Result of [`edt`]: squared distance (pixels²) to the nearest feature pixel
/// and that pixel's index (`u32::MAX` / `f32::INFINITY` if there is none).
pub struct Edt {
    pub d2: Vec<f32>,
    pub nearest: Vec<u32>,
}

impl Edt {
    #[inline]
    pub fn dist(&self, i: usize) -> f32 {
        self.d2[i].sqrt()
    }
}

/// Exact Euclidean distance transform of `feature` (w × h, top-down).
pub fn edt(feature: &[bool], w: usize, h: usize) -> Edt {
    assert_eq!(feature.len(), w * h);
    const NONE: u32 = u32::MAX;
    // Pass 1 (columns): nearest feature row per pixel, by a down and an up sweep.
    let mut fy = vec![NONE; w * h];
    let mut last = vec![NONE; w];
    for y in 0..h {
        for x in 0..w {
            if feature[y * w + x] {
                last[x] = y as u32;
            }
            fy[y * w + x] = last[x];
        }
    }
    last.iter_mut().for_each(|v| *v = NONE);
    for y in (0..h).rev() {
        for x in 0..w {
            if feature[y * w + x] {
                last[x] = y as u32;
            }
            let up = fy[y * w + x];
            let dn = last[x];
            let pick = match (up, dn) {
                (NONE, d) => d,
                (u, NONE) => u,
                (u, d) => {
                    if (y as i64 - u as i64).abs() <= (d as i64 - y as i64).abs() {
                        u
                    } else {
                        d
                    }
                }
            };
            fy[y * w + x] = pick;
        }
    }
    // Pass 2 (rows): lower envelope of parabolas f(q) = g(q)^2.
    let mut out: Vec<(f32, u32)> = vec![(f32::INFINITY, NONE); w * h];
    par_rows(&mut out, w, |y, orow| {
        let mut v: Vec<usize> = Vec::with_capacity(w);
        let mut z: Vec<f64> = Vec::with_capacity(w + 1);
        let f = |q: usize| -> f64 {
            let r = fy[y * w + q];
            if r == NONE {
                f64::INFINITY
            } else {
                let d = y as f64 - r as f64;
                d * d
            }
        };
        for q in 0..w {
            let fq = f(q);
            if !fq.is_finite() {
                continue;
            }
            loop {
                match v.last() {
                    None => {
                        v.push(q);
                        z.clear();
                        z.push(f64::NEG_INFINITY);
                        break;
                    }
                    Some(&p) => {
                        let s = ((fq + (q * q) as f64) - (f(p) + (p * p) as f64)) / (2.0 * (q as f64 - p as f64));
                        if s <= *z.last().unwrap() {
                            v.pop();
                            z.pop();
                            if v.is_empty() {
                                continue;
                            }
                        } else {
                            v.push(q);
                            z.push(s);
                            break;
                        }
                    }
                }
            }
        }
        if v.is_empty() {
            return;
        }
        let mut k = 0;
        for (x, o) in orow.iter_mut().enumerate() {
            while k + 1 < v.len() && z[k + 1] < x as f64 {
                k += 1;
            }
            let q = v[k];
            let dx = x as f64 - q as f64;
            let d2 = dx * dx + f(q);
            *o = (d2 as f32, fy[y * w + q] * w as u32 + q as u32);
        }
    });
    let (d2, nearest) = out.into_iter().unzip();
    Edt { d2, nearest }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn disk(w: usize, h: usize, cx: f64, cy: f64, r: f64) -> Vec<bool> {
        (0..w * h)
            .map(|i| {
                let (x, y) = ((i % w) as f64, (i / w) as f64);
                (x - cx).powi(2) + (y - cy).powi(2) <= r * r
            })
            .collect()
    }

    #[test]
    fn edt_matches_brute_force() {
        let (w, h) = (53, 41);
        let mut s = 7u64;
        let f: Vec<bool> = (0..w * h)
            .map(|_| {
                s = s.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
                (s >> 59) == 0
            })
            .collect();
        let e = edt(&f, w, h);
        for y in 0..h {
            for x in 0..w {
                let mut best = f64::INFINITY;
                for j in 0..w * h {
                    if f[j] {
                        let d = ((j % w) as f64 - x as f64).powi(2) + ((j / w) as f64 - y as f64).powi(2);
                        best = best.min(d);
                    }
                }
                let i = y * w + x;
                assert_eq!(e.d2[i] as f64, best, "({x},{y})");
                let n = e.nearest[i] as usize;
                let dn = ((n % w) as f64 - x as f64).powi(2) + ((n / w) as f64 - y as f64).powi(2);
                assert_eq!(dn, best);
            }
        }
    }

    #[test]
    fn edt_empty_is_infinite() {
        let e = edt(&[false; 12], 4, 3);
        assert!(e.d2.iter().all(|d| d.is_infinite()));
        assert!(e.nearest.iter().all(|&n| n == u32::MAX));
    }

    #[test]
    fn ms_single_pixel_and_square() {
        let l = marching_squares(&[true], 1, 1);
        assert_eq!(l.len(), 1);
        assert_eq!(l[0].len(), 4);
        // 3x3 square in a 5x5 image: one loop of 12 midpoints
        let mut m = vec![false; 25];
        for y in 1..4 {
            for x in 1..4 {
                m[y * 5 + x] = true;
            }
        }
        let l = marching_squares(&m, 5, 5);
        assert_eq!(l.len(), 1);
        assert_eq!(l[0].len(), 12);
        // outward normals point away from the centre (2,2)
        for i in 0..l[0].len() {
            let (px, py) = l[0].pts[i];
            let (nx, ny) = l[0].normal(i);
            let (vx, vy) = (px as f64 - 2.0, py as f64 - 2.0);
            assert!(vx * nx + vy * ny > 0.0, "vertex {i} ({px},{py}) normal ({nx},{ny})");
        }
    }

    #[test]
    fn ms_hole_and_two_components() {
        // ring: disk r=10 minus disk r=4 -> 2 loops; plus a separate blob -> 3
        let (w, h) = (60, 30);
        let a = disk(w, h, 15.0, 15.0, 10.0);
        let b = disk(w, h, 15.0, 15.0, 4.0);
        let c = disk(w, h, 45.0, 15.0, 5.0);
        let m: Vec<bool> = (0..w * h).map(|i| (a[i] && !b[i]) || c[i]).collect();
        let l = marching_squares(&m, w, h);
        assert_eq!(l.len(), 3);
        // every vertex sits between an inside and an outside pixel, normal points outside
        for lp in &l {
            for i in 0..lp.len() {
                let (px, py) = lp.pts[i];
                let (nx, ny) = lp.normal(i);
                let probe = |s: f64| {
                    let (x, y) = ((px as f64 + nx * s).round() as i64, (py as f64 + ny * s).round() as i64);
                    x >= 0 && y >= 0 && (x as usize) < w && (y as usize) < h && m[y as usize * w + x as usize]
                };
                assert!(!probe(0.75) || probe(-0.75), "normal must not point inside");
            }
        }
    }

    #[test]
    fn ms_is_deterministic() {
        let m = disk(80, 70, 33.3, 30.1, 21.0);
        assert_eq!(marching_squares(&m, 80, 70), marching_squares(&m, 80, 70));
    }
}
