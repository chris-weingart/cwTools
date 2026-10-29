//! Spline trimap: the generated trimap seen as three edges around the seed —
//! INNER (edge of the certain-FG core, = Erode In), MIDDLE (the seed's 0.5
//! iso line) and OUTER (edge of the unknown band, = Dilate Out) — plus an
//! optional OUTER 2 band, a gradient across the unknown band, and up to
//! [`MAX_POINTS`] control points that push the inner / outer edge locally.
//!
//! Identity guarantee: with no active control point, Band Gradient = Flat and
//! Outer 2 off, [`apply`] leaves the production trimap untouched (byte
//! identical by construction). Local changes are only computed where a
//! control point's offset reaches (|delta| >= 0.5 px).
//!
//! ANCHORING (edge-relative, stateless per frame): a control point stores
//! (U, V) = a position in the seed's bounding box (seed > 0.5, U 0..1 left →
//! right, V 0..1 bottom → top). Each frame the point is mapped through that
//! frame's bbox and snapped to the nearest MIDDLE contour vertex; the offset
//! then falls off along the contour (arc length, raised cosine over ±Width).
//! Growth is a dilation, so the pushed edge forms a rounded lobe whose
//! shoulders reach up to R + Out beyond ±Width.
//! The bbox follows translation and scale of the subject, the snap re-finds
//! the edge each frame. Failure cases: see README ("Anchoring").
//!
//! Offsets: Out = px the outer edge moves outward at the point (negative =
//! tighter band), In = px the inner edge moves into the subject (more
//! unknown inside). Both in full-resolution frame pixels. Where a point
//! acts, membership is decided with exact Euclidean distances:
//!   grow:   unknown-or-FG ⇐ dist(p, v) <= R_out + dOut(v) for some middle vertex v
//!           not FG        ⇐ dist(p, v) <= R_in + dIn(v)   for some v
//!   shrink: unknown-or-FG ⇔ dist(p, seed > 0) <= R_out + dOut(p)
//!           FG            ⇔ dist(p, seed < 255 after clamp) > R_in + dIn(p)
//! with R = kernel / 2 (the ellipse radius). Growth (offset > 0) is a
//! variable dilation: the union of discs of radius R + offset(v) around
//! every middle vertex v (smooth, no notches where the outline folds).
//! Shrink (offset < 0) uses the offset of the middle vertex nearest to p.
//!
//! All buffers here are at the model's processing resolution (the padded
//! crop), like production's trimap.

use crate::recipe::{seed_clamp, TrimapParams};
use cwtools_core::contour::{edt, marching_squares, Loop};

pub const MAX_POINTS: usize = 8;
pub const FALLOFF_FLAT: i32 = 0;
pub const FALLOFF_LINEAR: i32 = 1;
pub const FALLOFF_SMOOTH: i32 = 2;

/// Class map values (per proc pixel).
pub const C_BG: u8 = 0;
pub const C_OUTER2: u8 = 1;
pub const C_BAND: u8 = 2;
pub const C_CORE: u8 = 3;

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct CtrlPoint {
    pub on: bool,
    pub u: f64,
    pub v: f64,
    pub out_px: f64,
    pub in_px: f64,
    pub width_px: f64,
}

impl CtrlPoint {
    pub const OFF: CtrlPoint = CtrlPoint { on: false, u: 0.5, v: 0.5, out_px: 0.0, in_px: 0.0, width_px: 150.0 };
    pub fn active(&self) -> bool {
        self.on && self.width_px >= 0.5 && (self.out_px.abs() >= 0.5 || self.in_px.abs() >= 0.5) && self.u.is_finite() && self.v.is_finite()
    }
}

#[derive(Clone, Debug, PartialEq)]
pub struct SplineParams {
    pub enabled: bool,
    pub falloff: i32,
    pub inner_level: f64,
    pub outer_level: f64,
    pub bias: f64,
    pub outer2_on: bool,
    pub outer2_width: f64,
    pub outer2_level: f64,
    pub points: [CtrlPoint; MAX_POINTS],
}

impl SplineParams {
    pub const DEFAULT: SplineParams = SplineParams {
        enabled: true,
        falloff: FALLOFF_FLAT,
        inner_level: 0.75,
        outer_level: 0.25,
        bias: 0.5,
        outer2_on: false,
        outer2_width: 40.0,
        outer2_level: 0.2,
        points: [CtrlPoint::OFF; MAX_POINTS],
    };

    pub fn any_points(&self) -> bool {
        self.points.iter().any(|p| p.active())
    }
    /// True when [`apply`] would not change the production trimap.
    pub fn is_identity(&self) -> bool {
        !self.enabled || (self.falloff == FALLOFF_FLAT && !self.outer2_on && !self.any_points())
    }
    /// Extra crop padding (frame px) so local outward pushes and Outer 2 stay
    /// inside the ViTMatte crop. 0 for the identity case.
    pub fn extra_reach(&self) -> usize {
        if !self.enabled {
            return 0;
        }
        let out = self.points.iter().filter(|p| p.active()).map(|p| p.out_px).fold(0.0f64, f64::max);
        let o2 = if self.outer2_on { self.outer2_width.max(0.0) } else { 0.0 };
        let r = out + o2;
        if r.is_finite() && r > 0.0 {
            r.ceil() as usize
        } else {
            0
        }
    }
    /// Pixel quantities scaled by the render scale (proxy).
    pub fn scaled(&self, rs: f64) -> SplineParams {
        let mut s = self.clone();
        if !(rs.is_finite() && rs > 0.0) || (rs - 1.0).abs() < 1e-9 {
            return s;
        }
        s.outer2_width *= rs;
        for p in s.points.iter_mut() {
            p.out_px *= rs;
            p.in_px *= rs;
            p.width_px *= rs;
        }
        s
    }
}

/// Frame (top-down pixel index space) ↔ processing space of one crop.
/// proc = (frame - x0 + 0.5) * s - 0.5 (pixel-centre mapping of the resize).
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct CropMap {
    pub x0: f64,
    pub y0: f64,
    pub sx: f64,
    pub sy: f64,
}

impl CropMap {
    pub fn to_proc(&self, fx: f64, fy: f64) -> (f64, f64) {
        ((fx - self.x0 + 0.5) * self.sx - 0.5, (fy - self.y0 + 0.5) * self.sy - 0.5)
    }
    pub fn to_frame(&self, px: f64, py: f64) -> (f64, f64) {
        ((px + 0.5) / self.sx - 0.5 + self.x0, (py + 0.5) / self.sy - 0.5 + self.y0)
    }
    /// Mean frame px per proc px (>= 1 when the crop was downscaled).
    pub fn frame_per_proc(&self) -> f64 {
        0.5 * (1.0 / self.sx + 1.0 / self.sy)
    }
}

/// A control point resolved on this frame (proc coordinates).
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Resolved {
    pub anchor: (f64, f64),
    /// Unit outward normal (top-down proc coordinates).
    pub normal: (f64, f64),
}

pub struct Applied {
    /// Class per padded proc pixel (C_BG / C_OUTER2 / C_BAND / C_CORE).
    pub class: Vec<u8>,
    /// Middle (0.5) contour loops in proc coordinates, restricted to pw × ph.
    pub mid: Vec<Loop>,
    pub points: [Option<Resolved>; MAX_POINTS],
    /// Ellipse radii (proc px) of the global kernels.
    pub r_out: f64,
    pub r_in: f64,
}

/// Kernel radii (proc px) for the global Erode In / Dilate Out.
pub fn radii(tp: &TrimapParams) -> (f64, f64) {
    let (ke, kd) = match tp.asym {
        Some((a, b)) => (a.max(1), b.max(1)),
        None => (tp.kernel_size.max(1), tp.kernel_size.max(1)),
    };
    ((ke / 2) as f64, (kd / 2) as f64)
}

/// Is a MIDDLE vertex on the artificial border of the pw × ph region?
#[inline]
pub fn on_border(p: (f32, f32), pw: usize, ph: usize) -> bool {
    p.0 < 0.0 || p.1 < 0.0 || p.0 > (pw as f32 - 1.0) || p.1 > (ph as f32 - 1.0)
}

/// Middle contour of a padded proc seed (seed >= 128), restricted to pw × ph.
pub fn middle_loops(seed_pad: &[u8], padw: usize, pw: usize, ph: usize) -> Vec<Loop> {
    let mut m = vec![false; pw * ph];
    for y in 0..ph {
        for x in 0..pw {
            m[y * pw + x] = seed_pad[y * padw + x] >= 128;
        }
    }
    marching_squares(&m, pw, ph)
}

/// Map a (U, V) anchor through the frame bbox (inclusive pixel indices,
/// top-down) to a frame position (pixel-centre coordinates).
pub fn anchor_to_frame(u: f64, v: f64, bbox: (usize, usize, usize, usize)) -> (f64, f64) {
    let (xmin, xmax, ymin, ymax) = bbox;
    (xmin as f64 + u * (xmax as f64 - xmin as f64), ymax as f64 - v * (ymax as f64 - ymin as f64))
}

/// Inverse of [`anchor_to_frame`].
pub fn frame_to_anchor(fx: f64, fy: f64, bbox: (usize, usize, usize, usize)) -> (f64, f64) {
    let (xmin, xmax, ymin, ymax) = bbox;
    let bw = xmax as f64 - xmin as f64;
    let bh = ymax as f64 - ymin as f64;
    let u = if bw > 0.0 { (fx - xmin as f64) / bw } else { 0.5 };
    let v = if bh > 0.0 { (ymax as f64 - fy) / bh } else { 0.5 };
    (u, v)
}

/// Nearest non-border vertex: (loop, index, squared distance).
pub fn snap(loops: &[Loop], p: (f64, f64), pw: usize, ph: usize) -> Option<(usize, usize, f64)> {
    let mut best: Option<(usize, usize, f64)> = None;
    for (li, l) in loops.iter().enumerate() {
        for (i, &q) in l.pts.iter().enumerate() {
            if on_border(q, pw, ph) {
                continue;
            }
            let d = (q.0 as f64 - p.0).powi(2) + (q.1 as f64 - p.1).powi(2);
            if best.map(|b| d < b.2).unwrap_or(true) {
                best = Some((li, i, d));
            }
        }
    }
    best
}

/// Set every pixel (centre) within `r` of (cx, cy).
fn stamp_disc(m: &mut [bool], w: usize, h: usize, cx: f64, cy: f64, r: f64) {
    if !(r.is_finite() && r >= 0.0) {
        return;
    }
    let y0 = (cy - r).ceil().max(0.0) as i64;
    let y1 = ((cy + r).floor() as i64).min(h as i64 - 1);
    for y in y0..=y1 {
        let dy = y as f64 - cy;
        let hw = (r * r - dy * dy).max(0.0).sqrt();
        let x0 = (cx - hw).ceil().max(0.0) as i64;
        let x1 = ((cx + hw).floor() as i64).min(w as i64 - 1);
        if x1 >= x0 {
            m[y as usize * w + x0 as usize..=y as usize * w + x1 as usize].fill(true);
        }
    }
}

/// Moving average of a closed loop over ±`k` vertices.
fn smoothed(l: &Loop, k: usize) -> Vec<(f64, f64)> {
    let n = l.len();
    let k = k.min((n.saturating_sub(1)) / 2);
    let m = (2 * k + 1) as f64;
    (0..n)
        .map(|i| {
            let (mut x, mut y) = (0.0, 0.0);
            for d in 0..=2 * k {
                let q = l.pts[(i + n + d - k) % n];
                x += q.0 as f64;
                y += q.1 as f64;
            }
            (x / m, y / m)
        })
        .collect()
}

#[inline]
fn raised_cos(s: f64, w: f64) -> f64 {
    if s >= w {
        0.0
    } else {
        0.5 * (1.0 + (std::f64::consts::PI * s / w).cos())
    }
}

/// Per-vertex offsets (proc px) of all active points on the middle loops.
/// Returns (dOut, dIn) per loop and the resolved points.
#[allow(clippy::type_complexity)]
pub fn vertex_offsets(
    mid: &[Loop],
    sp: &SplineParams,
    map: &CropMap,
    bbox: (usize, usize, usize, usize),
    pw: usize,
    ph: usize,
) -> (Vec<Vec<f64>>, Vec<Vec<f64>>, [Option<Resolved>; MAX_POINTS]) {
    let mut dout: Vec<Vec<f64>> = mid.iter().map(|l| vec![0.0; l.len()]).collect();
    let mut din: Vec<Vec<f64>> = dout.clone();
    let mut res = [None; MAX_POINTS];
    let s = 0.5 * (map.sx + map.sy);
    for (k, p) in sp.points.iter().enumerate() {
        if !p.on || !p.u.is_finite() || !p.v.is_finite() {
            continue;
        }
        let (fx, fy) = anchor_to_frame(p.u, p.v, bbox);
        let Some((li, vi, _)) = snap(mid, map.to_proc(fx, fy), pw, ph) else { continue };
        let l = &mid[li];
        let q = l.pts[vi];
        res[k] = Some(Resolved { anchor: (q.0 as f64, q.1 as f64), normal: l.normal(vi) });
        if !p.active() {
            continue;
        }
        let (o, i_, w) = (p.out_px * s, p.in_px * s, (p.width_px * s).max(0.5));
        let n = l.len();
        // arc length on a lightly smoothed loop (the raw marching-squares
        // staircase over-measures arcs by up to ~10%)
        let sm = smoothed(l, 6);
        let seg = |j: usize| {
            let (a, b) = (sm[j % n], sm[(j + 1) % n]);
            ((b.0 - a.0).powi(2) + (b.1 - a.1).powi(2)).sqrt()
        };
        let mut add = |j: usize, arc: f64| {
            let wt = raised_cos(arc, w);
            dout[li][j] += o * wt;
            din[li][j] += i_ * wt;
        };
        add(vi, 0.0);
        // walk both directions until Width or a border vertex / full loop
        for dir in [1usize, n - 1] {
            let (mut j, mut arc) = (vi, 0.0);
            for _ in 0..n / 2 {
                let nj = (j + dir) % n;
                arc += if dir == 1 { seg(j) } else { seg(nj) };
                if arc >= w || on_border(l.pts[nj], pw, ph) {
                    break;
                }
                add(nj, arc);
                j = nj;
            }
        }
    }
    (dout, din, res)
}

/// Apply the spline edits to a production trimap (padded proc size) in place.
/// `seed_pad` = the padded proc seed the trimap came from (before clamp).
/// Returns None when nothing changes and no geometry is wanted.
#[allow(clippy::too_many_arguments)]
pub fn apply(
    tri: &mut [u8],
    seed_pad: &[u8],
    padw: usize,
    padh: usize,
    pw: usize,
    ph: usize,
    tp: &TrimapParams,
    sp: &SplineParams,
    map: &CropMap,
    bbox: (usize, usize, usize, usize),
    want_geometry: bool,
) -> Option<Applied> {
    let identity = sp.is_identity();
    if identity && !want_geometry {
        return None;
    }
    let n = padw * padh;
    let (r_in, r_out) = radii(tp);
    let mid = middle_loops(seed_pad, padw, pw, ph);
    let base_class = |t: u8| if t == 0 { C_BG } else if t == 255 { C_CORE } else { C_BAND };
    if identity {
        let class = tri.iter().map(|&t| base_class(t)).collect();
        let (_, _, points) = vertex_offsets(&mid, sp, map, bbox, pw, ph);
        return Some(Applied { class, mid, points, r_out, r_in });
    }

    let (dout_v, din_v, points) = vertex_offsets(&mid, sp, map, bbox, pw, ph);
    let any_pts = sp.any_points();

    // Growth (offset > 0) = variable dilation: union of discs of radius
    // R + offset around every middle vertex (smooth by construction, no
    // notches where the outline folds). Shrink (offset < 0) = nearest-vertex
    // rule: the offset of the middle vertex nearest to the pixel.
    let mut grow_out = vec![false; n];
    let mut grow_out2 = vec![false; n];
    let mut grow_in = vec![false; n];
    let (mut dout, mut din) = (vec![0f32; n], vec![0f32; n]);
    let s_avg = 0.5 * (map.sx + map.sy);
    let w2 = if sp.outer2_on { sp.outer2_width.max(0.0) * s_avg } else { 0.0 };
    let any_neg = any_pts && dout_v.iter().chain(din_v.iter()).flatten().any(|d| *d <= -0.5);
    if any_pts {
        let mut feat = vec![false; n];
        let mut fo = vec![0f32; n];
        let mut fi = vec![0f32; n];
        for (li, l) in mid.iter().enumerate() {
            for (j, &q) in l.pts.iter().enumerate() {
                if on_border(q, pw, ph) {
                    continue;
                }
                let (o, ii) = (dout_v[li][j], din_v[li][j]);
                let (qx, qy) = (q.0 as f64, q.1 as f64);
                if o >= 0.5 {
                    stamp_disc(&mut grow_out, padw, padh, qx, qy, r_out + o);
                    if sp.outer2_on {
                        stamp_disc(&mut grow_out2, padw, padh, qx, qy, r_out + o + w2);
                    }
                }
                if ii >= 0.5 {
                    stamp_disc(&mut grow_in, padw, padh, qx, qy, r_in + ii);
                }
                if any_neg {
                    let x = (q.0.round() as i64).clamp(0, pw as i64 - 1) as usize;
                    let y = (q.1.round() as i64).clamp(0, ph as i64 - 1) as usize;
                    let i = y * padw + x;
                    if !feat[i] || (o as f32) < fo[i] {
                        fo[i] = o as f32;
                    }
                    if !feat[i] || (ii as f32) < fi[i] {
                        fi[i] = ii as f32;
                    }
                    feat[i] = true;
                }
            }
        }
        if any_neg && feat.iter().any(|&f| f) {
            let e = edt(&feat, padw, padh);
            for i in 0..n {
                let k = e.nearest[i];
                if k != u32::MAX {
                    dout[i] = fo[k as usize].min(0.0);
                    din[i] = fi[k as usize].min(0.0);
                }
            }
        }
    }
    let any_shrink_in = din.iter().any(|d| *d <= -0.5);

    let mut clamped = seed_pad.to_vec();
    seed_clamp(&mut clamped, tp.seed_clamp);
    let need_sup = any_neg || sp.outer2_on;
    let d_sup = need_sup.then(|| edt(&clamped.iter().map(|&s| s > 0).collect::<Vec<_>>(), padw, padh));
    let d_nc = any_shrink_in.then(|| edt(&clamped.iter().map(|&s| s < 255).collect::<Vec<_>>(), padw, padh));
    let w2 = w2 as f32;

    let mut class = vec![C_BG; n];
    for i in 0..n {
        let base = tri[i];
        let (o, ii) = (dout[i], din[i]);
        let reach = (r_out as f32 + o).max(0.0);
        let outer_base = if o <= -0.5 { d_sup.as_ref().map(|e| e.dist(i) <= reach).unwrap_or(base != 0) } else { base != 0 };
        let outer = outer_base || grow_out[i];
        let core_base = if ii <= -0.5 {
            let r = (r_in as f32 + ii).max(0.0);
            d_nc.as_ref().map(|e| e.dist(i) > r).unwrap_or(base == 255)
        } else {
            base == 255
        };
        let core = core_base && !grow_in[i];
        class[i] = if core && outer {
            C_CORE
        } else if outer {
            C_BAND
        } else if sp.outer2_on && (grow_out2[i] || d_sup.as_ref().map(|e| e.dist(i) <= reach + w2).unwrap_or(false)) {
            C_OUTER2
        } else {
            C_BG
        };
    }

    let lvl = |x: f64| ((x.clamp(0.0, 1.0) * 255.0).round() as i64).clamp(1, 254) as u8;
    let unk = tp.unknown_level();
    let l2 = lvl(sp.outer2_level);
    if sp.falloff == FALLOFF_FLAT {
        for i in 0..n {
            tri[i] = match class[i] {
                C_CORE => 255,
                C_BAND => unk,
                C_OUTER2 => l2,
                _ => 0,
            };
        }
    } else {
        let a = edt(&class.iter().map(|&c| c == C_CORE).collect::<Vec<_>>(), padw, padh);
        let b = edt(&class.iter().map(|&c| c <= C_OUTER2).collect::<Vec<_>>(), padw, padh);
        let bias = sp.bias.clamp(0.01, 0.99);
        for i in 0..n {
            tri[i] = match class[i] {
                C_CORE => 255,
                C_OUTER2 => l2,
                C_BG => 0,
                _ => {
                    let (da, db) = (a.dist(i) as f64, b.dist(i) as f64);
                    let t = match (da.is_finite(), db.is_finite()) {
                        (true, true) => da / (da + db).max(1e-9),
                        (false, true) => 1.0,
                        (true, false) => 0.0,
                        _ => 0.5,
                    };
                    let t = if sp.falloff == FALLOFF_SMOOTH { t * t * (3.0 - 2.0 * t) } else { t };
                    let t = t / ((1.0 / bias - 2.0) * (1.0 - t) + 1.0);
                    lvl(sp.inner_level + (sp.outer_level - sp.inner_level) * t)
                }
            };
        }
    }
    Some(Applied { class, mid, points, r_out, r_in })
}

// ---------------------------------------------------------------------------
// Overlay geometry (canonical coordinates) + per-instance cache
// ---------------------------------------------------------------------------

/// Frame index space → OFX canonical coordinates.
#[derive(Clone, Copy, Debug)]
pub struct Canon {
    pub bx1: f64,
    pub by1: f64,
    pub h: f64,
    pub rs: f64,
}

impl Canon {
    pub fn pt(&self, fx: f64, fy: f64) -> (f64, f64) {
        ((self.bx1 + fx + 0.5) / self.rs, (self.by1 + self.h - 0.5 - fy) / self.rs)
    }
    /// Canonical → frame index space (inverse of [`pt`]).
    pub fn frame(&self, cx: f64, cy: f64) -> (f64, f64) {
        (cx * self.rs - self.bx1 - 0.5, self.by1 + self.h - 0.5 - cy * self.rs)
    }
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct GeoPoint {
    pub slot: usize,
    pub anchor: (f64, f64),
    /// Unit outward normal, canonical (y up).
    pub normal: (f64, f64),
    pub out_handle: (f64, f64),
    pub in_handle: (f64, f64),
}

#[derive(Clone, Debug, Default)]
pub struct Geometry {
    pub time: f64,
    pub inner: Vec<Vec<(f64, f64)>>,
    pub middle: Vec<Vec<(f64, f64)>>,
    pub outer: Vec<Vec<(f64, f64)>>,
    pub outer2: Vec<Vec<(f64, f64)>>,
    /// Seed bbox as pixel-centre canonical coords: (x_left, y_bottom, x_right, y_top).
    pub bbox_c: (f64, f64, f64, f64),
    /// Source bounds, canonical (x1, y1, x2, y2).
    pub frame_c: (f64, f64, f64, f64),
    /// Full-res frame px per proc px (kernel → canonical radius factor).
    pub fpp: f64,
    pub r_out_c: f64,
    pub r_in_c: f64,
    pub points: Vec<GeoPoint>,
}

impl Geometry {
    pub fn vertex_count(&self) -> usize {
        [&self.inner, &self.middle, &self.outer, &self.outer2].iter().map(|v| v.iter().map(|p| p.len()).sum::<usize>()).sum()
    }
    /// Nearest middle vertex to a canonical point: (point, outward normal, distance).
    pub fn nearest_middle(&self, p: (f64, f64)) -> Option<((f64, f64), (f64, f64), f64)> {
        let mut best: Option<((f64, f64), (f64, f64), f64)> = None;
        for pl in &self.middle {
            for i in 0..pl.len() {
                let q = pl[i];
                let d = ((q.0 - p.0).powi(2) + (q.1 - p.1).powi(2)).sqrt();
                if best.map(|b| d < b.2).unwrap_or(true) {
                    let a = pl[i.saturating_sub(1)];
                    let b = pl[(i + 1).min(pl.len() - 1)];
                    // canonical is y-up: outward normal of direction (dx,dy) = (-dy, dx)
                    let (dx, dy) = (b.0 - a.0, b.1 - a.1);
                    let l = (dx * dx + dy * dy).sqrt().max(1e-12);
                    best = Some((q, (-dy / l, dx / l), d));
                }
            }
        }
        best
    }
    /// Distance from a canonical point to the nearest vertex of `lines`.
    pub fn dist_to(lines: &[Vec<(f64, f64)>], p: (f64, f64)) -> f64 {
        lines.iter().flatten().map(|q| ((q.0 - p.0).powi(2) + (q.1 - p.1).powi(2)).sqrt()).fold(f64::INFINITY, f64::min)
    }
    /// (U, V) of a canonical point in this frame's bbox.
    pub fn uv(&self, p: (f64, f64)) -> (f64, f64) {
        let (x0, y0, x1, y1) = self.bbox_c;
        let u = if (x1 - x0).abs() > 1e-9 { (p.0 - x0) / (x1 - x0) } else { 0.5 };
        let v = if (y1 - y0).abs() > 1e-9 { (p.1 - y0) / (y1 - y0) } else { 0.5 };
        (u, v)
    }
}

/// Split loops at border vertices and drop collinear runs; map proc → canonical.
fn polylines(loops: &[Loop], pw: usize, ph: usize, map: &CropMap, cn: &Canon) -> Vec<Vec<(f64, f64)>> {
    let mut out = Vec::new();
    for l in loops {
        let n = l.len();
        if n < 2 {
            continue;
        }
        // start at a border vertex if there is one, so runs don't wrap
        let start = (0..n).find(|&i| on_border(l.pts[i], pw, ph));
        let closed = start.is_none();
        let s0 = start.unwrap_or(0);
        let mut cur: Vec<(f64, f64)> = Vec::new();
        let push_run = |cur: &mut Vec<(f64, f64)>, out: &mut Vec<Vec<(f64, f64)>>| {
            if cur.len() >= 2 {
                out.push(std::mem::take(cur));
            } else {
                cur.clear();
            }
        };
        let count = if closed { n + 1 } else { n };
        for k in 0..count {
            let q = l.pts[(s0 + k) % n];
            if on_border(q, pw, ph) {
                push_run(&mut cur, &mut out);
                continue;
            }
            let (fx, fy) = map.to_frame(q.0 as f64, q.1 as f64);
            let c = cn.pt(fx, fy);
            // drop the middle of three collinear points
            if cur.len() >= 2 {
                let a = cur[cur.len() - 2];
                let b = cur[cur.len() - 1];
                let cross = (b.0 - a.0) * (c.1 - a.1) - (b.1 - a.1) * (c.0 - a.0);
                if cross.abs() < 1e-6 {
                    cur.pop();
                }
            }
            cur.push(c);
        }
        push_run(&mut cur, &mut out);
    }
    out
}

/// Build the overlay geometry of one render.
#[allow(clippy::too_many_arguments)]
pub fn build_geometry(
    ap: &Applied,
    padw: usize,
    pw: usize,
    ph: usize,
    map: &CropMap,
    cn: &Canon,
    bbox: (usize, usize, usize, usize),
    sp_full: &SplineParams,
    outer2: bool,
    time: f64,
    frame_c: (f64, f64, f64, f64),
) -> Geometry {
    let sub = |f: &dyn Fn(u8) -> bool| -> Vec<bool> {
        let mut m = vec![false; pw * ph];
        for y in 0..ph {
            for x in 0..pw {
                m[y * pw + x] = f(ap.class[y * padw + x]);
            }
        }
        m
    };
    let inner = marching_squares(&sub(&|c| c == C_CORE), pw, ph);
    let outer = marching_squares(&sub(&|c| c >= C_BAND), pw, ph);
    let outer2l = if outer2 { marching_squares(&sub(&|c| c >= C_OUTER2), pw, ph) } else { Vec::new() };
    // frame px (at this render scale) per proc px; kernel param k gives a
    // canonical radius of k * fpp / 2 at any render scale.
    let fpp = map.frame_per_proc();
    let r_out_c = ap.r_out * fpp / cn.rs;
    let r_in_c = ap.r_in * fpp / cn.rs;
    let (xmin, xmax, ymin, ymax) = bbox;
    let bl = cn.pt(xmin as f64, ymax as f64);
    let tr = cn.pt(xmax as f64, ymin as f64);
    let mut points = Vec::new();
    for (k, r) in ap.points.iter().enumerate() {
        let Some(r) = r else { continue };
        let (fx, fy) = map.to_frame(r.anchor.0, r.anchor.1);
        let a = cn.pt(fx, fy);
        let nl = (r.normal.0.powi(2) + r.normal.1.powi(2)).sqrt().max(1e-12);
        let nrm = (r.normal.0 / nl, -r.normal.1 / nl);
        let p = &sp_full.points[k];
        let ro = (r_out_c + p.out_px).max(0.0);
        let ri = (r_in_c + p.in_px).max(0.0);
        points.push(GeoPoint {
            slot: k,
            anchor: a,
            normal: nrm,
            out_handle: (a.0 + nrm.0 * ro, a.1 + nrm.1 * ro),
            in_handle: (a.0 - nrm.0 * ri, a.1 - nrm.1 * ri),
        });
    }
    Geometry {
        time,
        inner: polylines(&inner, pw, ph, map, cn),
        middle: polylines(&ap.mid, pw, ph, map, cn),
        outer: polylines(&outer, pw, ph, map, cn),
        outer2: polylines(&outer2l, pw, ph, map, cn),
        bbox_c: (bl.0, bl.1, tr.0, tr.1),
        frame_c,
        fpp,
        r_out_c,
        r_in_c,
        points,
    }
}

use std::collections::HashMap;
use std::sync::{Arc, Mutex};

static GEOM: Mutex<Option<HashMap<usize, Vec<Arc<Geometry>>>>> = Mutex::new(None);
const GEOM_KEEP: usize = 24;

/// Remember the geometry of a render (per effect instance, last few times).
pub fn store_geometry(effect: usize, g: Geometry) {
    let mut lock = GEOM.lock().unwrap_or_else(|e| e.into_inner());
    let m = lock.get_or_insert_with(HashMap::new);
    let v = m.entry(effect).or_default();
    v.retain(|x| (x.time - g.time).abs() > 1e-6);
    v.push(Arc::new(g));
    if v.len() > GEOM_KEEP {
        v.remove(0);
    }
}

/// Geometry for `time` (exact), else the most recent one (stale = true).
pub fn geometry_for(effect: usize, time: f64) -> Option<(Arc<Geometry>, bool)> {
    let lock = GEOM.lock().unwrap_or_else(|e| e.into_inner());
    let v = lock.as_ref()?.get(&effect)?;
    if let Some(g) = v.iter().find(|g| (g.time - time).abs() < 1e-6) {
        return Some((g.clone(), false));
    }
    v.last().map(|g| (g.clone(), true))
}

pub fn clear_geometry(effect: usize) {
    let mut lock = GEOM.lock().unwrap_or_else(|e| e.into_inner());
    if let Some(m) = lock.as_mut() {
        m.remove(&effect);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::recipe::gen_trimap;

    fn disk_seed(w: usize, h: usize, cx: f64, cy: f64, r: f64) -> Vec<u8> {
        (0..w * h)
            .map(|i| {
                let (x, y) = ((i % w) as f64, (i / w) as f64);
                if (x - cx).powi(2) + (y - cy).powi(2) <= r * r {
                    255
                } else {
                    0
                }
            })
            .collect()
    }
    const ID: CropMap = CropMap { x0: 0.0, y0: 0.0, sx: 1.0, sy: 1.0 };

    fn bbox(seed: &[u8], w: usize, h: usize) -> (usize, usize, usize, usize) {
        crate::recipe::bbox_gt(seed, w, h, 128).unwrap()
    }

    #[test]
    fn identity_leaves_trimap_untouched() {
        let (w, h) = (200, 160);
        let s = disk_seed(w, h, 100.0, 80.0, 50.0);
        let tp = TrimapParams::HAIR;
        let t0 = gen_trimap(&s, w, h, &tp);
        let mut t = t0.clone();
        let sp = SplineParams::DEFAULT;
        assert!(sp.is_identity());
        assert!(apply(&mut t, &s, w, h, w, h, &tp, &sp, &ID, bbox(&s, w, h), false).is_none());
        let ap = apply(&mut t, &s, w, h, w, h, &tp, &sp, &ID, bbox(&s, w, h), true).unwrap();
        assert_eq!(t, t0);
        assert_eq!(ap.class.iter().filter(|&&c| c == C_BAND).count(), t0.iter().filter(|&&v| v == 128).count());
        // a point that is ON but has zero offsets is still the identity
        let mut sp2 = sp.clone();
        sp2.points[0].on = true;
        assert!(sp2.is_identity());
        assert_eq!(sp2.extra_reach(), 0);
    }

    /// Distance from the disk centre to the outer band edge along a ray.
    fn reach_along(tri: &[u8], w: usize, cx: f64, cy: f64, ang: f64) -> f64 {
        let (dx, dy) = (ang.cos(), ang.sin());
        let mut last = 0.0;
        for s in 0..400 {
            let r = s as f64 * 0.5;
            let (x, y) = ((cx + dx * r).round() as i64, (cy + dy * r).round() as i64);
            if x < 0 || y < 0 || x as usize >= w || y as usize >= tri.len() / w {
                break;
            }
            if tri[y as usize * w + x as usize] != 0 {
                last = r;
            }
        }
        last
    }

    #[test]
    fn local_outward_push_follows_spec() {
        let (w, h) = (400, 400);
        let (cx, cy, r) = (200.0, 200.0, 80.0);
        let s = disk_seed(w, h, cx, cy, r);
        let tp = TrimapParams::HAIR; // dilate 30 -> R_out 15
        let base = gen_trimap(&s, w, h, &tp);
        let mut sp = SplineParams::DEFAULT;
        // top of the subject (V = 1), 80 px out, width 100 px of arc
        sp.points[0] = CtrlPoint { on: true, u: 0.5, v: 1.0, out_px: 80.0, in_px: 0.0, width_px: 100.0 };
        let mut t = base.clone();
        apply(&mut t, &s, w, h, w, h, &tp, &sp, &ID, bbox(&s, w, h), false).unwrap();
        let up = -std::f64::consts::FRAC_PI_2; // top-down: up = -y
        let at = reach_along(&t, w, cx, cy, up);
        assert!((at - (r + 15.0 + 80.0)).abs() <= 2.0, "reach at anchor {at}");
        // Growth is a dilation (union of discs R + offset around the edge):
        // the lobe has rounded shoulders, so it is back to base only beyond
        // Width + R + Out of arc (100 + 15 + 80 = 195 px ≈ 2.45 rad on r=80).
        let far = reach_along(&t, w, cx, cy, up + 2.5);
        let far0 = reach_along(&base, w, cx, cy, up + 2.5);
        assert!((far - far0).abs() <= 1.0, "far {far} vs {far0}");
        // half-way along Width: at least the raised-cosine value (+40), below the peak
        let mid = reach_along(&t, w, cx, cy, up + 50.0 / r);
        assert!(mid >= r + 15.0 + 40.0 - 2.0 && mid < at, "mid {mid}");
        // monotone falloff on both sides
        for sgn in [1.0, -1.0] {
            let mut prev = f64::INFINITY;
            for k in 0..30 {
                let v = reach_along(&t, w, cx, cy, up + sgn * k as f64 * 0.1);
                assert!(v <= prev + 1.0, "k={k} {v} > {prev}");
                prev = v;
            }
        }
        // only the band changed: core identical, nothing outside got FG
        for i in 0..w * h {
            assert_eq!(t[i] == 255, base[i] == 255);
        }
    }

    #[test]
    fn inner_push_and_negative_out() {
        let (w, h) = (300, 300);
        let s = disk_seed(w, h, 150.0, 150.0, 90.0);
        let tp = TrimapParams::HAIR;
        let base = gen_trimap(&s, w, h, &tp);
        let mut sp = SplineParams::DEFAULT;
        sp.points[0] = CtrlPoint { on: true, u: 1.0, v: 0.5, out_px: -10.0, in_px: 30.0, width_px: 60.0 };
        let mut t = base.clone();
        apply(&mut t, &s, w, h, w, h, &tp, &sp, &ID, bbox(&s, w, h), false).unwrap();
        // right edge, row 150: base core ends ~ x = 150+90-2, band to ~ 150+90+15
        let row = |m: &[u8], v: u8| (0..w).filter(|&x| m[150 * w + x] == v).max().unwrap_or(0) as f64;
        let core0 = (0..w).filter(|&x| base[150 * w + x] == 255).max().unwrap() as f64;
        let core1 = (0..w).filter(|&x| t[150 * w + x] == 255).max().unwrap() as f64;
        assert!((core0 - core1 - 30.0).abs() <= 2.0, "core {core0} -> {core1}");
        let out0 = row(&base, 128);
        let out1 = row(&t, 128);
        assert!((out0 - out1 - 10.0).abs() <= 2.0, "outer {out0} -> {out1}");
    }

    /// Edge-relative anchoring: the same params on a translated + scaled
    /// subject move the local stretch with it.
    #[test]
    fn anchoring_follows_translate_and_scale() {
        let (w, h) = (640, 480);
        let tp = TrimapParams::HAIR;
        let mut sp = SplineParams::DEFAULT;
        // 30° above the right-hand side of the subject
        sp.points[0] = CtrlPoint { on: true, u: 0.95, v: 0.80, out_px: 60.0, in_px: 0.0, width_px: 80.0 };
        let centroid_of_extra = |cx: f64, cy: f64, r: f64| -> (f64, f64, f64) {
            // blob: disk + a smaller disk (non-symmetric so bbox != circle)
            let a = disk_seed(w, h, cx, cy, r);
            let b = disk_seed(w, h, cx - 0.6 * r, cy + 0.5 * r, 0.6 * r);
            let s: Vec<u8> = a.iter().zip(&b).map(|(&x, &y)| x.max(y)).collect();
            let base = gen_trimap(&s, w, h, &tp);
            let mut t = base.clone();
            apply(&mut t, &s, w, h, w, h, &tp, &sp, &ID, bbox(&s, w, h), false).unwrap();
            let (mut sx, mut sy, mut n) = (0.0, 0.0, 0.0);
            for i in 0..w * h {
                if t[i] != base[i] {
                    sx += (i % w) as f64;
                    sy += (i / w) as f64;
                    n += 1.0;
                }
            }
            (sx / n, sy / n, n)
        };
        let (x1, y1, n1) = centroid_of_extra(250.0, 240.0, 80.0);
        // translate by (+120, -60): the extra region moves by the same vector
        let (x2, y2, n2) = centroid_of_extra(370.0, 180.0, 80.0);
        assert!(((x2 - x1) - 120.0).abs() < 2.0 && ((y2 - y1) + 60.0).abs() < 2.0, "{x1},{y1} -> {x2},{y2}");
        assert!((n1 - n2).abs() / n1 < 0.03);
        // scale x1.5 around the centre: the extra region's direction from the
        // subject centre is preserved (same anchor on the edge)
        let (x3, y3, _) = centroid_of_extra(250.0, 240.0, 120.0);
        let a1 = (y1 - 240.0).atan2(x1 - 250.0);
        let a3 = (y3 - 240.0).atan2(x3 - 250.0);
        assert!((a1 - a3).abs() < 0.08, "angle {a1} vs {a3}");
    }

    #[test]
    fn gradient_and_outer2_levels() {
        let (w, h) = (300, 300);
        let s = disk_seed(w, h, 150.0, 150.0, 70.0);
        let mut tp = TrimapParams::HAIR;
        tp.asym = Some((20, 60));
        let base = gen_trimap(&s, w, h, &tp);
        let mut sp = SplineParams::DEFAULT;
        sp.falloff = FALLOFF_LINEAR;
        sp.outer2_on = true;
        sp.outer2_width = 20.0;
        let mut t = base.clone();
        let ap = apply(&mut t, &s, w, h, w, h, &tp, &sp, &ID, bbox(&s, w, h), false).unwrap();
        // along the row through the centre, right side: levels fall from inner to outer
        let row: Vec<u8> = (150..w).map(|x| t[150 * w + x]).collect();
        let band: Vec<u8> = row.iter().copied().filter(|&v| v != 0 && v != 255 && v != 51).collect();
        assert!(band.len() > 30);
        assert!(band.first().unwrap() > band.last().unwrap());
        assert!(band.windows(2).all(|p| p[0] >= p[1]));
        assert!(*band.first().unwrap() >= 180 && *band.last().unwrap() <= 75);
        // outer 2 band ~20 px wide at level 0.2 (51)
        let o2 = row.iter().filter(|&&v| v == 51).count();
        assert!((18..=22).contains(&o2), "outer2 {o2}");
        assert!(ap.class.iter().any(|&c| c == C_OUTER2));
        // core / BG untouched beyond outer2
        for i in 0..w * h {
            if base[i] == 255 {
                assert_eq!(t[i], 255);
            }
        }
    }

    #[test]
    fn geometry_maps_to_canonical() {
        let (w, h) = (200, 100);
        let s = disk_seed(w, h, 100.0, 50.0, 30.0);
        let tp = TrimapParams::HAIR;
        let mut t = gen_trimap(&s, w, h, &tp);
        let sp = SplineParams::DEFAULT;
        let bb = bbox(&s, w, h);
        let ap = apply(&mut t, &s, w, h, w, h, &tp, &sp, &ID, bb, true).unwrap();
        let cn = Canon { bx1: 0.0, by1: 0.0, h: h as f64, rs: 1.0 };
        let g = build_geometry(&ap, w, w, h, &ID, &cn, bb, &sp, false, 1.0, (0.0, 0.0, w as f64, h as f64));
        assert_eq!(g.middle.len(), 1);
        assert_eq!(g.outer.len(), 1);
        assert_eq!(g.inner.len(), 1);
        // canonical centre of the disk is (100.5, 49.5) (y flipped, pixel centres)
        let mean = |pl: &Vec<(f64, f64)>| {
            let n = pl.len() as f64;
            (pl.iter().map(|p| p.0).sum::<f64>() / n, pl.iter().map(|p| p.1).sum::<f64>() / n)
        };
        let (mx, my) = mean(&g.middle[0]);
        assert!((mx - 100.5).abs() < 1.0 && (my - 49.5).abs() < 1.0, "{mx},{my}");
        // outer is ~15 px outside the middle, inner ~2 px inside
        let rad = |pl: &Vec<(f64, f64)>| pl.iter().map(|p| ((p.0 - 100.5).powi(2) + (p.1 - 49.5).powi(2)).sqrt()).sum::<f64>() / pl.len() as f64;
        assert!((rad(&g.outer[0]) - rad(&g.middle[0]) - 15.0).abs() < 1.5);
        assert!((rad(&g.middle[0]) - rad(&g.inner[0]) - 2.0).abs() < 1.5);
        assert!((g.r_out_c - 15.0).abs() < 1e-9 && (g.r_in_c - 2.0).abs() < 1e-9);
        // uv of the bbox corners
        let (u, v) = g.uv((g.bbox_c.0, g.bbox_c.3));
        assert!(u.abs() < 1e-9 && (v - 1.0).abs() < 1e-9);
    }

    #[test]
    fn canon_roundtrip() {
        let cn = Canon { bx1: 10.0, by1: -4.0, h: 300.0, rs: 0.5 };
        let (x, y) = cn.pt(17.25, 123.5);
        let (a, b) = cn.frame(x, y);
        assert!((a - 17.25).abs() < 1e-9 && (b - 123.5).abs() < 1e-9);
        let m = CropMap { x0: 40.0, y0: 7.0, sx: 0.6, sy: 0.61 };
        let (p, q) = m.to_proc(100.0, 200.0);
        let (r, s) = m.to_frame(p, q);
        assert!((r - 100.0).abs() < 1e-9 && (s - 200.0).abs() < 1e-9);
    }
}
