//! Multi-object model: per-instance masks (from the SAM3 server's API v2
//! `instances: "all"`, or a single geometric-prompt mask), ordering, the
//! per-output-channel selection, the ID map and its tiny label font.
//!
//! Everything here is host-independent (top-down buffers at model-image
//! resolution) so it is unit-tested without an OFX host.

use crate::imgutil::Gray;

/// One detected object. `mask` is a soft probability map (u8, 0..255) at the
/// model-image size `w`x`h` (top-down). `bbox` = x0,y0,x1,y1 in those pixels.
#[derive(Clone, Debug)]
pub struct Instance {
    pub concept: String,
    /// Index of `concept` in the request's concept list (0-based).
    pub concept_index: usize,
    pub score: f32,
    pub bbox: [f32; 4],
    pub w: usize,
    pub h: usize,
    pub mask: Vec<u8>,
}

impl Instance {
    pub fn area(&self) -> usize {
        self.mask.iter().filter(|v| **v >= 128).count()
    }
    pub fn to_gray(&self) -> Gray {
        Gray { w: self.w, h: self.h, data: self.mask.iter().map(|v| *v as f32 / 255.0).collect() }
    }
    /// Centroid of the >= 0.5 region (model px), else the box centre.
    pub fn centroid(&self) -> (f32, f32) {
        let (mut sx, mut sy, mut n) = (0f64, 0f64, 0usize);
        for y in 0..self.h {
            for x in 0..self.w {
                if self.mask[y * self.w + x] >= 128 {
                    sx += x as f64;
                    sy += y as f64;
                    n += 1;
                }
            }
        }
        if n == 0 {
            ((self.bbox[0] + self.bbox[2]) * 0.5, (self.bbox[1] + self.bbox[3]) * 0.5)
        } else {
            ((sx / n as f64) as f32 + 0.5, (sy / n as f64) as f32 + 0.5)
        }
    }
}

/// Split a prompt into concepts: commas or '+' separate, whitespace trimmed,
/// case-insensitive de-duplication (first spelling kept), at most 16
/// (the server's cap).
pub fn parse_concepts(text: &str) -> Vec<String> {
    let mut out: Vec<String> = Vec::new();
    for raw in text.split([',', '+', ';']) {
        let c = raw.split_whitespace().collect::<Vec<_>>().join(" ");
        if c.is_empty() || out.iter().any(|o| o.eq_ignore_ascii_case(&c)) {
            continue;
        }
        out.push(c);
        if out.len() == 16 {
            break;
        }
    }
    out
}

/// Merge two concept lists (a first), same rules as [`parse_concepts`].
pub fn merge_concepts(a: &[String], b: &[String]) -> Vec<String> {
    parse_concepts(&a.iter().chain(b.iter()).cloned().collect::<Vec<_>>().join(","))
}

pub const ORDER_SERVER: i32 = 0; // concept order, then score (server order)
pub const ORDER_LEFT_RIGHT: i32 = 1;
pub const ORDER_TOP_BOTTOM: i32 = 2;
pub const ORDER_SIZE: i32 = 3;
pub const ORDER_SCORE: i32 = 4;

/// Re-order instances (stable). Object numbers shown to the user are the
/// 1-based positions AFTER ordering.
pub fn order(insts: &mut [Instance], mode: i32) {
    let key_f = |i: &Instance| -> f32 {
        match mode {
            ORDER_LEFT_RIGHT => (i.bbox[0] + i.bbox[2]) * 0.5,
            ORDER_TOP_BOTTOM => (i.bbox[1] + i.bbox[3]) * 0.5,
            ORDER_SIZE => -(i.area() as f32),
            ORDER_SCORE => -i.score,
            _ => 0.0,
        }
    };
    if mode == ORDER_SERVER {
        insts.sort_by(|a, b| a.concept_index.cmp(&b.concept_index).then(b.score.total_cmp(&a.score)));
    } else {
        insts.sort_by(|a, b| key_f(a).total_cmp(&key_f(b)));
    }
}

/// What one output channel carries.
pub const SEL_OFF: i32 = 0;
pub const SEL_INSTANCE: i32 = 1;
pub const SEL_CONCEPT: i32 = 2;
pub const SEL_ALL: i32 = 3;
/// Resolved per frame from the slot's viewer pick point (multi.rs) into an instance.
pub const SEL_PICK: i32 = 4;

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Sel {
    pub kind: i32,
    /// 1-based object number (SEL_INSTANCE) or concept number (SEL_CONCEPT).
    pub index: i32,
}

/// The instance under model-image point (x, y): the smallest one whose mask covers
/// it, else the one with the nearest centroid within 10% of the image diagonal
/// (the object moved off the pick point a little), else None.
pub fn instance_at(insts: &[Instance], x: f32, y: f32) -> Option<usize> {
    let mut best: Option<(usize, usize)> = None;
    for (k, i) in insts.iter().enumerate() {
        let (px, py) = (x.floor() as i64, y.floor() as i64);
        if px < 0 || py < 0 || px >= i.w as i64 || py >= i.h as i64 {
            continue;
        }
        if i.mask[py as usize * i.w + px as usize] >= 128 {
            let a = i.area();
            if best.map_or(true, |(_, ba)| a < ba) {
                best = Some((k, a));
            }
        }
    }
    if let Some((k, _)) = best {
        return Some(k);
    }
    let mut near: Option<(usize, f32)> = None;
    for (k, i) in insts.iter().enumerate() {
        let (cx, cy) = i.centroid();
        let d = ((cx - x).powi(2) + (cy - y).powi(2)).sqrt();
        let lim = 0.1 * ((i.w * i.w + i.h * i.h) as f32).sqrt();
        if d <= lim && near.map_or(true, |(_, bd)| d < bd) {
            near = Some((k, d));
        }
    }
    near.map(|(k, _)| k)
}

/// Max (union) of the instances `sel` picks, at model resolution. None =
/// nothing selected / no match (channel is black).
pub fn select(insts: &[Instance], sel: Sel) -> Option<Gray> {
    let pick: Vec<&Instance> = match sel.kind {
        SEL_INSTANCE => insts.get((sel.index - 1).max(0) as usize).filter(|_| sel.index >= 1).into_iter().collect(),
        SEL_CONCEPT => insts.iter().filter(|i| i.concept_index as i32 == sel.index - 1).collect(),
        SEL_ALL => insts.iter().collect(),
        _ => Vec::new(),
    };
    union_of(&pick)
}

pub fn union_of(pick: &[&Instance]) -> Option<Gray> {
    let first = pick.first()?;
    let (w, h) = (first.w, first.h);
    let mut acc = vec![0u8; w * h];
    for i in pick {
        if i.w != w || i.h != h {
            continue;
        }
        for (a, m) in acc.iter_mut().zip(&i.mask) {
            *a = (*a).max(*m);
        }
    }
    Some(Gray { w, h, data: acc.iter().map(|v| *v as f32 / 255.0).collect() })
}

/// Union of all instances, or only the highest-scoring one (`best_only`).
pub fn union_all(insts: &[Instance], best_only: bool) -> Option<Gray> {
    if best_only {
        let best = insts.iter().max_by(|a, b| a.score.total_cmp(&b.score))?;
        return union_of(&[best]);
    }
    union_of(&insts.iter().collect::<Vec<_>>())
}

/// "1 person .98 | 2 person .71 | 3 sofa .88" (numbers = current order).
pub fn list_text(insts: &[Instance]) -> String {
    if insts.is_empty() {
        return "no objects".into();
    }
    insts
        .iter()
        .enumerate()
        .map(|(k, i)| format!("{} {} {:.2}", k + 1, i.concept, i.score))
        .collect::<Vec<_>>()
        .join(" | ")
}

/// Multi-line variant for the Object List param (one object per line, with
/// concept number and box).
pub fn list_lines(insts: &[Instance], concepts: &[String]) -> String {
    let mut s = String::new();
    if !concepts.is_empty() {
        s += "concepts: ";
        s += &concepts.iter().enumerate().map(|(k, c)| format!("{}={}", k + 1, c)).collect::<Vec<_>>().join(", ");
        s += "\n";
    }
    for (k, i) in insts.iter().enumerate() {
        s += &format!(
            "#{} {} (concept {}) score {:.2} box {:.0},{:.0}-{:.0},{:.0}\n",
            k + 1,
            i.concept,
            i.concept_index + 1,
            i.score,
            i.bbox[0],
            i.bbox[1],
            i.bbox[2],
            i.bbox[3]
        );
    }
    if insts.is_empty() {
        s += "no objects\n";
    }
    s
}

/// Distinct, saturated colour for object number `k` (1-based); golden-ratio
/// hue steps so neighbours differ strongly.
pub fn palette(k: usize) -> [f32; 3] {
    let h = (k as f32 * 0.618_034).fract();
    let (s, v) = (0.85f32, 1.0f32);
    let i = (h * 6.0).floor();
    let f = h * 6.0 - i;
    let (p, q, t) = (v * (1.0 - s), v * (1.0 - f * s), v * (1.0 - (1.0 - f) * s));
    match i as i32 % 6 {
        0 => [v, t, p],
        1 => [q, v, p],
        2 => [p, v, t],
        3 => [p, q, v],
        4 => [t, p, v],
        _ => [v, p, q],
    }
}

/// Label map at model resolution: 0 = background, k = object number (1-based,
/// current order). Pixels where several objects are >= 0.5 go to the SMALLER
/// object (so "hair" stays visible inside "person").
pub fn id_labels(insts: &[Instance]) -> Option<(usize, usize, Vec<u16>)> {
    let first = insts.first()?;
    let (w, h) = (first.w, first.h);
    let mut lab = vec![0u16; w * h];
    let mut by_area: Vec<(usize, usize)> = insts.iter().enumerate().map(|(k, i)| (i.area(), k)).collect();
    by_area.sort_by(|a, b| b.0.cmp(&a.0)); // biggest first, smaller overwrite
    for (_, k) in by_area {
        let i = &insts[k];
        if i.w != w || i.h != h {
            continue;
        }
        for (l, m) in lab.iter_mut().zip(&i.mask) {
            if *m >= 128 {
                *l = (k + 1) as u16;
            }
        }
    }
    Some((w, h, lab))
}

// ---------------------------------------------------------------------------
// 3x5 digit font for ID map labels.
// ---------------------------------------------------------------------------
const DIGITS: [[u8; 5]; 10] = [
    [0b111, 0b101, 0b101, 0b101, 0b111],
    [0b010, 0b110, 0b010, 0b010, 0b111],
    [0b111, 0b001, 0b111, 0b100, 0b111],
    [0b111, 0b001, 0b111, 0b001, 0b111],
    [0b101, 0b101, 0b111, 0b001, 0b001],
    [0b111, 0b100, 0b111, 0b001, 0b111],
    [0b111, 0b100, 0b111, 0b101, 0b111],
    [0b111, 0b001, 0b010, 0b010, 0b010],
    [0b111, 0b101, 0b111, 0b101, 0b111],
    [0b111, 0b101, 0b111, 0b001, 0b111],
];

/// 3x5 letters and signs for viewer notes (upper case).
fn glyph(ch: u8) -> Option<[u8; 5]> {
    Some(match ch.to_ascii_uppercase() {
        b'0'..=b'9' => DIGITS[(ch - b'0') as usize],
        b'A' => [0b010, 0b101, 0b111, 0b101, 0b101],
        b'B' => [0b110, 0b101, 0b110, 0b101, 0b110],
        b'C' => [0b011, 0b100, 0b100, 0b100, 0b011],
        b'D' => [0b110, 0b101, 0b101, 0b101, 0b110],
        b'E' => [0b111, 0b100, 0b110, 0b100, 0b111],
        b'F' => [0b111, 0b100, 0b110, 0b100, 0b100],
        b'G' => [0b011, 0b100, 0b101, 0b101, 0b011],
        b'H' => [0b101, 0b101, 0b111, 0b101, 0b101],
        b'I' => [0b111, 0b010, 0b010, 0b010, 0b111],
        b'J' => [0b001, 0b001, 0b001, 0b101, 0b010],
        b'K' => [0b101, 0b101, 0b110, 0b101, 0b101],
        b'L' => [0b100, 0b100, 0b100, 0b100, 0b111],
        b'M' => [0b101, 0b111, 0b111, 0b101, 0b101],
        b'N' => [0b110, 0b101, 0b101, 0b101, 0b101],
        b'O' => [0b010, 0b101, 0b101, 0b101, 0b010],
        b'P' => [0b110, 0b101, 0b110, 0b100, 0b100],
        b'Q' => [0b010, 0b101, 0b101, 0b110, 0b011],
        b'R' => [0b110, 0b101, 0b110, 0b101, 0b101],
        b'S' => [0b011, 0b100, 0b010, 0b001, 0b110],
        b'T' => [0b111, 0b010, 0b010, 0b010, 0b010],
        b'U' => [0b101, 0b101, 0b101, 0b101, 0b111],
        b'V' => [0b101, 0b101, 0b101, 0b101, 0b010],
        b'W' => [0b101, 0b101, 0b111, 0b111, 0b101],
        b'X' => [0b101, 0b101, 0b010, 0b101, 0b101],
        b'Y' => [0b101, 0b101, 0b010, 0b010, 0b010],
        b'Z' => [0b111, 0b001, 0b010, 0b100, 0b111],
        b'/' => [0b001, 0b001, 0b010, 0b100, 0b100],
        b'-' => [0b000, 0b000, 0b111, 0b000, 0b000],
        b'.' => [0b000, 0b000, 0b000, 0b000, 0b010],
        b':' => [0b000, 0b010, 0b000, 0b010, 0b000],
        b' ' => [0; 5],
        _ => return None,
    })
}

/// Pixels of number `n` drawn with cell size `s`, centred on (cx, cy)
/// (top-down coords). Returns (x, y, is_ink) where !is_ink = 1-cell outline.
pub fn glyph_pixels(n: usize, cx: i32, cy: i32, s: i32) -> Vec<(i32, i32, bool)> {
    text_pixels(&n.to_string(), cx, cy, s)
}

/// Pixels of `txt` (see `glyph`; other characters are skipped), like `glyph_pixels`.
pub fn text_pixels(txt: &str, cx: i32, cy: i32, s: i32) -> Vec<(i32, i32, bool)> {
    let rows: Vec<[u8; 5]> = txt.bytes().filter_map(glyph).collect();
    let nd = rows.len() as i32;
    let (gw, gh) = (nd * 4 - 1, 5); // cells, 1-cell gap between characters
    let (x0, y0) = (cx - gw * s / 2, cy - gh * s / 2);
    let mut ink = std::collections::HashSet::new();
    for (di, d) in rows.iter().enumerate() {
        for (r, row) in d.iter().enumerate() {
            for c in 0..3 {
                if row & (0b100 >> c) != 0 {
                    ink.insert((di as i32 * 4 + c, r as i32));
                }
            }
        }
    }
    let mut out = Vec::new();
    for cy_ in -1..=gh {
        for cx_ in -1..=gw {
            let is_ink = ink.contains(&(cx_, cy_));
            let near = (-1..=1).any(|dy| (-1..=1).any(|dx| ink.contains(&(cx_ + dx, cy_ + dy))));
            if !is_ink && !near {
                continue;
            }
            for py in 0..s {
                for px in 0..s {
                    out.push((x0 + cx_ * s + px, y0 + cy_ * s + py, is_ink));
                }
            }
        }
    }
    out
}

#[cfg(test)]
mod tests {
    #[test]
    fn instance_at_prefers_covering_then_smallest_then_nearest() {
        let mk = |x0: usize, y0: usize, x1: usize, y1: usize| {
            let (w, h) = (20, 10);
            let mut mask = vec![0u8; w * h];
            for y in y0..y1 {
                for x in x0..x1 {
                    mask[y * w + x] = 255;
                }
            }
            Instance { concept: "c".into(), concept_index: 0, score: 1.0, bbox: [x0 as f32, y0 as f32, x1 as f32, y1 as f32], w, h, mask }
        };
        let v = vec![mk(0, 0, 10, 10), mk(2, 2, 5, 5), mk(15, 0, 17, 2)];
        assert_eq!(super::instance_at(&v, 3.0, 3.0), Some(1)); // inside both: the smaller
        assert_eq!(super::instance_at(&v, 8.0, 8.0), Some(0));
        assert_eq!(super::instance_at(&v, 17.5, 1.0), Some(2)); // just off: nearest centroid
        assert_eq!(super::instance_at(&v, 15.0, 9.5), None); // far from everything
        assert_eq!(super::instance_at(&[], 1.0, 1.0), None);
    }

    use super::*;

    fn inst(ci: usize, c: &str, score: f32, w: usize, h: usize, rect: (usize, usize, usize, usize)) -> Instance {
        let mut mask = vec![0u8; w * h];
        for y in rect.1..rect.3 {
            for x in rect.0..rect.2 {
                mask[y * w + x] = 255;
            }
        }
        Instance {
            concept: c.into(),
            concept_index: ci,
            score,
            bbox: [rect.0 as f32, rect.1 as f32, rect.2 as f32, rect.3 as f32],
            w,
            h,
            mask,
        }
    }

    #[test]
    fn concepts_split_trim_dedupe() {
        assert_eq!(parse_concepts("person, sofa ,dog"), vec!["person", "sofa", "dog"]);
        assert_eq!(parse_concepts(" person +  red   car ; Person,,"), vec!["person", "red car"]);
        assert!(parse_concepts(" , ").is_empty());
        assert_eq!(parse_concepts("person"), vec!["person"]);
        let many = (0..20).map(|i| format!("c{i}")).collect::<Vec<_>>().join(",");
        assert_eq!(parse_concepts(&many).len(), 16);
        assert_eq!(merge_concepts(&["a".into()], &["A".into(), "b".into()]), vec!["a", "b"]);
    }

    #[test]
    fn select_instance_concept_all() {
        let v = vec![
            inst(0, "person", 0.9, 8, 4, (0, 0, 2, 2)),
            inst(0, "person", 0.7, 8, 4, (4, 0, 6, 2)),
            inst(1, "dog", 0.8, 8, 4, (6, 2, 8, 4)),
        ];
        let g = select(&v, Sel { kind: SEL_INSTANCE, index: 2 }).unwrap();
        assert_eq!(g.data[4], 1.0);
        assert_eq!(g.data[0], 0.0);
        let g = select(&v, Sel { kind: SEL_CONCEPT, index: 1 }).unwrap();
        assert_eq!((g.data[0], g.data[4], g.data[2 * 8 + 6]), (1.0, 1.0, 0.0));
        let g = select(&v, Sel { kind: SEL_ALL, index: 0 }).unwrap();
        assert_eq!(g.data[3 * 8 + 7], 1.0);
        assert!(select(&v, Sel { kind: SEL_INSTANCE, index: 4 }).is_none());
        assert!(select(&v, Sel { kind: SEL_INSTANCE, index: 0 }).is_none());
        assert!(select(&v, Sel { kind: SEL_OFF, index: 1 }).is_none());
        assert!(select(&v, Sel { kind: SEL_CONCEPT, index: 3 }).is_none());
        let b = union_all(&v, true).unwrap();
        assert_eq!((b.data[0], b.data[4]), (1.0, 0.0));
    }

    #[test]
    fn ordering_modes() {
        let mut v = vec![
            inst(1, "dog", 0.8, 10, 10, (6, 6, 10, 10)),
            inst(0, "person", 0.7, 10, 10, (0, 0, 2, 2)),
            inst(0, "person", 0.9, 10, 10, (3, 0, 9, 5)),
        ];
        order(&mut v, ORDER_SERVER);
        assert_eq!((v[0].score, v[1].score, v[2].concept.as_str()), (0.9, 0.7, "dog"));
        order(&mut v, ORDER_LEFT_RIGHT);
        assert_eq!(v[0].bbox[0], 0.0);
        order(&mut v, ORDER_SIZE);
        assert_eq!(v[0].bbox[0], 3.0);
        order(&mut v, ORDER_SCORE);
        assert_eq!(v[2].score, 0.7);
        order(&mut v, ORDER_TOP_BOTTOM);
        assert_eq!(v[2].concept, "dog");
    }

    #[test]
    fn id_labels_smaller_wins() {
        let v = vec![inst(0, "person", 0.9, 10, 10, (0, 0, 10, 10)), inst(1, "hair", 0.8, 10, 10, (2, 2, 4, 4))];
        let (w, _, l) = id_labels(&v).unwrap();
        assert_eq!(l[0], 1);
        assert_eq!(l[3 * w + 3], 2);
        assert_ne!(palette(1), palette(2));
        assert!(list_text(&v).starts_with("1 person 0.90 | 2 hair"));
        assert!(list_lines(&v, &["person".into(), "hair".into()]).contains("#2 hair (concept 2)"));
    }

    #[test]
    fn glyphs_have_ink_and_outline() {
        let p = glyph_pixels(12, 50, 50, 2);
        assert!(p.iter().any(|q| q.2) && p.iter().any(|q| !q.2));
        let i = inst(0, "x", 1.0, 10, 10, (2, 2, 6, 6));
        let (cx, cy) = i.centroid();
        assert!((cx - 4.0).abs() < 0.01 && (cy - 4.0).abs() < 0.01);
    }
}
