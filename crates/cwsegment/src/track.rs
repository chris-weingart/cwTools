//! Object tracking (SAM3's video memory): objects picked on an anchor frame are
//! followed frame by frame.
//!
//! - track_init.onnx: the anchor frame's encoder features + an object mask
//!   (1008² letterboxed, 0/1) → low-res mask logits [288²], object score,
//!   object pointer [256], memory [64·72·72] of that frame
//! - track_step.onnx: the next frame's features + the object's memory bank
//!   (memories of earlier frames + their time offsets, object pointers +
//!   offsets) → the same four outputs for that frame
//!
//! [`Bank`] is the bookkeeping of transformers' Sam3TrackerVideoModel in
//! streaming mode (flame_inference/sam_onnx/sam3_track.py `Bank`, which matches
//! it bit for bit): at most 4 anchor (conditioning) frames, the 6 frames before
//! (after, tracking backward), object pointers of the anchor frames and of the
//! 15 frames before, offsets / 15.
//!
//! Frames are tracked as Flame renders them: frame f needs f-1 (forward) or
//! f+1 (backward) done. Results stay cached per node, so scrubbing back is free.

use crate::imgutil::Gray;
use crate::objects::Instance;
use std::collections::{BTreeMap, HashMap};
use std::sync::{Arc, Mutex};

pub const S: usize = 1008;
pub const LR: usize = 288;
pub const G: usize = 72;
pub const MEM_LEN: usize = 64 * G * G;
pub const PTR_LEN: usize = 256;
const NUM_MASKMEM: i64 = 7;
const MAX_COND: usize = 4;
const MAX_PTRS: i64 = 16;

pub const DIR_OFF: i32 = 0;
pub const DIR_FWD: i32 = 1;
pub const DIR_BACK: i32 = 2;
pub const DIR_BOTH: i32 = 3;

/// One frame's memory of one object.
#[derive(Clone)]
pub struct Entry {
    /// None once the frame is too far back for the spatial memory (pointer kept).
    pub mem: Option<Arc<Vec<f32>>>,
    pub ptr: Vec<f32>,
}

/// track_step.onnx inputs for one object on one frame.
pub struct StepInputs {
    pub feats: Vec<f32>,
    pub tpos: Vec<i64>,
    pub ptrs: Vec<f32>,
    pub offs: Vec<f32>,
}
impl StepInputs {
    pub fn n_mem(&self) -> usize {
        self.tpos.len()
    }
    pub fn n_ptr(&self) -> usize {
        self.offs.len()
    }
}

/// One object's memory bank. `cond` keeps insertion order (it orders the memory).
#[derive(Clone, Default)]
pub struct Bank {
    pub cond: Vec<(i64, Entry)>,
    pub track: BTreeMap<i64, Entry>,
}

impl Bank {
    /// Memory-bank inputs for frame `f` (sam3_track.py `Bank.inputs`).
    pub fn inputs(&self, f: i64, reverse: bool) -> StepInputs {
        let sign = if reverse { -1 } else { 1 };
        // anchor frames: all when <= MAX_COND, else the closest before / after + nearest
        let (sel, unsel): (Vec<&(i64, Entry)>, Vec<&(i64, Entry)>) = if self.cond.len() <= MAX_COND {
            (self.cond.iter().collect(), Vec::new())
        } else {
            let mut picked: Vec<i64> = Vec::new();
            if let Some(b) = self.cond.iter().map(|c| c.0).filter(|&t| t < f).max() {
                picked.push(b);
            }
            if let Some(a) = self.cond.iter().map(|c| c.0).filter(|&t| t >= f).min() {
                picked.push(a);
            }
            let mut rest: Vec<i64> = self.cond.iter().map(|c| c.0).filter(|t| !picked.contains(t)).collect();
            rest.sort_by_key(|t| (t - f).abs()); // stable, as Python's sorted
            picked.extend(rest.into_iter().take(MAX_COND.saturating_sub(picked.len())));
            // selected in pick order (Python dict insertion order), unselected in cond order
            let sel = picked.iter().filter_map(|t| self.cond.iter().find(|c| c.0 == *t)).collect();
            let unsel = self.cond.iter().filter(|c| !picked.contains(&c.0)).collect();
            (sel, unsel)
        };
        let mut feats: Vec<f32> = Vec::new();
        let mut tpos: Vec<i64> = Vec::new();
        for (_, e) in &sel {
            if let Some(m) = &e.mem {
                feats.extend_from_slice(m);
                tpos.push(0);
            }
        }
        for k in (1..NUM_MASKMEM).rev() {
            let prev = f - k * sign;
            let e = self.track.get(&prev).or_else(|| unsel.iter().find(|c| c.0 == prev).map(|c| &c.1));
            if let Some(Entry { mem: Some(m), .. }) = e {
                feats.extend_from_slice(m);
                tpos.push(k);
            }
        }
        let mut ptrs: Vec<f32> = Vec::new();
        let mut offs: Vec<f32> = Vec::new();
        for (t, e) in &self.cond {
            if if reverse { *t >= f } else { *t <= f } {
                ptrs.extend_from_slice(&e.ptr);
                offs.push(((f - t) * sign) as f32);
            }
        }
        for d in 1..MAX_PTRS {
            let r = if reverse { f + d } else { f - d };
            if r < 0 {
                break;
            }
            if let Some(e) = self.track.get(&r) {
                ptrs.extend_from_slice(&e.ptr);
                offs.push(d as f32);
            }
        }
        StepInputs { feats, tpos, ptrs, offs }
    }

    /// Drop what frames `fwd` (forward frontier) and `back` (backward frontier)
    /// can no longer reach: spatial memories more than 6 frames, pointers more
    /// than 15 frames away from both.
    pub fn forget(&mut self, fwd: i64, back: i64) {
        let dist = |t: i64| (t - fwd).abs().min((t - back).abs());
        self.track.retain(|t, _| dist(*t) < MAX_PTRS);
        for (t, e) in self.track.iter_mut() {
            if dist(*t) >= NUM_MASKMEM {
                e.mem = None;
            }
        }
    }
}

/// Picked objects: "2, 5 7" -> [2, 5, 7] (1-based object numbers, first spelling kept).
pub fn parse_list(s: &str) -> Vec<usize> {
    let mut out: Vec<usize> = Vec::new();
    for w in s.split(|c: char| !c.is_ascii_digit()) {
        if let Ok(n) = w.parse::<usize>() {
            if n >= 1 && !out.contains(&n) {
                out.push(n);
            }
        }
    }
    out
}
pub fn format_list(l: &[usize]) -> String {
    l.iter().map(|n| n.to_string()).collect::<Vec<_>>().join(", ")
}

/// Instance mask (model image, top-down, soft u8) -> 1008² letterboxed 0/1 (the
/// encoder sees the image top-left in 1008², see local.rs `image_tensor`).
pub fn letterbox_mask(i: &Instance) -> Vec<f32> {
    let mut m = vec![0f32; S * S];
    for y in 0..i.h.min(S) {
        for x in 0..i.w.min(S) {
            if i.mask[y * i.w + x] >= 128 {
                m[y * S + x] = 1.0;
            }
        }
    }
    m
}

/// What the tracker gives for one frame.
pub enum Frame {
    /// (object number, soft mask at model-image size, object score) per tracked object, list order.
    Objects(Vec<(usize, Gray, f32)>),
    /// Not tracked yet: the frame to render next to get there.
    NotYet(i64),
}

/// How many frames one Track Range press fetches at most (the frames wait in memory,
/// ~1.7 MB each at 1008 px, until the background job has tracked them).
pub const MAX_BATCH: usize = 300;

struct Session {
    key: String,
    anchor: i64,
    numbers: Vec<usize>,
    banks: Vec<Bank>,
    fwd: i64,
    back: i64,
}

/// Tracked results of a node, readable while a background job holds the session:
/// frame -> per object (sigmoid of the 288² logits as u8, object score).
struct Store {
    key: String,
    anchor: i64,
    back: i64,
    fwd: i64,
    numbers: Vec<usize>,
    frames: HashMap<i64, FrameMasks>,
}
static STORE: Mutex<Option<HashMap<usize, Store>>> = Mutex::new(None);

fn store_get(node: usize, key: &str, f: i64) -> Option<FrameMasks> {
    let g = STORE.lock().ok()?;
    let s = g.as_ref()?.get(&node)?;
    if s.key != key {
        return None;
    }
    s.frames.get(&f).cloned()
}
fn store_put(node: usize, s: &Session, f: i64, m: &FrameMasks) {
    if let Ok(mut g) = STORE.lock() {
        let map = g.get_or_insert_with(HashMap::new);
        let e = map.entry(node).or_insert_with(|| Store {
            key: s.key.clone(), anchor: s.anchor, back: s.back, fwd: s.fwd, numbers: s.numbers.clone(), frames: HashMap::new(),
        });
        if e.key != s.key {
            *e = Store { key: s.key.clone(), anchor: s.anchor, back: s.back, fwd: s.fwd, numbers: s.numbers.clone(), frames: HashMap::new() };
        }
        e.back = s.back;
        e.fwd = s.fwd;
        e.frames.insert(f, m.clone());
    }
}

/// A Track Range job running in the background.
#[derive(Clone, Debug, Default)]
pub struct Job {
    pub key: String,
    pub done: usize,
    pub total: usize,
    pub running: bool,
    pub cancel: bool,
    pub error: Option<String>,
}
static JOBS: Mutex<Option<HashMap<usize, Job>>> = Mutex::new(None);

pub fn job(node: usize) -> Option<Job> {
    JOBS.lock().ok()?.as_ref()?.get(&node).cloned()
}
pub fn job_update(node: usize, f: impl FnOnce(&mut Job)) {
    if let Ok(mut g) = JOBS.lock() {
        f(g.get_or_insert_with(HashMap::new).entry(node).or_default());
    }
}
/// Ask the node's running job to stop after the current frame.
pub fn cancel_job(node: usize) {
    job_update(node, |j| j.cancel = true);
}

static SESSIONS: Mutex<Option<HashMap<usize, Session>>> = Mutex::new(None);

static RUN: Mutex<Vec<usize>> = Mutex::new(Vec::new());
/// Track Range pressed: the node's next render tracks the whole range.
pub fn request_run(node: usize) {
    if let Ok(mut g) = RUN.lock() {
        if !g.contains(&node) {
            g.push(node);
        }
    }
}
/// Takes the node's Track Range request (true once per press).
pub fn take_run(node: usize) -> bool {
    RUN.lock().map(|mut g| {
        let had = g.contains(&node);
        g.retain(|n| *n != node);
        had
    }).unwrap_or(false)
}
/// Tracked range (back, fwd) of the node with this key, if tracking started.
pub fn range(node: usize, key: &str) -> Option<(i64, i64)> {
    let g = STORE.lock().ok()?;
    let s = g.as_ref()?.get(&node)?;
    (s.key == key && !s.frames.is_empty()).then_some((s.back, s.fwd))
}

/// Forget the node's tracking (a running job stops after its current frame).
pub fn reset(node: usize) {
    cancel_job(node);
    if let Ok(mut g) = SESSIONS.lock() {
        if let Some(m) = g.as_mut() {
            m.remove(&node);
        }
    }
    if let Ok(mut g) = STORE.lock() {
        if let Some(m) = g.as_mut() {
            m.remove(&node);
        }
    }
}
pub fn reset_all() {
    let nodes: Vec<usize> = JOBS.lock().ok().and_then(|g| g.as_ref().map(|m| m.keys().copied().collect())).unwrap_or_default();
    for n in nodes {
        cancel_job(n);
    }
    if let Ok(mut g) = SESSIONS.lock() {
        *g = None;
    }
    if let Ok(mut g) = STORE.lock() {
        *g = None;
    }
}

/// What the node has tracked: (anchor, first, last, object numbers).
/// Milliseconds Flame took per frame in the last Track Range fetch (0 = none yet).
static FETCH_MS: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);

pub fn fetch_ms() -> u64 {
    FETCH_MS.load(std::sync::atomic::Ordering::Relaxed)
}

pub fn set_fetch_ms(ms: u64) {
    FETCH_MS.store(ms, std::sync::atomic::Ordering::Relaxed);
}

pub fn status(node: usize) -> Option<(i64, i64, i64, Vec<usize>)> {
    let g = STORE.lock().ok()?;
    let s = g.as_ref()?.get(&node)?;
    (!s.frames.is_empty()).then(|| (s.anchor, s.back, s.fwd, s.numbers.clone()))
}

/// The cached 288² masks (u8 probabilities) of `frame` (tests).
#[cfg(test)]
pub fn raw(node: usize, frame: i64) -> Option<Vec<Vec<u8>>> {
    let g = STORE.lock().ok()?;
    Some(g.as_ref()?.get(&node)?.frames.get(&frame)?.iter().map(|x| x.0.clone()).collect())
}

fn to_u8(low: &[f32]) -> Vec<u8> {
    low.iter().map(|l| ((1.0 / (1.0 + (-l).exp())) * 255.0 + 0.5) as u8).collect()
}
/// u8 probabilities back to logits for the shared upscaler (local::logits_to_mask).
fn to_logits(p: &[u8]) -> Vec<f32> {
    p.iter()
        .map(|&v| {
            let q = (v as f32 / 255.0).clamp(0.5 / 255.0, 1.0 - 0.5 / 255.0);
            (q / (1.0 - q)).ln()
        })
        .collect()
}

/// The backend: model folder features + the two graphs (local.rs implements it;
/// tests use a fake).
pub trait Backend {
    type Feat;
    fn features(&mut self) -> Result<Self::Feat, String>;
    fn init(&mut self, f: &Self::Feat, mask: Vec<f32>) -> Result<Out, String>;
    fn step(&mut self, f: &Self::Feat, inp: &StepInputs) -> Result<Out, String>;
}
pub struct Out {
    pub low: Vec<f32>,
    pub obj: f32,
    pub ptr: Vec<f32>,
    pub mem: Vec<f32>,
}

pub struct Request<'a> {
    pub node: usize,
    pub frame: i64,
    pub anchor: i64,
    pub numbers: &'a [usize],
    pub dir: i32,
    /// Anything that changes the anchor objects (prompt, numbering, model, size).
    pub key: String,
    /// Detections on this frame when it is the anchor (numbered as shown).
    pub anchor_insts: Option<&'a [Instance]>,
    pub mw: usize,
    pub mh: usize,
}

/// Masks for `r.frame`, tracking one more frame when it is next to the tracked range.
pub fn frame<B: Backend>(r: &Request, be: &mut B) -> Result<Frame, String> {
    let taken = SESSIONS.lock().map_err(|_| "track lock poisoned".to_string())?.get_or_insert_with(HashMap::new).remove(&r.node);
    let mut s = match taken {
        Some(s) if s.key == r.key => s,
        _ => Session {
            key: r.key.clone(),
            anchor: r.anchor,
            numbers: r.numbers.to_vec(),
            banks: Vec::new(),
            fwd: r.anchor,
            back: r.anchor,
        },
    };
    let res = advance(&mut s, r, be);
    if let Ok(mut g) = SESSIONS.lock() {
        g.get_or_insert_with(HashMap::new).insert(r.node, s);
    }
    Ok(match res? {
        Step::Masks(objs) => Frame::Objects(
            objs.into_iter()
                .zip(r.numbers)
                .map(|((p, score), n)| (*n, crate::local::logits_to_mask(&to_logits(&p), r.mw, r.mh), score))
                .collect(),
        ),
        Step::NotYet(next) => Frame::NotYet(next),
    })
}

/// The stored masks of `r.frame` without tracking anything (renders while a job runs).
pub fn cached(r: &Request) -> Option<Frame> {
    let objs = store_get(r.node, &r.key, r.frame)?;
    Some(Frame::Objects(
        objs.into_iter()
            .zip(r.numbers)
            .map(|((p, score), n)| (*n, crate::local::logits_to_mask(&to_logits(&p), r.mw, r.mh), score))
            .collect(),
    ))
}

enum Step {
    Masks(FrameMasks),
    /// not reachable from the tracked range in an allowed direction: render this one next
    NotYet(i64),
}

type FrameMasks = Vec<(Vec<u8>, f32)>;

fn advance<B: Backend>(s: &mut Session, r: &Request, be: &mut B) -> Result<Step, String> {
    if let Some(m) = store_get(r.node, &r.key, r.frame) {
        return Ok(Step::Masks(m));
    }
    if s.banks.is_empty() {
        if r.frame != r.anchor {
            return Ok(Step::NotYet(r.frame)); // the message says: press Track Range
        }
        let insts = r.anchor_insts.ok_or_else(|| "internal: no detections on the anchor frame".to_string())?;
        let missing: Vec<usize> = r.numbers.iter().copied().filter(|n| *n == 0 || *n > insts.len()).collect();
        if !missing.is_empty() {
            return Err(format!(
                "object {} not on the anchor frame ({} objects found there)",
                format_list(&missing), insts.len()
            ));
        }
        let feat = be.features()?;
        let mut out: FrameMasks = Vec::new();
        let mut banks = Vec::new();
        for n in r.numbers {
            let o = be.init(&feat, letterbox_mask(&insts[n - 1]))?;
            check(&o)?;
            let mut b = Bank::default();
            b.cond.push((r.anchor, Entry { mem: Some(Arc::new(o.mem)), ptr: o.ptr }));
            banks.push(b);
            out.push((to_u8(&o.low), o.obj));
        }
        s.banks = banks;
        store_put(r.node, s, r.anchor, &out);
        return Ok(Step::Masks(out));
    }
    let fwd_ok = r.dir == DIR_FWD || r.dir == DIR_BOTH;
    let back_ok = r.dir == DIR_BACK || r.dir == DIR_BOTH;
    let reverse = if r.frame == s.fwd + 1 && fwd_ok {
        false
    } else if r.frame == s.back - 1 && back_ok {
        true
    } else {
        let next = if r.frame > s.anchor && fwd_ok { s.fwd + 1 } else if r.frame < s.anchor && back_ok { s.back - 1 } else { s.anchor };
        return Ok(Step::NotYet(next));
    };
    let feat = be.features()?;
    let mut out: FrameMasks = Vec::new();
    for b in s.banks.iter_mut() {
        let inp = b.inputs(r.frame, reverse);
        let o = be.step(&feat, &inp)?;
        check(&o)?;
        b.track.insert(r.frame, Entry { mem: Some(Arc::new(o.mem)), ptr: o.ptr });
        out.push((to_u8(&o.low), o.obj));
    }
    if reverse {
        s.back = r.frame;
    } else {
        s.fwd = r.frame;
    }
    let (fwd, back) = (s.fwd, s.back);
    for b in s.banks.iter_mut() {
        b.forget(fwd, back);
    }
    store_put(r.node, s, r.frame, &out);
    Ok(Step::Masks(out))
}

/// Model outputs must be finite and the right size (a non-finite memory would
/// poison every later frame).
fn check(o: &Out) -> Result<(), String> {
    if o.low.len() != LR * LR || o.ptr.len() != PTR_LEN || o.mem.len() != MEM_LEN {
        return Err(format!("unexpected tracker output sizes {} / {} / {}", o.low.len(), o.ptr.len(), o.mem.len()));
    }
    if !o.obj.is_finite() || o.ptr.iter().chain(&o.mem).any(|x| !x.is_finite()) {
        return Err("the tracker produced non-finite values (GPU / memory problem?)".into());
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn entry(v: f32) -> Entry {
        Entry { mem: Some(Arc::new(vec![v; 2])), ptr: vec![v; 1] }
    }

    /// Same choices as sam3_track.py `Bank.inputs` (memory = anchor + 6 back,
    /// pointers = anchor + 15 back, offsets).
    #[test]
    fn bank_matches_reference_forward() {
        let mut b = Bank::default();
        b.cond.push((10, entry(100.0)));
        for f in 11..=30 {
            b.track.insert(f, entry(f as f32));
        }
        let i = b.inputs(31, false);
        assert_eq!(i.tpos, vec![0, 6, 5, 4, 3, 2, 1]);
        assert_eq!(i.feats.iter().step_by(2).copied().collect::<Vec<_>>(), vec![100.0, 25.0, 26.0, 27.0, 28.0, 29.0, 30.0]);
        // anchor pointer (offset 21) + 15 frames back
        assert_eq!(i.offs[0], 21.0);
        assert_eq!(i.offs[1..].to_vec(), (1..16).map(|d| d as f32).collect::<Vec<_>>());
        assert_eq!(i.ptrs[1], 30.0);
    }

    #[test]
    fn bank_backward_and_early_frames() {
        let mut b = Bank::default();
        b.cond.push((10, entry(100.0)));
        b.track.insert(9, entry(9.0));
        b.track.insert(8, entry(8.0));
        let i = b.inputs(7, true);
        assert_eq!(i.tpos, vec![0, 2, 1]); // anchor 10 (offset 0), 9 = 2 after, 8 = 1 after
        assert_eq!(i.offs, vec![3.0, 1.0, 2.0]);
        // forward on frame 11: frames 9, 8 are 2/3 back
        let i = b.inputs(11, false);
        assert_eq!(i.tpos, vec![0, 3, 2]);
        assert_eq!(i.offs, vec![1.0, 2.0, 3.0]);
    }

    #[test]
    fn bank_more_than_four_anchors() {
        let mut b = Bank::default();
        for t in [0, 20, 40, 60, 80, 100] {
            b.cond.push((t, entry(t as f32)));
        }
        let i = b.inputs(50, false);
        // closest before (40), closest after (60), then the nearest two (20, 80: tie in cond order)
        assert_eq!(i.feats.iter().step_by(2).copied().collect::<Vec<_>>(), vec![40.0, 60.0, 20.0, 80.0]);
        assert_eq!(i.tpos, vec![0, 0, 0, 0]);
        // pointers: every anchor up to frame 50 in cond order
        assert_eq!(i.offs, vec![50.0, 30.0, 10.0]);
    }

    #[test]
    fn forget_keeps_what_both_frontiers_need() {
        let mut b = Bank::default();
        for f in 0..40 {
            b.track.insert(f, entry(f as f32));
        }
        b.forget(39, 30);
        assert!(!b.track.contains_key(&14)); // 16 from back frontier 30
        assert!(b.track.contains_key(&15));
        assert!(b.track[&23].mem.is_none()); // 7 from frontier 30
        assert!(b.track[&24].mem.is_some());
    }

    #[test]
    fn list_parsing() {
        assert_eq!(parse_list("2, 5 7,2"), vec![2, 5, 7]);
        assert_eq!(parse_list(" #3;0;x"), vec![3]);
        assert_eq!(format_list(&[2, 5]), "2, 5");
    }

    struct Fake {
        calls: Vec<String>,
    }
    impl Backend for Fake {
        type Feat = ();
        fn features(&mut self) -> Result<(), String> {
            Ok(())
        }
        fn init(&mut self, _f: &(), mask: Vec<f32>) -> Result<Out, String> {
            self.calls.push(format!("init {}", mask.iter().filter(|v| **v > 0.5).count()));
            Ok(Out { low: vec![5.0; LR * LR], obj: 10.0, ptr: vec![0.0; PTR_LEN], mem: vec![0.0; MEM_LEN] })
        }
        fn step(&mut self, _f: &(), inp: &StepInputs) -> Result<Out, String> {
            self.calls.push(format!("step mem={} ptr={}", inp.n_mem(), inp.n_ptr()));
            Ok(Out { low: vec![-5.0; LR * LR], obj: 3.0, ptr: vec![0.0; PTR_LEN], mem: vec![0.0; MEM_LEN] })
        }
    }

    #[test]
    fn session_tracks_in_render_order() {
        let mk = |w: usize, h: usize, x0: usize| Instance {
            concept: "a".into(), concept_index: 0, score: 1.0, bbox: [0.0; 4], w, h,
            mask: (0..w * h).map(|i| if i % w >= x0 { 255 } else { 0 }).collect(),
        };
        let insts = vec![mk(100, 50, 50), mk(100, 50, 90)];
        let mut be = Fake { calls: Vec::new() };
        let req = |frame: i64, a: Option<&'static [Instance]>| Request {
            node: 7, frame, anchor: 5, numbers: &[2, 1], dir: DIR_FWD, key: "k".into(), anchor_insts: a, mw: 100, mh: 50,
        };
        let leak: &'static [Instance] = Box::leak(insts.into_boxed_slice());
        reset(7);
        assert!(matches!(frame(&req(6, None), &mut be), Ok(Frame::NotYet(6)))); // anchor first
        match frame(&req(5, Some(leak)), &mut be).unwrap() {
            Frame::Objects(o) => {
                assert_eq!(o.iter().map(|x| x.0).collect::<Vec<_>>(), vec![2, 1]);
                assert!(o[0].1.data[0] > 0.99);
            }
            _ => panic!("anchor"),
        }
        assert_eq!(be.calls, vec!["init 500", "init 2500"]);
        match frame(&req(7, None), &mut be).unwrap() {
            Frame::NotYet(n) => assert_eq!(n, 6),
            _ => panic!("7 before 6"),
        }
        assert!(matches!(frame(&req(6, None), &mut be).unwrap(), Frame::Objects(_)));
        assert!(matches!(frame(&req(7, None), &mut be).unwrap(), Frame::Objects(_)));
        assert!(matches!(frame(&req(4, None), &mut be).unwrap(), Frame::NotYet(5))); // forward only
        let n = be.calls.len();
        assert!(matches!(frame(&req(6, None), &mut be).unwrap(), Frame::Objects(_))); // cached
        assert_eq!(be.calls.len(), n);
        assert_eq!(be.calls[2..], ["step mem=1 ptr=1", "step mem=1 ptr=1", "step mem=2 ptr=2", "step mem=2 ptr=2"]);
        assert_eq!(status(7).unwrap(), (5, 5, 7, vec![2, 1]));
        reset(7);
    }
}
