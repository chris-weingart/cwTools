//! SAM3 for cwSegment: encoder/decoder (+ text detector, tracker) ONNX, run
//! in this process (Local, onnxruntime) or on cwtools-infer (Remote, the same
//! files, crate::sam_remote). Everything around the graphs is shared, so the
//! two modes only differ in where a graph runs ([`Engine`]).
//!
//! - encoder.onnx: image [1,3,1008,1008] RGB 0..1 (display-referred) →
//!   image_embed [1,256,72,72], high_res_0 [1,32,288,288], high_res_1 [1,64,144,144]
//! - decoder.onnx: + point_coords [1,N,2], point_labels [1,N] (1 fg, 0 bg,
//!   2/3 box corners), mask_input [1,1,288,288], has_mask_input [1] →
//!   low_res_masks [1,4,288,288] logits, iou_predictions [1,4], object_score [1,1]
//!
//! Text prompts (concepts) need the SAM3 text models in the same folder:
//! - encoder.onnx v2 (also outputs det_fpn_0..2, the detector neck levels; the
//!   click-only v1 encoder has no text support), optional encoder_fp16.onnx
//!   (used on the GPU when present: half the VRAM, same detections)
//! - text_encoder.onnx: input_ids/attention_mask [1,32] -> text_features [1,32,256]
//!   (runs on the CPU; once per prompt text, cached)
//! - detector.onnx: det_fpn_0..2 + text_features + attention_mask ->
//!   pred_logits [1,200], presence_logit [1,1], pred_boxes [1,200,4] (xyxy 0..1
//!   of the 1008² frame), pred_masks [1,200,288,288] logits
//! - vocab.json + merges.txt (CLIP tokenizer, crate::clip_tok)
//! score(query) = sigmoid(logit) * sigmoid(presence), kept above Detect Thresh.
//!
//! Tracking (crate::track) needs track_init.onnx + track_step.onnx in the same
//! folder (SAM3 tracker memory path, flame_inference/sam_onnx/export_sam3_track.py);
//! they take this encoder's outputs.
//!
//! onnxruntime is loaded dynamically (ort `load-dynamic`) from the path in the
//! "ONNX Runtime" param — by default (empty) cwtools.conf `ort_lib` or Flame's
//! own bundled 1.22 library (auto-detected), so the plugin ships no ORT of its own. Embeddings are cached per frame_key so
//! prompt edits on the same frame only re-run the (small) decoder.

use crate::imgutil::{Gray, Rgb8};
use crate::objects::Instance;
use crate::params::Values;
use crate::prompt::Prompts;
use ort::session::RunOptions;
use ort::value::Tensor;
use std::sync::{Arc, Mutex};

pub const S: usize = 1008;
const LR: usize = 288;

pub(crate) struct Embeds {
    pub(crate) embed: Vec<f32>,
    pub(crate) hr0: Vec<f32>,
    pub(crate) hr1: Vec<f32>,
    /// Detector neck levels 0..2 (encoder v2 only).
    det: Option<[Vec<f32>; 3]>,
    /// Remote: the outputs stay on the server (empty vectors here).
    remote: Option<RemoteFeat>,
}

pub(crate) struct RemoteFeat {
    server: String,
    /// handle prefix of the kept encoder outputs
    prefix: String,
    /// the detector neck levels were kept too
    det: bool,
}

impl Drop for RemoteFeat {
    /// Embedding no longer cached (evicted / released): drop it on the server too.
    fn drop(&mut self) {
        let (server, prefix) = (std::mem::take(&mut self.server), std::mem::take(&mut self.prefix));
        std::thread::spawn(move || {
            if let Err(e) = cwtools_remote::forget(&server, Some(&prefix), &[], 5.0) {
                crate::log::log(&format!("remote: could not release {prefix} on the server: {e}"));
            }
        });
    }
}

/// Where the SAM3 graphs run.
enum Engine<'a> {
    Local(&'a Model),
    Remote(&'a RemoteSam),
}
impl Engine<'_> {
    fn dir(&self) -> &std::path::Path {
        match self {
            Engine::Local(m) => &m.dir,
            Engine::Remote(r) => &r.dir,
        }
    }
    fn tag(&self) -> String {
        match self {
            Engine::Local(_) => "local".into(),
            Engine::Remote(r) => format!("remote {}", r.server),
        }
    }
}

/// The encoder/decoder pair currently in use. Both sessions live in the
/// cwtools-core process-wide cache (shared by every cwSegment node, guarded
/// by the never-dropped CUDA keeper, released by "Release GPU Memory").
struct Model {
    encoder: Arc<SharedSession>,
    decoder: Arc<SharedSession>,
    gpu: bool,
    dir: std::path::PathBuf,
}

/// Pair key (model folder + GPU) of the sessions loaded last: switching folder
/// or GPU drops the old pair first (one SAM pair in VRAM, as before).
static PAIR_KEY: Mutex<String> = Mutex::new(String::new());
static EMB_CACHE: Mutex<Vec<(String, Arc<Embeds>)>> = Mutex::new(Vec::new());
const EMB_CAP: usize = 6;

pub fn clear_cache() {
    if let Ok(mut g) = EMB_CACHE.lock() {
        g.clear();
    }
}

use crate::sam_remote::{RemoteSam, FEATS_GONE};
use cwtools_core::ort_rt::{self, ConvSearch, Ep, ModelKey, Precision, SharedSession};

/// Session key for one SAM file. The CUDA options equal onnxruntime's
/// defaults (TF32 on, EXHAUSTIVE conv search, max workspace) = what the
/// earlier per-plugin sessions used, so masks stay byte-identical.
fn key_for(path: &std::path::Path, gpu: bool) -> ModelKey {
    ModelKey {
        path: path.to_string_lossy().into_owned(),
        ep: if gpu { Ep::Cuda } else { Ep::Cpu },
        precision: Precision::Tf32,
        conv_max_workspace: true,
        conv_search: ConvSearch::Exhaustive,
    }
}

/// The Model Folder, or the installed models when the node's folder does not
/// exist here (e.g. a setup saved on another workstation).
pub(crate) fn model_folder(v: &Values) -> std::path::PathBuf {
    let installed = &cwtools_core::config::get().segment_models;
    if !std::path::Path::new(&v.model_dir).is_dir() && std::path::Path::new(installed).is_dir() {
        crate::log::log(&format!("local: {} missing, using the installed models in {installed} (cwtools.conf)", v.model_dir));
        std::path::PathBuf::from(installed)
    } else {
        std::path::PathBuf::from(&v.model_dir)
    }
}

/// Run `f` on the node's engine: Mode Remote = cwtools-infer, Local = this process.
/// Remote encoder features the server dropped (restart, eviction) are recomputed once.
fn with_engine<R>(v: &Values, f: impl Fn(&Engine) -> Result<R, String>) -> Result<R, String> {
    if v.mode == 0 {
        let rs = RemoteSam { server: v.server.clone(), timeout_s: v.timeout, dir: model_folder(v) };
        match f(&Engine::Remote(&rs)) {
            Err(e) if e.starts_with(FEATS_GONE) => {
                crate::log::log(&format!("remote: {e}; encoding again"));
                forget_remote_embeds();
                f(&Engine::Remote(&rs))
            }
            r => r,
        }
    } else {
        with_model(v, |m| f(&Engine::Local(m)))
    }
}

fn forget_remote_embeds() {
    if let Ok(mut g) = EMB_CACHE.lock() {
        g.retain(|(_, e)| e.remote.is_none());
    }
}

fn with_model<R>(v: &Values, f: impl FnOnce(&Model) -> Result<R, String>) -> Result<R, String> {
    ort_rt::init_ort(&v.ort_lib)?;
    let pair = format!("{}|{}", v.model_dir, v.use_gpu);
    {
        let mut pk = PAIR_KEY.lock().map_err(|_| "model lock poisoned".to_string())?;
        if *pk != pair {
            if !pk.is_empty() {
                // drop old sessions first (VRAM); the CUDA keeper stays alive
                let n = ort_rt::release_all();
                crate::log::log(&format!("local: model folder/GPU changed, released {n} session(s)"));
            }
            clear_cache();
            *pk = pair;
        }
    }
    let dir_buf = model_folder(v);
    let dir = dir_buf.as_path();
    let fp16 = dir.join("encoder_fp16.onnx");
    // fp16 encoder only where CUDA can run it: macOS has no CUDA (onnxruntime falls back to the
    // CPU, where the fp32 encoder of segment-sam3-cpu is the right one)
    let enc = if v.use_gpu && !cfg!(target_os = "macos") && fp16.exists() { fp16 } else { find_model(dir, "encoder") };
    let dec = find_model(dir, "decoder");
    for p in [&enc, &dec] {
        if !p.exists() {
            return Err(format!("missing model file {}", p.display()));
        }
    }
    let (ke, kd) = (key_for(&enc, v.use_gpu), key_for(&dec, v.use_gpu));
    let fresh = !ort_rt::is_cached(&ke) || !ort_rt::is_cached(&kd);
    let t0 = std::time::Instant::now();
    let encoder = ort_rt::shared_session(&v.ort_lib, &ke)?;
    let decoder = ort_rt::shared_session(&v.ort_lib, &kd)?;
    if fresh {
        crate::log::log(&format!(
            "local: loaded models from {} in {:.1}s (encoder gpu={}, decoder gpu={})",
            dir.display(),
            t0.elapsed().as_secs_f64(),
            encoder.on_gpu,
            decoder.on_gpu
        ));
    }
    let gpu = encoder.on_gpu && decoder.on_gpu;
    f(&Model { encoder, decoder, gpu, dir: dir.to_path_buf() })
}

/// Run options for one session run: with Free VRAM/Frame the CUDA arena
/// shrinks back to its initial chunk after the run (weights stay loaded).
fn run_opts(v: &Values, s: &SharedSession) -> Result<RunOptions, String> {
    let mut ro = RunOptions::new().map_err(|e| format!("run options: {e}"))?;
    if v.free_vram && s.on_gpu {
        ro.add_config_entry("memory.enable_memory_arena_shrinkage", "gpu:0").map_err(|e| format!("run options: {e}"))?;
    }
    Ok(ro)
}

/// Release GPU Memory: drop every cached SAM session of this plugin (the CUDA
/// keeper stays) and the embedding cache. Returns the number released.
pub fn release_gpu() -> usize {
    let n = ort_rt::release_all();
    clear_cache();
    n
}

/// `<dir>/<role>.onnx`, else the first `*<role>*.onnx` in the folder.
fn find_model(dir: &std::path::Path, role: &str) -> std::path::PathBuf {
    let exact = dir.join(format!("{role}.onnx"));
    if exact.exists() {
        return exact;
    }
    if let Ok(rd) = std::fs::read_dir(dir) {
        let mut c: Vec<_> = rd
            .filter_map(|e| e.ok().map(|e| e.path()))
            .filter(|p| {
                let n = p.file_name().map(|n| n.to_string_lossy().to_lowercase()).unwrap_or_default();
                n.ends_with(".onnx") && n.contains(role) && !n.starts_with("text_") && !n.contains("_fp16")
            })
            .collect();
        c.sort();
        if let Some(p) = c.into_iter().next() {
            return p;
        }
    }
    exact
}

fn emb_get(k: &str) -> Option<Arc<Embeds>> {
    let g = EMB_CACHE.lock().ok()?;
    g.iter().find(|(kk, _)| kk == k).map(|(_, e)| e.clone())
}
fn emb_put(k: String, e: Arc<Embeds>) {
    if let Ok(mut g) = EMB_CACHE.lock() {
        g.retain(|(kk, _)| *kk != k);
        if g.len() >= EMB_CAP {
            g.remove(0);
        }
        g.push((k, e));
    }
}

fn extract(outputs: &ort::session::SessionOutputs, name: &str) -> Result<Vec<f32>, String> {
    let v = outputs.get(name).ok_or_else(|| format!("model output '{name}' missing"))?;
    let (_, d) = v.try_extract_tensor::<f32>().map_err(|e| format!("output {name}: {e}"))?;
    Ok(d.to_vec())
}

/// Letterbox the (already ≤1008) image top-left into a 1008² NCHW 0..1 tensor.
fn image_tensor(img: &Rgb8) -> Vec<f32> {
    let mut t = vec![0f32; 3 * S * S];
    for y in 0..img.h.min(S) {
        for x in 0..img.w.min(S) {
            let i = (y * img.w + x) * 3;
            for c in 0..3 {
                t[c * S * S + y * S + x] = img.data[i + c] as f32 / 255.0;
            }
        }
    }
    t
}

/// Encoder output for this frame (cached per frame key + model folder + engine).
/// `det` = the text detector will need the neck levels (Remote keeps them only then).
fn embeds(fk: &str, img: &Rgb8, v: &Values, eng: &Engine, det: bool) -> Result<Arc<Embeds>, String> {
    if img.w > S || img.h > S {
        return Err(format!("internal: local image {}x{} exceeds {S}", img.w, img.h));
    }
    let ckey = format!("{fk}|{}|{}", v.model_dir, eng.tag());
    if let Some(e) = if v.cache { emb_get(&ckey) } else { None } {
        if !det || e.remote.as_ref().map(|r| r.det).unwrap_or(true) {
            return Ok(e);
        }
    }
    let m = match eng {
        Engine::Local(m) => m,
        Engine::Remote(rs) => {
            let t0 = std::time::Instant::now();
            let prefix = rs.encode(img, &image_tensor(img), det)?;
            if crate::log::verbose() {
                crate::log::log(&format!("remote: encoder {:.0} ms (text={det})", t0.elapsed().as_secs_f64() * 1000.0));
            }
            let e = Arc::new(Embeds {
                embed: Vec::new(), hr0: Vec::new(), hr1: Vec::new(), det: None,
                remote: Some(RemoteFeat { server: rs.server.clone(), prefix, det }),
            });
            emb_put(ckey, e.clone());
            return Ok(e);
        }
    };
    let t0 = std::time::Instant::now();
    let x = Tensor::from_array((vec![1i64, 3, S as i64, S as i64], image_tensor(img))).map_err(|e| format!("tensor: {e}"))?;
    let ro = run_opts(v, &m.encoder)?;
    let mut es = m.encoder.session.lock().map_err(|_| "session lock poisoned".to_string())?;
    let has_det = es.outputs.iter().any(|o| o.name == "det_fpn_0");
    let out = es.run_with_options(ort::inputs!["image" => x], &ro).map_err(|e| format!("encoder run: {e}"))?;
    let det = if has_det {
        Some([extract(&out, "det_fpn_0")?, extract(&out, "det_fpn_1")?, extract(&out, "det_fpn_2")?])
    } else {
        None
    };
    let e = Arc::new(Embeds { embed: extract(&out, "image_embed")?, hr0: extract(&out, "high_res_0")?, hr1: extract(&out, "high_res_1")?, det, remote: None });
    if crate::log::verbose() {
        crate::log::log(&format!("local: encoder {:.0} ms (gpu={}, text={has_det})", t0.elapsed().as_secs_f64() * 1000.0, m.gpu));
    }
    emb_put(ckey, e.clone());
    Ok(e)
}

/// Text prompt only (no points / box): every detection of the text above Detect
/// Thresh, as one mask (union, or the best instance with Best Instance Only).
/// Points / box present: the tracker path (the text is ignored, as before).
pub fn segment(fk: &str, img: &Rgb8, p: &Prompts, v: &Values) -> Result<Gray, String> {
    if p.points.is_empty() && p.bbox.is_none() {
        if p.text.trim().is_empty() {
            return Err("Local mode needs a text prompt, points or a box".into());
        }
        return segment_text(fk, img, p.text.trim(), v);
    }
    with_engine(v, |eng| {
        let emb = embeds(fk, img, v, eng, false)?;

        let mut coords: Vec<f32> = Vec::new();
        let mut labels: Vec<f32> = Vec::new();
        for &(x, y, fg) in &p.points {
            coords.extend_from_slice(&[x, y]);
            labels.push(if fg { 1.0 } else { 0.0 });
        }
        if let Some(b) = p.bbox {
            coords.extend_from_slice(&[b[0], b[1], b[2], b[3]]);
            labels.extend_from_slice(&[2.0, 3.0]);
        }
        let (masks, iou, obj) = match (eng, &emb.remote) {
            (Engine::Remote(rs), Some(r)) => rs.decode(&r.prefix, coords, labels)?,
            (Engine::Local(m), None) => {
                let n = labels.len() as i64;
                let t = |shape: Vec<i64>, d: Vec<f32>| Tensor::from_array((shape, d)).map_err(|e| format!("tensor: {e}"));
                let ro = run_opts(v, &m.decoder)?;
                let mut ds = m.decoder.session.lock().map_err(|_| "session lock poisoned".to_string())?;
                let out = ds
                    .run_with_options(ort::inputs![
                        "image_embed" => t(vec![1, 256, 72, 72], emb.embed.clone())?,
                        "high_res_0" => t(vec![1, 32, 288, 288], emb.hr0.clone())?,
                        "high_res_1" => t(vec![1, 64, 144, 144], emb.hr1.clone())?,
                        "point_coords" => t(vec![1, n, 2], coords)?,
                        "point_labels" => t(vec![1, n], labels)?,
                        "mask_input" => t(vec![1, 1, LR as i64, LR as i64], vec![0f32; LR * LR])?,
                        "has_mask_input" => t(vec![1], vec![0f32])?,
                    ], &ro)
                    .map_err(|e| format!("decoder run: {e}"))?;
                let obj = extract(&out, "object_score").ok().and_then(|v| v.first().copied()).unwrap_or(1.0);
                (extract(&out, "low_res_masks")?, extract(&out, "iou_predictions")?, obj)
            }
            _ => return Err("internal: embedding of the other mode".into()),
        };
        if masks.len() < 4 * LR * LR || iou.len() < 4 {
            return Err(format!("unexpected decoder output sizes {} / {}", masks.len(), iou.len()));
        }
        let single_point = p.bbox.is_none() && p.points.len() == 1;
        let idx = if single_point {
            (1..4).max_by(|&a, &b| iou[a].partial_cmp(&iou[b]).unwrap_or(std::cmp::Ordering::Equal)).unwrap_or(0)
        } else {
            0
        };
        if crate::log::verbose() {
            crate::log::log(&format!("{}: decoder idx={idx} iou={iou:?} object_score={obj:.2}", eng.tag()));
        }
        if obj < 0.0 {
            return Ok(Gray { w: img.w, h: img.h, data: vec![0.0; img.w * img.h] });
        }
        Ok(logits_to_mask(&masks[idx * LR * LR..(idx + 1) * LR * LR], img.w, img.h))
    })
}

// ── tracking ───────────────────────────────────────────────────────────

/// crate::track backend: this frame's encoder features (cached as for every
/// Local render) + track_init.onnx / track_step.onnx from the model folder.
pub(crate) struct TrackBackend<'a> {
    pub v: &'a Values,
    pub fk: &'a str,
    pub img: &'a Rgb8,
}

pub(crate) struct TrackFeat {
    emb: Arc<Embeds>,
    dir: std::path::PathBuf,
}

impl TrackBackend<'_> {
    fn remote(&self, dir: &std::path::Path) -> RemoteSam {
        RemoteSam { server: self.v.server.clone(), timeout_s: self.v.timeout, dir: dir.to_path_buf() }
    }
}

fn feat_inputs(t: &dyn Fn(Vec<i64>, Vec<f32>) -> Result<Tensor<f32>, String>, e: &Embeds) -> Result<[Tensor<f32>; 3], String> {
    Ok([
        t(vec![1, 256, 72, 72], e.embed.clone())?,
        t(vec![1, 32, 288, 288], e.hr0.clone())?,
        t(vec![1, 64, 144, 144], e.hr1.clone())?,
    ])
}

fn track_out(out: &ort::session::SessionOutputs) -> Result<crate::track::Out, String> {
    Ok(crate::track::Out {
        low: extract(out, "low_res_mask")?,
        obj: extract(out, "object_score")?.first().copied().unwrap_or(f32::NAN),
        ptr: extract(out, "object_pointer")?,
        mem: crate::track::Mem::Local(Arc::new(extract(out, "maskmem")?)),
    })
}

fn track_session(v: &Values, dir: &std::path::Path, file: &str) -> Result<Arc<SharedSession>, String> {
    let p = dir.join(file);
    if !p.exists() {
        return Err(format!("tracking needs {file} in {} (SAM3 tracker models)", dir.display()));
    }
    ort_rt::shared_session(&v.ort_lib, &key_for(&p, v.use_gpu))
}

/// track_init.onnx on one object: encoder features + 1008² 0/1 mask.
pub(crate) fn run_track_init(v: &Values, dir: &std::path::Path, emb: &Embeds, mask: Vec<f32>) -> Result<crate::track::Out, String> {
    let s = track_session(v, dir, "track_init.onnx")?;
    let t = |shape: Vec<i64>, d: Vec<f32>| Tensor::from_array((shape, d)).map_err(|e| format!("tensor: {e}"));
    let [e, h0, h1] = feat_inputs(&t, emb)?;
    let ro = run_opts(v, &s)?;
    let mut ss = s.session.lock().map_err(|_| "session lock poisoned".to_string())?;
    let out = ss
        .run_with_options(ort::inputs![
            "image_embed" => e, "high_res_0" => h0, "high_res_1" => h1,
            "mask" => t(vec![1, 1, S as i64, S as i64], mask)?,
        ], &ro)
        .map_err(|e| format!("track_init run: {e}"))?;
    track_out(&out)
}

/// track_step.onnx on one object: encoder features + its memory bank inputs.
pub(crate) fn run_track_step(v: &Values, dir: &std::path::Path, emb: &Embeds, inp: &crate::track::StepInputs) -> Result<crate::track::Out, String> {
    let s = track_session(v, dir, "track_step.onnx")?;
    let t = |shape: Vec<i64>, d: Vec<f32>| Tensor::from_array((shape, d)).map_err(|e| format!("tensor: {e}"));
    let [e, h0, h1] = feat_inputs(&t, emb)?;
    let (n, p) = (inp.n_mem() as i64, inp.n_ptr() as i64);
    let t0 = std::time::Instant::now();
    let ro = run_opts(v, &s)?;
    let mut ss = s.session.lock().map_err(|_| "session lock poisoned".to_string())?;
    let out = ss
        .run_with_options(ort::inputs![
            "image_embed" => e, "high_res_0" => h0, "high_res_1" => h1,
            "mem_feats" => t(vec![n, 64, 72, 72], inp.feats_local()?)?,
            "mem_tpos" => Tensor::from_array((vec![n], inp.tpos.clone())).map_err(|e| format!("tensor: {e}"))?,
            "pointers" => t(vec![p, 256], inp.ptrs.clone())?,
            "pointer_offsets" => t(vec![p], inp.offs.clone())?,
        ], &ro)
        .map_err(|e| format!("track_step run: {e}"))?;
    if crate::log::verbose() {
        crate::log::log(&format!("local: track_step mem={n} ptr={p} {:.0} ms (gpu={})", t0.elapsed().as_secs_f64() * 1000.0, s.on_gpu));
    }
    track_out(&out)
}

impl crate::track::Backend for TrackBackend<'_> {
    type Feat = TrackFeat;

    fn features(&mut self) -> Result<TrackFeat, String> {
        if self.img.w > S || self.img.h > S {
            return Err(format!("internal: local image {}x{} exceeds {S}", self.img.w, self.img.h));
        }
        with_engine(self.v, |eng| Ok(TrackFeat { emb: embeds(self.fk, self.img, self.v, eng, false)?, dir: eng.dir().to_path_buf() }))
    }
    fn init(&mut self, f: &TrackFeat, mask: Vec<f32>, tag: &str) -> Result<crate::track::Out, String> {
        match &f.emb.remote {
            Some(r) => self.remote(&f.dir).track_init(&r.prefix, mask, tag),
            None => run_track_init(self.v, &f.dir, &f.emb, mask),
        }
    }
    fn step(&mut self, f: &TrackFeat, inp: &crate::track::StepInputs, tag: &str) -> Result<crate::track::Out, String> {
        match &f.emb.remote {
            Some(r) => self.remote(&f.dir).track_step(&r.prefix, inp, tag),
            None => run_track_step(self.v, &f.dir, &f.emb, inp),
        }
    }
    fn release(&mut self, handles: Vec<String>) {
        if self.v.mode == 0 {
            self.remote(std::path::Path::new(&self.v.model_dir)).release(&handles);
        }
    }
}

// ── text (concept) path ────────────────────────────────────────────────

const NQ: usize = 200;

/// Tokenizer of the folder loaded last (dir, tokenizer).
static TOKENIZER: Mutex<Option<(std::path::PathBuf, Arc<crate::clip_tok::ClipTokenizer>)>> = Mutex::new(None);
/// Text features per (folder, text): [1,32,256] + attention mask.
static TEXT_CACHE: Mutex<Vec<(String, Arc<(Vec<f32>, Vec<i64>)>)>> = Mutex::new(Vec::new());
const TEXT_CAP: usize = 32;

struct Detections {
    /// sigmoid(logit) * sigmoid(presence) per query
    scores: Vec<f32>,
    /// xyxy, 0..1 of the 1008² frame
    boxes: Vec<f32>,
    /// NQ x 288 x 288 logits (Local), or those of `sel` only (Remote)
    masks: Vec<f32>,
    /// Remote: the queries whose masks were downloaded (above Detect Thresh)
    sel: Option<Vec<usize>>,
}

impl Detections {
    /// 288² logits of query `i` (Remote: only a query above the threshold).
    fn mask(&self, i: usize) -> &[f32] {
        let k = match &self.sel {
            None => i,
            Some(s) => s.iter().position(|&q| q == i).expect("mask of a query below the threshold"),
        };
        &self.masks[k * LR * LR..(k + 1) * LR * LR]
    }
}

fn sigmoid(x: f32) -> f32 {
    1.0 / (1.0 + (-x).exp())
}

fn tokenizer(dir: &std::path::Path) -> Result<Arc<crate::clip_tok::ClipTokenizer>, String> {
    let mut g = TOKENIZER.lock().map_err(|_| "tokenizer lock poisoned".to_string())?;
    if let Some((d, t)) = g.as_ref() {
        if d == dir {
            return Ok(t.clone());
        }
    }
    let t = Arc::new(crate::clip_tok::ClipTokenizer::from_dir(dir)?);
    *g = Some((dir.to_path_buf(), t.clone()));
    Ok(t)
}

fn need(dir: &std::path::Path, f: &str) -> Result<std::path::PathBuf, String> {
    let p = dir.join(f);
    if p.exists() {
        Ok(p)
    } else {
        Err(format!("text prompts need {f} in {} (SAM3 text models)", dir.display()))
    }
}

/// Text features for `text` (text encoder on the CPU, here or on the server; cached).
fn text_features(v: &Values, eng: &Engine, text: &str) -> Result<Arc<(Vec<f32>, Vec<i64>)>, String> {
    let dir = eng.dir();
    let key = format!("{}|{}|{text}", dir.display(), eng.tag());
    if let Ok(g) = TEXT_CACHE.lock() {
        if let Some((_, f)) = g.iter().find(|(k, _)| *k == key) {
            return Ok(f.clone());
        }
    }
    let tok = tokenizer(dir)?;
    let (ids, mask) = tok.encode(text);
    let t0 = std::time::Instant::now();
    let feats = match eng {
        Engine::Remote(rs) => rs.text_features(ids, mask.clone())?,
        Engine::Local(_) => {
            let te = ort_rt::shared_session(&v.ort_lib, &key_for(&need(dir, "text_encoder.onnx")?, false))?;
            let n = crate::clip_tok::TEXT_LEN as i64;
            let mut s = te.session.lock().map_err(|_| "session lock poisoned".to_string())?;
            let out = s
                .run(ort::inputs![
                    "input_ids" => Tensor::from_array((vec![1, n], ids)).map_err(|e| format!("tensor: {e}"))?,
                    "attention_mask" => Tensor::from_array((vec![1, n], mask.clone())).map_err(|e| format!("tensor: {e}"))?,
                ])
                .map_err(|e| format!("text encoder run: {e}"))?;
            extract(&out, "text_features")?
        }
    };
    let f = Arc::new((feats, mask));
    if crate::log::verbose() {
        crate::log::log(&format!("{}: text encoder '{text}' {:.0} ms", eng.tag(), t0.elapsed().as_secs_f64() * 1000.0));
    }
    if let Ok(mut g) = TEXT_CACHE.lock() {
        if g.len() >= TEXT_CAP {
            g.remove(0);
        }
        g.push((key, f.clone()));
    }
    Ok(f)
}

/// Detections of `text`; Remote downloads the masks of the queries above `thr` only.
fn detect(v: &Values, eng: &Engine, emb: &Embeds, text: &str, thr: f32) -> Result<Detections, String> {
    let tf = text_features(v, eng, text)?;
    let m = match (eng, &emb.remote) {
        (Engine::Local(m), None) => m,
        (Engine::Remote(rs), Some(r)) => {
            let t0 = std::time::Instant::now();
            let (logits, presence, boxes, h) = rs.detect(&r.prefix, &tf.0, &tf.1)?;
            if logits.len() != NQ || boxes.len() != NQ * 4 {
                rs.release(&[h]);
                return Err(format!("unexpected detector output sizes {} / {}", logits.len(), boxes.len()));
            }
            let p = sigmoid(presence);
            let scores: Vec<f32> = logits.iter().map(|l| sigmoid(*l) * p).collect();
            if scores.iter().any(|x| !x.is_finite()) {
                rs.release(&[h]);
                return Err("detector produced non-finite scores".into());
            }
            let sel = kept(&scores, thr);
            let masks = rs.masks(&h, &sel)?;
            if crate::log::verbose() {
                let best = scores.iter().cloned().fold(0f32, f32::max);
                crate::log::log(&format!("{}: detector '{text}' {:.0} ms presence={p:.3} best={best:.3} kept={}",
                                         eng.tag(), t0.elapsed().as_secs_f64() * 1000.0, sel.len()));
            }
            return Ok(Detections { scores, boxes, masks, sel: Some(sel) });
        }
        _ => return Err("internal: embedding of the other mode".into()),
    };
    let det = emb.det.as_ref().ok_or_else(|| {
        format!("the encoder in {} has no text support (click-only v1 encoder); install the SAM3 text models", m.dir.display())
    })?;
    let ds = ort_rt::shared_session(&v.ort_lib, &key_for(&need(&m.dir, "detector.onnx")?, v.use_gpu))?;
    let t = |shape: Vec<i64>, d: Vec<f32>| Tensor::from_array((shape, d)).map_err(|e| format!("tensor: {e}"));
    let n = crate::clip_tok::TEXT_LEN as i64;
    let t0 = std::time::Instant::now();
    let ro = run_opts(v, &ds)?;
    let mut s = ds.session.lock().map_err(|_| "session lock poisoned".to_string())?;
    let out = s
        .run_with_options(ort::inputs![
            "det_fpn_0" => t(vec![1, 256, 288, 288], det[0].clone())?,
            "det_fpn_1" => t(vec![1, 256, 144, 144], det[1].clone())?,
            "det_fpn_2" => t(vec![1, 256, 72, 72], det[2].clone())?,
            "text_features" => t(vec![1, n, 256], tf.0.clone())?,
            "attention_mask" => Tensor::from_array((vec![1, n], tf.1.clone())).map_err(|e| format!("tensor: {e}"))?,
        ], &ro)
        .map_err(|e| format!("detector run: {e}"))?;
    let logits = extract(&out, "pred_logits")?;
    let presence = extract(&out, "presence_logit")?.first().copied().unwrap_or(0.0);
    let boxes = extract(&out, "pred_boxes")?;
    let masks = extract(&out, "pred_masks")?;
    if logits.len() != NQ || boxes.len() != NQ * 4 || masks.len() != NQ * LR * LR {
        return Err(format!("unexpected detector output sizes {} / {} / {}", logits.len(), boxes.len(), masks.len()));
    }
    let p = sigmoid(presence);
    let scores: Vec<f32> = logits.iter().map(|l| sigmoid(*l) * p).collect();
    if scores.iter().any(|x| !x.is_finite()) {
        return Err("detector produced non-finite scores".into());
    }
    if crate::log::verbose() {
        let best = scores.iter().cloned().fold(0f32, f32::max);
        crate::log::log(&format!("local: detector '{text}' {:.0} ms presence={p:.3} best={best:.3}", t0.elapsed().as_secs_f64() * 1000.0));
    }
    Ok(Detections { scores, boxes, masks, sel: None })
}

/// Query indices above `thr`, best first.
fn kept(scores: &[f32], thr: f32) -> Vec<usize> {
    let mut k: Vec<usize> = (0..scores.len()).filter(|&i| scores[i] > thr).collect();
    k.sort_by(|&a, &b| scores[b].partial_cmp(&scores[a]).unwrap_or(std::cmp::Ordering::Equal));
    k
}

fn segment_text(fk: &str, img: &Rgb8, text: &str, v: &Values) -> Result<Gray, String> {
    with_engine(v, |eng| {
        let emb = embeds(fk, img, v, eng, true)?;
        let d = detect(v, eng, &emb, text, v.score_threshold as f32)?;
        let mut k = kept(&d.scores, v.score_threshold as f32);
        if v.best_only {
            k.truncate(1);
        }
        if k.is_empty() {
            return Ok(Gray { w: img.w, h: img.h, data: vec![0.0; img.w * img.h] });
        }
        // union = max of the probabilities = sigmoid of the max logit
        let mut lr = d.mask(k[0]).to_vec();
        for &i in &k[1..] {
            for (a, b) in lr.iter_mut().zip(d.mask(i)) {
                *a = a.max(*b);
            }
        }
        Ok(logits_to_mask(&lr, img.w, img.h))
    })
}

/// Every detection of every concept as its own soft mask at the model-image
/// size (what the Remote server's `instances: "all"` returns): above Detect
/// Thresh, at most `max_instances` (highest scores), concept order then score.
pub fn segment_instances(fk: &str, img: &Rgb8, concepts: &[String], v: &Values, max_instances: i32) -> Result<Vec<Instance>, String> {
    with_engine(v, |eng| {
        let emb = embeds(fk, img, v, eng, true)?;
        let mut items: Vec<(usize, f32, [f32; 4], Vec<f32>)> = Vec::new();
        for (ci, c) in concepts.iter().enumerate() {
            let d = detect(v, eng, &emb, c, v.score_threshold as f32)?;
            for i in kept(&d.scores, v.score_threshold as f32) {
                let b = &d.boxes[i * 4..i * 4 + 4];
                let px = |x: f32, lim: usize| (x * S as f32).clamp(0.0, lim as f32);
                items.push((ci, d.scores[i], [px(b[0], img.w), px(b[1], img.h), px(b[2], img.w), px(b[3], img.h)],
                            d.mask(i).to_vec()));
            }
        }
        let cap = max_instances.clamp(1, 128) as usize;
        if items.len() > cap {
            let mut order: Vec<usize> = (0..items.len()).collect();
            order.sort_by(|&a, &b| items[b].1.partial_cmp(&items[a].1).unwrap_or(std::cmp::Ordering::Equal));
            let mut keep = order[..cap].to_vec();
            keep.sort();
            items = keep.into_iter().map(|i| std::mem::take(&mut items[i])).collect();
        }
        Ok(items
            .into_iter()
            .map(|(ci, score, bbox, lr)| {
                let g = logits_to_mask(&lr, img.w, img.h);
                Instance {
                    concept: concepts[ci].clone(),
                    concept_index: ci,
                    score,
                    bbox,
                    w: img.w,
                    h: img.h,
                    mask: g.data.iter().map(|p| (p.clamp(0.0, 1.0) * 255.0 + 0.5) as u8).collect(),
                }
            })
            .collect())
    })
}

/// Bilinear-sample the 288² logit grid (covering the 1008² letterboxed frame)
/// at each pixel of the w×h image region, then sigmoid.
pub fn logits_to_mask(lr: &[f32], w: usize, h: usize) -> Gray {
    let k = LR as f32 / S as f32;
    let mut data = vec![0f32; w * h];
    let maxi = LR as f32 - 1.0;
    for y in 0..h {
        let fy = ((y as f32 + 0.5) * k - 0.5).clamp(0.0, maxi);
        let y0 = fy.floor() as usize;
        let y1 = (y0 + 1).min(LR - 1);
        let ty = fy - y0 as f32;
        for x in 0..w {
            let fx = ((x as f32 + 0.5) * k - 0.5).clamp(0.0, maxi);
            let x0 = fx.floor() as usize;
            let x1 = (x0 + 1).min(LR - 1);
            let tx = fx - x0 as f32;
            let a = lr[y0 * LR + x0] + (lr[y0 * LR + x1] - lr[y0 * LR + x0]) * tx;
            let b = lr[y1 * LR + x0] + (lr[y1 * LR + x1] - lr[y1 * LR + x0]) * tx;
            let l = a + (b - a) * ty;
            data[y * w + x] = 1.0 / (1.0 + (-l).exp());
        }
    }
    Gray { w, h, data }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// End-to-end Local text path on real models (ignored; needs a GPU box):
    ///   CWSEG_E2E_DIR=<dir> CWSEG_ORT_LIB=<libonnxruntime.so> cargo test -p cwsegment --lib e2e -- --ignored
    /// <dir>/models = SAM3 text model folder, <dir>/image.rgb + image.json {"w","h"} (RGB8,
    /// top-down, <= 1008). Writes <dir>/rust_out.json (instances: concept, score, bbox,
    /// area) + rust_union_<concept>.u8 (segment(), w*h bytes) for the Python reference.
    #[test]
    #[ignore]
    fn e2e_local_text() {
        let dir = std::path::PathBuf::from(std::env::var("CWSEG_E2E_DIR").expect("CWSEG_E2E_DIR"));
        let lib = std::env::var("CWSEG_ORT_LIB").unwrap_or_default();
        let meta: serde_json::Value = serde_json::from_str(&std::fs::read_to_string(dir.join("image.json")).unwrap()).unwrap();
        let (w, h) = (meta["w"].as_u64().unwrap() as usize, meta["h"].as_u64().unwrap() as usize);
        let img = Rgb8 { w, h, data: std::fs::read(dir.join("image.rgb")).unwrap() };
        let gpu = std::env::var("CWSEG_CPU").is_err();
        let v = Values::for_test(dir.join("models").to_str().unwrap(), &lib, gpu);
        let concepts: Vec<String> = meta["concepts"].as_array().unwrap().iter().map(|c| c.as_str().unwrap().to_string()).collect();
        let t0 = std::time::Instant::now();
        let insts = segment_instances("e2e", &img, &concepts, &v, 32).unwrap();
        let t_inst = t0.elapsed().as_secs_f64();
        let mut out = Vec::new();
        for i in &insts {
            out.push(serde_json::json!({"concept": i.concept, "score": i.score, "bbox": i.bbox, "area": i.area()}));
        }
        for c in &concepts {
            let p = Prompts { text: c.clone(), points: Vec::new(), bbox: None };
            let m = segment("e2e", &img, &p, &v).unwrap();
            let u8s: Vec<u8> = m.data.iter().map(|x| (x.clamp(0.0, 1.0) * 255.0 + 0.5) as u8).collect();
            std::fs::write(dir.join(format!("rust_union_{}.u8", c.replace(' ', "_"))), u8s).unwrap();
        }
        let j = serde_json::json!({"instances": out, "seconds_instances": t_inst, "gpu": gpu});
        std::fs::write(dir.join("rust_out.json"), serde_json::to_string_pretty(&j).unwrap()).unwrap();
        eprintln!("{} instances in {t_inst:.2}s: {}", insts.len(), j);
    }
    /// Rust tracking (track::Bank + track_init/step) vs the Python reference on dumped
    /// features (ignored; flame_inference/sam_onnx/track_dump.py writes the dumps):
    ///   CWSEG_TRACK_DIR=<dir> CWSEG_ORT_LIB=<libonnxruntime.so> cargo test -p cwsegment --lib e2e_track -- --ignored
    /// <dir>/models = track_init/step.onnx, <dir>/f<i>_{embed,hr0,hr1}.f32, mask.u8 (1008²),
    /// ref<i>.u8 (sigmoid of the reference 288² logits). Writes rust<i>.u8.
    #[test]
    #[ignore]
    fn e2e_track() {
        use crate::track::{self, Backend, Frame, Request};
        let dir = std::path::PathBuf::from(std::env::var("CWSEG_TRACK_DIR").expect("CWSEG_TRACK_DIR"));
        let lib = std::env::var("CWSEG_ORT_LIB").unwrap_or_default();
        let v = Values::for_test(dir.join("models").to_str().unwrap(), &lib, std::env::var("CWSEG_CPU").is_err());
        let rd = |n: String| -> Vec<f32> {
            std::fs::read(dir.join(n)).unwrap().chunks_exact(4).map(|b| f32::from_le_bytes([b[0], b[1], b[2], b[3]])).collect()
        };
        struct Dump<'a> { v: &'a Values, dir: std::path::PathBuf, cur: Option<Arc<Embeds>> }
        impl Backend for Dump<'_> {
            type Feat = Arc<Embeds>;
            fn features(&mut self) -> Result<Arc<Embeds>, String> { Ok(self.cur.clone().unwrap()) }
            fn init(&mut self, f: &Arc<Embeds>, m: Vec<f32>, _tag: &str) -> Result<track::Out, String> { run_track_init(self.v, &self.dir, f, m) }
            fn step(&mut self, f: &Arc<Embeds>, i: &track::StepInputs, _tag: &str) -> Result<track::Out, String> { run_track_step(self.v, &self.dir, f, i) }
        }
        ort_rt::init_ort(&lib).unwrap();
        let n = (0..).take_while(|i| dir.join(format!("ref{i}.u8")).exists()).count();
        let mask: Vec<u8> = std::fs::read(dir.join("mask.u8")).unwrap();
        let inst = Instance { concept: "t".into(), concept_index: 0, score: 1.0, bbox: [0.0; 4], w: S, h: S, mask };
        let insts = vec![inst];
        let mut be = Dump { v: &v, dir: dir.join("models"), cur: None };
        let (mut worst, mut worst_f, mut iou_min) = (0i32, 0usize, 1f64);
        let t0 = std::time::Instant::now();
        track::reset(1);
        for i in 0..n {
            be.cur = Some(Arc::new(Embeds { embed: rd(format!("f{i}_embed.f32")), hr0: rd(format!("f{i}_hr0.f32")), hr1: rd(format!("f{i}_hr1.f32")), det: None, remote: None }));
            let r = Request { node: 1, frame: i as i64, anchor: 0, numbers: &[1], dir: track::DIR_FWD, key: "e2e".into(),
                              anchor_insts: (i == 0).then_some(&insts[..]), mw: S, mh: S };
            assert!(matches!(track::frame(&r, &mut be).unwrap(), Frame::Objects(_)));
            let got = track::raw(1, i as i64).unwrap().remove(0);
            let want = std::fs::read(dir.join(format!("ref{i}.u8"))).unwrap();
            std::fs::write(dir.join(format!("rust{i}.u8")), &got).unwrap();
            let d = got.iter().zip(&want).map(|(a, b)| (*a as i32 - *b as i32).abs()).max().unwrap();
            let (mut inter, mut uni) = (0usize, 0usize);
            for (a, b) in got.iter().zip(&want) {
                inter += (*a >= 128 && *b >= 128) as usize;
                uni += (*a >= 128 || *b >= 128) as usize;
            }
            let iou = if uni == 0 { 1.0 } else { inter as f64 / uni as f64 };
            iou_min = iou_min.min(iou);
            if d > worst { worst = d; worst_f = i; }
        }
        eprintln!("e2e_track: {n} frames in {:.1}s, IoU min {iou_min:.5}, worst |diff| {worst}/255 (frame {worst_f})", t0.elapsed().as_secs_f64());
        assert!(iou_min > 0.999, "Rust tracking differs from the reference");
    }

    /// Remote == Local on the whole pipeline (ignored; needs a GPU box + a cwtools-infer
    /// API 2 server with the same SAM3 files):
    ///   CWSEG_E2E_DIR=<dir> CWSEG_ORT_LIB=<lib> CWSEG_REMOTE=http://host:port cargo test -p cwsegment --lib e2e_remote -- --ignored
    /// <dir> as for e2e_local_text. Concept 0 is detected on a still, its first object is
    /// tracked over CWSEG_FRAMES (20) frames that pan the still 4 px a frame, once Local and
    /// once Remote; prints the differences (numbers only).
    #[test]
    #[ignore]
    fn e2e_remote_parity() {
        use crate::track::{self, Frame, Request};
        let dir = std::path::PathBuf::from(std::env::var("CWSEG_E2E_DIR").expect("CWSEG_E2E_DIR"));
        let lib = std::env::var("CWSEG_ORT_LIB").unwrap_or_default();
        let server = std::env::var("CWSEG_REMOTE").expect("CWSEG_REMOTE");
        let n: usize = std::env::var("CWSEG_FRAMES").ok().and_then(|s| s.parse().ok()).unwrap_or(20);
        let meta: serde_json::Value = serde_json::from_str(&std::fs::read_to_string(dir.join("image.json")).unwrap()).unwrap();
        let (w, h) = (meta["w"].as_u64().unwrap() as usize, meta["h"].as_u64().unwrap() as usize);
        let still = Rgb8 { w, h, data: std::fs::read(dir.join("image.rgb")).unwrap() };
        let concept = meta["concepts"][0].as_str().unwrap().to_string();
        let pan = |k: usize| {
            let mut d = vec![0u8; w * h * 3];
            for y in 0..h {
                for x in 0..w {
                    let sx = x.saturating_sub(4 * k);
                    d[(y * w + x) * 3..(y * w + x) * 3 + 3].copy_from_slice(&still.data[(y * w + sx) * 3..(y * w + sx) * 3 + 3]);
                }
            }
            Rgb8 { w, h, data: d }
        };
        let frames: Vec<Rgb8> = (0..n).map(pan).collect();
        let keys: Vec<String> = frames.iter().map(|f| format!("par_{}", crate::imgutil::frame_key(f))).collect();
        let run = |v: &Values, node: usize| -> (Vec<Instance>, Vec<Vec<u8>>, f64) {
            let t0 = std::time::Instant::now();
            let insts = segment_instances(&keys[0], &frames[0], &[concept.clone()], v, 32).unwrap();
            assert!(!insts.is_empty(), "no '{concept}' found");
            track::reset(node);
            let mut masks = Vec::new();
            for k in 0..n {
                let mut be = TrackBackend { v, fk: &keys[k], img: &frames[k] };
                let r = Request { node, frame: k as i64, anchor: 0, numbers: &[1], dir: track::DIR_FWD, key: "par".into(),
                                  anchor_insts: (k == 0).then_some(&insts[..]), mw: w, mh: h };
                assert!(matches!(track::frame(&r, &mut be).unwrap(), Frame::Objects(_)));
                masks.push(track::raw(node, k as i64).unwrap().remove(0));
            }
            track::reset(node);
            (insts, masks, t0.elapsed().as_secs_f64())
        };
        let mut lv = Values::for_test(dir.join("models").to_str().unwrap(), &lib, true);
        lv.cache = false;
        let mut rv = Values::for_test(dir.join("models").to_str().unwrap(), &lib, true);
        rv.cache = false;
        rv.mode = 0;
        rv.server = server.clone();
        let (li, lm, lt) = run(&lv, 11);
        let (ri, rm, rt) = run(&rv, 12);
        let score_d = li.iter().zip(&ri).map(|(a, b)| (a.score - b.score).abs()).fold(0f32, f32::max);
        let inst_d = li.iter().zip(&ri).map(|(a, b)| a.mask.iter().zip(&b.mask).map(|(x, y)| (*x as i32 - *y as i32).abs()).max().unwrap_or(0)).max().unwrap_or(0);
        let (mut worst, mut iou_min) = (0i32, 1f64);
        for (a, b) in lm.iter().zip(&rm) {
            worst = worst.max(a.iter().zip(b).map(|(x, y)| (*x as i32 - *y as i32).abs()).max().unwrap_or(0));
            let (mut i, mut u) = (0usize, 0usize);
            for (x, y) in a.iter().zip(b) {
                i += (*x >= 128 && *y >= 128) as usize;
                u += (*x >= 128 || *y >= 128) as usize;
            }
            iou_min = iou_min.min(if u == 0 { 1.0 } else { i as f64 / u as f64 });
        }
        let area = |m: &[u8]| m.iter().filter(|v| **v >= 128).count() as f64 / m.len() as f64 * 100.0;
        eprintln!("e2e_remote: '{concept}' local {} / remote {} instances, score max|diff| {score_d:.2e}, instance mask max|diff| {inst_d}/255", li.len(), ri.len());
        eprintln!("e2e_remote: {n} tracked frames, IoU min {iou_min:.5}, mask max|diff| {worst}/255, area {:.1}% -> {:.1}%", area(&lm[0]), area(&lm[n - 1]));
        eprintln!("e2e_remote: local {lt:.1} s, remote {rt:.1} s");
        assert_eq!(li.len(), ri.len(), "instance count differs");
        assert!(iou_min > 0.999, "Remote tracking differs from Local");
    }

    #[test]
    fn logits_map_left_half() {
        let mut lr = vec![-10f32; LR * LR];
        for y in 0..LR {
            for x in 0..LR / 2 {
                lr[y * LR + x] = 10.0;
            }
        }
        let m = logits_to_mask(&lr, S, 500);
        assert!(m.data[10 * S + 100] > 0.99);
        assert!(m.data[10 * S + 900] < 0.01);
    }
}
