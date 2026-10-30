//! Remote backend: the SAME SAM3 graphs as Local (encoder_fp16, decoder, text
//! encoder, detector, track_init / track_step) run on cwtools-infer (API 2),
//! the server named by Server URL / `remote_server` (the cwMatte / cwRefine
//! server). All pre- and post-processing stays in local.rs, so Remote equals
//! Local by construction (apart from GPU kernel differences).
//!
//! What stays on the server (its tensor store, by handle):
//! - the encoder outputs of a frame (`<proc>/e/<n>/<output>`), used by the
//!   decoder, the detector and the tracker without coming back here;
//! - the tracker's memory of each object and frame (`<proc>/t/<node>/<gen>/...`),
//!   concatenated on the server for every track_step;
//! - the detector's 200 masks, until the few above Detect Thresh are fetched.
//!
//! So a frame uploads its image (lossless, ~1-2 MB) and downloads small results.
//! Dropped encoder features / memories are released on the server (`/forget`);
//! what a crashed Flame leaves behind the server drops after its idle time.

use crate::imgutil::Rgb8;
use cwtools_remote::{self as rc, plugin, FetchItem, RemoteError, TensorIn};
use std::path::{Path, PathBuf};
use std::sync::OnceLock;

pub const S: usize = 1008;
const LR: usize = 288;

/// Marks an error caused by encoder features the server no longer has: the
/// caller drops its embedding cache entry and runs the encoder again.
pub const FEATS_GONE: &str = "remote features gone:";

/// Where the graphs run and how to name them.
pub(crate) struct RemoteSam {
    pub server: String,
    pub timeout_s: f64,
    /// Model folder: its SHA256SUMS names the files the server must have (the
    /// files themselves are not needed here), plus the tokenizer for text prompts.
    pub dir: PathBuf,
}

/// This process's handle prefix (unique per Flame session and workstation).
pub(crate) fn proc_id() -> &'static str {
    static P: OnceLock<String> = OnceLock::new();
    P.get_or_init(|| {
        let t = std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).map(|d| d.as_nanos()).unwrap_or(0);
        format!("cws{:x}{:x}", std::process::id(), (t as u64) & 0xffff_ffff)
    })
}

/// Server handle of a tracker memory named by crate::track (`t/<node>/<gen>/<frame>/<object>`).
pub(crate) fn mem_handle(tag: &str) -> String {
    format!("{}/{tag}", proc_id())
}

fn next_id() -> u64 {
    static N: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(1);
    N.fetch_add(1, std::sync::atomic::Ordering::Relaxed)
}

const ENC_OUT: [&str; 3] = ["image_embed", "high_res_0", "high_res_1"];
const DET_OUT: [&str; 3] = ["det_fpn_0", "det_fpn_1", "det_fpn_2"];

impl RemoteSam {
    fn sha(&self, file: &str) -> Result<String, String> {
        let p = self.dir.join(file);
        rc::model_sha256(&p.to_string_lossy()).map_err(|e| {
            format!("Remote needs the sha256 of {file}: not listed in {}/SHA256SUMS ({e}) - install the SAM3 model sets (or their checksums)", self.dir.display())
        })
    }

    fn err(&self, e: RemoteError, model: &str) -> String {
        if e.code == "ref_missing" {
            if e.missing.iter().any(|h| h.contains("/e/")) {
                return format!("{FEATS_GONE} {}", e.msg);
            }
            return "Remote: the server no longer has this node's tracking memory (restarted, or idle too long) - \
                    press Reset Track, then Track Range"
                .into();
        }
        if e.status == 409 {
            plugin::forget_models();
        }
        crate::log::log(&format!("remote {} {model}: {e}", self.server));
        plugin::artist_message(&e, &self.server, model, self.timeout_s)
    }

    fn run(&self, model: &str, file: &str, inputs: Vec<TensorIn>, outputs: &[&str], keep: Vec<(String, String)>, cpu: bool) -> Result<rc::Response, String> {
        self.drop_ended();
        let req = rc::Request {
            model: model.into(),
            sha256: self.sha(file)?,
            inputs,
            ep: Some(if cpu { rc::Ep::sam3_cpu() } else { rc::Ep::sam3() }),
            outputs: Some(outputs.iter().map(|s| s.to_string()).collect()),
            keep,
            output_dtype: None,
            output_quantize: None,
            output_level: None,
            level: rc::DEFAULT_LEVEL,
            timeout_s: self.timeout_s,
            webp: true,
        };
        let t0 = std::time::Instant::now();
        let r = rc::infer(&self.server, &req).map_err(|e| self.err(e, model))?;
        if crate::log::verbose() {
            let t = &r.timings;
            crate::log::log(&format!(
                "remote: {model} {:.0} ms (up {} KB, down {} KB, server {})",
                t0.elapsed().as_secs_f64() * 1000.0, t.up_bytes >> 10, t.down_bytes >> 10, t.server
            ));
        }
        Ok(r)
    }

    fn out(r: &rc::Response, name: &str, n: usize) -> Result<Vec<f32>, String> {
        let t = r.get(name).ok_or_else(|| format!("Remote: response has no '{name}'"))?;
        if n > 0 && t.data.len() != n {
            return Err(format!("Remote: {name} has {} values, expected {n}", t.data.len()));
        }
        Ok(t.data.clone())
    }

    /// Server-side memories of ended track sessions.
    fn drop_ended(&self) {
        let ended = crate::track::take_ended();
        for p in ended {
            let prefix = format!("{}/{p}", proc_id());
            if let Err(e) = rc::forget(&self.server, Some(&prefix), &[], 5.0) {
                crate::log::log(&format!("remote: could not release {prefix} on the server: {e}"));
            }
        }
    }

    pub fn release(&self, handles: &[String]) {
        if handles.is_empty() {
            return;
        }
        if let Err(e) = rc::forget(&self.server, None, handles, 5.0) {
            crate::log::log(&format!("remote: could not release {} tensor(s) on the server: {e}", handles.len()));
        }
    }

    /// encoder_fp16 on the frame; its outputs stay on the server. Returns the handle prefix.
    /// `img_t` is the 1008² letterboxed tensor local.rs feeds the encoder, `img` its source pixels.
    pub fn encode(&self, img: &Rgb8, img_t: &[f32], det: bool) -> Result<String, String> {
        // the tensor's indices = the u8 pixels it was made from (0 in the letterbox)
        let mut idx = vec![0u8; 3 * S * S];
        for y in 0..img.h.min(S) {
            for x in 0..img.w.min(S) {
                let i = (y * img.w + x) * 3;
                for c in 0..3 {
                    idx[c * S * S + y * S + x] = img.data[i + c];
                }
            }
        }
        let input = TensorIn::lut8_from_source("image", &[1, 3, S, S], img_t, idx)?;
        let prefix = format!("{}/e/{}/", proc_id(), next_id());
        let names: Vec<&str> = if det { ENC_OUT.iter().chain(DET_OUT.iter()).copied().collect() } else { ENC_OUT.to_vec() };
        let keep = names.iter().map(|n| (n.to_string(), format!("{prefix}{n}"))).collect();
        self.run("encoder_fp16", "encoder_fp16.onnx", vec![input], &[], keep, false)?;
        Ok(prefix)
    }

    fn feat_refs(prefix: &str) -> Vec<TensorIn> {
        ENC_OUT.iter().map(|n| TensorIn::refs(n, vec![format!("{prefix}{n}")], 0)).collect()
    }

    /// decoder.onnx: (low_res_masks [4·288²], iou_predictions [4], object_score).
    pub fn decode(&self, prefix: &str, coords: Vec<f32>, labels: Vec<f32>) -> Result<(Vec<f32>, Vec<f32>, f32), String> {
        let n = labels.len();
        let mut inp = Self::feat_refs(prefix);
        inp.push(TensorIn::f32("point_coords", &[1, n, 2], coords));
        inp.push(TensorIn::f32("point_labels", &[1, n], labels));
        inp.push(TensorIn::f32("mask_input", &[1, 1, LR, LR], vec![0f32; LR * LR]));
        inp.push(TensorIn::f32("has_mask_input", &[1], vec![0f32]));
        let r = self.run("decoder", "decoder.onnx", inp, &["low_res_masks", "iou_predictions", "object_score"], Vec::new(), false)?;
        let obj = Self::out(&r, "object_score", 0)?.first().copied().unwrap_or(1.0);
        Ok((Self::out(&r, "low_res_masks", 4 * LR * LR)?, Self::out(&r, "iou_predictions", 4)?, obj))
    }

    /// text_encoder.onnx on the server's CPU (as Local): [1,32,256].
    pub fn text_features(&self, ids: Vec<i64>, mask: Vec<i64>) -> Result<Vec<f32>, String> {
        let n = crate::clip_tok::TEXT_LEN;
        let inp = vec![TensorIn::i64("input_ids", &[1, n], ids), TensorIn::i64("attention_mask", &[1, n], mask)];
        let r = self.run("text_encoder", "text_encoder.onnx", inp, &["text_features"], Vec::new(), true)?;
        Self::out(&r, "text_features", n * 256)
    }

    /// detector.onnx: (pred_logits [200], presence_logit, pred_boxes [200·4]); the
    /// 200 masks stay on the server until [`RemoteSam::masks`].
    pub fn detect(&self, prefix: &str, tf: &[f32], mask: &[i64]) -> Result<(Vec<f32>, f32, Vec<f32>, String), String> {
        let n = crate::clip_tok::TEXT_LEN;
        let mut inp: Vec<TensorIn> = DET_OUT.iter().map(|d| TensorIn::refs(d, vec![format!("{prefix}{d}")], 0)).collect();
        inp.push(TensorIn::f32("text_features", &[1, n, 256], tf.to_vec()));
        inp.push(TensorIn::i64("attention_mask", &[1, n], mask.to_vec()));
        let h = format!("{}/d/{}", proc_id(), next_id());
        let r = self.run("detector", "detector.onnx", inp, &["pred_logits", "presence_logit", "pred_boxes"],
                         vec![("pred_masks".into(), h.clone())], false)?;
        let presence = Self::out(&r, "presence_logit", 0)?.first().copied().unwrap_or(0.0);
        Ok((Self::out(&r, "pred_logits", 0)?, presence, Self::out(&r, "pred_boxes", 0)?, h))
    }

    /// The detector masks of these queries (288² logits each, in `sel` order); drops the rest.
    pub fn masks(&self, handle: &str, sel: &[usize]) -> Result<Vec<f32>, String> {
        let out = if sel.is_empty() {
            Vec::new()
        } else {
            let t = rc::fetch(&self.server, &[FetchItem { handle: handle.into(), axis: 1, index: Some(sel.to_vec()) }], self.timeout_s)
                .map_err(|e| self.err(e, "detector"))?;
            let d = t.into_iter().next().map(|t| t.data).unwrap_or_default();
            if d.len() != sel.len() * LR * LR {
                return Err(format!("Remote: {} mask values for {} detections", d.len(), sel.len()));
            }
            d
        };
        self.release(&[handle.to_string()]);
        Ok(out)
    }

    fn track_outs(r: &rc::Response, mem: String) -> Result<crate::track::Out, String> {
        Ok(crate::track::Out {
            low: Self::out(r, "low_res_mask", LR * LR)?,
            obj: Self::out(r, "object_score", 0)?.first().copied().unwrap_or(f32::NAN),
            ptr: Self::out(r, "object_pointer", crate::track::PTR_LEN)?,
            mem: crate::track::Mem::Remote(mem),
        })
    }

    /// track_init.onnx: the memory stays on the server as `mem_handle(tag)`.
    pub fn track_init(&self, prefix: &str, mask: Vec<f32>, tag: &str) -> Result<crate::track::Out, String> {
        let mut inp = Self::feat_refs(prefix);
        inp.push(TensorIn::f32_auto("mask", &[1, 1, S, S], mask));
        let h = mem_handle(tag);
        let r = self.run("track_init", "track_init.onnx", inp, &["low_res_mask", "object_score", "object_pointer"],
                         vec![("maskmem".into(), h.clone())], false)?;
        Self::track_outs(&r, h)
    }

    /// track_step.onnx with the bank's memories concatenated on the server.
    pub fn track_step(&self, prefix: &str, inp: &crate::track::StepInputs, tag: &str) -> Result<crate::track::Out, String> {
        let mut x = Self::feat_refs(prefix);
        x.push(TensorIn::refs("mem_feats", inp.feats_remote()?, 0));
        x.push(TensorIn::i64("mem_tpos", &[inp.n_mem()], inp.tpos.clone()));
        x.push(TensorIn::f32("pointers", &[inp.n_ptr(), crate::track::PTR_LEN], inp.ptrs.clone()));
        x.push(TensorIn::f32("pointer_offsets", &[inp.n_ptr()], inp.offs.clone()));
        let h = mem_handle(tag);
        let r = self.run("track_step", "track_step.onnx", x, &["low_res_mask", "object_score", "object_pointer"],
                         vec![("maskmem".into(), h.clone())], false)?;
        Self::track_outs(&r, h)
    }
}

/// Test Server: /health (API 2 with the tensor store) + the SAM3 files on the server match.
pub fn test_server(server: &str, dir: &Path) -> Result<String, String> {
    let h = rc::get_json(server, "/health", 5.0).map_err(|e| plugin::artist_message(&e, server, "SAM3", 5.0))?;
    let feats: Vec<&str> = h["features"].as_array().map(|a| a.iter().filter_map(|x| x.as_str()).collect()).unwrap_or_default();
    if h["service"] != "cwtools-infer" || !feats.contains(&"refs") || !feats.contains(&"fetch") {
        return Err(format!("{server} is not cwtools-infer 0.4 or newer (API 2 tensor store) - update the server"));
    }
    plugin::forget_models();
    let mut missing = Vec::new();
    for f in ["encoder_fp16", "decoder", "text_encoder", "detector", "track_init", "track_step"] {
        let sha = rc::model_sha256(&dir.join(format!("{f}.onnx")).to_string_lossy());
        match sha {
            Ok(s) => match plugin::model_info(server, f, &s, 5.0) {
                Ok(m) if m.sha256 == s && m.servable => {}
                _ => missing.push(f),
            },
            Err(_) => missing.push(f),
        }
    }
    let store = &h["store"];
    if missing.is_empty() {
        Ok(format!("cwtools-infer API {} - all SAM3 graphs match, tensor store {} / {} MB", h["api"], store["mb"], store["cap_mb"]))
    } else {
        Err(format!("server reachable, but these SAM3 graphs are missing or differ from {}: {}", dir.display(), missing.join(", ")))
    }
}
