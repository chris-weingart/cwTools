//! Local backend: the core ONNX of the Model choice (ViTMatte / MEMatte /
//! custom; `pixel_values` [1,4,h,w] →
//! `alphas` [1,1,h,w], dynamic h/w multiples of 32) through Flame's bundled
//! onnxruntime 1.22, via the cwtools-core process-wide session cache: every
//! cwRefine node in the Flame process shares one session per
//! (model path, EP, GPU math).

use crate::params::Values;
use cwtools_core::ort_rt::{self, Ep, ModelKey, Precision};
use ort::session::RunOptions;
use ort::value::Tensor;
use std::sync::Mutex;

pub fn model_key(v: &Values) -> ModelKey {
    ModelKey {
        path: v.model_path.clone(),
        ep: if v.use_gpu { Ep::Cuda } else { Ep::Cpu },
        precision: if v.tf32 { Precision::Tf32 } else { Precision::Fp32 },
        conv_max_workspace: true, // ORT default (the validated cwRefine setup)
        conv_search: crate::params::conv_search_for(&v.model_name),
    }
}

pub struct RunInfo {
    pub ms: f64,
    pub on_gpu: bool,
}

/// Run the core on one padded crop.
pub fn infer(v: &Values, pixel_values: Vec<f32>, padh: usize, padw: usize) -> Result<(Vec<f32>, RunInfo), String> {
    if v.model_path.is_empty() {
        return Err("Model = Custom Path but Custom Model is empty".into());
    }
    let key = model_key(v);
    let shared = ort_rt::shared_session(&v.ort_lib, &key)?;
    let t0 = std::time::Instant::now();
    let input = Tensor::from_array((vec![1i64, 4, padh as i64, padw as i64], pixel_values)).map_err(|e| format!("tensor: {e}"))?;
    let mut sess = shared.session.lock().map_err(|_| "session lock poisoned".to_string())?;
    let mut ro = RunOptions::new().map_err(|e| format!("run options: {e}"))?;
    if v.free_vram && shared.on_gpu {
        // Shrink the CUDA arena back to its initial chunk at the end of this run.
        ro.add_config_entry("memory.enable_memory_arena_shrinkage", "gpu:0").map_err(|e| format!("run options: {e}"))?;
    }
    let run_err = match sess.run_with_options(ort::inputs!["pixel_values" => input], &ro) {
        Ok(out) => {
            let a = out.get("alphas").ok_or("model has no 'alphas' output (wrong ONNX? need a cwRefine core: pixel_values -> alphas)")?;
            let (shape, data) = a.try_extract_tensor::<f32>().map_err(|e| format!("alphas: {e}"))?;
            if data.len() != padh * padw {
                return Err(format!("alphas shape {shape:?}, expected [1,1,{padh},{padw}]"));
            }
            return Ok((data.to_vec(), RunInfo { ms: t0.elapsed().as_secs_f64() * 1000.0, on_gpu: shared.on_gpu }));
        }
        Err(e) => e.to_string(),
    };
    drop(sess);
    cwtools_core::log::log(&format!("{} {padw}x{padh}: {run_err}", v.model_name));
    if is_oom(&run_err) {
        // Give the whole arena back (Flame needs it); the next render reloads.
        ort_rt::release_path(&v.model_path);
        let remote = if cwtools_remote::plugin::reachable(&v.server, 2.0) { ", or set Device = Remote (server reachable)" } else { "" };
        // MEMatte needs ~1/3 of ViTMatte's VRAM at the same crop.
        let alt = if v.model_name == crate::params::MODELS[crate::params::MODEL_VITMATTE as usize].label { ", set Model = MEMatte" } else { "" };
        return Err(format!(
            "GPU out of memory for a {padw}x{padh} {} crop - lower Max Crop (1536 / 1024){alt} or free VRAM{remote}; model released",
            v.model_name
        ));
    }
    Err(format!("{} run ({padw}x{padh}) failed: {}", v.model_name, run_err.chars().take(160).collect::<String>()))
}

fn is_oom(e: &str) -> bool {
    let l = e.to_lowercase();
    l.contains("failed to allocate memory") || l.contains("out of memory") || l.contains("cudaerrormemoryallocation")
}

// ---------------------------------------------------------------------------
// Result cache: crop-size refined alpha keyed on the exact model input.
// ---------------------------------------------------------------------------
struct Entry {
    key: String,
    alpha: Vec<f32>,
}
static RESULTS: Mutex<Vec<Entry>> = Mutex::new(Vec::new());
const RESULTS_CAP: usize = 12;

pub fn result_key(v: &Values, input_hash: u64) -> String {
    if v.remote {
        // Remote results are their own entries (other GPU = rare 1-level flips).
        let sha = cwtools_remote::model_sha256(&v.model_path).unwrap_or_default();
        return format!("{input_hash:016x}|remote|{}|{sha}|{}", v.server, if v.tf32 { "tf32" } else { "fp32" });
    }
    format!("{input_hash:016x}|{}", model_key(v).describe())
}
pub fn result_get(key: &str) -> Option<Vec<f32>> {
    let mut g = RESULTS.lock().ok()?;
    let i = g.iter().position(|e| e.key == key)?;
    let e = g.remove(i);
    let a = e.alpha.clone();
    g.push(e); // most recent last
    Some(a)
}
pub fn result_put(key: String, alpha: &[f32]) {
    if let Ok(mut g) = RESULTS.lock() {
        g.retain(|e| e.key != key);
        if g.len() >= RESULTS_CAP {
            g.remove(0);
        }
        g.push(Entry { key, alpha: alpha.to_vec() });
    }
}
pub fn clear_results() {
    if let Ok(mut g) = RESULTS.lock() {
        g.clear();
    }
}
