//! Local backend: a BiRefNet ONNX (static [1,3,S,S] fp32 RGB → [1,1,S,S]
//! logits) through Flame's bundled onnxruntime 1.22, via the cwtools-core
//! process-wide session cache (every cwMatte node in the Flame process shares
//! one session per model path + EP; the core's CUDA keeper makes releasing
//! sessions safe).
//!
//! Model metadata (input size, I/O names) comes from the session; the
//! normalisation and sigmoid come from the model's `.json` sidecar (Flame
//! Inference format) when there is one.

use crate::engine::{IMAGENET_MEAN, IMAGENET_STD};
use crate::params::Values;
use cwtools_core::ort_rt::{self, ConvSearch, Ep, ModelKey, Precision, SharedSession};
use ort::session::RunOptions;
use ort::value::{Tensor, ValueType};
use std::sync::{Arc, Mutex};

pub fn model_key(v: &Values) -> ModelKey {
    ModelKey {
        path: v.model_path(),
        ep: if v.use_gpu { Ep::Cuda } else { Ep::Cpu },
        // fp32 GEMMs (the validation recipe ran with TF32 off).
        precision: Precision::Fp32,
        // With ORT's defaults (heuristic, max workspace) cuDNN's pick for
        // BiRefNet's decoder convs wants a 13 GB workspace on a 3090 — it can
        // never fit next to Flame. Default (implicit GEMM) + capped workspace.
        conv_max_workspace: false,
        conv_search: conv_search(),
    }
}

/// Conv algorithm search for BiRefNet (env CWMATTE_CONV=heuristic|exhaustive|default overrides it, for diagnostics).
fn conv_search() -> ConvSearch {
    match std::env::var("CWMATTE_CONV").unwrap_or_default().as_str() {
        "heuristic" => ConvSearch::Heuristic,
        "default" => ConvSearch::Default,
        "exhaustive" => ConvSearch::Exhaustive,
        // Measured on an RTX 3090 (Flame holding 14 GB): Default = 4.7 GB peak
        // at 1024, 0.40 s; Heuristic + Exhaustive both OOM (13 GB request).
        _ => ConvSearch::Default,
    }
}

#[derive(Clone, Debug, PartialEq)]
pub struct Meta {
    pub input: String,
    pub output: String,
    pub sw: usize,
    pub sh: usize,
    pub mean: [f32; 3],
    pub std: [f32; 3],
    /// Some(true) = apply sigmoid, Some(false) = output is already a matte,
    /// None = decide from the values.
    pub sigmoid: Option<bool>,
}

/// Normalisation + sigmoid from a Flame-Inference sidecar JSON.
pub fn parse_sidecar(js: &str) -> Result<([f32; 3], [f32; 3], Option<bool>), String> {
    let v: serde_json::Value = serde_json::from_str(js).map_err(|e| format!("sidecar json: {e}"))?;
    let md = &v["ModelDescription"];
    let inp = &md["Inputs"][0];
    let arr3 = |x: &serde_json::Value| -> Option<[f32; 3]> {
        let a = x.as_array()?;
        (a.len() == 3).then(|| [a[0].as_f64().unwrap_or(0.0) as f32, a[1].as_f64().unwrap_or(0.0) as f32, a[2].as_f64().unwrap_or(1.0) as f32])
    };
    let norm = &inp["Normalization"];
    let (mean, std) = if norm.is_null() {
        ([0.0; 3], [1.0; 3]) // baked normalisation
    } else {
        (arr3(&norm["Mean"]).unwrap_or(IMAGENET_MEAN), arr3(&norm["StandardDeviation"]).unwrap_or(IMAGENET_STD))
    };
    if std.iter().any(|s| *s == 0.0 || !s.is_finite()) {
        return Err("sidecar: zero / invalid StandardDeviation".into());
    }
    let sig = match md["Outputs"][0]["PostProcessing"].as_str() {
        Some(s) if s.eq_ignore_ascii_case("sigmoid") => Some(true),
        Some(_) => Some(false),
        None => None,
    };
    Ok((mean, std, sig))
}

fn sidecar_for(model: &str) -> Option<String> {
    let p = std::path::Path::new(model).with_extension("json");
    std::fs::read_to_string(p).ok()
}

static META: Mutex<Vec<(String, Meta)>> = Mutex::new(Vec::new());

fn meta_for(shared: &SharedSession) -> Result<Meta, String> {
    let path = &shared.key.path;
    if let Some(m) = META.lock().ok().and_then(|g| g.iter().find(|(p, _)| p == path).map(|(_, m)| m.clone())) {
        return Ok(m);
    }
    let sess = shared.session.lock().map_err(|_| "session lock poisoned".to_string())?;
    let inp = sess.inputs.first().ok_or("model has no inputs")?;
    let out = sess.outputs.first().ok_or("model has no outputs")?;
    let (sw, sh) = match &inp.input_type {
        ValueType::Tensor { shape, .. } if shape.len() == 4 => {
            if shape[1] != 3 && shape[1] != -1 {
                return Err(format!("model input '{}' has {} channels, need 3 (RGB)", inp.name, shape[1]));
            }
            let (h, w) = (shape[2], shape[3]);
            if h <= 0 || w <= 0 {
                // Dynamic H/W: BiRefNet's training size.
                (1024, 1024)
            } else {
                (w as usize, h as usize)
            }
        }
        t => return Err(format!("model input '{}' is {t:?}, need a [1,3,S,S] tensor", inp.name)),
    };
    if let ValueType::Tensor { ty, .. } = &inp.input_type {
        if *ty != ort::tensor::TensorElementType::Float32 {
            return Err(format!("model input is {ty:?}; cwMatte feeds float32 (use the *_gs / *_gs_fp16 exports)"));
        }
    }
    let (mean, std, sigmoid) = match sidecar_for(path) {
        Some(js) => parse_sidecar(&js)?,
        None => (IMAGENET_MEAN, IMAGENET_STD, None),
    };
    let m = Meta { input: inp.name.clone(), output: out.name.clone(), sw, sh, mean, std, sigmoid };
    cwtools_core::log::log(&format!("model {path}: input {} {sw}x{sh}, output {}, sigmoid {:?}", m.input, m.output, m.sigmoid));
    if let Ok(mut g) = META.lock() {
        g.retain(|(p, _)| p != path);
        g.push((path.clone(), m.clone()));
    }
    Ok(m)
}

pub struct Loaded {
    pub shared: Arc<SharedSession>,
    pub meta: Meta,
}

/// Load (or reuse) the shared session for the current params.
pub fn load(v: &Values) -> Result<Loaded, String> {
    if let Some(e) = v.gpu_only_error() {
        return Err(e);
    }
    let key = model_key(v);
    if key.path.is_empty() {
        return Err("Model = Custom Path but Custom Model is empty".into());
    }
    let shared = ort_rt::shared_session(&v.ort_lib, &key).map_err(|e| oom_hint(&key.path, e, Some(&v.server)))?;
    let meta = meta_for(&shared)?;
    Ok(Loaded { shared, meta })
}

pub struct RunInfo {
    pub ms: f64,
    pub on_gpu: bool,
}

/// Run the model on one normalised [1,3,sh,sw] tensor → raw output (S×S).
pub fn infer(v: &Values, l: &Loaded, input: Vec<f32>) -> Result<(Vec<f32>, RunInfo), String> {
    let m = &l.meta;
    let t0 = std::time::Instant::now();
    let tensor = Tensor::from_array((vec![1i64, 3, m.sh as i64, m.sw as i64], input)).map_err(|e| format!("tensor: {e}"))?;
    let mut sess = l.shared.session.lock().map_err(|_| "session lock poisoned".to_string())?;
    let mut ro = RunOptions::new().map_err(|e| format!("run options: {e}"))?;
    if v.free_vram && l.shared.on_gpu {
        // Shrink the CUDA arena back to its initial chunk at the end of this run.
        ro.add_config_entry("memory.enable_memory_arena_shrinkage", "gpu:0").map_err(|e| format!("run options: {e}"))?;
    }
    let run_err = match sess.run_with_options(ort::inputs![m.input.as_str() => tensor], &ro) {
        Ok(out) => {
            let a = out.get(m.output.as_str()).ok_or_else(|| format!("model has no '{}' output", m.output))?;
            let (shape, data) = a.try_extract_tensor::<f32>().map_err(|e| format!("output: {e}"))?;
            if data.len() != m.sw * m.sh {
                return Err(format!("output shape {shape:?}, expected [1,1,{},{}]", m.sh, m.sw));
            }
            return Ok((data.to_vec(), RunInfo { ms: t0.elapsed().as_secs_f64() * 1000.0, on_gpu: l.shared.on_gpu }));
        }
        Err(e) => e.to_string(),
    };
    drop(sess);
    cwtools_core::log::log(&format!("BiRefNet {}x{}: {run_err}", m.sw, m.sh));
    let path = l.shared.key.path.clone();
    if is_oom(&run_err) {
        // Give the whole arena + weights back (Flame needs it); the next render reloads.
        ort_rt::release_path(&path);
        return Err(oom_hint(&path, run_err, Some(&v.server)));
    }
    Err(format!("BiRefNet run failed: {}", run_err.chars().take(160).collect::<String>()))
}

pub fn is_oom(e: &str) -> bool {
    let l = e.to_lowercase();
    l.contains("failed to allocate memory")
        || l.contains("out of memory")
        || l.contains("cudaerrormemoryallocation")
        || l.contains("cuda_error_out_of_memory")
        || l.contains("bfc_arena")
}

/// Turn an OOM error into the artist-facing message; other errors pass through.
/// `server`: the node's Remote server; when it answers /health (2 s) the
/// message suggests Device = Remote first (HR models do not fit next to Flame).
fn oom_hint(path: &str, e: String, server: Option<&str>) -> String {
    if !is_oom(&e) {
        return e;
    }
    let remote = match server {
        Some(s) if cwtools_remote::plugin::reachable(s, 2.0) => "set Device = Remote (the server is reachable and has the VRAM), or ",
        _ => "",
    };
    format!(
        "GPU out of memory for {} - {remote}{}free VRAM (Release GPU Memory on other nodes) and re-render; model released",
        std::path::Path::new(path).file_name().map(|n| n.to_string_lossy().into_owned()).unwrap_or_default(),
        smaller_model_hint(path)
    )
}

/// Which smaller Model to suggest after an OOM (empty for the smallest ones).
fn smaller_model_hint(path: &str) -> &'static str {
    let f = std::path::Path::new(path).file_name().map(|n| n.to_string_lossy().into_owned()).unwrap_or_default();
    if f.contains("_HR-") {
        "switch Model to General 1536 / Lite, or "
    } else if f.contains("_1536_") {
        "switch Model to Lite, or "
    } else if f.contains("swin_v1_tiny") {
        "turn Use GPU off (CPU), or "
    } else {
        ""
    }
}

// ---------------------------------------------------------------------------
// Result cache: S×S matte keyed on the exact model input + model.
// ---------------------------------------------------------------------------
struct Entry {
    key: String,
    matte: Arc<Vec<f32>>,
}
static RESULTS: Mutex<Vec<Entry>> = Mutex::new(Vec::new());
/// 2048² f32 = 16 MB per entry.
const RESULTS_CAP: usize = 8;

pub fn result_key(v: &Values, input_hash: u64) -> String {
    if v.remote {
        // Remote results are their own entries (other GPU = small kernel differences).
        let (path, _, sha) = crate::remote::identity(v);
        return format!("{input_hash:016x}|remote|{}|{path}|{}", v.server, sha.unwrap_or_default());
    }
    format!("{input_hash:016x}|{}", model_key(v).describe())
}
pub fn result_get(key: &str) -> Option<Arc<Vec<f32>>> {
    let mut g = RESULTS.lock().ok()?;
    let i = g.iter().position(|e| e.key == key)?;
    let e = g.remove(i);
    let m = e.matte.clone();
    g.push(e); // most recent last
    Some(m)
}
pub fn result_put(key: String, matte: Arc<Vec<f32>>) {
    if let Ok(mut g) = RESULTS.lock() {
        g.retain(|e| e.key != key);
        if g.len() >= RESULTS_CAP {
            g.remove(0);
        }
        g.push(Entry { key, matte });
    }
}
pub fn clear_results() {
    if let Ok(mut g) = RESULTS.lock() {
        g.clear();
    }
    if let Ok(mut g) = META.lock() {
        g.clear();
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn sidecar_birefnet_fp16() {
        let js = r#"{"ModelDescription":{"Inputs":[{"Name":"input_image","Normalization":{"Mean":[0.485,0.456,0.406],
            "StandardDeviation":[0.229,0.224,0.225]}}],"Outputs":[{"Name":"output_image","PostProcessing":"Sigmoid"}]}}"#;
        let (m, s, sig) = parse_sidecar(js).unwrap();
        assert_eq!(m, IMAGENET_MEAN);
        assert_eq!(s, IMAGENET_STD);
        assert_eq!(sig, Some(true));
        let baked = r#"{"ModelDescription":{"Inputs":[{"Name":"x"}],"Outputs":[{"Name":"y","PostProcessing":"None"}]}}"#;
        assert_eq!(parse_sidecar(baked).unwrap(), ([0.0; 3], [1.0; 3], Some(false)));
        assert!(parse_sidecar("nope").is_err());
    }
    #[test]
    fn oom_messages() {
        assert!(is_oom("Failed to allocate memory for requested buffer of size 1234"));
        let m = oom_hint("/m/BiRefNet_HR-matting-epoch_135_gs_fp16.onnx", "CUDA failure 2: out of memory".into(), None);
        assert!(m.contains("General 1536") && m.contains("BiRefNet_HR-matting"), "{m}");
        let m = oom_hint("/m/BiRefNet_dynamic-general-epoch_174_1536_gs_fp16.onnx", "out of memory".into(), None);
        assert!(m.contains("Lite") && !m.contains("General 1536 /"), "{m}");
        let m = oom_hint("/m/BiRefNet-general-bb_swin_v1_tiny-epoch_232_gs.onnx", "out of memory".into(), None);
        assert!(m.contains("Use GPU off"), "{m}");
        let m = oom_hint("/m/custom.onnx", "out of memory".into(), None);
        assert!(m.contains("- free VRAM"), "{m}");
        assert_eq!(oom_hint("/m/x.onnx", "other".into(), None), "other");
    }
}
