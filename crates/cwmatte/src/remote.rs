//! Remote backend: the SAME BiRefNet ONNX on the cwtools-infer server
//! (server/API.md). The plugin keeps every pre/post step: it sends exactly
//! the normalised [1,3,S,S] tensor the Local path would feed the model
//! (lossless lut8 from the stretched u8 square, table read back from the
//! tensor), the model name + sha256 and the Local EP options
//! (`Ep::birefnet()`), and gets the raw float32 output back; sigmoid, resize
//! back and the matte knobs run here exactly as Local.
//!
//! Download stays float32: fp16 logits would add up to ~1.1e-4 to the matte
//! (|x|·sigmoid'(x)/2048 peaks at x≈2.4; measured 1.0-1.2e-4), over the
//! 1e-4 budget, so there is no fp16 option.

use crate::engine::{IMAGENET_MEAN, IMAGENET_STD};
use crate::local::{parse_sidecar, Meta};
use crate::params::Values;
use cwtools_core::log;
use cwtools_remote::{self as rc, plugin};

/// Model name on the server + sha256 of this node's model file (the GPU file
/// of the Model entry: Remote always runs on the server's GPU).
pub fn identity(v: &Values) -> (String, String, Result<String, String>) {
    let mut r = v.clone();
    r.remote = true;
    let path = r.model_path();
    (path.clone(), rc::model_name(&path), rc::model_sha256(&path))
}

/// Model metadata from the server's /models (input size / names) + the
/// sidecar (local file next to the model as for Local, else the server's).
pub fn meta(v: &Values) -> Result<(Meta, String, String), String> {
    let (path, name, sha) = identity(v);
    if path.is_empty() {
        return Err("Model = Custom Path but Custom Model is empty".into());
    }
    let sha = sha.map_err(|e| format!("Remote: cannot read this node's model sha256 ({e})"))?;
    let mi = plugin::model_info(&v.server, &name, &sha, v.timeout_s.min(15.0))
        .map_err(|e| plugin::artist_message(&e, &v.server, &name, v.timeout_s.min(15.0)))?;
    if mi.sha256 != sha {
        plugin::forget_models();
        return Err(format!(
            "Remote: the server's {name} is a different file than this node's (sha256 {} vs {}) - install the same model on both sides",
            &mi.sha256[..mi.sha256.len().min(12)],
            &sha[..sha.len().min(12)]
        ));
    }
    if !mi.servable {
        return Err(format!("Remote: the server refuses {name} (state {})", mi.state));
    }
    if mi.in_shape.len() != 4 || mi.in_shape[1] != 3 || mi.in_shape[2] <= 0 || mi.in_shape[3] <= 0 {
        return Err(format!("Remote: {name} input shape {:?}, need a static [1,3,S,S]", mi.in_shape));
    }
    if mi.in_dtype != "float32" {
        return Err(format!("Remote: {name} input is {}; cwMatte feeds float32 (use the *_gs / *_gs_fp16 exports)", mi.in_dtype));
    }
    let (mean, std, sigmoid) = match std::fs::read_to_string(std::path::Path::new(&path).with_extension("json")) {
        Ok(js) => parse_sidecar(&js)?,
        Err(_) if !mi.sidecar.is_null() => parse_sidecar(&mi.sidecar.to_string())?,
        Err(_) => (IMAGENET_MEAN, IMAGENET_STD, None),
    };
    let m = Meta {
        input: mi.input.clone(),
        output: mi.output.clone(),
        sw: mi.in_shape[3] as usize,
        sh: mi.in_shape[2] as usize,
        mean,
        std,
        sigmoid,
    };
    Ok((m, name, sha))
}

/// Run the model on the server: `input` = the normalised tensor, `square` =
/// the u8 RGB square (HWC) it was made from → raw output (S×S, float32).
pub fn infer(v: &Values, m: &Meta, name: &str, sha: &str, input: Vec<f32>, square: &[u8], time: f64) -> Result<Vec<f32>, String> {
    let t0 = std::time::Instant::now();
    let plane = m.sw * m.sh;
    let mut idx = vec![0u8; 3 * plane];
    for i in 0..plane {
        for ch in 0..3 {
            idx[ch * plane + i] = square[i * 3 + ch];
        }
    }
    let shape = [1usize, 3, m.sh, m.sw];
    let t = match rc::TensorIn::lut8_from_source(&m.input, &shape, &input, idx) {
        Ok(t) => t,
        Err(e) => {
            log::log(&format!("remote: lut8 not possible ({e}) - sending float32"));
            rc::TensorIn::f32_auto(&m.input, &shape, input)
        }
    };
    let req = rc::Request {
        model: name.to_string(),
        sha256: sha.to_string(),
        inputs: vec![t],
        ep: Some(rc::Ep::birefnet()),
        outputs: Some(vec![m.output.clone()]),
        output_dtype: None,
        output_quantize: None,
        output_level: None,
        level: rc::DEFAULT_LEVEL,
        timeout_s: v.timeout_s,
        webp: true,
    };
    let r = rc::infer(&v.server, &req).map_err(|e| {
        log::log(&format!("remote {} {name}: {e}", v.server));
        if e.status == 409 {
            plugin::forget_models();
        }
        plugin::artist_message(&e, &v.server, name, v.timeout_s)
    })?;
    let o = r.get(&m.output).ok_or_else(|| format!("Remote: response has no '{}'", m.output))?;
    if o.data.len() != plane {
        return Err(format!("Remote: output has {} values, expected {}x{}", o.data.len(), m.sw, m.sh));
    }
    let tm = &r.timings;
    log::log(&format!(
        "t={time} remote {name} {}x{} total {:.0} ms (encode {:.0} http {:.0} decode {:.0}; up {} KB down {} KB; server run {} load {} queue {} ms)",
        m.sw, m.sh, t0.elapsed().as_secs_f64() * 1e3, tm.encode_ms, tm.http_ms, tm.decode_ms, tm.up_bytes / 1024, tm.down_bytes / 1024,
        tm.server["run"], tm.server["load"], tm.server["queue"]
    ));
    Ok(o.data.clone())
}
