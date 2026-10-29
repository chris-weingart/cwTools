//! Remote backend: the SAME core ONNX (Model choice) on the cwtools-infer server
//! (server/API.md). The plugin keeps every pre/post step: it sends exactly the
//! `pixel_values` tensor the Local path would feed the model (lossless lut8,
//! the table read back from that tensor), the model name + sha256 and the
//! Local EP options (`Ep::vitmatte(tf32)`), and gets the float alphas back
//! (native dtype, lossless). The rest (unpad, Lanczos back, paste) is
//! `engine::finish_f32`, as Local.

use crate::engine::{self, Prepared};
use crate::params::Values;
use cwtools_core::log;
use cwtools_remote::{self as rc, plugin};

/// Model name on the server + sha256 of this node's model file.
pub fn identity(v: &Values) -> (String, Result<String, String>) {
    (rc::model_name(&v.model_path), rc::model_sha256(&v.model_path))
}

/// Refine one prepared crop on the server → crop-size float alpha.
pub fn infer(v: &Values, prep: &Prepared, time: f64) -> Result<Vec<f32>, String> {
    let p = prep.plan;
    if v.model_path.is_empty() {
        return Err("Model = Custom Path but Custom Model is empty".into());
    }
    let (name, sha) = identity(v);
    let sha = sha.map_err(|e| format!("Remote: cannot read this node's model sha256 ({e})"))?;
    let t0 = std::time::Instant::now();
    let pv = prep.pixel_values();
    let (padw, padh) = (p.padw, p.padh);
    let np = padw * padh;
    // u8 source planes of pixel_values: display RGB (HWC -> CHW) + trimap.
    let mut idx = vec![0u8; 4 * np];
    for i in 0..np {
        for ch in 0..3 {
            idx[ch * np + i] = prep.img_pad[i * 3 + ch];
        }
        idx[3 * np + i] = prep.tri_pad[i];
    }
    let shape = [1usize, 4, padh, padw];
    let input = match rc::TensorIn::lut8_from_source("pixel_values", &shape, &pv, idx) {
        Ok(t) => t,
        Err(e) => {
            log::log(&format!("remote: lut8 not possible ({e}) - sending float32"));
            rc::TensorIn::f32_auto("pixel_values", &shape, pv)
        }
    };
    let req = rc::Request {
        model: name.clone(),
        sha256: sha,
        inputs: vec![input],
        ep: Some(rc::Ep::vitmatte(v.tf32)),
        outputs: Some(vec!["alphas".into()]),
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
        plugin::artist_message(&e, &v.server, &name, v.timeout_s)
    })?;
    let a = &r.get("alphas").ok_or("Remote: response has no 'alphas'")?.data;
    if a.len() != np {
        return Err(format!("Remote: alphas has {} values, expected {padw}x{padh}", a.len()));
    }
    let crop = engine::finish_f32(&p, a)?;
    let tm = &r.timings;
    log::log(&format!(
        "t={time} remote crop {}x{} pad {padw}x{padh} {} total {:.0} ms (encode {:.0} http {:.0} decode {:.0}; up {} KB down {} KB; server run {} load {} queue {} ms)",
        p.cw(), p.ch(), if v.tf32 { "tf32" } else { "fp32" },
        t0.elapsed().as_secs_f64() * 1e3, tm.encode_ms, tm.http_ms, tm.decode_ms, tm.up_bytes / 1024, tm.down_bytes / 1024,
        tm.server["run"], tm.server["load"], tm.server["queue"]
    ));
    Ok(crop)
}
