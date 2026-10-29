//! Remote backend: a SAM3 server over HTTP (API v1 single mask, API v2 per-instance masks).
//!
//! Binary encoding: JPEG body + `X-SAM3-Request` JSON header, `Accept: image/png`
//! → PNG mask body + `X-SAM3-Result` JSON header. Once a frame_key has been
//! uploaded, later prompt changes on the same frame send an EMPTY body (server
//! keeps the embedding); on HTTP 409 (evicted) we re-send with the image. On a
//! slow uplink this saves most of the time.

use crate::imgutil::{self, Gray, Rgb8};
use crate::objects::Instance;
use crate::prompt::Prompts;
use serde_json::{json, Value};
use std::collections::HashSet;
use std::io::Read;
use std::sync::Mutex;

#[derive(Clone, Debug)]
pub struct RemoteOpts {
    pub server: String,
    pub timeout_s: f64,
    pub best_only: bool,
    pub score_threshold: f64,
    pub jpeg_quality: u8,
}

pub struct RemoteResult {
    pub mask: Gray, // soft probability 0..1 at the sent size
    pub info: Value,
}

/// frame_keys this process has already uploaded (per server URL).
static SENT: Mutex<Option<HashSet<String>>> = Mutex::new(None);

pub fn forget_sent() {
    if let Ok(mut g) = SENT.lock() {
        *g = None;
    }
}
fn was_sent(k: &str) -> bool {
    SENT.lock().map(|g| g.as_ref().map(|s| s.contains(k)).unwrap_or(false)).unwrap_or(false)
}
fn mark_sent(k: &str, sent: bool) {
    if let Ok(mut g) = SENT.lock() {
        let s = g.get_or_insert_with(HashSet::new);
        if sent {
            s.insert(k.to_string());
        } else {
            s.remove(k);
        }
    }
}

use cwtools_core::http::{base, err_str, shared_agent, tmo, JsonBody};

pub fn health(server: &str, timeout_s: f64) -> Result<String, String> {
    if server.trim().is_empty() {
        return Err(crate::params::NO_SERVER_MSG.to_string());
    }
    let r = shared_agent().get(&format!("{}/health", base(server))).timeout(tmo(timeout_s)).call().map_err(err_str)?;
    let v: Value = r.into_json_value()?;
    Ok(format!(
        "model_loaded={} device={} cache={} api={}",
        v["model_loaded"],
        v["device"],
        v["cache"],
        if v["api"].is_null() { "1 (no per-instance masks)".to_string() } else { v["api"].to_string() }
    ))
}

fn request_json(frame_key: &str, img: &Rgb8, p: &Prompts, o: &RemoteOpts) -> Value {
    let mut j = json!({ "frame_key": frame_key, "soft": true });
    if !p.text.is_empty() {
        j["text"] = json!(p.text);
        j["instances"] = json!(if o.best_only { "best" } else { "union" });
        j["score_threshold"] = json!(o.score_threshold);
    }
    if let Some(b) = p.bbox {
        j["box"] = json!([b[0], b[1], b[2], b[3]]);
    }
    if !p.points.is_empty() {
        let pts: Vec<Value> = p.points.iter().map(|(x, y, pos)| json!([x, y, if *pos { 1 } else { 0 }])).collect();
        j["points"] = Value::Array(pts);
    }
    let _ = img;
    j
}

/// POST /segment with the binary encoding. Sends an EMPTY body when this
/// frame_key was uploaded before (server keeps the embedding) and re-sends
/// the JPEG on HTTP 409. Returns (X-SAM3-Result JSON, content type, body,
/// uploaded).
fn post_segment(frame_key: &str, img: &Rgb8, hdr: &str, o: &RemoteOpts) -> Result<(Value, String, Vec<u8>, bool), String> {
    if o.server.trim().is_empty() {
        return Err(crate::params::NO_SERVER_MSG.to_string());
    }
    let url = format!("{}/segment", base(&o.server));
    let ag = shared_agent();
    let send = |body: &[u8]| -> Result<ureq::Response, ureq::Error> {
        ag.post(&url)
            .timeout(tmo(o.timeout_s))
            .set("Content-Type", "image/jpeg")
            .set("Accept", "image/png")
            .set("X-SAM3-Request", hdr)
            .send_bytes(body)
    };
    let mut uploaded = false;
    let resp = if was_sent(frame_key) {
        match send(&[]) {
            Ok(r) => r,
            Err(ureq::Error::Status(409, _)) => {
                mark_sent(frame_key, false);
                let jpg = imgutil::encode_jpeg(img, o.jpeg_quality)?;
                uploaded = true;
                send(&jpg).map_err(err_str)?
            }
            Err(e) => return Err(err_str(e)),
        }
    } else {
        let jpg = imgutil::encode_jpeg(img, o.jpeg_quality)?;
        uploaded = true;
        send(&jpg).map_err(err_str)?
    };
    mark_sent(frame_key, true);
    let info: Value = resp
        .header("X-SAM3-Result")
        .and_then(|h| serde_json::from_str(h).ok())
        .unwrap_or(Value::Null);
    let ctype = resp.content_type().to_string();
    let mut body = Vec::new();
    resp.into_reader()
        .take(256 << 20)
        .read_to_end(&mut body)
        .map_err(|e| format!("read body: {e}"))?;
    Ok((info, ctype, body, uploaded))
}

/// Segment one frame. `img` is the display-referred, downscaled frame (top-down).
pub fn segment(frame_key: &str, img: &Rgb8, p: &Prompts, o: &RemoteOpts) -> Result<RemoteResult, String> {
    let hdr = request_json(frame_key, img, p, o).to_string();
    let t0 = std::time::Instant::now();
    let (info, ctype, body, uploaded) = post_segment(frame_key, img, &hdr, o)?;

    let mask = if ctype.contains("json") {
        // Server answered JSON (no mask path, or ignored Accept).
        let v: Value = serde_json::from_slice(&body).map_err(|e| format!("bad JSON: {e}"))?;
        if v["ok"] == json!(false) {
            return Err(format!("server: {}", v["error"]));
        }
        match v["mask_png"].as_str() {
            Some(b64) => {
                use base64::Engine;
                let png = base64::engine::general_purpose::STANDARD.decode(b64).map_err(|e| e.to_string())?;
                imgutil::decode_png_gray(&png)?
            }
            None => return Err(format!("server returned no mask (path={})", v["path"])),
        }
    } else {
        imgutil::decode_png_gray(&body)?
    };
    if crate::log::verbose() {
        crate::log::log(&format!(
            "remote {} uploaded={} {:.0}ms mask={}x{} info={}",
            frame_key,
            uploaded,
            t0.elapsed().as_secs_f64() * 1000.0,
            mask.w,
            mask.h,
            info
        ));
    }
    Ok(RemoteResult { mask, info })
}

/// API v2: every instance of every concept as its own soft mask (at the sent
/// image size). Needs a v2 server (`instances: "all"`); a v1 server answers
/// `path: encode_only` and this returns a clear error.
pub fn segment_instances(
    frame_key: &str,
    img: &Rgb8,
    concepts: &[String],
    exemplar_box: Option<[f32; 4]>,
    o: &RemoteOpts,
    max_instances: i32,
) -> Result<Vec<Instance>, String> {
    let mut j = json!({
        "frame_key": frame_key,
        "soft": true,
        "instances": "all",
        "concepts": concepts,
        "score_threshold": o.score_threshold,
        "max_instances": max_instances.clamp(1, 128),
    });
    if let Some(b) = exemplar_box {
        j["box"] = json!([b[0], b[1], b[2], b[3]]);
    }
    let hdr = j.to_string();
    let t0 = std::time::Instant::now();
    let (info, ctype, body, uploaded) = post_segment(frame_key, img, &hdr, o)?;
    let (meta, png): (Value, Option<Vec<u8>>) = if ctype.contains("json") {
        let v: Value = serde_json::from_slice(&body).map_err(|e| format!("bad JSON: {e}"))?;
        if v["ok"] == json!(false) {
            return Err(format!("server: {}", v["error"]));
        }
        let png = match v["mask_png"].as_str() {
            Some(b64) => {
                use base64::Engine;
                Some(base64::engine::general_purpose::STANDARD.decode(b64).map_err(|e| e.to_string())?)
            }
            None => None,
        };
        (v, png)
    } else {
        (info, Some(body))
    };
    if meta["path"] != json!("instances") {
        return Err(format!(
            "server has no per-instance mode (path={}; needs a SAM3 server with API v2)",
            meta["path"]
        ));
    }
    let list = meta["instances"].as_array().cloned().unwrap_or_default();
    let mut out = Vec::new();
    if !list.is_empty() {
        let png = png.ok_or("server listed instances but sent no mask strip")?;
        let strip = imgutil::decode_png_gray(&png)?;
        let n = meta["strip"]["n"].as_u64().unwrap_or(list.len() as u64) as usize;
        let h = meta["strip"]["h"].as_u64().map(|v| v as usize).unwrap_or(if n > 0 { strip.h / n } else { 0 });
        let w = strip.w;
        if n != list.len() || h == 0 || strip.h != n * h {
            return Err(format!("bad mask strip {}x{} for {} instances of height {h}", strip.w, strip.h, list.len()));
        }
        for (k, d) in list.iter().enumerate() {
            let bx = d["box"].as_array().map(|a| a.iter().map(|v| v.as_f64().unwrap_or(0.0) as f32).collect::<Vec<_>>()).unwrap_or_default();
            let mask: Vec<u8> = strip.data[k * w * h..(k + 1) * w * h].iter().map(|v| (v * 255.0 + 0.5) as u8).collect();
            out.push(Instance {
                concept: d["concept"].as_str().unwrap_or("?").to_string(),
                concept_index: d["concept_index"].as_u64().unwrap_or(0) as usize,
                score: d["score"].as_f64().unwrap_or(0.0) as f32,
                bbox: [
                    *bx.first().unwrap_or(&0.0),
                    *bx.get(1).unwrap_or(&0.0),
                    *bx.get(2).unwrap_or(&0.0),
                    *bx.get(3).unwrap_or(&0.0),
                ],
                w,
                h,
                mask,
            });
        }
    }
    if crate::log::verbose() {
        crate::log::log(&format!(
            "remote instances {} uploaded={} {:.0}ms n={} concepts={:?} timings={}",
            frame_key,
            uploaded,
            t0.elapsed().as_secs_f64() * 1000.0,
            out.len(),
            concepts,
            meta["timings_ms"]
        ));
    }
    Ok(out)
}
