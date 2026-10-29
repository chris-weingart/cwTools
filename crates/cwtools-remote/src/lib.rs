//! Client for the cwtools-infer server (Remote mode of cwRefine / cwMatte).
//!
//! The plugin keeps all pre/post-processing and sends exactly the tensors it
//! would feed the model locally, in the lossless CWT1 container
//! (server/API.md); the server runs the same ONNX file (checked by sha256)
//! with onnxruntime 1.22 and the same CUDA EP options and returns the raw
//! model outputs. Remote == Local by construction, up to GPU kernel
//! differences.
//!
//! Wire format: `b"CWT1" | u32 LE header len | header JSON | blob area`.
//! Codecs (all lossless; zstd frames carry a content checksum):
//! - `lut8-grad-zstd`: per channel (axis 1) a 256-entry value table + uint8
//!   indices, 2-D gradient filtered (dx then dy, mod 256), zstd. For tensors
//!   that are per-channel functions of 8-bit pixels (both plugins' inputs).
//! - `xor-shuf-zstd`: floats, XOR with the left neighbour (bit pattern),
//!   byte-shuffle, zstd. For model outputs.
//! - `zstd`, `raw`.

pub mod plugin;

use serde_json::{json, Value};
use std::io::Read;
use std::time::{Duration, Instant};

pub const MAGIC: &[u8; 4] = b"CWT1";
pub const CONTENT_TYPE: &str = "application/x-cwtensor";
/// zstd level for uploads: 3 is ~200 MB/s and within ~5% of level 9 on plates.
pub const DEFAULT_LEVEL: i32 = 3;

// ---------------------------------------------------------------------------
// Tensors
// ---------------------------------------------------------------------------

/// Input tensor payload.
pub enum Data {
    /// float32, sent `xor-shuf-zstd` (lossless, any values).
    F32(Vec<f32>),
    /// float32 as per-channel LUT + u8 indices (`lut8-grad-zstd`).
    /// `idx` is NCHW (same layout as the tensor), `lut[c][k]` the value of index k in channel c.
    Lut8 { idx: Vec<u8>, lut: Vec<[f32; 256]> },
    /// uint8 tensor, sent `zstd`.
    U8(Vec<u8>),
}

pub struct TensorIn {
    pub name: String,
    pub shape: Vec<usize>,
    pub data: Data,
}

impl TensorIn {
    /// Float tensor, auto-packed as `lut8` when every channel (axis 1) takes
    /// at most 256 distinct values (bit-exact: the table holds the exact f32
    /// bit patterns), else `xor-shuf-zstd`.
    pub fn f32_auto(name: &str, shape: &[usize], data: Vec<f32>) -> TensorIn {
        match lut8_from_f32(shape, &data) {
            Some((idx, lut)) => TensorIn { name: name.into(), shape: shape.to_vec(), data: Data::Lut8 { idx, lut } },
            None => TensorIn { name: name.into(), shape: shape.to_vec(), data: Data::F32(data) },
        }
    }
    /// Float tensor that the plugin computed per channel from 8-bit planes
    /// (`idx`, NCHW, = the u8 pixel values the tensor was made from). The
    /// value table is read back FROM THE TENSOR and every element is checked
    /// against it, so the server's `lut[idx]` is bit-identical to `data` by
    /// construction (no normalisation formula is duplicated). Using the real
    /// pixel values as indices keeps the planes natural images, which the
    /// WebP / gradient coders compress well. Err if some index maps to two
    /// different values (then send `f32_auto`).
    pub fn lut8_from_source(name: &str, shape: &[usize], data: &[f32], idx: Vec<u8>) -> Result<TensorIn, String> {
        if shape.len() < 3 || data.len() != shape.iter().product::<usize>() || idx.len() != data.len() {
            return Err(format!("{name}: lut8 needs NCHW data and indices of the same size"));
        }
        let (n, c) = (shape[0], shape[1]);
        let plane: usize = shape[2..].iter().product();
        let mut lut = vec![[0f32; 256]; c];
        for ch in 0..c {
            let mut set = [false; 256];
            for b in 0..n {
                let base = (b * c + ch) * plane;
                for i in base..base + plane {
                    let k = idx[i] as usize;
                    if !set[k] {
                        set[k] = true;
                        lut[ch][k] = data[i];
                    } else if lut[ch][k].to_bits() != data[i].to_bits() {
                        return Err(format!("{name}: channel {ch} index {k} maps to two values"));
                    }
                }
            }
        }
        Ok(TensorIn { name: name.into(), shape: shape.to_vec(), data: Data::Lut8 { idx, lut } })
    }

    pub fn numel(&self) -> usize {
        self.shape.iter().product()
    }
}

/// Split a float NCHW-like tensor (channel axis 1) into per-channel LUTs +
/// u8 indices. None if some channel has more than 256 distinct values.
pub fn lut8_from_f32(shape: &[usize], data: &[f32]) -> Option<(Vec<u8>, Vec<[f32; 256]>)> {
    if shape.len() < 3 || data.len() != shape.iter().product::<usize>() {
        return None;
    }
    let (n, c) = (shape[0], shape[1]);
    let plane: usize = shape[2..].iter().product();
    let mut idx = vec![0u8; data.len()];
    let mut luts = vec![[0f32; 256]; c];
    for ch in 0..c {
        // open-addressing table of f32 bit patterns -> index (<= 256 entries)
        let mut keys = [u32::MAX; 1024];
        let mut vals = [0u8; 1024];
        let mut used = [false; 1024];
        let mut count = 0usize;
        let mut last = (u32::MAX, 0u8, false);
        for b in 0..n {
            let base = (b * c + ch) * plane;
            for i in base..base + plane {
                let bits = data[i].to_bits();
                if last.2 && last.0 == bits {
                    idx[i] = last.1;
                    continue;
                }
                let mut h = (bits.wrapping_mul(0x9E37_79B1) >> 22) as usize;
                let k = loop {
                    if !used[h] {
                        if count == 256 {
                            return None;
                        }
                        used[h] = true;
                        keys[h] = bits;
                        vals[h] = count as u8;
                        luts[ch][count] = data[i];
                        count += 1;
                        break vals[h];
                    }
                    if keys[h] == bits {
                        break vals[h];
                    }
                    h = (h + 1) & 1023;
                };
                idx[i] = k;
                last = (bits, k, true);
            }
        }
        // Renumber by ascending value: for a monotone normalisation of 8-bit
        // pixels the index is then (a rank of) the pixel value, which keeps
        // neighbouring pixels close for the gradient filter + zstd.
        let mut order: Vec<usize> = (0..count).collect();
        order.sort_by(|&a, &b| luts[ch][a].total_cmp(&luts[ch][b]));
        let mut remap = [0u8; 256];
        let mut sorted = [0f32; 256];
        for (rank, &old) in order.iter().enumerate() {
            remap[old] = rank as u8;
            sorted[rank] = luts[ch][old];
        }
        luts[ch] = sorted;
        for b in 0..n {
            let base = (b * c + ch) * plane;
            for v in &mut idx[base..base + plane] {
                *v = remap[*v as usize];
            }
        }
    }
    Some((idx, luts))
}

/// Model output (float outputs are returned as f32; float16 is widened).
#[derive(Debug, Clone)]
pub struct TensorOut {
    pub name: String,
    pub shape: Vec<usize>,
    pub dtype: String,
    /// float outputs (float16 widened); empty for uint8 outputs
    pub data: Vec<f32>,
    /// uint8 outputs (`output_quantize`); empty otherwise
    pub u8: Vec<u8>,
}

// ---------------------------------------------------------------------------
// Codecs
// ---------------------------------------------------------------------------

fn zstd_c(raw: &[u8], level: i32) -> Result<Vec<u8>, String> {
    let mut c = zstd::bulk::Compressor::new(level).map_err(|e| format!("zstd: {e}"))?;
    c.set_parameter(zstd::zstd_safe::CParameter::ChecksumFlag(true)).map_err(|e| format!("zstd: {e}"))?;
    c.include_contentsize(true).map_err(|e| format!("zstd: {e}"))?;
    c.compress(raw).map_err(|e| format!("zstd: {e}"))
}

fn zstd_d(z: &[u8], size: usize) -> Result<Vec<u8>, String> {
    let out = zstd::bulk::decompress(z, size).map_err(|e| format!("zstd decode: {e}"))?;
    if out.len() != size {
        return Err(format!("zstd decode: {} bytes, expected {size}", out.len()));
    }
    Ok(out)
}

/// 2-D gradient filter (dx then dy, wrapping) on every `h × w` plane.
fn grad_enc(idx: &[u8], h: usize, w: usize) -> Vec<u8> {
    let mut out = vec![0u8; idx.len()];
    for (pi, plane) in idx.chunks_exact(h * w).enumerate() {
        let o = &mut out[pi * h * w..(pi + 1) * h * w];
        for y in 0..h {
            for x in 0..w {
                let v = plane[y * w + x];
                let l = if x > 0 { plane[y * w + x - 1] } else { 0 };
                let (u, ul) = if y > 0 {
                    (plane[(y - 1) * w + x], if x > 0 { plane[(y - 1) * w + x - 1] } else { 0 })
                } else {
                    (0, 0)
                };
                // e = (v - l) - (u - ul)
                o[y * w + x] = v.wrapping_sub(l).wrapping_sub(u.wrapping_sub(ul));
            }
        }
    }
    out
}

fn xor_shuf_f32(data: &[f32], row: usize) -> Vec<u8> {
    let n = data.len();
    let mut out = vec![0u8; n * 4];
    for (r, chunk) in data.chunks(row.max(1)).enumerate() {
        let mut prev = 0u32;
        for (j, v) in chunk.iter().enumerate() {
            let b = v.to_bits();
            let x = b ^ prev;
            prev = b;
            let i = r * row + j;
            let le = x.to_le_bytes();
            out[i] = le[0];
            out[n + i] = le[1];
            out[2 * n + i] = le[2];
            out[3 * n + i] = le[3];
        }
    }
    out
}

fn xor_unshuf(raw: &[u8], isz: usize, row: usize) -> Vec<u64> {
    let n = raw.len() / isz;
    let mut out = vec![0u64; n];
    let row = row.max(1);
    let mut prev = 0u64;
    for i in 0..n {
        if i % row == 0 {
            prev = 0;
        }
        let mut x = 0u64;
        for k in 0..isz {
            x |= (raw[k * n + i] as u64) << (8 * k);
        }
        let v = x ^ prev;
        prev = v;
        out[i] = v;
    }
    out
}

fn f16_to_f32(h: u16) -> f32 {
    let s = ((h >> 15) as u32) << 31;
    let e = ((h >> 10) & 0x1f) as u32;
    let m = (h & 0x3ff) as u32;
    let bits = if e == 0 {
        if m == 0 {
            s
        } else {
            // subnormal
            let mut e2 = 127 - 15 + 1;
            let mut m2 = m;
            while m2 & 0x400 == 0 {
                m2 <<= 1;
                e2 -= 1;
            }
            s | ((e2 as u32) << 23) | ((m2 & 0x3ff) << 13)
        }
    } else if e == 31 {
        s | 0x7f80_0000 | (m << 13)
    } else {
        s | ((e + 127 - 15) << 23) | (m << 13)
    };
    f32::from_bits(bits)
}

/// Lossless WebP (method 0 = fastest: ~150 ms for a 2K frame, ~20% smaller
/// than gradient + zstd on plates).
fn webp_lossless(rgb: &[u8], w: usize, h: usize) -> Result<Vec<u8>, String> {
    let mut cfg = webp::WebPConfig::new().map_err(|_| "webp config".to_string())?;
    cfg.lossless = 1;
    cfg.method = 0;
    cfg.quality = 0.0;
    cfg.exact = 1;
    let enc = webp::Encoder::from_rgb(rgb, w as u32, h as u32);
    let mem = enc.encode_advanced(&cfg).map_err(|e| format!("webp encode: {e:?}"))?;
    Ok(mem.to_vec())
}

struct Blobs {
    area: Vec<u8>,
}
impl Blobs {
    fn push(&mut self, b: &[u8]) -> Value {
        let r = json!([self.area.len(), b.len()]);
        self.area.extend_from_slice(b);
        r
    }
}

fn encode_tensor(t: &TensorIn, blobs: &mut Blobs, level: i32, webp: bool) -> Result<Value, String> {
    if t.shape.is_empty() {
        return Err(format!("{}: scalar tensors are not supported", t.name));
    }
    let n = t.numel();
    let mut d = json!({"name": t.name, "shape": t.shape});
    match &t.data {
        Data::F32(v) => {
            if v.len() != n {
                return Err(format!("{}: {} values for shape {:?}", t.name, v.len(), t.shape));
            }
            d["dtype"] = json!("float32");
            d["codec"] = json!("xor-shuf-zstd");
            d["data"] = blobs.push(&zstd_c(&xor_shuf_f32(v, *t.shape.last().unwrap()), level)?);
        }
        Data::U8(v) => {
            if v.len() != n {
                return Err(format!("{}: {} values for shape {:?}", t.name, v.len(), t.shape));
            }
            d["dtype"] = json!("uint8");
            d["codec"] = json!("zstd");
            d["data"] = blobs.push(&zstd_c(v, level)?);
        }
        Data::Lut8 { idx, lut } => {
            if idx.len() != n || t.shape.len() < 3 || lut.len() != t.shape[1] {
                return Err(format!("{}: lut8 needs idx of {n} values and {} tables", t.name, t.shape.get(1).unwrap_or(&0)));
            }
            let c = t.shape[1];
            let (h, w) = (t.shape[t.shape.len() - 2], t.shape[t.shape.len() - 1]);
            let mut lb = Vec::with_capacity(lut.len() * 1024);
            for l in lut {
                for v in l {
                    lb.extend_from_slice(&v.to_le_bytes());
                }
            }
            d["dtype"] = json!("float32");
            d["lut"] = blobs.push(&lb);
            d["crc32"] = json!(crc32fast::hash(idx));
            let plane = h * w;
            if webp && t.shape.len() == 4 && t.shape[0] == 1 && c >= 3 && w <= 16383 && h <= 16383 {
                // channels 0..3 as one lossless WebP RGB image, the rest grad-zstd
                let mut rgb = vec![0u8; plane * 3];
                for i in 0..plane {
                    rgb[i * 3] = idx[i];
                    rgb[i * 3 + 1] = idx[plane + i];
                    rgb[i * 3 + 2] = idx[2 * plane + i];
                }
                d["codec"] = json!("lut8-webp");
                d["webp"] = blobs.push(&webp_lossless(&rgb, w, h)?);
                if c > 3 {
                    d["data"] = blobs.push(&zstd_c(&grad_enc(&idx[3 * plane..], h, w), level)?);
                }
            } else {
                d["codec"] = json!("lut8-grad-zstd");
                let x = if c >= 3 {
                    // cross-channel decorrelation (c0 - c1, c1, c2 - c1); per batch item
                    let mut x = idx.clone();
                    for b in 0..t.shape[0] {
                        let base = b * c * plane;
                        for i in 0..plane {
                            let g = idx[base + plane + i];
                            x[base + i] = idx[base + i].wrapping_sub(g);
                            x[base + 2 * plane + i] = idx[base + 2 * plane + i].wrapping_sub(g);
                        }
                    }
                    d["xform"] = json!("gdiff");
                    x
                } else {
                    idx.clone()
                };
                d["data"] = blobs.push(&zstd_c(&grad_enc(&x, h, w), level)?);
            }
        }
    }
    Ok(d)
}

fn decode_tensor(d: &Value, area: &[u8]) -> Result<TensorOut, String> {
    let name = d["name"].as_str().unwrap_or("?").to_string();
    let shape: Vec<usize> = d["shape"].as_array().ok_or("output: no shape")?.iter().map(|x| x.as_u64().unwrap_or(0) as usize).collect();
    let dtype = d["dtype"].as_str().unwrap_or("").to_string();
    let codec = d["codec"].as_str().unwrap_or("raw");
    let n: usize = shape.iter().product();
    let isz = match dtype.as_str() {
        "float32" => 4,
        "float16" => 2,
        "uint8" => 1,
        t => return Err(format!("{name}: unsupported output dtype {t}")),
    };
    let blob = |k: &str| -> Result<&[u8], String> {
        let r = d[k].as_array().ok_or(format!("{name}: no blob {k}"))?;
        let (o, l) = (r[0].as_u64().unwrap_or(0) as usize, r[1].as_u64().unwrap_or(0) as usize);
        area.get(o..o + l).ok_or(format!("{name}: blob {k} out of range"))
    };
    let row = *shape.last().unwrap_or(&1);
    if codec == "grad-zstd" {
        if isz != 1 || shape.len() < 2 {
            return Err(format!("{name}: grad-zstd needs a uint8 tensor with H, W"));
        }
        let (h, w) = (shape[shape.len() - 2], shape[shape.len() - 1]);
        let mut v = zstd_d(blob("data")?, n)?;
        grad_dec(&mut v, h, w);
        return Ok(TensorOut { name, shape, dtype, data: Vec::new(), u8: v });
    }
    let bits: Vec<u64> = match codec {
        "raw" | "zstd" => {
            let raw = if codec == "raw" { blob("data")?.to_vec() } else { zstd_d(blob("data")?, n * isz)? };
            if raw.len() != n * isz {
                return Err(format!("{name}: {} bytes, expected {}", raw.len(), n * isz));
            }
            raw.chunks_exact(isz).map(|c| c.iter().rev().fold(0u64, |a, &b| (a << 8) | b as u64)).collect()
        }
        "xor-shuf-zstd" => xor_unshuf(&zstd_d(blob("data")?, n * isz)?, isz, row),
        c => return Err(format!("{name}: unsupported output codec {c}")),
    };
    let data = match isz {
        4 => bits.iter().map(|&b| f32::from_bits(b as u32)).collect(),
        2 => bits.iter().map(|&b| f16_to_f32(b as u16)).collect(),
        _ => bits.iter().map(|&b| b as f32).collect(),
    };
    if isz == 1 {
        return Ok(TensorOut { name, shape, dtype, data: Vec::new(), u8: bits.iter().map(|&b| b as u8).collect() });
    }
    Ok(TensorOut { name, shape, dtype, data, u8: Vec::new() })
}

/// Inverse of `grad_enc`: cumulative sums over y then x (wrapping), in place.
fn grad_dec(v: &mut [u8], h: usize, w: usize) {
    for p in v.chunks_exact_mut(h * w) {
        for y in 1..h {
            for x in 0..w {
                p[y * w + x] = p[y * w + x].wrapping_add(p[(y - 1) * w + x]);
            }
        }
        for y in 0..h {
            for x in 1..w {
                p[y * w + x] = p[y * w + x].wrapping_add(p[y * w + x - 1]);
            }
        }
    }
}

/// Build a CWT1 body.
pub fn pack(header: &Value, area: &[u8]) -> Vec<u8> {
    let hj = serde_json::to_vec(header).unwrap_or_default();
    let mut out = Vec::with_capacity(8 + hj.len() + area.len());
    out.extend_from_slice(MAGIC);
    out.extend_from_slice(&(hj.len() as u32).to_le_bytes());
    out.extend_from_slice(&hj);
    out.extend_from_slice(area);
    out
}

/// Split a CWT1 body into header + blob area.
pub fn unpack(body: &[u8]) -> Result<(Value, &[u8]), String> {
    if body.len() < 8 || &body[..4] != MAGIC {
        return Err("response is not a CWT1 container".into());
    }
    let hl = u32::from_le_bytes([body[4], body[5], body[6], body[7]]) as usize;
    let hj = body.get(8..8 + hl).ok_or("truncated CWT1 header")?;
    let h: Value = serde_json::from_slice(hj).map_err(|e| format!("CWT1 header: {e}"))?;
    Ok((h, &body[8 + hl..]))
}

// ---------------------------------------------------------------------------
// Requests
// ---------------------------------------------------------------------------

/// CUDA EP options; part of the server's session key. Must equal what the
/// plugin uses locally for the same model (cwrefine/cwmatte `local::model_key`).
#[derive(Clone, Debug, PartialEq)]
pub struct Ep {
    /// "fp32" | "tf32"
    pub precision: &'static str,
    /// "heuristic" | "default" | "exhaustive"
    pub conv_search: &'static str,
    pub conv_max_workspace: bool,
}
impl Ep {
    /// cwRefine (ViTMatte core): heuristic search, max workspace, fp32 / tf32.
    pub fn vitmatte(tf32: bool) -> Ep {
        Ep { precision: if tf32 { "tf32" } else { "fp32" }, conv_search: "heuristic", conv_max_workspace: true }
    }
    /// cwMatte (BiRefNet): default (implicit GEMM) search, capped workspace, fp32 math.
    pub fn birefnet() -> Ep {
        Ep { precision: "fp32", conv_search: "default", conv_max_workspace: false }
    }
}

pub struct Request {
    /// Model name = ONNX file stem on the server (GET /models).
    pub model: String,
    /// sha256 of the model file the client would run locally. A different
    /// file on the server is an error (HTTP 409 sha_mismatch), never a silent difference.
    pub sha256: String,
    pub inputs: Vec<TensorIn>,
    pub ep: Option<Ep>,
    /// Output names to return (None = all).
    pub outputs: Option<Vec<String>>,
    /// Some("float16") = halve the download (LOSSY); None = native dtype (lossless).
    pub output_dtype: Option<&'static str>,
    /// Some("u8_trunc255") = the server returns float outputs as uint8
    /// `(x * 255).clamp(0, 255) as u8` (= cwRefine's alpha_to_u8, bit-identical);
    /// ~10x smaller download. Read them from `TensorOut::u8`.
    pub output_quantize: Option<&'static str>,
    /// zstd level the server uses for the outputs (None = server default 3).
    pub output_level: Option<i32>,
    pub level: i32,
    pub timeout_s: f64,
    /// Send lut8 tensors' first 3 channels as lossless WebP (default true;
    /// false = gradient + zstd, ~20% bigger, ~3x faster to encode).
    pub webp: bool,
}

#[derive(Debug, Clone, Default)]
pub struct Timings {
    pub encode_ms: f64,
    pub http_ms: f64,
    pub decode_ms: f64,
    pub up_bytes: usize,
    pub down_bytes: usize,
    /// Server-side timings (decode / queue / load / run / encode / total), ms.
    pub server: Value,
}

pub struct Response {
    pub outputs: Vec<TensorOut>,
    pub header: Value,
    pub timings: Timings,
}

impl Response {
    pub fn get(&self, name: &str) -> Option<&TensorOut> {
        self.outputs.iter().find(|o| o.name == name)
    }
}

/// Error from the server or transport. `code` = server error code
/// ("sha_mismatch", "gpu_oom", "busy", "nonfinite_output", ...) or "transport".
#[derive(Debug, Clone)]
pub struct RemoteError {
    pub status: u16,
    pub code: String,
    pub msg: String,
}
impl std::fmt::Display for RemoteError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        if self.status > 0 {
            write!(f, "server {} {}: {}", self.status, self.code, self.msg)
        } else {
            write!(f, "{}: {}", self.code, self.msg)
        }
    }
}

fn local_err(code: &str, msg: impl Into<String>) -> RemoteError {
    RemoteError { status: 0, code: code.into(), msg: msg.into() }
}

pub fn agent() -> &'static ureq::Agent {
    static A: std::sync::OnceLock<ureq::Agent> = std::sync::OnceLock::new();
    A.get_or_init(|| {
        ureq::AgentBuilder::new()
            .timeout_connect(Duration::from_secs(4))
            .max_idle_connections_per_host(4)
            .build()
    })
}

/// Empty server (nothing configured): fail before any network call.
fn no_server(server: &str) -> Result<(), RemoteError> {
    if server.trim().is_empty() {
        Err(local_err("no_server", "no server configured"))
    } else {
        Ok(())
    }
}

fn base(url: &str) -> String {
    url.trim().trim_end_matches('/').to_string()
}

fn status_err(e: ureq::Error) -> RemoteError {
    match e {
        ureq::Error::Status(code, r) => {
            let body = r.into_string().unwrap_or_default();
            let v: Value = serde_json::from_str(&body).unwrap_or(Value::Null);
            RemoteError {
                status: code,
                code: v["code"].as_str().unwrap_or("http").to_string(),
                msg: v["error"].as_str().map(|s| s.to_string()).unwrap_or_else(|| body.chars().take(200).collect()),
            }
        }
        ureq::Error::Transport(t) => transport_err(&t),
    }
}

/// Classify a transport failure: "timeout" (no answer in time), "unreachable"
/// (connect / DNS failed), else "transport".
fn transport_err(t: &ureq::Transport) -> RemoteError {
    let msg = t.to_string();
    let io_timeout = std::error::Error::source(t)
        .and_then(|s| s.downcast_ref::<std::io::Error>())
        .map(|e| matches!(e.kind(), std::io::ErrorKind::TimedOut | std::io::ErrorKind::WouldBlock))
        .unwrap_or(false);
    // Connect failures first: a connect timeout means the host is not there
    // (VPN down / wrong IP), not that the server is slow.
    let code = if matches!(t.kind(), ureq::ErrorKind::ConnectionFailed | ureq::ErrorKind::Dns) {
        "unreachable"
    } else if io_timeout || msg.contains("timed out") {
        "timeout"
    } else {
        "transport"
    };
    local_err(code, msg)
}

fn io_err(e: std::io::Error) -> RemoteError {
    let code = if matches!(e.kind(), std::io::ErrorKind::TimedOut | std::io::ErrorKind::WouldBlock) { "timeout" } else { "transport" };
    local_err(code, e.to_string())
}

/// GET `/health` or `/models`.
pub fn get_json(server: &str, path: &str, timeout_s: f64) -> Result<Value, RemoteError> {
    no_server(server)?;
    let r = agent()
        .get(&format!("{}{path}", base(server)))
        .timeout(Duration::from_secs_f64(timeout_s.max(0.5)))
        .call()
        .map_err(status_err)?;
    let s = r.into_string().map_err(io_err)?;
    serde_json::from_str(&s).map_err(|e| local_err("bad_json", e.to_string()))
}

/// POST `/infer`.
pub fn infer(server: &str, req: &Request) -> Result<Response, RemoteError> {
    no_server(server)?;
    let t0 = Instant::now();
    let mut blobs = Blobs { area: Vec::new() };
    let mut descs = Vec::new();
    for t in &req.inputs {
        descs.push(encode_tensor(t, &mut blobs, req.level, req.webp).map_err(|e| local_err("encode", e))?);
    }
    let mut h = json!({"model": req.model, "sha256": req.sha256, "inputs": descs});
    if let Some(ep) = &req.ep {
        h["ep"] = json!({"precision": ep.precision, "conv_search": ep.conv_search, "conv_max_workspace": ep.conv_max_workspace});
    }
    if let Some(o) = &req.outputs {
        h["outputs"] = json!(o);
    }
    if let Some(d) = req.output_dtype {
        h["output_dtype"] = json!(d);
    }
    if let Some(q) = req.output_quantize {
        h["output_quantize"] = json!(q);
    }
    if let Some(l) = req.output_level {
        h["output_level"] = json!(l);
    }
    let body = pack(&h, &blobs.area);
    drop(blobs);
    let t1 = Instant::now();
    let r = agent()
        .post(&format!("{}/infer", base(server)))
        .timeout(Duration::from_secs_f64(req.timeout_s.max(1.0)))
        .set("Content-Type", CONTENT_TYPE)
        .send_bytes(&body)
        .map_err(status_err)?;
    let mut out = Vec::new();
    r.into_reader()
        .take(1 << 31)
        .read_to_end(&mut out)
        .map_err(io_err)?;
    let t2 = Instant::now();
    let (hdr, area) = unpack(&out).map_err(|e| local_err("bad_response", e))?;
    if hdr["sha256"].as_str() != Some(req.sha256.as_str()) {
        return Err(local_err("sha_mismatch", "response is for a different model file"));
    }
    let mut outputs = Vec::new();
    for d in hdr["outputs"].as_array().ok_or_else(|| local_err("bad_response", "no outputs"))? {
        let t = decode_tensor(d, area).map_err(|e| local_err("bad_response", e))?;
        if t.data.iter().any(|v| !v.is_finite()) {
            return Err(local_err("nonfinite_output", format!("{}: NaN/Inf in the response", t.name)));
        }
        outputs.push(t);
    }
    let t3 = Instant::now();
    let ms = |a: Instant, b: Instant| (b - a).as_secs_f64() * 1e3;
    let timings = Timings {
        encode_ms: ms(t0, t1),
        http_ms: ms(t1, t2),
        decode_ms: ms(t2, t3),
        up_bytes: body.len(),
        down_bytes: out.len(),
        server: hdr["timings_ms"].clone(),
    };
    Ok(Response { outputs, header: hdr, timings })
}

// ---------------------------------------------------------------------------
// Model identity
// ---------------------------------------------------------------------------

/// sha256 of a model file: from a `SHA256SUMS` next to it when it lists the
/// file (the delivery manifests), else hashed (cached per path+size+mtime).
pub fn model_sha256(path: &str) -> Result<String, String> {
    let p = std::path::Path::new(path);
    let fname = p.file_name().and_then(|n| n.to_str()).ok_or("bad model path")?;
    if let Some(dir) = p.parent() {
        if let Ok(s) = std::fs::read_to_string(dir.join("SHA256SUMS")) {
            for line in s.lines() {
                let mut it = line.split_whitespace();
                if let (Some(h), Some(f)) = (it.next(), it.next()) {
                    if f.trim_start_matches('*') == fname && h.len() == 64 {
                        return Ok(h.to_lowercase());
                    }
                }
            }
        }
    }
    sha256_file(path)
}

/// sha256 of a file (cached per path + size + mtime for the process).
pub fn sha256_file(path: &str) -> Result<String, String> {
    use sha2::{Digest, Sha256};
    static CACHE: std::sync::Mutex<Vec<(String, u64, i128, String)>> = std::sync::Mutex::new(Vec::new());
    let md = std::fs::metadata(path).map_err(|e| format!("{path}: {e}"))?;
    let mt = md
        .modified()
        .ok()
        .and_then(|t| t.duration_since(std::time::UNIX_EPOCH).ok())
        .map(|d| d.as_nanos() as i128)
        .unwrap_or(0);
    if let Ok(g) = CACHE.lock() {
        if let Some(e) = g.iter().find(|e| e.0 == path && e.1 == md.len() && e.2 == mt) {
            return Ok(e.3.clone());
        }
    }
    let mut f = std::fs::File::open(path).map_err(|e| format!("{path}: {e}"))?;
    let mut h = Sha256::new();
    let mut buf = vec![0u8; 1 << 20];
    loop {
        let n = f.read(&mut buf).map_err(|e| format!("{path}: {e}"))?;
        if n == 0 {
            break;
        }
        h.update(&buf[..n]);
    }
    let hex: String = h.finalize().iter().map(|b| format!("{b:02x}")).collect();
    if let Ok(mut g) = CACHE.lock() {
        g.push((path.to_string(), md.len(), mt, hex.clone()));
    }
    Ok(hex)
}

/// Model name on the server for a local model path (file stem).
pub fn model_name(path: &str) -> String {
    std::path::Path::new(path).file_stem().map(|s| s.to_string_lossy().into_owned()).unwrap_or_default()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn lut8_roundtrip_and_limits() {
        let shape = [1usize, 2, 3, 4];
        let data: Vec<f32> = (0..24).map(|i| ((i % 5) as f32 - 2.0) / 3.0).collect();
        let (idx, lut) = lut8_from_f32(&shape, &data).unwrap();
        for c in 0..2 {
            for i in 0..12 {
                assert_eq!(lut[c][idx[c * 12 + i] as usize].to_bits(), data[c * 12 + i].to_bits());
            }
        }
        let many: Vec<f32> = (0..600).map(|i| i as f32).collect();
        assert!(lut8_from_f32(&[1, 1, 20, 30], &many).is_none());
    }

    #[test]
    fn grad_and_xor_inverse() {
        let (h, w) = (5usize, 7usize);
        let v: Vec<u8> = (0..2 * h * w).map(|i| (i * 37 % 251) as u8).collect();
        let mut d = grad_enc(&v, h, w);
        grad_dec(&mut d, h, w);
        assert_eq!(d, v);
        let f: Vec<f32> = (0..35).map(|i| (i as f32 * 0.37).sin()).collect();
        let back: Vec<f32> = xor_unshuf(&xor_shuf_f32(&f, 7), 4, 7).iter().map(|&b| f32::from_bits(b as u32)).collect();
        assert_eq!(back.iter().map(|x| x.to_bits()).collect::<Vec<_>>(), f.iter().map(|x| x.to_bits()).collect::<Vec<_>>());
    }

    #[test]
    fn f16_widening() {
        assert_eq!(f16_to_f32(0x3c00), 1.0);
        assert_eq!(f16_to_f32(0xc000), -2.0);
        assert_eq!(f16_to_f32(0x0001), 2f32.powi(-24));
        assert!(f16_to_f32(0x7e00).is_nan());
    }

    #[test]
    fn container_roundtrip() {
        let mut blobs = Blobs { area: Vec::new() };
        let t = TensorIn::f32_auto("x", &[1, 1, 2, 3], vec![0.0, 0.5, 1.0, 0.5, 0.0, 1.0]);
        assert!(matches!(t.data, Data::Lut8 { .. }));
        let d = encode_tensor(&t, &mut blobs, 3, false).unwrap();
        let body = pack(&json!({"inputs": [d]}), &blobs.area);
        let (h, area) = unpack(&body).unwrap();
        assert_eq!(h["inputs"][0]["codec"], "lut8-grad-zstd");
        assert_eq!(area.len(), blobs.area.len());
    }
}
