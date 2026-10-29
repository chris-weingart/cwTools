//! Run a cwRefine core ONNX (`pixel_values` [1,4,h,w] -> `alphas` [1,1,h,w])
//! outside Flame, through Flame's bundled onnxruntime and the exact session
//! setup cwRefine uses (cwtools-core shared session, ModelKey from
//! `local::model_key` defaults). Lets a new core be checked against the
//! library cwRefine will load without a Flame GPU.
//!
//!   cargo run --release -p cwrefine --example core_run -- \
//!       --model core.onnx --input case.npy --out alphas.f32 \
//!       [--lib /opt/Autodesk/lib64/<ver>/libonnxruntime.so.1.22.0] [--cpu] [--tf32] [--runs 3]
//!       [--ws-capped] [--search heuristic|exhaustive|default]
//!
//! `--input` is a float32 C-order .npy of shape [1,4,h,w]; `--out` receives
//! the raw little-endian float32 alphas. Timings and on_gpu go to stdout.

use cwtools_core::ort_rt::{self, ConvSearch, Ep, ModelKey, Precision};
use ort::value::Tensor;

fn arg(args: &[String], name: &str) -> Option<String> {
    args.iter()
        .position(|a| a == name)
        .and_then(|i| args.get(i + 1).cloned())
}

/// Minimal .npy (v1/v2/v3) float32 C-order reader; returns (shape, data).
fn read_npy(path: &str) -> Result<(Vec<usize>, Vec<f32>), String> {
    let b = std::fs::read(path).map_err(|e| format!("{path}: {e}"))?;
    if b.len() < 10 || &b[..6] != b"\x93NUMPY" {
        return Err(format!("{path}: not a .npy"));
    }
    let (hlen, hstart) = if b[6] == 1 {
        (u16::from_le_bytes([b[8], b[9]]) as usize, 10)
    } else {
        (u32::from_le_bytes([b[8], b[9], b[10], b[11]]) as usize, 12)
    };
    let header = String::from_utf8_lossy(&b[hstart..hstart + hlen]).to_string();
    if !header.contains("'<f4'") || header.contains("'fortran_order': True") {
        return Err(format!(
            "{path}: need little-endian float32 C-order, got {header}"
        ));
    }
    let shape_s = header
        .split("'shape': (")
        .nth(1)
        .and_then(|s| s.split(')').next())
        .ok_or("npy: no shape")?;
    let shape: Vec<usize> = shape_s
        .split(',')
        .filter_map(|s| s.trim().parse().ok())
        .collect();
    let data: Vec<f32> = b[hstart + hlen..]
        .chunks_exact(4)
        .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
        .collect();
    if data.len() != shape.iter().product::<usize>() {
        return Err(format!("npy: {} values for shape {shape:?}", data.len()));
    }
    Ok((shape, data))
}

fn main() -> Result<(), String> {
    let args: Vec<String> = std::env::args().collect();
    let model = arg(&args, "--model").ok_or("--model required")?;
    let input = arg(&args, "--input").ok_or("--input required")?;
    let out = arg(&args, "--out");
    let lib = arg(&args, "--lib").unwrap_or_default(); // empty = cwtools.conf ort_lib / auto-detect
    let runs: usize = arg(&args, "--runs")
        .and_then(|s| s.parse().ok())
        .unwrap_or(1);
    let key = ModelKey {
        path: model,
        ep: if args.iter().any(|a| a == "--cpu") {
            Ep::Cpu
        } else {
            Ep::Cuda
        },
        precision: if args.iter().any(|a| a == "--tf32") {
            Precision::Tf32
        } else {
            Precision::Fp32
        },
        conv_max_workspace: !args.iter().any(|a| a == "--ws-capped"),
        conv_search: match arg(&args, "--search").as_deref() {
            Some("exhaustive") => ConvSearch::Exhaustive,
            Some("default") => ConvSearch::Default,
            _ => ConvSearch::Heuristic,
        },
    };

    let (shape, data) = read_npy(&input)?;
    if shape.len() != 4 || shape[0] != 1 || shape[1] != 4 {
        return Err(format!("input shape {shape:?}, expected [1,4,h,w]"));
    }
    let (h, w) = (shape[2], shape[3]);

    let shared = ort_rt::shared_session(&lib, &key)?;
    println!(
        "lib {lib}\nmodel {} on_gpu={} load_ms={:.0}",
        key.describe(),
        shared.on_gpu,
        shared.load_ms
    );
    let mut sess = shared.session.lock().map_err(|_| "session lock poisoned")?;
    let mut last = Vec::new();
    for r in 0..runs.max(1) {
        let t = Tensor::from_array((vec![1i64, 4, h as i64, w as i64], data.clone()))
            .map_err(|e| format!("tensor: {e}"))?;
        let t0 = std::time::Instant::now();
        let o = sess
            .run(ort::inputs!["pixel_values" => t])
            .map_err(|e| format!("run: {e}"))?;
        let ms = t0.elapsed().as_secs_f64() * 1000.0;
        let a = o.get("alphas").ok_or("no 'alphas' output")?;
        let (oshape, od) = a
            .try_extract_tensor::<f32>()
            .map_err(|e| format!("alphas: {e}"))?;
        println!("run {r}: {w}x{h} {ms:.1} ms out {oshape:?}");
        last = od.to_vec();
    }
    let nonfinite = last.iter().filter(|x| !x.is_finite()).count();
    println!("nonfinite {nonfinite}");
    if let Some(p) = out {
        let bytes: Vec<u8> = last.iter().flat_map(|x| x.to_le_bytes()).collect();
        std::fs::write(&p, bytes).map_err(|e| format!("{p}: {e}"))?;
    }
    Ok(())
}
