//! Parity + timing driver for the REAL render core (`render::solve_window`),
//! outside any OFX host. Feeds a plate + matte (PFM, bottom-up rows = OFX
//! order) as one RGBA buffer, prints per-stage times and an FNV-1a hash of the
//! raw output bits. The hash format matches the v0.2.0 bench harness
//! (cwfill-bench, `lib` variant) so a v0.3.0 hash can be compared against the
//! v0.2.0 hash for the same flags on the same machine — they must be equal.
//!
//! cargo run --release --example parity -- plate.pfm matte.pfm [--sb 0|1]
//!   [--om 0..3] [--despill S] [--dso 0|1] [--dth T] [--screen 0..4] [--tone E]
//!   [--cs 0..2] [--aff A] [--smul S] [--auto 0|1] [--q 0..2]
//!   [--threads N] [--reps N] [--cache-test D2]
//!
//! --cache-test D2: render once through the cache (miss), then again with
//! despill strength D2 (must hit), and check the hit's hash against a fresh
//! uncached render with D2.

use cwfill::ofx_api::OfxRectI;
use cwfill::params::Values;
use cwfill::render::{solve_window, CacheCtx, RenderInfo, View};

fn read_pfm(path: &str) -> (usize, usize, usize, Vec<f32>) {
    let bytes = std::fs::read(path).expect("read pfm");
    let mut pos = 0;
    let mut tok = || {
        while bytes[pos].is_ascii_whitespace() {
            pos += 1;
        }
        let s = pos;
        while !bytes[pos].is_ascii_whitespace() {
            pos += 1;
        }
        String::from_utf8_lossy(&bytes[s..pos]).to_string()
    };
    let magic = tok();
    let w: usize = tok().parse().unwrap();
    let h: usize = tok().parse().unwrap();
    let scale: f32 = tok().parse().unwrap();
    pos += 1;
    assert!(scale < 0.0, "expect little-endian pfm");
    let depth = if magic == "PF" { 3 } else { 1 };
    let data = bytes[pos..pos + w * h * depth * 4]
        .chunks_exact(4)
        .map(|b| f32::from_le_bytes([b[0], b[1], b[2], b[3]]))
        .collect();
    (w, h, depth, data)
}

/// params.rs defaults (v0.3.0).
fn defaults() -> Values {
    Values {
        self_mul_strength: 0.0,
        self_mul_mode: 1,
        soft_band_only: true,
        band_dilation: 2,
        pin_hard: true,
        pin_low: 0.002,
        pin_high: 0.998,
        regularization: 1e-5,
        quality: 2,
        gradient_weight: 1.0,
        image_affinity: 0.0,
        auto_solve: false,
        input_colorspace: 0,
        roi_padding: 64,
        output_mode: 0,
        despill_screen: 1,
        residual_despill: 0.0,
        despill_range: 0.5,
        despill_softness: 0.5,
        screen_colour: [0.0, 1.0, 0.0],
        despill_soft_only: false,
        despill_matte_thresh: 1.0,
        edge_tone: 0.0,
        edge_tone_spread: 32.0,
    }
}

fn fnv(v: &[f32]) -> u64 {
    let mut h = 0xcbf29ce484222325u64;
    for x in v {
        for byte in x.to_bits().to_le_bytes() {
            h ^= byte as u64;
            h = h.wrapping_mul(0x100000001b3);
        }
    }
    h
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let get = |k: &str, d: &str| -> String {
        args.iter().position(|a| a == k).map(|i| args[i + 1].clone()).unwrap_or_else(|| d.to_string())
    };
    let (w, h, d, plate) = read_pfm(&args[1]);
    let (mw, mh, md, matte) = read_pfm(&args[2]);
    assert!(d == 3 && md == 1 && mw == w && mh == h);
    let mut v = defaults();
    v.soft_band_only = get("--sb", "1") == "1";
    v.output_mode = get("--om", "0").parse().unwrap();
    v.residual_despill = get("--despill", "0").parse().unwrap();
    v.despill_soft_only = get("--dso", "0") == "1";
    v.despill_matte_thresh = get("--dth", "1.0").parse().unwrap();
    v.edge_tone = get("--tone", "0").parse().unwrap();
    v.edge_tone_spread = get("--tone-spread", "32").parse().unwrap();
    v.despill_screen = get("--screen", "1").parse().unwrap();
    v.input_colorspace = get("--cs", "0").parse().unwrap();
    v.image_affinity = get("--aff", "0").parse().unwrap();
    v.self_mul_strength = get("--smul", "0").parse().unwrap();
    v.auto_solve = get("--auto", "0") == "1";
    v.quality = get("--q", "2").parse().unwrap();
    if let Some(c) = args.iter().position(|a| a == "--custom") {
        let rgb: Vec<f32> = args[c + 1].split(',').map(|x| x.parse().unwrap()).collect();
        v.despill_screen = 4;
        v.screen_colour = [rgb[0], rgb[1], rgb[2]];
    }
    v.despill_range = get("--range", "0.5").parse().unwrap();
    v.despill_softness = get("--soft", "0.5").parse().unwrap();
    let threads: usize = get("--threads", "1").parse().unwrap();
    let reps: usize = get("--reps", "1").parse().unwrap();

    let mut src = vec![0f32; w * h * 4];
    for i in 0..w * h {
        src[i * 4..i * 4 + 3].copy_from_slice(&plate[i * 3..i * 3 + 3]);
        src[i * 4 + 3] = matte[i];
    }
    let win = OfxRectI { x1: 0, y1: 0, x2: w as i32, y2: h as i32 };
    let view = |buf: &mut [f32]| View { data: buf.as_mut_ptr() as *mut u8, bounds: win, row_bytes: (w * 16) as i32, n_comps: 4 };
    let src_v = view(&mut src);
    let mut out = vec![0f32; w * h * 4];

    let run = |v: &Values, out: &mut [f32], ctx: Option<CacheCtx>| -> RenderInfo {
        let out_v = view(out);
        solve_window(&src_v, &out_v, &win, v, ctx, threads).1
    };
    let ms = |s: f64| s * 1000.0;
    let line = |tag: &str, t: &RenderInfo, hash: u64| {
        println!(
            "{}x{} sb={} om={} despill={} dso={} cs={} aff={} threads={} {} crop={}x{} | gather {:.1} band {:.1} crop+enc {:.1} SOLVE {:.1} output {:.1} | TOTAL {:.1} ms{} | out_hash {:016x}",
            w, h, v.soft_band_only as u8, v.output_mode, v.residual_despill, v.despill_soft_only as u8,
            v.input_colorspace, v.image_affinity, threads, tag, t.crop.0, t.crop.1,
            ms(t.gather), ms(t.band), ms(t.crop_encode), ms(t.solve), ms(t.output), ms(t.total),
            if t.cache_hit { " (cache HIT)" } else { "" }, hash
        );
    };

    if let Some(i) = args.iter().position(|a| a == "--cache-test") {
        let d2: f32 = args[i + 1].parse().unwrap();
        let ctx = Some(CacheCtx { instance: 1, time: 1001.0 });
        let t_miss = run(&v, &mut out, ctx);
        line("cache-miss", &t_miss, fnv(&out));
        let mut v2 = v;
        v2.residual_despill = d2;
        let mut best: Option<RenderInfo> = None;
        for _ in 0..reps.max(1) {
            let t = run(&v2, &mut out, ctx);
            assert!(t.cache_hit, "despill-only change must hit the cache");
            if best.as_ref().map_or(true, |b| t.total < b.total) {
                best = Some(t);
            }
        }
        let hit_hash = fnv(&out);
        let t_fresh = run(&v2, &mut out, None);
        let fresh_hash = fnv(&out);
        println!("  despill {} -> {}: hit {:.1} ms vs fresh {:.1} ms | hit {:016x} fresh {:016x} {}",
            v.residual_despill, d2, ms(best.unwrap().total), ms(t_fresh.total), hit_hash, fresh_hash,
            if hit_hash == fresh_hash { "MATCH" } else { "MISMATCH" });
        assert_eq!(hit_hash, fresh_hash, "cache hit must reproduce a fresh render");
        return;
    }

    let mut best: Option<RenderInfo> = None;
    for _ in 0..reps.max(1) {
        let t = run(&v, &mut out, None);
        if best.as_ref().map_or(true, |b| t.total < b.total) {
            best = Some(t);
        }
    }
    line("v0.3", &best.unwrap(), fnv(&out));
    if let Some(i) = args.iter().position(|a| a == "--dump") {
        // Raw little-endian f32 RGBA, OFX (bottom-up) row order.
        let bytes: Vec<u8> = out.iter().flat_map(|x| x.to_le_bytes()).collect();
        std::fs::write(&args[i + 1], bytes).expect("dump");
    }
}

