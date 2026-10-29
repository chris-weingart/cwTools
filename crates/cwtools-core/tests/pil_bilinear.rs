//! `resample` vs Pillow BILINEAR (fixtures from parity/make_matte_fixtures.py,
//! Pillow 11.3): 8-bit must be bit-exact, mode "F" within float rounding.
//!
//! Optional full-size check on real plates: set CWMATTE_PIL_DIR to a
//! folder with plate_1029.ppm / moto.ppm and their *_pil2048.ppm /
//! *_pil1536.ppm / *_pil1024.ppm / *_pil768.ppm Pillow resizes, then
//! `cargo test --release -p cwtools-core --test pil_bilinear -- --ignored`.

use cwtools_core::resample::{resize_f32, resize_u8, Filter};
use serde_json::Value;

fn fixtures() -> Value {
    let p = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/pil_bilinear.json");
    serde_json::from_str(&std::fs::read_to_string(p).unwrap()).unwrap()
}
fn us(v: &Value, k: &str) -> usize {
    v[k].as_u64().unwrap() as usize
}

#[test]
fn u8_bilinear_is_bit_exact_to_pillow() {
    let f = fixtures();
    for c in f["u8"].as_array().unwrap() {
        let src: Vec<u8> = c["src"].as_array().unwrap().iter().map(|x| x.as_u64().unwrap() as u8).collect();
        let want: Vec<u8> = c["out"].as_array().unwrap().iter().map(|x| x.as_u64().unwrap() as u8).collect();
        let (sw, sh, ch, dw, dh) = (us(c, "sw"), us(c, "sh"), us(c, "c"), us(c, "dw"), us(c, "dh"));
        let got = resize_u8(&src, sw, sh, ch, dw, dh, Filter::Bilinear);
        let bad = got.iter().zip(&want).filter(|(a, b)| a != b).count();
        assert_eq!(bad, 0, "{sw}x{sh}x{ch} -> {dw}x{dh}: {bad} values differ");
    }
}

#[test]
fn f32_bilinear_matches_pillow() {
    let f = fixtures();
    for c in f["f32"].as_array().unwrap() {
        let src: Vec<f32> = c["src"].as_array().unwrap().iter().map(|x| x.as_f64().unwrap() as f32).collect();
        let want: Vec<f32> = c["out"].as_array().unwrap().iter().map(|x| x.as_f64().unwrap() as f32).collect();
        let (sw, sh, dw, dh) = (us(c, "sw"), us(c, "sh"), us(c, "dw"), us(c, "dh"));
        let got = resize_f32(&src, sw, sh, dw, dh, Filter::Bilinear);
        let max = got.iter().zip(&want).map(|(a, b)| (a - b).abs()).fold(0f32, f32::max);
        assert!(max <= 1e-6, "{sw}x{sh} -> {dw}x{dh}: max diff {max}");
    }
}

fn read_ppm(p: &std::path::Path) -> (usize, usize, Vec<u8>) {
    let b = std::fs::read(p).unwrap();
    // P6\n<w> <h>\n255\n (Pillow writes single spaces / newlines)
    let mut fields = Vec::new();
    let mut i = 0;
    while fields.len() < 4 {
        while b[i].is_ascii_whitespace() {
            i += 1;
        }
        let s = i;
        while !b[i].is_ascii_whitespace() {
            i += 1;
        }
        fields.push(String::from_utf8_lossy(&b[s..i]).to_string());
    }
    assert_eq!(fields[0], "P6");
    let (w, h): (usize, usize) = (fields[1].parse().unwrap(), fields[2].parse().unwrap());
    (w, h, b[i + 1..i + 1 + w * h * 3].to_vec())
}

#[test]
#[ignore]
fn real_plates_bit_exact_to_pillow() {
    let dir = std::path::PathBuf::from(std::env::var("CWMATTE_PIL_DIR").expect("set CWMATTE_PIL_DIR"));
    for stem in ["plate_1029", "moto"] {
        let (w, h, px) = read_ppm(&dir.join(format!("{stem}.ppm")));
        for s in [2048usize, 1536, 1024, 768] {
            let (_, _, want) = read_ppm(&dir.join(format!("{stem}_pil{s}.ppm")));
            let t = std::time::Instant::now();
            let got = resize_u8(&px, w, h, 3, s, s, Filter::Bilinear);
            let bad = got.iter().zip(&want).filter(|(a, b)| a != b).count();
            eprintln!("{stem} {w}x{h} -> {s}: {bad} differing bytes, {:?}", t.elapsed());
            assert_eq!(bad, 0);
        }
    }
}
