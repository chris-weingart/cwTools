//! cwMatte pre/post-processing vs the Python validation recipe (fixtures from
//! parity/make_matte_fixtures.py: numpy float32 ImageNet normalisation of all
//! 256 levels, numpy sigmoid).

use cwmatte::engine::{finish, normalise, prepare, Roi, IMAGENET_MEAN, IMAGENET_STD};
use serde_json::Value;

fn fx() -> Value {
    let p = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/imagenet_norm.json");
    serde_json::from_str(&std::fs::read_to_string(p).unwrap()).unwrap()
}

#[test]
fn imagenet_normalisation_is_bit_exact_to_numpy() {
    let f = fx();
    // One 256x1 image whose pixel k = (k, k, k): every level on every channel.
    let img: Vec<u8> = (0..=255u8).flat_map(|k| [k, k, k]).collect();
    let t = normalise(&img, 256, 1, IMAGENET_MEAN, IMAGENET_STD);
    for c in 0..3 {
        let want: Vec<f32> = f["norm"][c].as_array().unwrap().iter().map(|x| x.as_f64().unwrap() as f32).collect();
        let got = &t[c * 256..(c + 1) * 256];
        for k in 0..256 {
            assert_eq!(got[k].to_bits(), want[k].to_bits(), "channel {c} level {k}: {} vs {}", got[k], want[k]);
        }
    }
}

#[test]
fn nchw_layout() {
    // 2x1 image: pixel0 = (255,0,0), pixel1 = (0,255,0)
    let t = normalise(&[255, 0, 0, 0, 255, 0], 2, 1, [0.0; 3], [1.0; 3]);
    assert_eq!(t, vec![1.0, 0.0, 0.0, 1.0, 0.0, 0.0]);
}

#[test]
fn sigmoid_matches_numpy() {
    let f = fx();
    let logits: Vec<f32> = f["logits"].as_array().unwrap().iter().map(|x| x.as_f64().unwrap() as f32).collect();
    let want: Vec<f32> = f["sigmoid"].as_array().unwrap().iter().map(|x| x.as_f64().unwrap() as f32).collect();
    let got = finish(&logits, Some(true)).unwrap();
    for (g, w) in got.iter().zip(&want) {
        assert!((g - w).abs() <= 1e-7, "{g} vs {w}");
    }
}

#[test]
fn prepare_stretches_to_square_and_keys_on_content() {
    let (w, h) = (64usize, 20usize);
    let rgb: Vec<u8> = (0..w * h * 3).map(|i| (i % 251) as u8).collect();
    let a = prepare(&rgb, Roi::full(w, h), 32, 32);
    assert_eq!(a.square.len(), 32 * 32 * 3);
    let b = prepare(&rgb, Roi::full(w, h), 32, 32);
    assert_eq!(a.key, b.key);
    let mut rgb2 = rgb.clone();
    rgb2[100] = rgb2[100].wrapping_add(128);
    assert_ne!(prepare(&rgb2, Roi::full(w, h), 32, 32).key, a.key);
    assert_ne!(prepare(&rgb, Roi::full(w, h), 16, 16).key, a.key);
}
