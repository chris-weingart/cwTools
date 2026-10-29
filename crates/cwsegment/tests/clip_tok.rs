//! cwSegment's CLIP tokenizer vs the facebook/sam3 tokenizer (transformers), on
//! tests/data/clip_tok_golden.json. Needs the tokenizer files (SAM License, not in
//! the repo): CWSEGMENT_TOKENIZER_DIR=<dir with vocab.json + merges.txt>; skips
//! without it.
use cwsegment::clip_tok::ClipTokenizer;

#[test]
fn matches_transformers() {
    let Ok(dir) = std::env::var("CWSEGMENT_TOKENIZER_DIR") else {
        eprintln!("skipped: set CWSEGMENT_TOKENIZER_DIR");
        return;
    };
    let tok = ClipTokenizer::from_dir(std::path::Path::new(&dir)).unwrap();
    let golden: serde_json::Value =
        serde_json::from_str(include_str!("data/clip_tok_golden.json")).unwrap();
    let mut bad = Vec::new();
    let cases = golden.as_array().unwrap();
    for c in cases {
        let text = c["text"].as_str().unwrap();
        let want: Vec<i64> = c["ids"].as_array().unwrap().iter().map(|v| v.as_i64().unwrap()).collect();
        let n = c["n"].as_u64().unwrap() as usize;
        let (ids, mask) = tok.encode(text);
        if ids != want || mask.iter().sum::<i64>() as usize != n {
            bad.push(format!("{text:?}: got {:?} want {:?}", &ids[..n.max(4).min(32)], &want[..n.max(4).min(32)]));
        }
    }
    assert!(bad.is_empty(), "{} of {} differ:\n{}", bad.len(), cases.len(), bad.join("\n"));
}
